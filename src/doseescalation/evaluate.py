import math
import os
import plotly.graph_objects as go
from collections import Counter
from doseescalation.dose_escalator import DoseEscalatorBase, SEEDADoseEscalator, SEEDAOriginalDoseEscalator, SEEDAPlateauDoseEscalator
from doseescalation.simulated_env import SimulatedEnv
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Sequence, Tuple


DEFAULT_COLORS = [
    '#1f77b4',  # muted blue
    '#ff7f0e',  # safety orange
    '#2ca02c',  # cooked asparagus green
    '#d62728',  # brick red
    '#9467bd',  # muted purple
    '#8c564b',  # chestnut brown
    '#e377c2',  # raspberry yogurt pink
    '#7f7f7f',  # middle gray
    '#bcbd22',  # curry yellow-green
    '#17becf'   # blue-teal
]


def simulate(
    cohort_sizes: Sequence[int],
    dose_escalator: DoseEscalatorBase,
    env: SimulatedEnv,
    n_efficate: int = 0,
    efficacy_env: Optional[SimulatedEnv] = None,
) -> Tuple[List, List, List]:
    """
    Run a simulated trial, returning three per-cohort sequences:

    - ``allocations``: the dose actually given to each cohort (the real
      exploration/trial policy). Use this for allocation-percentage and
      toxicity-burden metrics.
    - ``recommendations``: the MTD the algorithm would declare if the trial
      stopped at that cohort (its best current estimate). Use this for
      correct-dose-selection metrics.
    - ``n_dles``: the number of dose-limiting events observed at the
      allocated dose for each cohort.

    For CRM and 3 + 3 the recommendation equals the allocation (no
    exploration/recommendation split); for UCB / SEEDA / SEEDA Plateau the two
    differ, and the recommendation is read via non-training mode.

    ``efficacy_env`` (optional, only used by SEEDA / SEEDA Plateau) is an env
    that samples the number of efficacy responders for the allocated dose, e.g.
    ``Binomial(cohort, q_dose)``. If omitted, the constant ``n_efficate`` is
    used instead (dose-independent efficacy).
    """
    _needs_efficacy = isinstance(
        dose_escalator,
        (SEEDADoseEscalator, SEEDAOriginalDoseEscalator, SEEDAPlateauDoseEscalator)
    )
    # UCB / SEEDA / SEEDA Plateau expose a separate recommendation through
    # non-training mode, while CRM / 3 + 3 have no such split (recommend == allocate):
    _can_recommend = hasattr(dose_escalator, "train")
    allocations = []
    recommendations = []
    n_dles = []
    for cohort in cohort_sizes:
        # Recommendation is the MTD the algorithm would declare if it stopped now:
        if _can_recommend:
            dose_escalator.train(False)
            recommendations.append(dose_escalator.propose())
            dose_escalator.train(True)
        else:
            recommendations.append(dose_escalator.propose())

        # Allocation is the dose actually given to this cohort (exploration policy):
        dose_level_index = dose_escalator.propose()
        allocations.append(dose_level_index)
        n_dle = env(dose_level_index, cohort)
        n_dles.append(n_dle)

        if _needs_efficacy:
            # If an efficacy_env is given, sample this cohort's responders from
            # it (dose-dependent), otherwise fall back to the constant n_efficate:
            n_eff = (
                efficacy_env(dose_level_index, cohort)
                if efficacy_env is not None else n_efficate
            )
            dose_escalator.update(dose_level_index, cohort, n_dle, n_eff)
        else:
            dose_escalator.update(dose_level_index, cohort, n_dle)
    return allocations, recommendations, n_dles


def _get_env_algos(
    dose_proposals_map: Dict[str, Dict[str, List[int]]],
) -> Tuple[List[str], List[str]]:
    env_names = list(dose_proposals_map.keys())
    algo_names = list(dose_proposals_map[env_names[0]].keys())
    for env_name in env_names:
        if set(algo_names) != set(dose_proposals_map[env_name].keys()):
            raise ValueError(
                "All environments must have the same set of algorithms"
            )
    return env_names, algo_names


def plot_dose_proposals(
    n_dose_levels: int,
    n_trials: int,
    dose_proposals_map: Dict[str, Dict[str, List[int]]],
    correct_mtds: Dict[str, int],
    mtds: Optional[Dict[str, int]] = None,
    unit_width: Optional[int] = 200,
    unit_height: Optional[int] = 200,
    title_text: Optional[str] = None,
    show_fig: Optional[bool] = False,
    img_path: Optional[str] = None,
    html_path: Optional[str] = None,
) -> None:
    """
    Plot the dose proposals for each trial design
    under each environment. Please note the input
    dose_proposals_map should have the following structure:
    {
        "a = 0.4": {
            "3 + 3": [0] * 100 + [1] * 17 + [2] * 3,
            "CRM": [0] * 80 + [1] * 30 + [2] * 10,
            "DP-SL": [0] * 82 + [1] * 31 + [2] * 7,
        },
        "Cohort 1": ...
    }
    i.e. environments on the top level and
    algorithms on the second level.

    Please also feel free to copy and modify this for your own use.
    """
    env_names, algo_names = _get_env_algos(dose_proposals_map)

    # initialize figure with subplots
    fig = make_subplots(
        rows=len(algo_names),
        cols=len(env_names),
        shared_xaxes=True,
        shared_yaxes=True,
        column_titles=env_names,
        row_titles=algo_names,
        x_title="Dose Level",
        y_title="Number Of Trials",
    )

    # Row titles default to fully vertical (textangle=90), which makes longer
    # algorithm names (e.g. "SEEDA Plateau (Modified L1)") overlap with
    # neighbouring rows. A 45-degree angle keeps every label readable within
    # the default row height:
    for annotation in fig.layout.annotations:
        if annotation.text in algo_names:
            annotation.textangle = 45

    # Add traces for each algorithm in each environment:
    #   - grey  : all proposals (base layer)
    #   - yellow: toxicity MTD (from `mtds`, when provided and differs from optimal)
    #   - green : optimal biological dose (from `correct_mtds`, top layer)
    shown_other = shown_opt = shown_tox_mtd = False
    for row_idx, algo in enumerate(algo_names):
        for col_idx, env in enumerate(env_names):
            proposals = dose_proposals_map[env][algo]
            # correct_mtds[env] may be a single dose (shared by every algorithm)
            # or a {algorithm: dose} dict, for per-algorithm correct doses.
            correct = correct_mtds[env]
            if isinstance(correct, dict):
                correct = correct[algo]
            correct_props = [m for m in proposals if m == correct]

            # Resolve the toxicity MTD for this env:
            tox_mtd = mtds[env] if mtds is not None else None
            if isinstance(tox_mtd, dict):
                tox_mtd = tox_mtd[algo]
            tox_mtd_props = [m for m in proposals if m == tox_mtd] if tox_mtd is not None else []

            # Base layer — all doses in grey:
            fig.add_trace(
                go.Histogram(
                    x=proposals,
                    name="Other",
                    marker_color='#7f7f7f',
                    showlegend=not shown_other,
                    xbins=dict(start=-0.5, end=n_dose_levels - 0.5, size=1),
                ),
                row=row_idx + 1,
                col=col_idx + 1,
            )

            # Yellow overlay — toxicity MTD (only when it differs from the optimal dose):
            if tox_mtd is not None and tox_mtd != correct:
                fig.add_trace(
                    go.Histogram(
                        x=tox_mtd_props,
                        name="Toxicity MTD",
                        marker_color='#FFD700',
                        showlegend=not shown_tox_mtd,
                        xbins=dict(start=-0.5, end=n_dose_levels - 0.5, size=1),
                    ),
                    row=row_idx + 1,
                    col=col_idx + 1,
                )
                shown_tox_mtd = shown_tox_mtd or len(tox_mtd_props) > 0

            # Green overlay — optimal biological dose (top layer):
            fig.add_trace(
                go.Histogram(
                    x=correct_props,
                    name="Optimal dose",
                    marker_color=DEFAULT_COLORS[2],
                    showlegend=not shown_opt,
                    xbins=dict(start=-0.5, end=n_dose_levels - 0.5, size=1),
                ),
                row=row_idx + 1,
                col=col_idx + 1,
            )

            fig.update_xaxes(
                range=[-0.5, n_dose_levels - 0.5],
                # Display doses as 1..n (paper's numbering) while the
                # underlying data/bins stay 0-indexed:
                tickmode="array",
                tickvals=list(range(n_dose_levels)),
                ticktext=[str(i + 1) for i in range(n_dose_levels)],
                row=row_idx + 1,
                col=col_idx + 1,
            )
            fig.update_yaxes(
                range=[0, n_trials],
                row=row_idx + 1,
                col=col_idx + 1,
            )

            shown_other = shown_other or len(proposals) > 0
            shown_opt = shown_opt or len(correct_props) > 0

    title_text = title_text or "Number of dose allocations"
    # Extra right margin so the 45-degree row-title labels aren't clipped at
    # the canvas edge; added on top of (not eating into) the plot width:
    row_label_margin = 200
    fig.update_layout(
        autosize=False,
        width=unit_width * (len(env_names) + 1) + row_label_margin,
        height=unit_height * len(algo_names),
        barmode="overlay",
        title_text=title_text,
        legend_title_text="Dose",
        # Pushed further right (x=1.3) so it clears the top row's row-title
        # label (e.g. "3 + 3"), which otherwise sits directly under it:
        legend=dict(x=1.3, xanchor="left", y=1, yanchor="top"),
        margin=dict(r=row_label_margin),
    )

    if img_path:
        os.makedirs(os.path.dirname(img_path) or ".", exist_ok=True)
        fig.write_image(img_path)
    if html_path:
        os.makedirs(os.path.dirname(html_path) or ".", exist_ok=True)
        fig.write_html(html_path)
    if show_fig:
        fig.show()


def plot_acc_progression(
    n_cohorts: int,
    dose_proposals_map: Dict[str, Dict[str, List[int]]],
    correct_mtds: Dict[str, int],
    unit_width: Optional[int] = 600,
    unit_height: Optional[int] = 400,
    title_text: Optional[str] = None,
    show_fig: Optional[bool] = False,
    img_path: Optional[str] = None,
    html_path: Optional[str] = None,
):
    """
    Plot the dose proposal accuracies for each trial design
    under each environment. Please note the input
    dose_proposals_map should have the following structure:
    {
        "a = 0.4": {
            "Cohort 1": {
                "3 + 3": [0] * 100 + [1] * 17 + [2] * 3,
                "CRM": [0] * 80 + [1] * 30 + [2] * 10,
                "DP-SL": [0] * 82 + [1] * 31 + [2] * 7,
            }
            "Cohort 2": {...},
            ...,
        },
        "a = 1.3": ...
    }
    i.e. environments on the top level, cohort numbers on the
    second level and algorithms on the third level.

    Please also feel free to copy and modify this for your own use.
    """

    a, _ = _get_env_algos(dose_proposals_map)
    cohort_names, algos = _get_env_algos(list(dose_proposals_map.values())[0])

    n_rows = round(math.sqrt(len(a)))
    n_cols = math.ceil(len(a) / n_rows)

    # initialize figure with subplots
    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        shared_xaxes=True,
        shared_yaxes=True,
        x_title="Cohort Number",
        y_title="Accuracy",
        subplot_titles=a,
    )

    for a_idx, a in enumerate(a):
        row_idx = a_idx // n_cols
        col_idx = a_idx % n_cols
        correct_for_a = correct_mtds[a]
        for algo_idx, algo in enumerate(algos):
            # correct_for_a may be a single dose or a {algorithm: dose} dict.
            mtd = (correct_for_a[algo] if isinstance(correct_for_a, dict)
                   else correct_for_a)
            accs = []
            for cohort in cohort_names:
                dlis = dose_proposals_map[a][cohort][algo]
                acc = sum([dli == mtd for dli in dlis]) / len(dlis)
                accs.append(acc)

            fig.add_trace(
                go.Scatter(
                    x=list(range(n_cohorts)),
                    y=accs,
                    name=algo,
                    mode='lines',
                    marker_color=DEFAULT_COLORS[algo_idx],
                    showlegend=a_idx == 0,
                ),
                row=row_idx + 1, col=col_idx + 1,
            )

    title_text = title_text or "Dose Proposal Accuracy Progression"

    fig.update_layout(
        autosize=False,
        width=unit_width * n_cols,
        height=unit_height * n_rows,
        title_text=title_text,
        legend_title_text="Design",
    )

    if img_path:
        os.makedirs(os.path.dirname(img_path) or ".", exist_ok=True)
        fig.write_image(img_path)
    if html_path:
        os.makedirs(os.path.dirname(html_path) or ".", exist_ok=True)
        fig.write_html(html_path)
    if show_fig:
        fig.show()


def plot_n_dles(
    n_dle_map: Dict[str, Dict[str, List[int]]],
    unit_height: Optional[int] = 200,
    title_text: Optional[str] = None,
    show_fig: Optional[bool] = False,
    img_path: Optional[str] = None,
    html_path: Optional[str] = None,
) -> None:
    """
    Plot the n_dles for each trial design.
    Please note the input n_dle_map should
    have the following structure:
    {
        "a = 0.4": {
            "3 + 3": [1, 2, 4, 2, 1, 2, 4, 3, ...],
            "CRM": ...,
            "DP-SL": ...,
        },
        "a = 1.0": ...
    }
    i.e. environments on the top level and
    algorithms on the second level.

    Please also feel free to copy and modify this for your own use.
    """
    env_names, algo_names = _get_env_algos(n_dle_map)
    max_n_dle = max([
        n_dle
        for env_map in n_dle_map.values()
        for n_dles in env_map.values()
        for n_dle in n_dles
    ])

    # initialize figure with subplots
    fig = make_subplots(
        rows=len(env_names),
        shared_xaxes=True,
        row_titles=env_names,
        x_title="Number Of DLEs",
        y_title="Number Of Trials",
    )

    # add traces for each algorithm in each environment
    max_y = 0
    for row_idx, env in enumerate(env_names):
        for algo_idx, algo in enumerate(algo_names):
            binned_data = sorted(Counter(n_dle_map[env][algo]).items())
            values = [data[1] for data in binned_data]
            max_y = max(max_y, max(values))
            fig.add_trace(
                go.Scatter(
                    x=[data[0] for data in binned_data],
                    y=values,
                    name=algo,
                    mode='lines+markers',
                    marker_color=DEFAULT_COLORS[algo_idx],
                    marker_opacity=0.5,
                    line_width=0.75,
                    showlegend=row_idx == 0,
                ),
                row=row_idx + 1, col=1,
            )
        fig.update_xaxes(
            range=[-0.5, max_n_dle + 0.5],
            dtick=1,
            row=row_idx + 1, col=1,
        )
    for row_idx in range(len(env_names)):
        fig.update_yaxes(
            range=[0, math.ceil(max_y / 10) * 10],
            row=row_idx + 1, col=1,
        )

    title_text = (
        title_text or
        "Number of DLEs for each trial design"
    )
    fig.update_layout(
        autosize=False,
        width=1200,
        height=unit_height * len(algo_names),
        title_text=title_text,
        legend_title_text="Design",
    )

    if img_path:
        os.makedirs(os.path.dirname(img_path) or ".", exist_ok=True)
        fig.write_image(img_path)
    if html_path:
        os.makedirs(os.path.dirname(html_path) or ".", exist_ok=True)
        fig.write_html(html_path)
    if show_fig:
        fig.show()
