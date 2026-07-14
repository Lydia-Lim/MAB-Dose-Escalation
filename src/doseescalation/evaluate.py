import math
import os
import numpy as np
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
) -> Tuple[List, List, List, List]:
    """
    Run a simulated trial, returning four per-cohort sequences:

    - ``allocations``: the dose actually given to each cohort (the real
      exploration/trial policy). Use this for allocation-percentage and
      toxicity-burden metrics.
    - ``recommendations``: the MTD the algorithm would declare if the trial
      stopped at that cohort (its best current estimate). Use this for
      correct-dose-selection metrics.
    - ``n_dles``: the number of dose-limiting events observed at the
      allocated dose for each cohort.
    - ``safe_sets``: the design's per-dose safety classification at each cohort
      (a boolean list per cohort, or ``None`` for designs like 3 + 3 that have
      no safety model). Use this for the Type I / II (false-alarm /
      miss-detection) error metrics of Figure 1 in the SEEDA paper.

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
    # Designs that expose a per-dose safety classification (SEEDA admissible set,
    # or estimated toxicity <= TTL for UCB / CRM); 3 + 3 has none:
    _has_safe = hasattr(dose_escalator, "safe_doses")
    allocations = []
    recommendations = []
    n_dles = []
    safe_sets = []
    for cohort in cohort_sizes:
        # Recommendation is the MTD the algorithm would declare if it stopped now:
        if _can_recommend:
            dose_escalator.train(False)
            recommendations.append(dose_escalator.propose())
            dose_escalator.train(True)
        else:
            recommendations.append(dose_escalator.propose())

        # Per-dose safety classification at this cohort (for Type I / II error
        # rates); None for designs without a safety model (e.g. 3 + 3):
        safe_sets.append(list(dose_escalator.safe_doses()) if _has_safe else None)

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
    return allocations, recommendations, n_dles, safe_sets


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
    # the canvas edge; added on top of (not eating into) the plot width. Sized
    # for the longest algorithm labels (e.g. "SEEDA Plateau (Two-sided,
    # decoupled, NoLog)"):
    row_label_margin = 340
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

    # Match the sample-efficiency plot's size, with the legend moved outside to
    # the right so every design stays visible:
    fig.update_layout(
        autosize=False,
        width=unit_width * n_cols + 350,
        height=unit_height * n_rows + 150,
        title_text=title_text,
        legend_title_text="Design",
        legend=dict(x=1.02, xanchor="left", y=1, yanchor="top"),
        margin=dict(r=340),
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


def plot_sample_efficiency(
    n_cohorts: int,
    cohort_size: int,
    dose_proposals_map: Dict[str, Dict[str, Dict[str, List[int]]]],
    correct_mtds: Dict[str, int],
    accuracy_step: float = 0.02,
    min_patients: int = 6,
    algos: Optional[List[str]] = None,
    unit_width: Optional[int] = 600,
    unit_height: Optional[int] = 400,
    title_text: Optional[str] = None,
    show_fig: Optional[bool] = False,
    img_path: Optional[str] = None,
    html_path: Optional[str] = None,
):
    """
    Reproduce Figure 3 of the SEEDA paper: the minimum number of trial
    participants needed to reach a given recommendation accuracy. This is the
    inverse of the accuracy-vs-cohort curve in ``plot_acc_progression``: for each
    target accuracy alpha, y = min number of patients whose cohort's
    recommendation accuracy first reaches alpha.

    Recommendation accuracy is the paper's successful-recommendation probability
    P[d_hat(n) = k*] (Section 2.2, Corollary 2), estimated over the trials at
    each n. Figure 3 is exactly the inverse of that accuracy-vs-n curve: for a
    target accuracy alpha, the minimum n at which the ensemble accuracy first
    reaches alpha. Recruitment starts from a minimum of ``min_patients`` patients
    (the paper's early-stopping setup, Section 5.1.3) -- no per-trial stopping
    rule is applied.

    ``dose_proposals_map`` has the same structure as ``plot_acc_progression``:
    {env: {cohort: {algo: [recommendations across trials]}}}. Pass ``algos`` to
    restrict which designs are drawn (the paper excludes the MTD-only designs
    3 + 3 / CRM here, since sample efficiency targets the optimal dose).
    """
    envs, _ = _get_env_algos(dose_proposals_map)
    cohort_names, all_algos = _get_env_algos(list(dose_proposals_map.values())[0])
    algos = algos or all_algos

    n_rows = round(math.sqrt(len(envs)))
    n_cols = math.ceil(len(envs) / n_rows)

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        shared_xaxes=True,
        shared_yaxes=True,
        x_title="Recommendation Accuracy",
        y_title="Number of Patients",
        subplot_titles=envs,
    )

    # Accuracy targets from 0 up to (but not including) 1:
    n_steps = int(round(1 / accuracy_step))
    accuracy_grid = [i * accuracy_step for i in range(n_steps)]

    for env_idx, env in enumerate(envs):
        row_idx = env_idx // n_cols
        col_idx = env_idx % n_cols
        correct_for_env = correct_mtds[env]
        for algo_idx, algo in enumerate(algos):
            mtd = (correct_for_env[algo] if isinstance(correct_for_env, dict)
                   else correct_for_env)
            # Recommendation accuracy after each cohort:
            accs = []
            for cohort in cohort_names:
                dlis = dose_proposals_map[env][cohort][algo]
                accs.append(sum(dli == mtd for dli in dlis) / len(dlis))
            # Cumulative patients seen by the end of each cohort:
            patients = [(t + 1) * cohort_size for t in range(len(accs))]

            # For each target accuracy, the fewest patients that first reach it:
            xs, ys = [], []
            for target in accuracy_grid:
                reached = [patients[t] for t in range(len(accs))
                           if patients[t] >= min_patients and accs[t] >= target]
                if reached:
                    xs.append(target)
                    ys.append(min(reached))

            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=ys,
                    name=algo,
                    mode='lines',
                    marker_color=DEFAULT_COLORS[algo_idx % len(DEFAULT_COLORS)],
                    showlegend=env_idx == 0,
                ),
                row=row_idx + 1, col=col_idx + 1,
            )

    title_text = title_text or (
        "Sample efficiency: patients to reach a given recommendation accuracy"
    )

    # Larger canvas with the legend moved outside to the right, so the many
    # designs stay readable:
    fig.update_layout(
        autosize=False,
        width=unit_width * n_cols + 350,
        height=unit_height * n_rows + 150,
        title_text=title_text,
        legend_title_text="Design",
        legend=dict(x=1.02, xanchor="left", y=1, yanchor="top"),
        margin=dict(r=340),
    )

    if img_path:
        os.makedirs(os.path.dirname(img_path) or ".", exist_ok=True)
        fig.write_image(img_path)
    if html_path:
        os.makedirs(os.path.dirname(html_path) or ".", exist_ok=True)
        fig.write_html(html_path)
    if show_fig:
        fig.show()


def plot_efficacy_and_violation(
    n_cohorts: int,
    n_trials: int,
    alloc_map: Dict[str, Dict[str, List[int]]],
    efficacy_probs: Sequence[float],
    toxicity_probs: Sequence[float],
    ttl: float,
    algos: Optional[List[str]] = None,
    unit_width: Optional[int] = 600,
    unit_height: Optional[int] = 400,
    title_text: Optional[str] = None,
    show_fig: Optional[bool] = False,
    img_path: Optional[str] = None,
    html_path: Optional[str] = None,
):
    """
    Reproduce Figure 2 of the SEEDA paper: efficacy per patient (left) and the
    safety-violation percentage (right), versus the number of cohorts.

    - Efficacy per patient: the cumulative mean of the allocated doses' true
      efficacy, averaged over trials. This conforms to the paper's "effective
      treatment" objective (Section 2.2): the cumulative efficacy sum_t X_t per
      patient. We use the expectation E[X_t] = q_{I(t)}, which equals the paper's
      observed curve in expectation over trials (but is smoother).
    - Safety violation %: the percentage of trials in which the cumulative mean
      true toxicity of the allocated doses exceeds the MTD threshold ``ttl``.
      This conforms to the paper's safety constraint (Section 2.2, Eq. (2)),
      proven for the average true toxicity in Theorem 2, i.e.
      P[(1/t) sum_{s<=t} p_{I(s)} > theta].

    ``alloc_map`` is the pooled per-cohort allocation map {env: {algo: [dose,
    ...]}}, flattened in trial-then-cohort order (as produced by extending each
    trial's allocations); it is reshaped to (n_trials, n_cohorts) here.
    """
    envs, all_algos = _get_env_algos(alloc_map)
    algos = algos or all_algos
    eff = np.asarray(efficacy_probs, dtype=float)
    tox = np.asarray(toxicity_probs, dtype=float)
    cohort_ax = np.arange(1, n_cohorts + 1)

    fig = make_subplots(
        rows=len(envs),
        cols=2,
        shared_xaxes=True,
        column_titles=["Efficacy per Patient", "Safety Violation (%)"],
        row_titles=envs if len(envs) > 1 else None,
        x_title="Number of Cohorts",
    )

    for env_idx, env in enumerate(envs):
        for algo_idx, algo in enumerate(algos):
            allocs = np.asarray(alloc_map[env][algo]).reshape(n_trials, n_cohorts)
            # Cumulative mean over cohorts, per trial:
            eff_cum = np.cumsum(eff[allocs], axis=1) / cohort_ax
            tox_cum = np.cumsum(tox[allocs], axis=1) / cohort_ax
            eff_curve = eff_cum.mean(axis=0)
            viol_curve = (tox_cum > ttl).mean(axis=0) * 100.0

            for col, curve in ((1, eff_curve), (2, viol_curve)):
                fig.add_trace(
                    go.Scatter(
                        x=list(range(n_cohorts)),
                        y=curve,
                        name=algo,
                        mode='lines',
                        marker_color=DEFAULT_COLORS[algo_idx % len(DEFAULT_COLORS)],
                        legendgroup=algo,
                        showlegend=(env_idx == 0 and col == 1),
                    ),
                    row=env_idx + 1, col=col,
                )

    fig.update_layout(
        autosize=False,
        width=unit_width * 2 + 350,
        height=unit_height * len(envs) + 150,
        title_text=title_text or "Efficacy per Patient and Safety Violation",
        legend_title_text="Design",
        legend=dict(x=1.02, xanchor="left", y=1, yanchor="top"),
        margin=dict(r=340),
    )

    if img_path:
        os.makedirs(os.path.dirname(img_path) or ".", exist_ok=True)
        fig.write_image(img_path)
    if html_path:
        os.makedirs(os.path.dirname(html_path) or ".", exist_ok=True)
        fig.write_html(html_path)
    if show_fig:
        fig.show()


def plot_error_rates(
    n_cohorts: int,
    safe_map: Dict[str, Dict[str, Dict[str, List[Optional[List[bool]]]]]],
    toxicity_probs: Sequence[float],
    ttl: float,
    algos: Optional[List[str]] = None,
    unit_width: Optional[int] = 600,
    unit_height: Optional[int] = 400,
    title_text: Optional[str] = None,
    show_fig: Optional[bool] = False,
    img_path: Optional[str] = None,
    html_path: Optional[str] = None,
):
    """
    Reproduce Figure 1 of the SEEDA paper: Type I (false alarm) and Type II
    (miss detection) error rates versus the number of cohorts.

    - Type I (false alarm): the fraction of truly-safe doses (p_k <= theta) that
      the design declares unsafe.
    - Type II (miss detection): the fraction of truly-unsafe doses (p_k > theta)
      that the design declares safe.
    Both are averaged over trials, per cohort.

    This conforms to the paper's definitions (Section J of the supplementary
    material): e1 = sum_k 1{p_k <= theta} 1{p_hat_k(n) > theta} (false alarm) and
    e2 = sum_k 1{p_k > theta} 1{p_hat_k(n) <= theta} (miss detection), which are
    exactly the type-I / type-II events analysed in Lemma 1 and Lemma 2. The
    per-dose "declared safe" test (p_hat_k(n) <= theta) is the admissible set
    {k : p_k(a_hat + alpha_t) <= theta} for SEEDA / SEEDA-Plateau (as in Lemma
    1/2) and the estimated toxicity <= theta for UCB / CRM.

    ``safe_map`` has the same nested structure as ``plot_acc_progression``'s
    input: {env: {cohort: {algo: [safe_set, ... one per trial]}}}, where each
    safe_set is a per-dose boolean list (or ``None`` for designs with no safety
    model, e.g. 3 + 3 — those designs are skipped).
    """
    envs, _ = _get_env_algos(safe_map)
    cohort_names, all_algos = _get_env_algos(list(safe_map.values())[0])
    algos = algos or all_algos
    tox = list(toxicity_probs)
    safe_idx = [k for k in range(len(tox)) if tox[k] <= ttl]
    unsafe_idx = [k for k in range(len(tox)) if tox[k] > ttl]

    fig = make_subplots(
        rows=len(envs),
        cols=2,
        shared_xaxes=True,
        column_titles=["Type I Error (%)", "Type II Error (%)"],
        row_titles=envs if len(envs) > 1 else None,
        x_title="Number of Cohorts",
    )

    for env_idx, env in enumerate(envs):
        for algo_idx, algo in enumerate(algos):
            # Skip designs with no safety model (all None):
            first = safe_map[env][cohort_names[0]][algo]
            if not first or all(s is None for s in first):
                continue
            type1, type2 = [], []
            for cohort in cohort_names:
                sets = [s for s in safe_map[env][cohort][algo] if s is not None]
                if not sets:
                    type1.append(np.nan)
                    type2.append(np.nan)
                    continue
                arr = np.asarray(sets, dtype=bool)  # (trials, n_doses)
                # False alarm: safe dose declared unsafe; miss: unsafe declared safe:
                type1.append((~arr[:, safe_idx]).mean() * 100 if safe_idx else np.nan)
                type2.append(arr[:, unsafe_idx].mean() * 100 if unsafe_idx else np.nan)

            for col, curve in ((1, type1), (2, type2)):
                fig.add_trace(
                    go.Scatter(
                        x=list(range(n_cohorts)),
                        y=curve,
                        name=algo,
                        mode='lines',
                        marker_color=DEFAULT_COLORS[algo_idx % len(DEFAULT_COLORS)],
                        legendgroup=algo,
                        showlegend=(env_idx == 0 and col == 1),
                    ),
                    row=env_idx + 1, col=col,
                )

    fig.update_layout(
        autosize=False,
        width=unit_width * 2 + 350,
        height=unit_height * len(envs) + 150,
        title_text=title_text or "Type I (false alarm) and Type II (miss detection) error rates",
        legend_title_text="Design",
        legend=dict(x=1.02, xanchor="left", y=1, yanchor="top"),
        margin=dict(r=340),
    )

    if img_path:
        os.makedirs(os.path.dirname(img_path) or ".", exist_ok=True)
        fig.write_image(img_path)
    if html_path:
        os.makedirs(os.path.dirname(html_path) or ".", exist_ok=True)
        fig.write_html(html_path)
    if show_fig:
        fig.show()


def plot_dose_curves(
    dose_levels: Sequence[float],
    efficacy_probs: Sequence[float],
    dose_toxicity_curve,
    ttl: float,
    a_values: Sequence[float],
    a_star: float,
    optimal_dose: Optional[int] = None,
    unit_width: Optional[int] = 600,
    unit_height: Optional[int] = 400,
    title_text: Optional[str] = None,
    show_fig: Optional[bool] = False,
    img_path: Optional[str] = None,
    html_path: Optional[str] = None,
):
    """
    Plot the dose-toxicity and dose-efficacy curves of the scenario.

    Left: the dose-toxicity model p_k(a) = ((tanh(d_k) + 1) / 2)^a evaluated for
    each value of ``a`` in ``a_values`` (one line each), with the MTD threshold
    theta (``ttl``) drawn as a dashed line. At a = ``a_star`` the curve passes
    through the scenario's true toxicity probabilities by construction, since the
    dose levels d_k were back-solved from those probabilities.

    Right: the dose-efficacy curve ``efficacy_probs`` (independent of a); the
    optimal biological dose k* (``optimal_dose``, 0-indexed) is marked if given.

    Doses are shown 1-indexed (1..K), matching the paper's numbering.
    """
    dl = np.asarray(dose_levels)
    doses = list(range(1, len(dl) + 1))

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=["Dose-toxicity curve", "Dose-efficacy curve"],
        x_title="Dose level",
    )

    # Left: dose-toxicity model for each a.
    for i, a in enumerate(a_values):
        p = dose_toxicity_curve(dl, a)
        is_star = a == a_star
        fig.add_trace(
            go.Scatter(
                x=doses,
                y=list(np.asarray(p, dtype=float)),
                name=f"a = {a:g}" + (" (a*)" if is_star else ""),
                mode='lines+markers',
                marker_color=DEFAULT_COLORS[i % len(DEFAULT_COLORS)],
                line=dict(width=3 if is_star else 1.5),
            ),
            row=1, col=1,
        )
    fig.add_hline(
        y=ttl, line_dash="dash", line_color="gray",
        annotation_text="θ (MTD threshold)", annotation_position="top left",
        row=1, col=1,
    )

    # Right: dose-efficacy curve.
    fig.add_trace(
        go.Scatter(
            x=doses,
            y=list(np.asarray(efficacy_probs, dtype=float)),
            name="Efficacy",
            mode='lines+markers',
            marker_color=DEFAULT_COLORS[2],
            showlegend=False,
        ),
        row=1, col=2,
    )
    if optimal_dose is not None:
        fig.add_vline(
            x=optimal_dose + 1, line_dash="dot", line_color="green",
            annotation_text="k* (optimal dose)", annotation_position="top right",
            row=1, col=2,
        )

    fig.update_yaxes(title_text="Toxicity probability", range=[0, 1], row=1, col=1)
    fig.update_yaxes(title_text="Efficacy probability", range=[0, 1], row=1, col=2)
    fig.update_xaxes(dtick=1, row=1, col=1)
    fig.update_xaxes(dtick=1, row=1, col=2)

    fig.update_layout(
        autosize=False,
        width=unit_width * 2,
        height=unit_height,
        title_text=title_text or "Dose-toxicity and dose-efficacy curves",
        legend_title_text="Toxicity model",
    )

    if img_path:
        os.makedirs(os.path.dirname(img_path) or ".", exist_ok=True)
        fig.write_image(img_path)
    if html_path:
        os.makedirs(os.path.dirname(html_path) or ".", exist_ok=True)
        fig.write_html(html_path)
    if show_fig:
        fig.show()
