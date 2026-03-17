import math
import plotly.graph_objects as go
from collections import Counter
from doseescalation.dose_escalator import DoseEscalatorBase
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
    env: SimulatedEnv
) -> Tuple[List, List]:
    dlis = []
    n_dles = []
    for cohort in cohort_sizes:
        dose_level_index = dose_escalator.propose()
        dlis.append(dose_level_index)
        n_dle = env(dose_level_index, cohort)
        n_dles.append(n_dle)
        dose_escalator.update(dose_level_index, cohort, n_dle)
    return dlis, n_dles


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

    # add traces for each algorithm in each environment
    shown_no_legend = shown_yes_legend = False
    for row_idx, algo in enumerate(algo_names):
        for col_idx, env in enumerate(env_names):
            proposals = dose_proposals_map[env][algo]
            correct_mtd = [
                mtd for mtd in proposals
                if mtd == correct_mtds[env]
            ]
            fig.add_trace(
                go.Histogram(
                    x=proposals,
                    name="No",
                    marker_color=DEFAULT_COLORS[-1],
                    showlegend=not shown_no_legend,
                    xbins=dict(  # bins used for histogram
                        start=-0.5,
                        end=n_dose_levels - 0.5,
                        size=1,
                    ),
                ),
                row=row_idx + 1,
                col=col_idx + 1,
            )
            fig.update_xaxes(
                range=[-0.5, n_dose_levels - 0.5],
                dtick=1,
                row=row_idx + 1,
                col=col_idx + 1,
            )
            fig.add_trace(
                go.Histogram(
                    x=correct_mtd,
                    name="Yes",
                    marker_color=DEFAULT_COLORS[2],
                    showlegend=not shown_yes_legend,
                    xbins=dict(  # bins used for histogram
                        start=-0.5,
                        end=n_dose_levels - 0.5,
                        size=1,
                    ),
                ),
                row=row_idx + 1,
                col=col_idx + 1,
            )
            fig.update_xaxes(
                range=[-0.5, n_dose_levels - 0.5],
                dtick=1,
                row=row_idx + 1,
                col=col_idx + 1,
            )
            fig.update_yaxes(
                range=[0, n_trials],
                row=row_idx + 1,
                col=col_idx + 1,
            )

            # make sure we only show these legends once
            shown_no_legend = (
                shown_no_legend or len(proposals) > 0
            )
            shown_yes_legend = (
                shown_yes_legend or len(correct_mtd) > 0
            )

    title_text = title_text or "Number of dose allocations"
    fig.update_layout(
        autosize=False,
        width=unit_width * (len(env_names) + 1),
        height=unit_height * len(algo_names),
        barmode="overlay",
        title_text=title_text,
        legend_title_text="True MTD",
    )

    if img_path:
        fig.write_image(img_path)
    if html_path:
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
        mtd = correct_mtds[a]
        for algo_idx, algo in enumerate(algos):
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
        fig.write_image(img_path)
    if html_path:
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
        fig.write_image(img_path)
    if html_path:
        fig.write_html(html_path)
    if show_fig:
        fig.show()
