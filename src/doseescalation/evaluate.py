import inspect
import math
import os
import re
import textwrap
import warnings
import numpy as np
import plotly.graph_objects as go
from doseescalation.dose_escalator import (
    DoseEscalatorBase, UCBDoseEscalator, SEEDADoseEscalator,
    SEEDAPlateauTwoSidedDecoupledDoseEscalator,
)
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

# Fixed per-design colors matching the SEEDA paper's own Figures 1-3 (basic
# MATLAB colors), used by plot_error_rates / plot_efficacy_and_violation /
# plot_sample_complexity so a given design has the same color across all three
# figures, regardless of each figure's `algos` subset/order:
PAPER_COLORS = {
    "3 + 3": "#00FFFF",   # cyan
    "CRM": "#000000",     # black
    "UCB": "#0000FF",     # blue
    "SEEDA": "#FF0000",   # red
    "SEEDA Plateau": "#00FF00",     # green
    "SEEDA Plateau (ours)": "#FF00FF",  # magenta
    "SEEDA Plateau (fixed)": "#FF8C00"  # orange
}

# Display names for the three Plateau renderings. The persisted CSVs and every
# map key still carry the internal names, so the rename happens at draw time and
# nothing has to be re-simulated. Internal names stay put; only what a reader
# sees changes.
DISPLAY_NAMES = {
    "SEEDA Plateau": "Reference",
    "SEEDA Plateau (fixed)": "Descending",
    "SEEDA Plateau (ours)": "Anchored",
}


def display_name(algo: str) -> str:
    """The label a reader sees for an internal design name."""
    return DISPLAY_NAMES.get(algo, algo)


# Base font size for every figure. Individual sizes below are set relative to
# this, so the whole set scales together rather than drifting apart.
BASE_FONT = 16


def simulate(
    cohort_sizes: Sequence[int],
    dose_escalator: DoseEscalatorBase,
    env: SimulatedEnv,
    n_efficate: int = 0,
    efficacy_env: Optional[SimulatedEnv] = None,
) -> Tuple[List, List, List, List, List]:
    """
    Run a simulated trial, returning five per-cohort sequences:

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
    - ``enrolled``: per-cohort boolean, ``True`` while the design is still
      running and actually dosing a new cohort, ``False`` once it has
      terminated (3 + 3 after it declares an MTD). Model-based designs that run
      the whole horizon are ``True`` throughout. Use this to freeze
      per-patient cumulative metrics (efficacy / safety violation) at a
      stopped design's real trial length, matching the paper's Figure 2 (a
      real 3 + 3 stops after ~5-6 cohorts rather than dosing its MTD forever).

    For CRM and 3 + 3 the recommendation equals the allocation (no
    exploration/recommendation split); for UCB / SEEDA / SEEDA Plateau the two
    differ, and the recommendation is read via non-training mode.

    ``efficacy_env`` (optional, only used by SEEDA / SEEDA Plateau) is an env
    that samples the number of efficacy responders for the allocated dose, e.g.
    ``Binomial(cohort, q_dose)``. If omitted, the constant ``n_efficate`` is
    used instead (dose-independent efficacy).
    """
    # UCB-1 (efficacy bandit) and the SEEDA family all consume efficacy feedback,
    # which they declare by taking ``n_efficate`` in ``update``. Asking the
    # signature rather than the class keeps designs defined outside this package
    # -- the diagnosis implementations of Algorithm 2 and of the shared MATLAB --
    # on the same path. Verified equivalent to the previous isinstance check for
    # every design in the pipeline: CRM and 3 + 3 False, SEEDA / UCB / all three
    # Plateau variants True.
    _needs_efficacy = "n_efficate" in inspect.signature(
        dose_escalator.update
    ).parameters
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
    enrolled = []
    for cohort in cohort_sizes:
        # Once a design has terminated (3 + 3 in stage 2), the trial is over:
        # it enrols no more patients. We still record its held allocation /
        # recommendation so every trial has the same length, but flag the
        # cohort as not enrolled and draw no new outcomes.
        is_active = not dose_escalator.stopped
        enrolled.append(is_active)

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

        if not is_active:
            # Trial has ended: no patient dosed this cohort, no update.
            n_dles.append(0)
            continue

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
    return allocations, recommendations, n_dles, safe_sets, enrolled


def _wrap_label(text: str, width: int = 16) -> str:
    """Break a long column/subplot title onto multiple lines (via <br>) so
    neighboring titles don't overlap when columns are narrow -- e.g. the
    scenario battery's scenario names as plot_dose_proposals column titles.

    A trailing parenthetical (e.g. "(k*=3, MTD=4)") always starts its own
    line, wrapped separately from the name that precedes it -- rather than
    being split wherever textwrap happens to land inside it.
    """
    m = re.match(r"^(.*\S)\s*(\([^()]*\))$", text)
    if m:
        name, paren = m.group(1), m.group(2)
        lines = (textwrap.wrap(name, width=width, break_long_words=False)
                 + textwrap.wrap(paren, width=width, break_long_words=False))
    else:
        lines = textwrap.wrap(text, width=width, break_long_words=False)
    return "<br>".join(lines)


def _nanmean_cols(m):
    """
    Column means ignoring NaN, quiet about all-NaN columns.

    The error-rate curves blank the round-robin warm-up cohorts to NaN on
    purpose, so those columns are legitimately all-NaN and NaN is the intended
    result -- numpy's "Mean of empty slice" warning is noise here, suppressed at
    the source rather than filtered by the caller.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(m, axis=0)


def _wrap_all(labels: Sequence[str], width: int = 16) -> List[str]:
    return [_wrap_label(str(t), width) for t in labels]


def _n_lines(labels: Sequence[str]) -> int:
    """Tallest wrapped label, in lines."""
    return max((str(t).count("<br>") + 1 for t in labels), default=1)


def _shrink_panel_titles(fig, labels: Sequence[str], font_size: int = 11) -> None:
    """Shrink the auto-generated row/column/subplot title annotations that
    ``make_subplots`` created from ``labels`` (already wrapped). Long scenario
    names otherwise render at the default 16px and collide with each other and
    with the figure title."""
    wanted = set(str(t) for t in labels)
    for ann in fig.layout.annotations:
        if ann.text in wanted:
            ann.font = dict(size=font_size)


def _rgba(hex_color: str, alpha: float) -> str:
    """'#00FF00' -> 'rgba(0,255,0,alpha)', for translucent CI bands."""
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def _add_ci_band(fig, x, lo, hi, color, row, col, alpha=0.18):
    """Shade [lo, hi] behind a curve, as two traces with 'tonexty' fill."""
    fig.add_trace(
        go.Scatter(x=list(x), y=list(hi), mode="lines",
                   line=dict(width=0), hoverinfo="skip",
                   showlegend=False),
        row=row, col=col,
    )
    fig.add_trace(
        go.Scatter(x=list(x), y=list(lo), mode="lines",
                   line=dict(width=0), fill="tonexty",
                   fillcolor=_rgba(color, alpha), hoverinfo="skip",
                   showlegend=False),
        row=row, col=col,
    )


N_BOOT = 1000          # bootstrap resamples for every CI in this module
BOOT_CI = (2.5, 97.5)  # percentile endpoints -> 95% CI
BOOT_SEED = 0


def _bootstrap_band(per_trial, n_boot: int = N_BOOT, seed: int = BOOT_SEED):
    """
    Bootstrap 95% CI for the column means of a (n_trials, n_points) array of
    per-trial values, resampling TRIALS with replacement.

    Trials are the resampling unit everywhere in this project: cohorts within a
    trial come from a sequential adaptive design and are not exchangeable, but
    whole trials are independent replicates. See experiments/bootstrap.py.

    Implemented in the multinomial-weights form: drawing how many times each
    trial appears and taking a weighted mean is identical to gathering a
    resampled index array, but is a single matrix product instead of an
    (n_boot x n_trials x n_points) gather -- seconds rather than many minutes,
    and megabytes rather than gigabytes.
    """
    x = np.asarray(per_trial, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    n_trials = x.shape[0]
    if n_trials < 2:
        flat = x.mean(axis=0)
        return flat, flat
    rng = np.random.default_rng(seed)
    w = rng.multinomial(n_trials, np.full(n_trials, 1.0 / n_trials), size=n_boot)
    boot = (w @ x) / n_trials                      # (n_boot, n_points)
    return (np.percentile(boot, BOOT_CI[0], axis=0),
            np.percentile(boot, BOOT_CI[1], axis=0))


def _top_margin_for(labels: Sequence[str], base: int = 70, per_line: int = 15) -> int:
    """Top margin leaving room for `labels` used as COLUMN titles plus the
    figure title above them -- without it, multi-line column titles grow
    upward into the title."""
    return base + per_line * _n_lines(labels)


def _widened_range(default_max: float, values: Sequence[float]) -> List[float]:
    """[0, default_max], widened to fit `values` when they exceed it -- so a
    curve that genuinely runs past the paper's axis convention (e.g. under
    misspecification) is shown rather than silently clipped off-screen."""
    finite = [v for v in values if v is not None and not np.isnan(v)]
    data_max = max(finite) if finite else 0
    return [0, max(default_max, data_max * 1.05)]


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
    ci_n_trials: Optional[int] = None,
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
    col_titles = _wrap_all(env_names, 20)
    fig = make_subplots(
        rows=len(algo_names),
        cols=len(env_names),
        shared_xaxes=True,
        shared_yaxes=True,
        column_titles=col_titles,
        row_titles=[display_name(a) for a in algo_names],
        x_title="Dose Level",
        y_title="Number Of Trials",
    )
    _shrink_panel_titles(fig, col_titles, font_size=13)

    # Row titles default to fully vertical (textangle=90), which makes longer
    # algorithm names (e.g. "SEEDA Plateau (Modified L1)") overlap with
    # neighboring rows. A 45-degree angle keeps every label readable within
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

            # Yellow overlay — the TRUE toxicity MTD, i.e. the highest genuinely
            # safe dose (only drawn when it differs from the optimal dose):
            if tox_mtd is not None and tox_mtd != correct:
                fig.add_trace(
                    go.Histogram(
                        x=tox_mtd_props,
                        name="True MTD",
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
                    name="True optimal dose (k*)",
                    marker_color=DEFAULT_COLORS[2],
                    showlegend=not shown_opt,
                    xbins=dict(start=-0.5, end=n_dose_levels - 0.5, size=1),
                ),
                row=row_idx + 1,
                col=col_idx + 1,
            )

            # Bootstrap error bars on each bar. `ci_n_trials` is the number of
            # INDEPENDENT trials behind this list; the list itself is
            # trial-then-cohort flattened, so reshaping recovers the per-trial
            # dose distribution and the bar height is total x mean fraction.
            # Cohorts inside a trial are dependent, so a plain binomial
            # interval on the pooled counts would understate the width --
            # hence resampling trials.
            if ci_n_trials and len(proposals) % ci_n_trials == 0:
                per_trial_dose = np.asarray(proposals).reshape(ci_n_trials, -1)
                frac = np.stack(
                    [(per_trial_dose == k).mean(axis=1) for k in range(n_dose_levels)],
                    axis=1,
                )                                     # (n_trials, n_doses)
                lo_f, hi_f = _bootstrap_band(frac)
                total = len(proposals)
                center = frac.mean(axis=0) * total
                fig.add_trace(
                    go.Scatter(
                        x=list(range(n_dose_levels)),
                        y=list(center),
                        mode="markers",
                        marker=dict(size=1, color="rgba(0,0,0,0)"),
                        error_y=dict(
                            type="data", symmetric=False,
                            array=list(np.asarray(hi_f) * total - center),
                            arrayminus=list(center - np.asarray(lo_f) * total),
                            thickness=1.2, width=3, color="#333333",
                        ),
                        hoverinfo="skip", showlegend=False,
                    ),
                    row=row_idx + 1, col=col_idx + 1,
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
    # Top margin sized to the tallest wrapped column title, with the figure
    # title pinned above it -- otherwise multi-line scenario names grow upward
    # and run straight through the title.
    top_margin = _top_margin_for(col_titles)
    fig.update_layout(
        autosize=False,
        font=dict(size=BASE_FONT),
        width=unit_width * (len(env_names) + 1) + row_label_margin,
        height=unit_height * len(algo_names) + top_margin,
        barmode="overlay",
        title_text=title_text,
        title=dict(y=1, yanchor="top", pad=dict(t=15)),
        legend_title_text="Dose",
        legend=dict(x=1.02, xanchor="left", y=1, yanchor="top"),
        margin=dict(r=row_label_margin, t=top_margin),
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
    show_ci: bool = False,
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
        subplot_titles=_wrap_all(a, 45),
    )
    _shrink_panel_titles(fig, _wrap_all(a, 45), font_size=14)

    for a_idx, a in enumerate(a):
        row_idx = a_idx // n_cols
        col_idx = a_idx % n_cols
        correct_for_a = correct_mtds[a]
        for algo_idx, algo in enumerate(algos):
            # correct_for_a may be a single dose or a {algorithm: dose} dict.
            mtd = (correct_for_a[algo] if isinstance(correct_for_a, dict)
                   else correct_for_a)
            # (n_trials, n_cohorts) 0/1: did this trial recommend k* at this
            # cohort? Column means are the accuracy curve; bootstrapping the
            # ROWS (trials) gives the band.
            hits = np.array(
                [[dli == mtd for dli in dose_proposals_map[a][cohort][algo]]
                 for cohort in cohort_names], dtype=float
            ).T
            accs = hits.mean(axis=0)

            color = PAPER_COLORS.get(algo, DEFAULT_COLORS[algo_idx])
            if show_ci:
                lo, hi = _bootstrap_band(hits)
                _add_ci_band(fig, range(n_cohorts), lo, hi, color,
                             row_idx + 1, col_idx + 1)
            fig.add_trace(
                go.Scatter(
                    x=list(range(n_cohorts)),
                    y=list(accs),
                    name=display_name(algo),
                    mode='lines',
                    marker_color=color,
                    showlegend=a_idx == 0,
                ),
                row=row_idx + 1, col=col_idx + 1,
            )

    title_text = title_text or "Dose Proposal Accuracy Progression"

    # Match the sample-efficiency plot's size, with the legend moved outside to
    # the right so every design stays visible:
    fig.update_layout(
        autosize=False,
        font=dict(size=BASE_FONT),
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


def plot_alloc_acc_progression(
    n_cohorts: int,
    n_trials: int,
    alloc_map: Dict[str, Dict[str, List[int]]],
    correct_mtds: Dict[str, int],
    algos: Optional[List[str]] = None,
    show_ci: bool = False,
    unit_width: Optional[int] = 600,
    unit_height: Optional[int] = 400,
    title_text: Optional[str] = None,
    show_fig: Optional[bool] = False,
    img_path: Optional[str] = None,
    html_path: Optional[str] = None,
):
    """
    Companion to ``plot_acc_progression``: that one tracks RECOMMENDATION
    accuracy (the algorithm's declared best-guess dose) over cohorts; this one
    tracks ALLOCATION accuracy instead -- the fraction of patients actually
    dosed at the correct dose, cohort by cohort, averaged over trials.

    ``alloc_map`` is the pooled per-cohort allocation map {env: {algo: [dose,
    ...]}}, flattened in trial-then-cohort order (as produced by extending
    each trial's allocations) -- the same shape ``plot_efficacy_and_violation``
    and ``plot_dose_proposals`` take, reshaped here to (n_trials, n_cohorts).
    ``correct_mtds`` may be a single dose per env or a {algorithm: dose} dict
    (e.g. ``correct_doses``, scoring against k* rather than the toxicity MTD).
    """
    envs, all_algos = _get_env_algos(alloc_map)
    algos = algos or all_algos

    n_rows = round(math.sqrt(len(envs)))
    n_cols = math.ceil(len(envs) / n_rows)

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        shared_xaxes=True,
        shared_yaxes=True,
        x_title="Cohort Number",
        y_title="Accuracy",
        subplot_titles=_wrap_all(envs, 45),
    )
    _shrink_panel_titles(fig, _wrap_all(envs, 45), font_size=14)

    for env_idx, env in enumerate(envs):
        row_idx = env_idx // n_cols
        col_idx = env_idx % n_cols
        correct_for_env = correct_mtds[env]
        for algo_idx, algo in enumerate(algos):
            correct = (correct_for_env[algo] if isinstance(correct_for_env, dict)
                       else correct_for_env)
            allocs = np.asarray(alloc_map[env][algo]).reshape(n_trials, n_cohorts)
            hits = (allocs == correct).astype(float)   # (n_trials, n_cohorts)
            acc = hits.mean(axis=0)

            color = PAPER_COLORS.get(
                algo, DEFAULT_COLORS[algo_idx % len(DEFAULT_COLORS)]
            )
            if show_ci:
                lo, hi = _bootstrap_band(hits)
                _add_ci_band(fig, range(n_cohorts), lo, hi, color,
                             row_idx + 1, col_idx + 1)
            fig.add_trace(
                go.Scatter(
                    x=list(range(n_cohorts)),
                    y=acc,
                    name=display_name(algo),
                    mode='lines',
                    marker_color=color,
                    legendgroup=algo,
                    showlegend=(env_idx == 0),
                ),
                row=row_idx + 1, col=col_idx + 1,
            )

    fig.update_layout(
        autosize=False,
        font=dict(size=BASE_FONT),
        width=unit_width * n_cols + 350,
        height=unit_height * n_rows + 150,
        title_text=title_text or "Dose Allocation Accuracy Progression",
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
    show_ci: bool = False,
    # ^ efficacy_probs / toxicity_probs may each be a flat Sequence[float]
    # (shared across every env, the original a-sweep use) or a Dict[env,
    # Sequence[float]] (one curve per env, e.g. when envs are scenarios with
    # different efficacy/toxicity curves) -- same scalar-or-dict convention
    # as plot_dose_proposals' correct_mtds.
    enrolled_map: Optional[Dict[str, Dict[str, List[bool]]]] = None,
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
    - Safety violation %: the running fraction of patients allocated to unsafe
      doses (true toxicity p_k > ``ttl``), averaged over trials. This is the
      paper's Fig 2 metric -- the MATLAB ``count_exceed`` = cumulative unsafe
      allocations / patients, i.e. Section 2.2's first safety formulation and
      Corollary 1's E[sum_{k: p_k > theta} N_k(n) / n]. (We previously used
      P[(1/t) sum p_{I(s)} > theta], the Theorem 2 event; that saturates near
      100% for any design that sits on an unsafe dose, masking the others.)

    ``alloc_map`` is the pooled per-cohort allocation map {env: {algo: [dose,
    ...]}}, flattened in trial-then-cohort order (as produced by extending each
    trial's allocations); it is reshaped to (n_trials, n_cohorts) here.

    ``enrolled_map`` (optional, same shape as ``alloc_map``) flags, per cohort,
    whether the design was still dosing new patients. Both cumulative means are
    then taken over enrolled cohorts only and forward-filled once a design has
    terminated, so a design that stops early (3 + 3) freezes at its true trial
    length instead of being credited with dosing its MTD for the whole horizon.
    If omitted, every cohort is treated as enrolled (the previous behavior).
    """
    envs, all_algos = _get_env_algos(alloc_map)
    algos = algos or all_algos

    fig = make_subplots(
        rows=len(envs),
        cols=2,
        shared_xaxes=True,
        column_titles=["Efficacy per Patient", "Safety Violation (%)"],
        row_titles=_wrap_all(envs, 34) if len(envs) > 1 else None,
        x_title="Number of Cohorts",
    )
    _shrink_panel_titles(fig, _wrap_all(envs, 34), font_size=13)

    for env_idx, env in enumerate(envs):
        eff = np.asarray(
            efficacy_probs[env] if isinstance(efficacy_probs, dict) else efficacy_probs,
            dtype=float,
        )
        tox = np.asarray(
            toxicity_probs[env] if isinstance(toxicity_probs, dict) else toxicity_probs,
            dtype=float,
        )
        row_eff, row_viol = [], []
        for algo_idx, algo in enumerate(algos):
            allocs = np.asarray(alloc_map[env][algo]).reshape(n_trials, n_cohorts)
            if enrolled_map is not None:
                enr = np.asarray(
                    enrolled_map[env][algo], dtype=float
                ).reshape(n_trials, n_cohorts)
            else:
                enr = np.ones((n_trials, n_cohorts))
            # Patients dosed up to each cohort (freezes once a design stops):
            n_patients = np.cumsum(enr, axis=1)
            n_patients = np.where(n_patients == 0, 1, n_patients)
            # Cumulative mean over ENROLLED cohorts, per trial (held after stop):
            eff_cum = np.cumsum(eff[allocs] * enr, axis=1) / n_patients
            eff_curve = eff_cum.mean(axis=0)      # eff_cum: (n_trials, n_cohorts)
            # Safety violation = running fraction of enrolled patients allocated to
            # UNSAFE doses (p_k > theta), averaged over trials -- the paper's Fig 2
            # metric (MATLAB count_exceed; Corollary 1's E[sum_{unsafe} N_k(n)/n]).
            # NOT P[cumulative avg toxicity > theta]: that is a per-trial 0/1 event
            # that saturates near 100% for any design sitting on an unsafe dose, so
            # CRM/KL-UCB/UCB pin to the top and mask each other.
            unsafe = (tox > ttl).astype(float)
            viol_cum = (np.cumsum(unsafe[allocs] * enr, axis=1)
                        / n_patients) * 100.0      # (n_trials, n_cohorts)
            viol_curve = viol_cum.mean(axis=0)
            row_eff.extend(eff_curve)
            row_viol.extend(viol_curve)

            color = PAPER_COLORS.get(
                algo, DEFAULT_COLORS[algo_idx % len(DEFAULT_COLORS)]
            )
            for col, curve, per_trial in ((1, eff_curve, eff_cum),
                                          (2, viol_curve, viol_cum)):
                if show_ci:
                    lo, hi = _bootstrap_band(per_trial)
                    _add_ci_band(fig, range(n_cohorts), lo, hi, color,
                                 env_idx + 1, col)
                fig.add_trace(
                    go.Scatter(
                        x=list(range(n_cohorts)),
                        y=list(curve),
                        name=display_name(algo),
                        mode='lines',
                        marker_color=color,
                        legendgroup=algo,
                        showlegend=(env_idx == 0 and col == 1),
                    ),
                    row=env_idx + 1, col=col,
                )

        # Paper Fig 2 y-axis ranges: efficacy per patient 0-0.7, safety
        # violation 0-120% -- widened per row if genuinely exceeded, rather
        # than silently clipping the curve off-screen.
        fig.update_yaxes(range=_widened_range(0.7, row_eff), row=env_idx + 1, col=1)
        fig.update_yaxes(range=_widened_range(120, row_viol), row=env_idx + 1, col=2)

    fig.update_layout(
        autosize=False,
        font=dict(size=BASE_FONT),
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
    show_ci: bool = False,
    # ^ toxicity_probs may be a flat Sequence[float] (shared across every env)
    # or a Dict[env, Sequence[float]] (one curve per env) -- same
    # scalar-or-dict convention as plot_dose_proposals' correct_mtds.
    algos: Optional[List[str]] = None,
    skip_initial_cohorts: int = 0,
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

    ``skip_initial_cohorts`` blanks the first N cohorts (plotted as NaN). Set it
    to the number of doses to drop the sample-each-dose-once round-robin: during
    that phase SEEDA/Plateau's ``safe_doses`` returns all-unsafe (nothing is
    classified yet), so Type I is a spurious 100% that clips the y-axis and hides
    the real early peak. Skipping it lets the true near-x=0 transient show.

    - Type I (false alarm): the fraction of truly-safe doses (p_k <= theta) that
      the design declares unsafe.
    - Type II (miss detection): the fraction of truly-unsafe doses (p_k > theta)
      that the design declares safe.
    Both are averaged over trials, per cohort.

    This conforms to the paper's definitions (Section J of the supplementary
    material): e1 = sum_k 1{p_k <= theta} 1{p_hat_k(n) > theta} (false alarm) and
    e2 = sum_k 1{p_k > theta} 1{p_hat_k(n) <= theta} (miss detection), which are
    exactly the type-I / type-II events analyzed in Lemma 1 and Lemma 2. The
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

    fig = make_subplots(
        rows=len(envs),
        cols=2,
        shared_xaxes=True,
        column_titles=["Type I Error (%)", "Type II Error (%)"],
        row_titles=_wrap_all(envs, 34) if len(envs) > 1 else None,
        x_title="Number of Cohorts",
    )
    _shrink_panel_titles(fig, _wrap_all(envs, 34), font_size=13)

    for env_idx, env in enumerate(envs):
        tox = list(toxicity_probs[env] if isinstance(toxicity_probs, dict) else toxicity_probs)
        safe_idx = [k for k in range(len(tox)) if tox[k] <= ttl]
        unsafe_idx = [k for k in range(len(tox)) if tox[k] > ttl]
        row_type1, row_type2 = [], []
        for algo_idx, algo in enumerate(algos):
            # Skip designs with no safety model (all None):
            first = safe_map[env][cohort_names[0]][algo]
            if not first or all(s is None for s in first):
                continue
            # Keep the PER-TRIAL error at each cohort (columns), not just its
            # mean, so the band can bootstrap over trials like every other
            # figure. NaN columns are the blanked warm-up.
            pt1, pt2 = [], []
            for c_i, cohort in enumerate(cohort_names):
                # Blank the round-robin warm-up (all-unsafe -> spurious 100% Type I):
                sets = ([] if c_i < skip_initial_cohorts
                        else [s for s in safe_map[env][cohort][algo] if s is not None])
                if not sets:
                    pt1.append(None)
                    pt2.append(None)
                    continue
                arr = np.asarray(sets, dtype=bool)  # (trials, n_doses)
                # False alarm: safe dose declared unsafe; miss: unsafe declared safe:
                pt1.append((~arr[:, safe_idx]).mean(axis=1) * 100 if safe_idx else None)
                pt2.append(arr[:, unsafe_idx].mean(axis=1) * 100 if unsafe_idx else None)

            n_tr = max((len(v) for v in pt1 + pt2 if v is not None), default=0)

            def _stack(per_cohort):
                """(n_trials, n_cohorts), NaN on blanked/absent cohorts."""
                cols = [np.full(n_tr, np.nan) if v is None else v for v in per_cohort]
                return np.column_stack(cols) if n_tr else None

            m1, m2 = _stack(pt1), _stack(pt2)
            type1 = ([np.nan] * len(cohort_names) if m1 is None
                     else _nanmean_cols(m1))
            type2 = ([np.nan] * len(cohort_names) if m2 is None
                     else _nanmean_cols(m2))
            row_type1.extend(type1)
            row_type2.extend(type2)

            color = PAPER_COLORS.get(
                algo, DEFAULT_COLORS[algo_idx % len(DEFAULT_COLORS)]
            )
            for col, curve, per_trial in ((1, type1, m1), (2, type2, m2)):
                if show_ci and per_trial is not None:
                    lo, hi = _bootstrap_band(per_trial)
                    _add_ci_band(fig, range(n_cohorts), lo, hi, color,
                                 env_idx + 1, col)
                fig.add_trace(
                    go.Scatter(
                        x=list(range(n_cohorts)),
                        y=list(curve),
                        name=display_name(algo),
                        mode='lines',
                        marker_color=color,
                        legendgroup=algo,
                        showlegend=(env_idx == 0 and col == 1),
                    ),
                    row=env_idx + 1, col=col,
                )

        # Paper Fig 1 y-axis ranges: Type I error 0-10%, Type II error 0-30% --
        # widened per row when a design's error rate genuinely exceeds it
        # (e.g. under toxicity misspecification), rather than silently
        # clipping the curve off-screen.
        fig.update_yaxes(range=_widened_range(10, row_type1), row=env_idx + 1, col=1)
        fig.update_yaxes(range=_widened_range(30, row_type2), row=env_idx + 1, col=2)

    fig.update_layout(
        autosize=False,
        font=dict(size=BASE_FONT),
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
        font=dict(size=BASE_FONT),
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


def plot_scenario_curves(
    scenarios: List[Dict],
    unit_width: Optional[int] = 500,
    unit_height: Optional[int] = 350,
    row_title_width: Optional[int] = 150,
    title_text: Optional[str] = None,
    show_fig: Optional[bool] = False,
    img_path: Optional[str] = None,
    html_path: Optional[str] = None,
):
    """
    One row per scenario, two columns (dose-toxicity curve | dose-efficacy
    curve), gridded for comparing a scenario battery side by side.

    The toxicity panel shows the TRUE curve -- the one the environment actually
    draws DLEs from. Where a scenario is misspecified, the NOMINAL curve is
    overlaid dashed: that is the curve the dose grid was calibrated to, so the
    gap between the two lines is the misspecification itself. Both are plotted
    against the dose index, so neither depends on which `a` parameterises which
    grid.

    ``row_title_width`` is the right margin held back for the rotated scenario
    names, in pixels, and it sets the canvas width along with it. It matters for
    print: legibility is the ratio of the font size to the canvas width, since
    LaTeX scales the whole image to the text width, so an over-wide gutter makes
    the text smaller on the page rather than larger.

    Each entry in ``scenarios`` is a dict:
        name             -- row label
        toxicity_probs   -- the TRUE toxicity curve driving outcomes
        nominal_toxicity_probs -- optional; the curve the dose grid was built
                             from. Drawn dashed, and only when it differs from
                             ``toxicity_probs`` (i.e. only for the misspecified
                             scenarios -- elsewhere it would be a duplicate).
        efficacy_probs   -- the efficacy curve driving outcomes
        ttl
        optimal_dose     -- 0-indexed k*, or None
    """
    n = len(scenarios)
    fig = make_subplots(
        rows=n, cols=2,
        column_titles=["Dose-toxicity curve", "Dose-efficacy curve"],
        row_titles=_wrap_all([s["name"] for s in scenarios], 30) if n > 1 else None,
        x_title="Dose level",
        # Wide enough that the second column's y-axis title clears the first
        # column's panel; at the plotly default the two collide.
        horizontal_spacing=0.16,
    )
    _shrink_panel_titles(fig, _wrap_all([s["name"] for s in scenarios], 30),
                         font_size=BASE_FONT - 2)

    shown_nominal = False
    for row_idx, sc in enumerate(scenarios):
        tox = np.asarray(sc["toxicity_probs"], dtype=float)
        eff = np.asarray(sc["efficacy_probs"], dtype=float)
        doses = list(range(1, len(tox) + 1))

        fig.add_trace(
            go.Scatter(
                x=doses, y=list(tox), name="True toxicity (generates the DLEs)",
                mode='lines+markers', marker_color=DEFAULT_COLORS[0],
                legendgroup="true_tox", showlegend=(row_idx == 0),
                # Explicit ranks: the nominal curve first appears several rows
                # down, so trace order alone would list it after efficacy.
                legendrank=1,
            ),
            row=row_idx + 1, col=1,
        )
        # Nominal curve, dashed, only where it differs -- elsewhere it would sit
        # exactly on top of the true curve. Same color as the true curve so it
        # reads as the same quantity, dashed so the gap between them is legible.
        nominal = sc.get("nominal_toxicity_probs")
        if nominal is not None and not np.allclose(nominal, tox):
            fig.add_trace(
                go.Scatter(
                    x=doses, y=list(np.asarray(nominal, dtype=float)),
                    name="Nominal toxicity (dose-grid calibration only)",
                    mode='lines', line=dict(color=DEFAULT_COLORS[0], dash="dash"),
                    legendgroup="nom_tox", showlegend=(not shown_nominal),
                    legendrank=2,
                ),
                row=row_idx + 1, col=1,
            )
            shown_nominal = True
        # add_hline with annotation_text=None still renders a "new text"
        # placeholder in this plotly version -- omit the annotation kwargs
        # entirely (rather than passing None) on every row but the first.
        hline_kwargs = (
            dict(annotation_text="θ", annotation_position="top left")
            if row_idx == 0 else {}
        )
        fig.add_hline(
            y=sc["ttl"], line_dash="dash", line_color="gray",
            row=row_idx + 1, col=1, **hline_kwargs,
        )

        fig.add_trace(
            go.Scatter(
                x=doses, y=list(eff), name="Efficacy", mode='lines+markers',
                marker_color=DEFAULT_COLORS[2],
                legendgroup="eff", showlegend=(row_idx == 0), legendrank=3,
            ),
            row=row_idx + 1, col=2,
        )
        if sc.get("optimal_dose") is not None:
            vline_kwargs = (
                dict(annotation_text="k*", annotation_position="top right")
                if row_idx == 0 else {}
            )
            fig.add_vline(
                x=sc["optimal_dose"] + 1, line_dash="dot", line_color="green",
                row=row_idx + 1, col=2, **vline_kwargs,
            )

        fig.update_yaxes(range=[0, 1], row=row_idx + 1, col=1)
        fig.update_yaxes(range=[0, 1], row=row_idx + 1, col=2)
        fig.update_xaxes(dtick=1, row=row_idx + 1, col=1)
        fig.update_xaxes(dtick=1, row=row_idx + 1, col=2)

    fig.update_yaxes(title_text="Toxicity probability", col=1)
    fig.update_yaxes(title_text="Efficacy probability", col=2)

    fig.update_layout(
        autosize=False,
        font=dict(size=BASE_FONT),
        width=unit_width * 2 + row_title_width + 110,
        height=unit_height * n + 150,
        title_text=title_text or "Dose-toxicity and dose-efficacy curves by scenario",
        # Legend across the top: the row titles occupy the right margin, so a
        # right-hand legend would sit on top of them. It has to clear the
        # column titles too, which sit just above the plotting area -- hence
        # the generous top margin and the offset below.
        legend=dict(orientation="h", yanchor="bottom", y=1.055,
                    xanchor="left", x=0, font=dict(size=BASE_FONT - 2)),
        margin=dict(r=row_title_width, t=170),
        title=dict(y=1, yanchor="top", pad=dict(t=12)),
    )

    if img_path:
        os.makedirs(os.path.dirname(img_path) or ".", exist_ok=True)
        fig.write_image(img_path)
    if html_path:
        os.makedirs(os.path.dirname(html_path) or ".", exist_ok=True)
        fig.write_html(html_path)
    if show_fig:
        fig.show()
    return fig


def _sample_complexity_band(d, rounds: Optional[int] = None):
    """
    (lo, hi) in PATIENTS for one design's sample-complexity curve.

    Uncertainty here is the LARGEST in the battery: the horizon is a mean over
    ~20 independent searches, against 1000 trials behind every other figure.

    Prefers a bootstrap over the individual per-round searches (column
    ``answers``, written by ``run_simulation.sample_complexity``). CSVs written
    before that column existed fall back to mean +/- 1.96*std/sqrt(n_rounds),
    the same interval by a different route -- but that needs the number of
    rounds, from the ``n_rounds`` column or the ``rounds`` argument. If neither
    is available this returns (None, None) and NO band is drawn: guessing
    n_rounds = 1 would silently plot +/- 1.96*std, the spread of a single
    search, which is sqrt(20) ~ 4.5x too wide and would badly overstate the
    uncertainty.
    """
    if d.empty or "n_cohorts" not in d:
        return None, None
    # answers/std are in COHORTS; the plot's y axis is PATIENTS.
    scale = (d["n_patients"] / d["n_cohorts"].replace(0, np.nan)).median()
    scale = 1.0 if not np.isfinite(scale) else float(scale)

    if "answers" in d.columns and d["answers"].notna().all():
        los, his = [], []
        for raw in d["answers"]:
            a = np.array([float(x) for x in str(raw).split(";") if x != ""])
            lo, hi = _bootstrap_band(a[:, None])
            los.append(float(lo[0]) * scale)
            his.append(float(hi[0]) * scale)
        return np.array(los), np.array(his)

    n_rounds = d["n_rounds"] if "n_rounds" in d.columns else rounds
    if "n_cohorts_std" in d.columns and n_rounds is not None:
        half = (1.96 * d["n_cohorts_std"]
                / np.sqrt(np.maximum(n_rounds, 1)) * scale).to_numpy()
        center = d["n_patients"].to_numpy()
        return np.maximum(center - half, 0), center + half
    return None, None


def plot_sample_complexity(
    sample_complexity_df,
    show_ci: bool = False,
    ci_rounds: Optional[int] = None,
    show_censored: bool = True,
    show_markers: bool = False,
    unit_width: Optional[int] = 600,
    unit_height: Optional[int] = 400,
    y_max: Optional[int] = None,
    title_text: Optional[str] = None,
    show_fig: Optional[bool] = False,
    img_path: Optional[str] = None,
    html_path: Optional[str] = None,
):
    """Figure 3, built the way the authors' samplecomplexity.m builds it.

    Takes the frame from ``run_simulation.sample_complexity``: one row per
    (design, delta) holding the mean horizon over independent searches, each of
    which ran fresh trials at every candidate horizon. Running fresh trials per
    horizon keeps the estimate well behaved near a design's asymptote.

    Targets that no search met within n_max are still plotted, at the ceiling the
    search exits on -- as the MATLAB does, which has no notion of censoring: it
    returns whatever n the loop ends on and averages it. Those points are what
    make the paper's curves climb to ~900 patients (n_max = 300 cohorts x 3), and
    the steep rise is the mean sliding from real answers toward the ceiling as
    more rounds fail to reach the target. ``show_censored=False`` drops them if
    you want only points every round actually reached.

    If ``sample_complexity_df`` has a ``scenario`` column with more than one
    distinct value (e.g. several ``run_simulation.sample_complexity()`` calls
    concatenated, one per scenario), the figure grids one subplot per scenario.
    A single scenario, or no ``scenario`` column at all, draws a flat
    single-panel figure.
    """
    # First-appearance order in the ORIGINAL (pre-sort) frame, not sorted --
    # so the caller's scenario ordering (e.g. reference scenario listed first)
    # is preserved rather than scrambled alphabetically or by the design/
    # accuracy sort below.
    scenarios = (list(dict.fromkeys(sample_complexity_df["scenario"]))
                if "scenario" in sample_complexity_df.columns else [None])
    df = sample_complexity_df.sort_values(["design", "accuracy"])
    if not show_censored:
        df = df[~df["censored"]]

    single = len(scenarios) <= 1
    n_rows = 1 if single else round(math.sqrt(len(scenarios)))
    n_cols = 1 if single else math.ceil(len(scenarios) / n_rows)
    sub_titles = None if single else _wrap_all(scenarios, 45)
    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=sub_titles,
    )
    if sub_titles:
        _shrink_panel_titles(fig, sub_titles, font_size=14)

    for s_idx, scen in enumerate(scenarios):
        d_scen = df if scen is None else df[df["scenario"] == scen]
        row_idx, col_idx = s_idx // n_cols, s_idx % n_cols
        for algo_idx, algo in enumerate(d_scen["design"].unique()):
            d = d_scen[d_scen["design"] == algo]
            color = PAPER_COLORS.get(
                algo, DEFAULT_COLORS[algo_idx % len(DEFAULT_COLORS)]
            )
            if show_ci:
                lo_p, hi_p = _sample_complexity_band(d, rounds=ci_rounds)
                if lo_p is not None:
                    _add_ci_band(fig, d["accuracy"], lo_p, hi_p, color,
                                 row_idx + 1, col_idx + 1)
            # Lines only by default, matching the paper's Figure 3. Markers are
            # available because the delta grid is coarse (10 points), so a plain
            # line implies more resolution than there is.
            fig.add_trace(
                go.Scatter(
                    x=d["accuracy"], y=d["n_patients"],
                    name=display_name(algo),
                    mode="lines+markers" if show_markers else "lines",
                    marker_color=color,
                    marker=dict(
                        size=8,
                        symbol=["circle-open" if c else "circle"
                                for c in d["censored"]],
                    ),
                    legendgroup=algo, showlegend=(s_idx == 0),
                ),
                row=row_idx + 1, col=col_idx + 1,
            )
        fig.update_yaxes(
            range=[0, y_max] if y_max is not None else None,
            rangemode="tozero" if y_max is None else None,
            row=row_idx + 1, col=col_idx + 1,
        )

    fig.update_xaxes(title_text="Recommendation Accuracy")
    fig.update_yaxes(title_text="Number of Patients", col=1)
    fig.update_layout(
        autosize=False,
        font=dict(size=BASE_FONT),
        width=(unit_width or 600) * n_cols + 350,
        height=(unit_height or 400) * n_rows + 150,
        title_text=title_text or (
            "Sample complexity: patients to reach a given recommendation accuracy"
        ),
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
    return fig


def plot_scenario_dashboard(
    label: str,
    n_cohorts: int,
    n_trials: int,
    n_levels: int,
    true_toxicity: Sequence[float],
    nominal_toxicity: Sequence[float],
    efficacy: Sequence[float],
    ttl: float,
    optimal_dose: int,
    rec_final: Dict[str, List[int]],
    alloc: Dict[str, List[int]],
    cohort_rec: Dict[str, Dict[str, List[int]]],
    cohort_safe: Dict[str, Dict[str, List[Optional[List[bool]]]]],
    enrolled: Dict[str, List[bool]],
    algos: Sequence[str],
    safety_algos: Optional[Sequence[str]] = None,
    sample_complexity=None,
    sc_rounds: Optional[int] = None,
    show_ci: bool = True,
    unit_width: int = 620,
    unit_height: int = 300,
    show_fig: bool = False,
    img_path: Optional[str] = None,
    html_path: Optional[str] = None,
):
    """
    Every metric for ONE scenario on a single page, designs overlaid in each
    panel. Complements the per-metric figures (one metric across all
    scenarios): use those to compare scenarios, this to read one scenario whole.

    Ten panels, 5 rows x 2 columns:
        setup      dose-toxicity curve        | dose-efficacy curve
        outcome    recommendation % per dose  | allocation % per dose
        learning   recommendation accuracy    | allocation accuracy
        safety     type I error               | type II error
        tradeoff   efficacy per patient       | safety violation %

    ``safety_algos`` restricts the error-rate row to designs that have a safety
    model (3 + 3 has none). Bands and error bars use the same bootstrap over
    trials as everywhere else.
    """
    algos = list(algos)
    safety_algos = list(safety_algos) if safety_algos is not None else algos
    doses = list(range(1, n_levels + 1))
    cohort_names = list(cohort_rec.keys())
    tox = np.asarray(true_toxicity, dtype=float)
    nom = np.asarray(nominal_toxicity, dtype=float)
    eff_true = np.asarray(efficacy, dtype=float)
    safe_idx = [k for k in range(n_levels) if tox[k] <= ttl]
    unsafe_idx = [k for k in range(n_levels) if tox[k] > ttl]

    has_sc = sample_complexity is not None and len(sample_complexity)
    n_rows = 6 if has_sc else 5
    titles = [
        "Dose-toxicity curve", "Dose-efficacy curve",
        # Row 2 reports each trial's FINAL pick, one per trial. That is NOT the
        # trial-average over all rounds that the recommendation-allocation
        # tables report, so the title says which.
        "Final recommendation % per dose", "Allocation % per dose (all cohorts)",
        "Recommendation accuracy progression", "Allocation accuracy progression",
        "Type I error rate", "Type II error rate",
        "Efficacy per patient", "Safety violation rate",
    ]
    specs = [[{}, {}]] * 5
    if has_sc:
        titles.append("Patients needed to reach a recommendation accuracy")
        specs = specs + [[{"colspan": 2}, None]]

    fig = make_subplots(
        rows=n_rows, cols=2, specs=specs, subplot_titles=titles,
        vertical_spacing=0.075, horizontal_spacing=0.085,
    )

    def color(a):
        return PAPER_COLORS.get(a, DEFAULT_COLORS[algos.index(a) % len(DEFAULT_COLORS)])

    # ---- row 1: the scenario itself (true solid, nominal dashed) ---------
    fig.add_trace(go.Scatter(x=doses, y=list(tox), mode="lines+markers",
                             marker_color=DEFAULT_COLORS[0], showlegend=False),
                  row=1, col=1)
    if not np.allclose(nom, tox):
        fig.add_trace(go.Scatter(x=doses, y=list(nom), mode="lines",
                                 line=dict(color=DEFAULT_COLORS[0], dash="dash"),
                                 showlegend=False), row=1, col=1)
    fig.add_hline(y=ttl, line_dash="dash", line_color="gray",
                  annotation_text="θ", annotation_position="top left",
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=doses, y=list(eff_true), mode="lines+markers",
                             marker_color=DEFAULT_COLORS[2], showlegend=False),
                  row=1, col=2)
    for c in (1, 2):
        fig.add_vline(x=optimal_dose + 1, line_dash="dot", line_color="green",
                      row=1, col=c)
        fig.update_yaxes(range=[0, 1], row=1, col=c)
        fig.update_xaxes(dtick=1, row=1, col=c)

    # ---- row 2: where each design ends up -------------------------------
    for src, col in ((rec_final, 1), (alloc, 2)):
        for a in algos:
            arr = np.asarray(src[a]).reshape(n_trials, -1)
            frac = np.stack([(arr == k).mean(axis=1) for k in range(n_levels)],
                            axis=1) * 100.0
            mean = frac.mean(axis=0)
            err = None
            if show_ci:
                lo, hi = _bootstrap_band(frac)
                err = dict(type="data", symmetric=False,
                           array=list(np.asarray(hi) - mean),
                           arrayminus=list(mean - np.asarray(lo)),
                           thickness=1, width=2, color="#444")
            fig.add_trace(
                go.Bar(x=doses, y=list(mean), name=display_name(a), marker_color=color(a),
                       error_y=err, legendgroup=a, showlegend=(col == 1)),
                row=2, col=col,
            )
        fig.update_xaxes(dtick=1, row=2, col=col)

    # ---- row 3: how fast they get there ---------------------------------
    for a in algos:
        hits = np.array([[d == optimal_dose for d in cohort_rec[c][a]]
                         for c in cohort_names], dtype=float).T
        allocs = np.asarray(alloc[a]).reshape(n_trials, n_cohorts)
        ahits = (allocs == optimal_dose).astype(float)
        for col, m in ((1, hits), (2, ahits)):
            if show_ci:
                lo, hi = _bootstrap_band(m)
                _add_ci_band(fig, range(n_cohorts), lo, hi, color(a), 3, col)
            fig.add_trace(
                go.Scatter(x=list(range(n_cohorts)), y=list(m.mean(axis=0)),
                           mode="lines", marker_color=color(a), name=display_name(a),
                           legendgroup=a, showlegend=False),
                row=3, col=col,
            )

    # ---- row 4: safety classification errors ----------------------------
    for a in safety_algos:
        first = cohort_safe[cohort_names[0]][a]
        if not first or all(s is None for s in first):
            continue
        p1, p2 = [], []
        for c_i, c in enumerate(cohort_names):
            sets = ([] if c_i < n_levels
                    else [s for s in cohort_safe[c][a] if s is not None])
            if not sets:
                p1.append(None)
                p2.append(None)
                continue
            arr = np.asarray(sets, dtype=bool)
            p1.append((~arr[:, safe_idx]).mean(axis=1) * 100 if safe_idx else None)
            p2.append(arr[:, unsafe_idx].mean(axis=1) * 100 if unsafe_idx else None)
        n_tr = max((len(v) for v in p1 + p2 if v is not None), default=0)
        if not n_tr:
            continue
        for col, per in ((1, p1), (2, p2)):
            m = np.column_stack([np.full(n_tr, np.nan) if v is None else v
                                 for v in per])
            if show_ci:
                lo, hi = _bootstrap_band(m)
                _add_ci_band(fig, range(n_cohorts), lo, hi, color(a), 4, col)
            fig.add_trace(
                go.Scatter(x=list(range(n_cohorts)), y=list(_nanmean_cols(m)),
                           mode="lines", marker_color=color(a), name=display_name(a),
                           legendgroup=a, showlegend=False),
                row=4, col=col,
            )

    # ---- row 5: the efficacy/safety tradeoff ----------------------------
    unsafe = (tox > ttl).astype(float)
    for a in algos:
        allocs = np.asarray(alloc[a]).reshape(n_trials, n_cohorts)
        enr = np.asarray(enrolled[a], dtype=float).reshape(n_trials, n_cohorts)
        n_pat = np.cumsum(enr, axis=1)
        n_pat = np.where(n_pat == 0, 1, n_pat)
        eff_cum = np.cumsum(eff_true[allocs] * enr, axis=1) / n_pat
        viol_cum = np.cumsum(unsafe[allocs] * enr, axis=1) / n_pat * 100.0
        for col, m in ((1, eff_cum), (2, viol_cum)):
            if show_ci:
                lo, hi = _bootstrap_band(m)
                _add_ci_band(fig, range(n_cohorts), lo, hi, color(a), 5, col)
            fig.add_trace(
                go.Scatter(x=list(range(n_cohorts)), y=list(m.mean(axis=0)),
                           mode="lines", marker_color=color(a), name=display_name(a),
                           legendgroup=a, showlegend=False),
                row=5, col=col,
            )

    # ---- row 6: sample complexity, spanning both columns ----------------
    # Same data and the same band as the standalone figure -- this is that run's
    # rows for this scenario, not a re-derivation. The band is wide because the
    # horizon is a mean over ~20 searches, against 1000 trials everywhere else.
    if has_sc:
        for a in algos:
            d = sample_complexity[sample_complexity["design"] == a]
            if d.empty:
                continue
            d = d.sort_values("accuracy")
            lo, hi = _sample_complexity_band(d, rounds=sc_rounds)
            if show_ci and lo is not None:
                _add_ci_band(fig, list(d["accuracy"]), lo, hi, color(a), 6, 1)
            fig.add_trace(
                go.Scatter(x=list(d["accuracy"]), y=list(d["n_patients"]),
                           mode="lines+markers", marker_color=color(a),
                           name=display_name(a), legendgroup=a,
                           showlegend=False),
                row=6, col=1,
            )
        fig.update_xaxes(title_text="Recommendation accuracy target",
                         row=6, col=1)
        fig.update_yaxes(title_text="Patients needed",
                         title_font=dict(size=BASE_FONT - 3), row=6, col=1)

    # Efficacy per patient is a proportion, and on a scenario where every design
    # sits at the same value the autoscale would otherwise expand floating-point
    # noise ~1e-16 wide into the full axis and plot rounding error as signal.
    fig.update_yaxes(range=[0, 1], row=5, col=1)

    for r, c, t in ((1, 1, "Toxicity prob."), (1, 2, "Efficacy prob."),
                    (2, 1, "% of trials"), (2, 2, "% of cohorts"),
                    (3, 1, "Accuracy"), (3, 2, "Accuracy"),
                    (4, 1, "Type I (%)"), (4, 2, "Type II (%)"),
                    (5, 1, "Efficacy rate"), (5, 2, "Violation rate")):
        fig.update_yaxes(title_text=t, title_font=dict(size=BASE_FONT - 3),
                         row=r, col=c)
    for c in (1, 2):
        fig.update_xaxes(title_text="Dose level", row=1, col=c)
        fig.update_xaxes(title_text="Dose level", row=2, col=c)
        for r in (3, 4, 5):
            fig.update_xaxes(title_text="Cohort number", row=r, col=c)
    fig.update_xaxes(title_font=dict(size=BASE_FONT - 3),
                     tickfont=dict(size=BASE_FONT - 4))
    fig.update_yaxes(tickfont=dict(size=BASE_FONT - 4))
    # Panel titles: make_subplots leaves their size unset, so they would render
    # at the plotly default rather than following BASE_FONT.
    for ann in fig.layout.annotations:
        if ann.font.size is None:
            ann.font = dict(size=BASE_FONT - 2)

    fig.update_layout(
        autosize=False, barmode="group", font=dict(size=BASE_FONT),
        width=unit_width * 2 + 300, height=unit_height * n_rows + 150,
        title_text=label,
        # The legend sits just under the title; a large top margin used to leave
        # a band of empty page between the two.
        legend=dict(orientation="h", yanchor="bottom", y=1.028,
                    xanchor="left", x=0, font=dict(size=BASE_FONT - 2)),
        margin=dict(t=135, r=60),
        title=dict(y=1, yanchor="top", pad=dict(t=10),
                   font=dict(size=BASE_FONT + 4)),
    )

    if img_path:
        os.makedirs(os.path.dirname(img_path) or ".", exist_ok=True)
        fig.write_image(img_path)
    if html_path:
        os.makedirs(os.path.dirname(html_path) or ".", exist_ok=True)
        fig.write_html(html_path)
    if show_fig:
        fig.show()
    return fig
