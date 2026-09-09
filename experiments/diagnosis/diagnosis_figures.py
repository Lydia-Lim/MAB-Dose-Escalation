"""Diagnosis figures: where SEEDA-Plateau's recommendation rule fails.

Two figures, both on the paper's main scenario, both with trial-level bootstrap
95% confidence intervals -- trials are the resampling unit everywhere in this
project, as in ``experiments/bootstrap.py``.

``fig_flat_test``
    The L1 flat test's tolerance for each adjacent pair of doses, cohort by
    cohort, against the TRUE efficacy gap that pair has to resolve. A pair reads
    as flat whenever the tolerance exceeds the gap, so any curve sitting above
    its own dashed reference line is a real rise being scored as a plateau. This
    is the mechanism behind the printed algorithm's collapse onto dose 1: its
    tolerance never falls below the 0.25 rise, so the scan finds its first flat
    pair at the bottom of the admissible set and stops there.

``fig_l1_l2``
    Which of the two terms of ``min(L1, L2)`` binds, and where the resulting
    recommendation sits relative to the toxicity MTD the design itself
    estimated. The MATLAB's ascending overwrite can only return the MTD or the
    dose below it, so its recommendation is confined to two values whatever the
    efficacy data say; the printed scan has no such floor and lands far below.

Both read the designs' own ``flat_threshold`` and the ``_last_l1`` / ``_last_l2``
recorded inside ``propose``, so nothing here re-derives a rule that the
implementations already state.

Imported by ``diagnosis_tables.py``, which passes the run folder so the figures
land beside the tables of the same run. Runnable alone for a quick look.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path
from typing import Callable, Dict, List, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "experiments"))
sys.path.insert(0, str(HERE))

from doseescalation.evaluate import _bootstrap_band
from doseescalation.run_simulation import dose_toxic_curve
from doseescalation.dose_escalator import (
    SEEDAPlateauFixedDoseEscalator,
    SEEDAPlateauOursDoseEscalator,
    SEEDAPlateauTwoSidedDecoupledDoseEscalator,
)

from _seeda_plateau_matlab import SEEDAPlateauMatlabDoseEscalator
from _seeda_plateau_naive import SEEDAPlateauNaiveDoseEscalator

# Short display labels. The thesis renamed all renderings of the algorithms
# to be as clear as possible. It says "Algorithm 2 as printed" where the
# distinction from the other renderings matters; the tables and figures cannot
# afford the words, and every row here is a rendering of Algorithm 2 anyway.
NAME_PRINTED = "Algorithm 2"
NAME_MATLAB = "MATLAB"
NAME_REFERENCE = "Reference"
NAME_DESCENDING = "Descending"
NAME_ANCHORED = "Anchored"

# The three renderings of the published design, then our two variants.
DIAGNOSIS_RENDERINGS = [NAME_PRINTED, NAME_MATLAB, NAME_REFERENCE]
VARIANTS = [NAME_DESCENDING, NAME_ANCHORED]

# One color per design, held across every figure.
RENDERING_COLORS = {
    NAME_PRINTED: "#8c564b",      # brown
    NAME_MATLAB: "#9467bd",       # purple
    NAME_REFERENCE: "#2ca02c",    # green
    NAME_DESCENDING: "#ff8c00",   # orange
    NAME_ANCHORED: "#d62728",     # red
}
# Anchored's test is one-sided and anchored to the MTD rather than pairwise, so
# its tolerance is one number per round instead of one per adjacent pair.
ANCHORED_DESIGNS = {NAME_ANCHORED}
IMG_SCALE = 2

# Bootstrap band styling for the two line figures. At 1000 trials the interval
# is 2--7% of the mean, which on these log axes is about the width of the line
# it surrounds, so the fill alone at a low alpha reads as nothing at all. The
# hairline edges are what make the band findable; the fill is kept light so it
# does not bury the dashed reference line where the two cross.
BAND_FILL_ALPHA = 0.24
BAND_EDGE_ALPHA = 0.40
BAND_EDGE_WIDTH = 0.8


def plateau_factories(cfg, n_cohorts: int) -> Dict[str, Callable]:
    """The three renderings, built exactly as their own runners build them."""
    grid = cfg.dose_levels_plateau
    delta_1 = 1.0 / n_cohorts

    def printed():
        return SEEDAPlateauNaiveDoseEscalator(
            grid, cfg.ttl, dose_toxic_curve,
            ucb_coefficient=cfg.plateau_c, delta_1=delta_1, eta=cfg.eta,
            a0=cfg.a_plateau, use_log_bonus=True, is_training=True, no_skip=True,
        )

    def matlab():
        return SEEDAPlateauMatlabDoseEscalator(
            grid, cfg.ttl, dose_toxic_curve, n_cohorts=n_cohorts,
            a0=cfg.a_plateau, a_max=cfg.a_max_plateau,
            ucb_coefficient=cfg.plateau_c,
            eta=cfg.eta, l1_coefficient=0.2, l1_mode="min",
            is_training=True,
        )

    def reference():
        return SEEDAPlateauTwoSidedDecoupledDoseEscalator(
            dose_levels=grid, target_toxicity_level=cfg.ttl,
            dose_toxicity_curve=dose_toxic_curve,
            ucb_coefficient=cfg.plateau_c, delta_1=delta_1,
            l1_coefficient=cfg.l1_coefficient,
            a_init=cfg.a_plateau, a_max=cfg.a_max_plateau,
            seed=0, eta=cfg.eta, no_skip=True,
        )

    def descending():
        return SEEDAPlateauFixedDoseEscalator(
            dose_levels=grid, target_toxicity_level=cfg.ttl,
            dose_toxicity_curve=dose_toxic_curve,
            ucb_coefficient=cfg.plateau_c, delta_1=delta_1,
            l1_coefficient=cfg.l1_coefficient,
            a_init=cfg.a_plateau, a_max=cfg.a_max_plateau,
            seed=0, eta=cfg.eta, no_skip=True,
        )

    def anchored():
        return SEEDAPlateauOursDoseEscalator(
            dose_levels=grid, target_toxicity_level=cfg.ttl,
            dose_toxicity_curve=dose_toxic_curve,
            ucb_coefficient=cfg.plateau_c, delta_1=delta_1,
            l1_coefficient=cfg.l1_coefficient,
            a_init=cfg.a_plateau, a_max=cfg.a_max_ours,
            p_smoothing=cfg.p_smoothing, tox_margin_c=cfg.tox_margin_c,
            seed=0, eta=cfg.eta, no_skip=True,
        )

    return {NAME_PRINTED: printed,
            NAME_MATLAB: matlab,
            NAME_REFERENCE: reference,
            NAME_DESCENDING: descending,
            NAME_ANCHORED: anchored}


def trace_design(factory, cfg, n_cohorts: int, seed: int,
                 anchored: bool = False) -> dict:
    """Run one design and record, per cohort, what the recommendation was made of.

    Returns arrays shaped (n_trials, n_cohorts[, n_pairs]). ``threshold`` is NaN
    on cohorts where a dose in the pair has no observations yet -- the tolerance
    is undefined there rather than infinite, and averaging in an infinity would
    destroy the panel.

    ``anchored`` designs test each candidate against the MTD rather than against
    its neighbor, so they have ONE tolerance per round, taken at that round's
    own estimated MTD, and it is stored in column 0 with the rest left NaN.
    """
    K = cfg.n_levels
    rng = np.random.default_rng(seed)
    tox = np.asarray(cfg.toxicity_probs, dtype=float)
    eff = np.asarray(cfg.efficacy_probs, dtype=float)

    thresholds = np.full((cfg.n_trials, n_cohorts, K - 1), np.nan)
    rec = np.zeros((cfg.n_trials, n_cohorts), dtype=int)
    l1 = np.full((cfg.n_trials, n_cohorts), -1, dtype=int)
    l2 = np.full((cfg.n_trials, n_cohorts), -1, dtype=int)

    for trial in range(cfg.n_trials):
        escalator = factory()
        for t in range(n_cohorts):
            escalator.train(True)
            allocated = escalator.propose()

            escalator.train(False)
            rec[trial, t] = escalator.propose()
            l1[trial, t] = getattr(escalator, "_last_l1", -1)
            l2[trial, t] = getattr(escalator, "_last_l2", -1)
            targets = [l2[trial, t]] if anchored else range(K - 1)
            for slot, m in enumerate(targets):
                if m < 0:
                    continue
                with np.errstate(divide="ignore", invalid="ignore"):
                    width = escalator.flat_threshold(m)
                if np.isfinite(width):
                    thresholds[trial, t, slot] = width

            escalator.train(True)
            escalator.update(
                allocated, cfg.cohort_size,
                int(rng.binomial(cfg.cohort_size, tox[allocated])),
                int(rng.binomial(cfg.cohort_size, eff[allocated])),
            )

    return {"threshold": thresholds, "rec": rec, "l1": l1, "l2": l2}


def _post_warmup(trace: dict, K: int) -> np.ndarray:
    """Mask of the rounds on which L1 and L2 are both defined, for every design.

    The printed algorithm and the MATLAB both return the round-robin dose
    outright during the first K cohorts, so neither computes a recommendation
    there; Reference does compute one. Dropping the warm-up for all three keeps
    the panel comparing the same rounds rather than three different windows.
    """
    valid = (trace["l1"] >= 0) & (trace["l2"] >= 0)
    valid[:, :K] = False
    return valid


def _nan_bootstrap(per_trial: np.ndarray):
    """Bootstrap band for column means of an array that may hold NaNs.

    ``_bootstrap_band`` takes a plain mean over trials, so a single NaN cohort
    would wipe out the whole column. Trials are still the resampling unit; the
    columns just average over the trials that have a value there.
    """
    x = np.asarray(per_trial, dtype=float)
    mask = np.isfinite(x).astype(float)
    filled = np.where(np.isfinite(x), x, 0.0)
    lo_s, hi_s = _bootstrap_band(filled)
    lo_n, hi_n = _bootstrap_band(mask)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.divide(lo_s, lo_n), np.divide(hi_s, hi_n)


def _write_series(rows: List[dict], csv_path: Path | None) -> None:
    """Persist a figure's plotted series, so a number can be quoted from the
    data rather than read off a plot. Nothing else consumes these files."""
    if csv_path is None:
        return
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(csv_path, index=False)


def fig_flat_test(traces: Dict[str, dict], cfg, n_cohorts: int, path: Path,
                  csv_path: Path | None = None):
    """Panel per rendering: L1's tolerance per adjacent pair vs the true gap."""
    eff = np.asarray(cfg.efficacy_probs, dtype=float)
    true_gaps = np.abs(np.diff(eff))
    names = list(traces)
    K = cfg.n_levels
    pair_colors = ["#4c78a8", "#f58518", "#54a24b", "#b279a2", "#9d755d"]

    fig = make_subplots(
        rows=1, cols=len(names), shared_yaxes=True, horizontal_spacing=0.045,
        subplot_titles=names,
    )
    x = list(range(1, n_cohorts + 1))
    rows: List[dict] = []

    for col, name in enumerate(names, start=1):
        thresholds = traces[name]["threshold"]
        for m in range(K - 1):
            series = thresholds[:, :, m]
            # The warm-up cohorts are all-NaN in every column until the pair's
            # two doses have been sampled; nanmean says so rather than erroring.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                mean = np.nanmean(series, axis=0)
            lo, hi = _nan_bootstrap(series)
            color = pair_colors[m % len(pair_colors)]
            rising = true_gaps[m] > 0
            label = (f"Doses {m + 1}–{m + 2}"
                     + (f"  (gap {true_gaps[m]:.2f})" if rising
                        else "  (plateau)"))
            edge = dict(color=_rgba(color, BAND_EDGE_ALPHA),
                        width=BAND_EDGE_WIDTH)
            rows.extend(
                {"design": name, "pair": f"{m + 1}-{m + 2}",
                 "true_gap": float(true_gaps[m]), "cohort": c,
                 "tolerance_mean": float(mean[i]),
                 "ci_lo": float(lo[i]), "ci_hi": float(hi[i])}
                for i, c in enumerate(x)
            )
            fig.add_trace(
                go.Scatter(x=x, y=list(hi), mode="lines", line=edge,
                           hoverinfo="skip", showlegend=False),
                row=1, col=col,
            )
            fig.add_trace(
                go.Scatter(x=x, y=list(lo), mode="lines", line=edge,
                           fill="tonexty",
                           fillcolor=_rgba(color, BAND_FILL_ALPHA),
                           hoverinfo="skip", showlegend=False),
                row=1, col=col,
            )
            fig.add_trace(
                go.Scatter(x=x, y=list(mean), mode="lines", name=label,
                           legendgroup=label, showlegend=(col == 1),
                           line=dict(color=color, width=2,
                                     dash="solid" if rising else "dot")),
                row=1, col=col,
            )

        # The only gap the test can actually get wrong. The plateau pairs are
        # genuinely flat, so no positive tolerance misclassifies them and there
        # is nothing to draw for them (log axis, and their gap is exactly zero).
        fig.add_hline(
            y=float(np.max(true_gaps)),
            line=dict(color="black", width=1.5, dash="dash"), row=1, col=col,
        )

    top_gap = float(np.max(true_gaps))
    fig.add_annotation(
        x=n_cohorts, y=np.log10(top_gap), xref="x", yref="y",
        text=f"true gap = {top_gap:.2f}", showarrow=False,
        xanchor="right", yanchor="bottom", font=dict(size=13, color="black"),
    )
    fig.update_xaxes(title_text="Cohort", title_font_size=15, tickfont_size=13)
    # Log scale: the printed algorithm's tolerance runs an order of magnitude
    # above the MATLAB's, and a linear axis clips its two widest pairs entirely.
    fig.update_yaxes(type="log", range=[np.log10(0.02), np.log10(4.0)],
                     tickvals=[0.02, 0.05, 0.1, 0.25, 0.5, 1, 2, 4],
                     tickfont_size=13)
    fig.update_yaxes(title_text="L1 flat-test tolerance",
                     title_font_size=15, row=1, col=1)
    fig.update_layout(
        title=dict(
            text=("L1 declares a pair flat whenever its tolerance exceeds the "
                  f"true efficacy gap<br><sub>Paper's main scenario, "
                  f"n = {n_cohorts} cohorts; mean over {cfg.n_trials} trials, "
                  f"bootstrap 95% CI</sub>"),
            font=dict(size=17),
        ),
        legend=dict(title_text="Adjacent pair", font=dict(size=13),
                    x=1.01, xanchor="left", y=1, yanchor="top"),
        width=1180, height=460, margin=dict(t=110, r=170),
        template="plotly_white",
    )
    for annotation in fig.layout.annotations[:len(names)]:
        annotation.font.size = 15
    _save(fig, path)
    _write_series(rows, csv_path)


def fig_l1_l2(traces: Dict[str, dict], cfg, n_cohorts: int, mtd: int,
              path: Path, csv_path: Path | None = None):
    """Which term binds, and how far the recommendation sits below the MTD.

    ``mtd`` is the scenario's TRUE MTD, used only to say in the subtitle what the
    offsets are not measured against -- the axis itself is each design's own L2.
    """
    names = list(traces)
    K = cfg.n_levels
    offsets = list(range(-(K - 1), 1))

    fig = make_subplots(
        rows=1, cols=2, horizontal_spacing=0.13,
        subplot_titles=(
            "Share of rounds where L1 is the binding term",
            "Recommendation minus the design's own estimated MTD (L2)",
        ),
    )

    rows: List[dict] = []

    for name in names:
        tr = traces[name]
        color = RENDERING_COLORS[name]
        valid = _post_warmup(tr, K)
        binds = np.where(valid, (tr["l1"] < tr["l2"]).astype(float), np.nan)
        per_trial = np.nanmean(binds, axis=1) * 100
        lo, hi = _bootstrap_band(per_trial[:, None])
        rows.append({"panel": "L1 binds", "design": name, "offset": "",
                     "pct_of_rounds": float(np.nanmean(per_trial)),
                     "ci_lo": float(lo[0]), "ci_hi": float(hi[0])})
        fig.add_trace(
            go.Bar(x=[name], y=[float(np.nanmean(per_trial))],
                   marker_color=color, showlegend=False,
                   error_y=dict(type="data", symmetric=False,
                                array=[float(hi[0] - np.nanmean(per_trial))],
                                arrayminus=[float(np.nanmean(per_trial) - lo[0])],
                                color="#444", thickness=1.4, width=6)),
            row=1, col=1,
        )

        # Offset of the recommendation from the MTD the design itself estimated,
        # so the confinement is read off the design's own belief rather than the
        # scenario's answer.
        offset = np.where(valid, tr["rec"] - tr["l2"], np.nan)
        shares, los, his = [], [], []
        for off in offsets:
            per_trial_share = np.nanmean(
                np.where(valid, (offset == off).astype(float), np.nan), axis=1
            ) * 100
            mean = float(np.nanmean(per_trial_share))
            lo, hi = _bootstrap_band(per_trial_share[:, None])
            shares.append(mean)
            los.append(mean - float(lo[0]))
            his.append(float(hi[0]) - mean)
            rows.append({"panel": "rec - L2", "design": name, "offset": off,
                         "pct_of_rounds": mean,
                         "ci_lo": float(lo[0]), "ci_hi": float(hi[0])})
        fig.add_trace(
            go.Bar(x=[str(o) for o in offsets], y=shares, name=name,
                   marker_color=color,
                   error_y=dict(type="data", symmetric=False, array=his,
                                arrayminus=los, color="#444",
                                thickness=1.4, width=4)),
            row=1, col=2,
        )

    fig.update_yaxes(title_text="% of rounds", range=[0, 105],
                     title_font_size=15, tickfont_size=13, row=1, col=1)
    fig.update_yaxes(title_text="% of rounds", range=[0, 105],
                     title_font_size=15, tickfont_size=13, row=1, col=2)
    fig.update_xaxes(tickfont_size=13, row=1, col=1)
    fig.update_xaxes(title_text="Recommendation − estimated MTD (doses)",
                     title_font_size=15, tickfont_size=13, row=1, col=2)
    fig.update_layout(
        barmode="group",
        title=dict(
            text=("What min(L1, L2) resolves to: how often the plateau term "
                  "binds, and how far below the MTD it reaches"
                  f"<br><sub>Paper's main scenario, "
                  f"n = {n_cohorts} cohorts; mean over {cfg.n_trials} trials, "
                  "bootstrap 95% CI."
                  "<br>Offsets are against each design's OWN estimated MTD (L2), "
                  f"not the scenario's true MTD of dose {mtd + 1}: an unclamped "
                  "a_hat can place L2 higher, which is how offsets past −3 "
                  "arise.</sub>"),
            font=dict(size=17),
        ),
        legend=dict(font=dict(size=13), x=1.01, xanchor="left", y=1,
                    yanchor="top"),
        width=1180, height=510, margin=dict(t=165, r=200),
        template="plotly_white",
    )
    for annotation in fig.layout.annotations[:2]:
        annotation.font.size = 15
    _save(fig, path)
    _write_series(rows, csv_path)


def comparisons(name: str, cfg, mtd: int) -> List[tuple]:
    """The judgements a design's test actually has to get right, and their true gaps.

    Returns ``(label, true_gap, threshold_column)`` for every comparison with a
    REAL efficacy difference to resolve. Comparisons whose true gap is zero are
    left out: those pairs are genuinely flat, so no positive tolerance can
    misclassify them and they carry no information about resolving power.

    The pairwise designs judge each dose against its neighbor; Anchored judges
    each dose against the MTD, so it answers a harder question -- the gap from
    dose 1 to the plateau is the sum of the steps, not one step.
    """
    eff = np.asarray(cfg.efficacy_probs, dtype=float)
    if name in ANCHORED_DESIGNS:
        return [(f"Dose {k + 1} vs MTD", abs(eff[mtd] - eff[k]), 0)
                for k in range(mtd) if abs(eff[mtd] - eff[k]) > 0]
    gaps = np.abs(np.diff(eff))
    return [(f"Doses {m + 1}–{m + 2}", gaps[m], m)
            for m in range(len(gaps)) if gaps[m] > 0]


def fig_resolving_power(traces: Dict[str, dict], cfg, n_cohorts: int, mtd: int,
                        path: Path, csv_path: Path | None = None):
    """Tolerance as a MULTIPLE of the true gap it has to resolve, for every design.

    Dividing by the gap is what makes the five comparable: the pairwise tests and
    Anchored's MTD-anchored test are applied to different differences, so their
    raw widths are not on the same footing, but "how many times larger than the
    difference it must detect" is. Above 1, the test cannot tell a real efficacy
    rise from noise and will call it a plateau.
    """
    names = list(traces)
    fig = make_subplots(
        rows=1, cols=len(names), shared_yaxes=True, horizontal_spacing=0.03,
        subplot_titles=names,
    )
    x = list(range(1, n_cohorts + 1))
    pair_colors = ["#4c78a8", "#f58518", "#54a24b", "#b279a2", "#9d755d"]
    # The pairwise designs share their comparisons, so a per-panel legend would
    # repeat the same four entries five times.
    seen: set = set()
    rows: List[dict] = []

    for col, name in enumerate(names, start=1):
        thresholds = traces[name]["threshold"]
        for j, (label, gap, column) in enumerate(comparisons(name, cfg, mtd)):
            series = thresholds[:, :, column] / gap
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                mean = np.nanmean(series, axis=0)
            lo, hi = _nan_bootstrap(series)
            color = pair_colors[j % len(pair_colors)]
            legend_label = f"{label}  (gap {gap:.2f})"
            edge = dict(color=_rgba(color, BAND_EDGE_ALPHA),
                        width=BAND_EDGE_WIDTH)
            rows.extend(
                {"design": name, "comparison": label, "true_gap": float(gap),
                 "cohort": c, "ratio_mean": float(mean[i]),
                 "ci_lo": float(lo[i]), "ci_hi": float(hi[i])}
                for i, c in enumerate(x)
            )
            fig.add_trace(
                go.Scatter(x=x, y=list(hi), mode="lines", line=edge,
                           hoverinfo="skip", showlegend=False), row=1, col=col)
            fig.add_trace(
                go.Scatter(x=x, y=list(lo), mode="lines", line=edge,
                           fill="tonexty",
                           fillcolor=_rgba(color, BAND_FILL_ALPHA),
                           hoverinfo="skip", showlegend=False), row=1, col=col)
            fig.add_trace(
                go.Scatter(x=x, y=list(mean), mode="lines", name=legend_label,
                           legendgroup=legend_label,
                           showlegend=legend_label not in seen,
                           line=dict(color=color, width=2)),
                row=1, col=col)
            seen.add(legend_label)
        fig.add_hline(y=1.0, line=dict(color="black", width=1.5, dash="dash"),
                      row=1, col=col)

    fig.add_annotation(
        x=n_cohorts, y=0.0, xref="x", yref="y",
        text="tolerance = true gap", showarrow=False, xanchor="right",
        yanchor="bottom", font=dict(size=13, color="black"),
    )
    fig.update_xaxes(title_text="Cohort", title_font_size=15, tickfont_size=13)
    fig.update_yaxes(type="log", range=[np.log10(0.05), np.log10(20)],
                     tickvals=[0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 20],
                     tickfont_size=13)
    fig.update_yaxes(title_text="Tolerance ÷ true efficacy gap",
                     title_font_size=15, row=1, col=1)
    fig.update_layout(
        title=dict(
            text=("Above the line, the plateau test cannot tell a real efficacy "
                  "rise from noise"
                  f"<br><sub>Paper's main scenario, n = {n_cohorts} cohorts; "
                  f"mean over {cfg.n_trials} trials, bootstrap 95% CI. Anchored "
                  "measures each dose against the MTD, the others against the "
                  "neighboring dose.</sub>"),
            font=dict(size=17),
        ),
        legend=dict(title_text="Comparison", font=dict(size=13),
                    x=1.01, xanchor="left", y=1, yanchor="top"),
        width=1500, height=470, margin=dict(t=115, r=210),
        template="plotly_white",
    )
    for annotation in fig.layout.annotations[:len(names)]:
        annotation.font.size = 15
    _save(fig, path)
    _write_series(rows, csv_path)


def _rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def _save(fig, path: Path):
    # PNG only, as the battery and the random-scenario study do -- neither
    # writes the interactive HTML, and the thesis only ever uses the image.
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_image(str(path), scale=IMG_SCALE)


def build_figures(cfg, n_cohorts: int, out: Path, mtd: int,
                  seed: int = 0, data_out: Path | None = None) -> Dict[str, dict]:
    """Run all five designs and write the diagnosis and variant figures into ``out``.

    ``fig_*`` are the diagnosis figures and carry the three renderings of the
    published design only. ``fig_variants_*`` repeat them with Descending and
    Anchored added, for the chapter that introduces the variants -- the same
    question asked of our own designs, so the answer is comparable.

    ``data_out``, when given, receives one CSV per figure holding the series it
    plots, means and interval bounds alike, so a number can be quoted from the
    data rather than read off a plot.
    """
    traces = {}
    for name, factory in plateau_factories(cfg, n_cohorts).items():
        print(f"  tracing {name} ...", flush=True)
        traces[name] = trace_design(factory, cfg, n_cohorts, seed,
                                    anchored=name in ANCHORED_DESIGNS)

    def csv(stem: str) -> Path | None:
        return None if data_out is None else data_out / f"{stem}.csv"

    renderings = {k: traces[k] for k in DIAGNOSIS_RENDERINGS}
    fig_flat_test(renderings, cfg, n_cohorts, out / "fig_flat_test.png",
                  csv("fig_flat_test"))
    fig_l1_l2(renderings, cfg, n_cohorts, mtd, out / "fig_l1_l2.png",
              csv("fig_l1_l2"))

    fig_resolving_power(traces, cfg, n_cohorts, mtd,
                        out / "fig_variants_resolving_power.png",
                        csv("fig_variants_resolving_power"))
    fig_l1_l2(traces, cfg, n_cohorts, mtd, out / "fig_variants_l1_l2.png",
              csv("fig_variants_l1_l2"))
    return traces


def summarize(traces: Dict[str, dict], cfg, mtd: int) -> None:
    """Print the numbers the chapter quotes, so they are never read off a plot."""
    print("\n=== resolving power at the final cohort (tolerance / true gap) ===")
    for name, tr in traces.items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            final = np.nanmean(tr["threshold"][:, -1, :], axis=0)
        cells = "  ".join(
            f"{label}: {final[column] / gap:.2f}x"
            for label, gap, column in comparisons(name, cfg, mtd)
        )
        print(f"{name:>12}  {cells}")

    print("\n=== recommendation relative to the design's own estimated MTD ===")
    for name, tr in traces.items():
        offset = np.where(_post_warmup(tr, cfg.n_levels),
                          tr["rec"] - tr["l2"], np.nan)
        shares = {
            off: 100 * np.nanmean((offset == off).astype(float)
                                  [np.isfinite(offset)])
            for off in range(-(cfg.n_levels - 1), 1)
        }
        cells = "  ".join(f"{off}: {v:.2f}%" for off, v in shares.items()
                          if v > 0.005)
        print(f"{name:>12}  {cells}")


def main():
    import argparse
    import datetime as dt

    from scenario_battery import SCENARIOS, make_config, TABLE2_COHORTS

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=None)
    args = parser.parse_args()

    stamp = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out = HERE / "runs" / stamp / "Results" / "Figures"
    sc = SCENARIOS[0]
    assert sc.name == "Paper's main scenario", sc.name
    cfg = make_config(sc, label=f"diagnosis-figures {stamp}")
    if args.trials is not None:
        cfg.n_trials = args.trials

    traces = build_figures(cfg, TABLE2_COHORTS, out, sc.tox_mtd,
                           data_out=out.parent.parent / "Data")
    summarize(traces, cfg, sc.tox_mtd)
    print(f"\nWrote figures into {out}")


if __name__ == "__main__":
    main()
