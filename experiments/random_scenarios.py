"""
Random-scenario study: how does each design perform *in general*?

The nine hand-built scenarios in scenario_battery.py each answer a specific
question, but they are chosen, so they cannot answer "which design would I pick
knowing nothing about the trial". This samples scenarios at random instead and
averages over them.

⚠ THE SCENARIO DISTRIBUTION IS THE CLAIM. There is no neutral way to sample
scenarios, and the prior decides the headline. Concretely: the published Plateau
rule can only ever return the MTD or MTD-1, so if the plateau onset usually sits
at MTD-1 it scores ~99%, and if the onset is drawn uniformly it scores ~0
whenever the onset is lower. Same code, opposite conclusion, purely from the
prior. This script therefore reports BOTH the pooled average and a breakdown by
stratum, and the write-up should quote the stratified table first.

GENERATOR (K = 6 doses, theta from scenario_battery):
  toxicity  monotone non-decreasing, at least 2 safe doses and (usually) an
            unsafe one, so there is a real MTD to find.
  efficacy  increase-then-plateau (the paper's Assumption 1): a plateau onset N
            is drawn UNIFORMLY over the safe doses, values rise to it and are
            flat after. Since k* = min(N, MTD), that makes k* uniform over the
            safe doses -- the point of the exercise.
  model     half the scenarios are well specified (the true curve is in the
            power-of-tanh family the designs assume); half draw their true
            curve from an off-family logistic instead. Reported separately, not
            pooled, so the two effects are never blended.

STRATIFIERS reported: distance MTD - k*, and well-specified vs misspecified.

Averages are over SCENARIOS, so the scenario is the resampling unit for the
confidence interval -- the same principle as bootstrap.py, one level up. We
report the between-scenario standard deviation too: that is the spread of a
design's performance across the scenario space, which is the thing a
practitioner actually cares about, and it is much larger than the CI.

SIMULATION BUDGET matches scenario_battery.py exactly (1000 trials, 300
cohorts). Between-scenario variance dominates the average, so fewer trials per
scenario would buy more scenarios for the same compute -- but keeping the
battery's budget means the two studies differ only in how scenarios are chosen,
which is the whole point of the comparison.

FIGURES are the battery's own figures, averaged over scenarios: the per-cohort
curves (accuracy progression, error rates, efficacy and safety violation) are
kept for every scenario and averaged within each stratum, with a band that
bootstraps SCENARIOS. Rows are well-specified / misspecified, columns are the
strata. The only bar charts are the dose distributions, and those are histograms
over dose offsets, where bars are the right form. Everything is persisted, so
`--redraw <run dir>` rebuilds the figures without re-simulating.

Run:
    python experiments/random_scenarios.py
    python experiments/random_scenarios.py --n-scenarios 50   # quick look
    python experiments/random_scenarios.py --redraw experiments/plots/<run>
"""

import argparse
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scenario_battery import (
    A_NOMINAL,
    ALGOS,
    COHORT_SIZE,
    CURVE_UNIT_HEIGHT,
    CURVE_UNIT_WIDTH,
    SHARED,
    TTL,
    Scenario,
    make_config,
)
from doseescalation.evaluate import (
    BASE_FONT,
    display_name,
    plot_scenario_curves,
)
from doseescalation.run_simulation import (
    a_key,
    inv_dose_toxic,
    logistic_toxic_curve,
    run_designs,
)

N_SCENARIOS = 200      # scenarios sampled (half well-specified, half not)
# Example curves drawn for the write-up. Four fills one figure of a shape that
# keeps the full text width in LaTeX; more would spill into a second file and
# the thesis wants a single illustrative figure here.
N_CURVE_EXAMPLES = 4
# Deliberately IDENTICAL to scenario_battery.py (N_TRIALS, FIGURES_COHORTS), so
# the random-scenario figures and the battery's figures are directly
# comparable and no result can be attributed to a change of simulation budget.
N_TRIALS = 1000
N_COHORTS = 300
N_LEVELS = 6
SEED = 0
N_BOOT = 1000          # resamples over SCENARIOS, for the CI on each average

# Scalar metrics -> the summary TABLES (one number per scenario).
METRICS = [
    ("rec_at_kstar", "Recommendation % at k* (all rounds)"),
    ("rec_final_at_kstar", "Final recommendation % at k*"),
    ("alloc_at_kstar", "Allocation % at k*"),
    ("unsafe_alloc_pct", "Allocation % to genuinely unsafe doses"),
    ("efficacy_per_patient", "Efficacy per patient (raw)"),
    ("efficacy_attained_pct", "Efficacy attained (% of what k* offers)"),
    ("type1_final", "Type I error % (end of trial)"),
    ("type2_final", "Type II error % (end of trial)"),
]

# Per-cohort curves -> the FIGURES. These are the same quantities the battery
# plots per scenario; here each curve is averaged across scenarios within a
# stratum, so the figures read exactly like the battery's.
#
# NOTE ON EFFICACY. Every metric here is scenario-relative, so that averaging
# across scenarios compares like with like: the accuracies are measured against
# each scenario's own k*, type I/II are fractions of its own safe/unsafe sets,
# and efficacy is divided by the efficacy available AT k*, giving "how much of
# the attainable efficacy this design collected". Raw efficacy per patient would
# not pool, because the generator draws each scenario's efficacy ceiling from
# U(0.35, 0.85). The scale tops out at exactly 100%: the plateau onset is drawn
# among the safe doses, so k* is the onset and its efficacy is the plateau level,
# which is the maximum over ALL doses. Dosing above the MTD therefore buys no
# efficacy whatsoever in this generator -- only toxicity.
CURVE_METRICS = {
    "rec_accuracy": "Recommendation accuracy at k* (%)",
    "alloc_accuracy": "Allocation accuracy at k* (%)",
    "type1": "Type I error (%)",
    "type2": "Type II error (%)",
    "efficacy_attained": "Efficacy attained (% of what k* offers)",
    "safety_violation": "Safety violation (%)",
}

# Dose distributions, expressed relative to k* so they pool across scenarios.
OFFSET_RANGE = list(range(-4, 3))          # dose - k*
# Short y-axis labels. The figure title already carries the full description,
# so repeating it on the axis wastes width and makes the two rows' labels
# collide; and a bare "%" says nothing about what is being measured.
AXIS_LABELS = {
    "rec_accuracy": "Accuracy (%)",
    "alloc_accuracy": "Accuracy (%)",
    "type1": "Type I error (%)",
    "type2": "Type II error (%)",
    "efficacy_attained": "Efficacy attained (%)",
    "safety_violation": "Violation (%)",
    "rec_offset": "Recommendation (%)",
    "alloc_offset": "Allocation (%)",
}

OFFSET_METRICS = {
    "rec_offset": "Recommendation % by dose relative to k*",
    "alloc_offset": "Allocation % by dose relative to k*",
}


def _monotone_toxicity(rng) -> list:
    """
    Monotone non-decreasing toxicity with at least 2 safe doses, and an unsafe
    dose in most draws. Built from sorted uniforms so the spacing varies rather
    than being fixed by a parametric form.
    """
    for _ in range(200):
        p = np.sort(rng.uniform(0.005, 0.85, size=N_LEVELS))
        n_safe = int((p <= TTL).sum())
        if 2 <= n_safe <= N_LEVELS:
            return [float(round(x, 4)) for x in p]
    return [0.02, 0.05, 0.12, 0.25, 0.45, 0.60]


def _plateau_efficacy(rng, onset: int) -> list:
    """
    Increase-then-plateau efficacy (Assumption 1): strictly rising to `onset`
    (0-indexed), flat at the plateau level thereafter.
    """
    top = float(rng.uniform(0.35, 0.85))
    if onset == 0:
        rise = []
    else:
        rise = sorted(rng.uniform(0.02, top - 0.02, size=onset))
    q = list(rise) + [top] * (N_LEVELS - onset)
    return [float(round(x, 4)) for x in q]


def _offfamily_toxicity(rng, nominal) -> list:
    """
    A TRUE toxicity curve outside the assumed power-of-tanh family: a
    two-parameter logistic on the designs' own dose grid. Monotone by
    construction (beta_1 > 0), and rejected unless it leaves at least one safe
    and one unsafe dose so the scenario stays well posed.
    """
    grid = [inv_dose_toxic(v, A_NOMINAL) for v in nominal]
    for _ in range(200):
        b0 = float(rng.uniform(-8.0, -1.0))
        b1 = float(rng.uniform(0.5, 6.0))
        p = np.asarray(logistic_toxic_curve(grid, b0, b1), dtype=float)
        n_safe = int((p <= TTL).sum())
        if 1 <= n_safe <= N_LEVELS - 1:
            return [float(round(x, 4)) for x in p]
    return None


def sample_scenario(rng, misspecified: bool):
    """One random Scenario, plus the metadata used to stratify it."""
    for _ in range(200):
        nominal = _monotone_toxicity(rng)
        true = _offfamily_toxicity(rng, nominal) if misspecified else None
        env = true if true is not None else nominal
        safe = [k for k in range(N_LEVELS) if env[k] <= TTL]
        if len(safe) < 2:
            continue
        mtd = max(safe)
        # Onset uniform over the safe doses => k* = min(onset, mtd) uniform too.
        onset = int(rng.integers(0, mtd + 1))
        eff = _plateau_efficacy(rng, onset)
        best = max(eff[k] for k in safe)
        kstar = min(k for k in safe if eff[k] == best)
        nom_safe = [k for k in range(N_LEVELS) if nominal[k] <= TTL]
        if not nom_safe:
            continue
        sc = Scenario(
            name="random", toxicity_probs=nominal, efficacy_probs=eff,
            optimal_dose=kstar, tox_mtd=max(nom_safe),
            true_toxicity_probs=true,
        )
        return sc, dict(mtd=mtd, kstar=kstar, gap=mtd - kstar,
                        n_safe=len(safe), misspecified=bool(misspecified))
    return None, None


def evaluate(sc: Scenario, meta: dict, n_trials: int, n_cohorts: int):
    """
    Run every design once on one scenario.

    Returns ``(rows, curves, offsets)``:
      rows    one scalar summary row per design (for the tables)
      curves  {metric: (n_designs, n_cohorts)} -- the per-cohort curves, kept so
              the study can average the SAME figures the battery draws (accuracy
              progression, error rates, efficacy/violation) across scenarios,
              rather than collapsing each scenario to a single number.
      offsets {metric: (n_designs, n_offsets)} -- the dose distribution
              expressed RELATIVE to k* (dose - k*), which is what makes it
              poolable: "dose 3" means different things in different scenarios,
              but "one dose below k*" does not.
    """
    cfg = replace(make_config(sc, "random"), n_trials=n_trials)
    res = run_designs(cfg, n_cohorts=n_cohorts)
    df, k, key = res.trials, meta["kstar"], a_key(1.0)
    env = np.asarray(sc.true_toxicity_probs or sc.toxicity_probs, dtype=float)
    unsafe = env > TTL
    eff = np.asarray(sc.efficacy_probs, dtype=float)

    safe_cols = [f"safe_{j + 1}" for j in range(N_LEVELS)]
    safe_idx = [j for j in range(N_LEVELS) if not unsafe[j]]
    unsafe_idx = [j for j in range(N_LEVELS) if unsafe[j]]

    rows = []
    curves = {m: [] for m in CURVE_METRICS}
    offsets = {m: [] for m in OFFSET_METRICS}
    for algo in cfg.algos:
        g = df[df.design == algo].sort_values(["trial", "cohort"])
        rec = g.dose_rec.to_numpy().reshape(n_trials, n_cohorts)
        alc = g.dose_alloc.to_numpy().reshape(n_trials, n_cohorts)
        enr = g.enrolled.to_numpy().astype(float).reshape(n_trials, n_cohorts)

        # --- per-cohort curves, averaged over this scenario's trials ---------
        curves["rec_accuracy"].append((rec == k).mean(axis=0) * 100)
        curves["alloc_accuracy"].append((alc == k).mean(axis=0) * 100)
        n_pat = np.cumsum(enr, axis=1)
        n_pat = np.where(n_pat == 0, 1, n_pat)
        eff_pp = (np.cumsum(eff[alc] * enr, axis=1) / n_pat).mean(axis=0)
        # As a share of the efficacy k* offers, so it pools across scenarios.
        curves["efficacy_attained"].append(eff_pp / eff[k] * 100)
        curves["safety_violation"].append(
            (np.cumsum(unsafe[alc] * enr, axis=1) / n_pat).mean(axis=0) * 100)

        # --- dose distribution relative to k* --------------------------------
        for name, arr in (("rec_offset", rec), ("alloc_offset", alc)):
            off = arr - k
            offsets[name].append(
                np.array([(off == o).mean() * 100 for o in OFFSET_RANGE]))

        # Type I / II per cohort, from the design's own safety classification.
        # 3 + 3 has no safety model and writes NaN, so it drops out of these.
        # dtype=float: the safe_* columns mix bools with NaN, so pandas hands
        # back an object array otherwise.
        sm = g[safe_cols].to_numpy(dtype=float).reshape(n_trials, n_cohorts, N_LEVELS)
        if np.isnan(sm).all():
            t1 = t2 = float("nan")
            curves["type1"].append(np.full(n_cohorts, np.nan))
            curves["type2"].append(np.full(n_cohorts, np.nan))
        else:
            cls = sm > 0.5
            c1 = ((~cls[:, :, safe_idx]).mean(axis=(0, 2)) * 100
                  if safe_idx else np.full(n_cohorts, np.nan))
            c2 = (cls[:, :, unsafe_idx].mean(axis=(0, 2)) * 100
                  if unsafe_idx else np.full(n_cohorts, np.nan))
            curves["type1"].append(c1)
            curves["type2"].append(c2)
            t1, t2 = float(c1[-1]), float(c2[-1])

        rows.append({
            "design": algo, **meta,
            # Table 2 convention: fraction of ALL rounds, trial-averaged.
            "rec_at_kstar": float((rec == k).mean() * 100),
            # And the final pick, which is the other natural summary.
            "rec_final_at_kstar": float((rec[:, -1] == k).mean() * 100),
            "alloc_at_kstar": float((alc == k).mean() * 100),
            "unsafe_alloc_pct": float(unsafe[alc].mean() * 100),
            "efficacy_per_patient": float(eff[alc].mean()),
            "efficacy_attained_pct": float(eff[alc].mean() / eff[k] * 100),
            "type1_final": t1,
            "type2_final": t2,
        })
    return (rows,
            {m: np.vstack(v) for m, v in curves.items()},
            {m: np.vstack(v) for m, v in offsets.items()})


def plot_example_curves(sampled, n_examples, img_path, rng, rows_per_figure=4):
    """
    Draw a random subset of the sampled scenarios in the same format as the
    battery's dose-curve figure: one row per scenario, toxicity | efficacy.

    These scenarios have no names, so each row is labelled with the facts that
    matter for reading it -- where k* and the MTD fell, and whether the toxicity
    model was misspecified. For the misspecified ones the nominal curve is
    overlaid dashed, exactly as in the battery figure.

    ``sampled`` is the list of (Scenario, meta) actually used in the run, so the
    figure shows real draws from the study rather than fresh ones.

    Written in chunks of ``rows_per_figure`` and returned as a list of paths. A
    single tall figure is scaled down by LaTeX to fit the page height, which
    costs it most of the text width and shrinks its labels with it; chunks that
    are close to square keep the full width. The first chunk keeps ``img_path``
    unchanged so existing references to it still resolve.
    """
    if not sampled:
        return []
    take = min(n_examples, len(sampled))
    picks = rng.choice(len(sampled), size=take, replace=False)
    entries = []
    for i in sorted(picks):
        sc, meta = sampled[i]
        tag = "misspecified" if meta["misspecified"] else "well specified"
        entries.append(dict(
            name=f"#{meta['scenario_id']} ({tag}) "
                 f"(k*={meta['kstar'] + 1}, MTD={meta['mtd'] + 1})",
            toxicity_probs=sc.true_toxicity_probs or sc.toxicity_probs,
            nominal_toxicity_probs=sc.toxicity_probs,
            efficacy_probs=sc.efficacy_probs,
            ttl=TTL,
            optimal_dose=meta["kstar"],
        ))
    img_path = Path(img_path)
    paths = []
    for start in range(0, len(entries), rows_per_figure):
        chunk = entries[start:start + rows_per_figure]
        part = start // rows_per_figure + 1
        out = (img_path if part == 1 else
               img_path.with_name(f"{img_path.stem}_{part}{img_path.suffix}"))
        title = f"{take} randomly chosen scenarios from the study"
        if len(entries) > rows_per_figure:
            title += f" ({start + 1}–{start + len(chunk)})"
        plot_scenario_curves(
            chunk, img_path=str(out),
            unit_width=CURVE_UNIT_WIDTH, unit_height=CURVE_UNIT_HEIGHT,
            title_text=title,
        )
        paths.append(out)
    return paths


def _strata(meta_df):
    """[(label, boolean mask)] -- pooled first, then each MTD - k* stratum."""
    out = [("all scenarios", np.ones(len(meta_df), dtype=bool))]
    for g in sorted(meta_df.gap.unique()):
        out.append((f"MTD - k* = {g}", (meta_df.gap == g).to_numpy()))
    return out


SPEC_ROWS = [(False, "Well-specified toxicity"), (True, "Misspecified toxicity")]


def _panel_titles(meta_df, spec_rows, strata):
    """Row-major panel titles, each carrying the number of scenarios behind it."""
    titles = []
    for mis, spec in spec_rows:
        spec_mask = (meta_df.misspecified == mis).to_numpy()
        for lab, mask in strata:
            titles.append(f"{spec}<br>{lab} (n={int((spec_mask & mask).sum())})")
    return titles


def _finish(fig, n_cols, title, unit_width, unit_height):
    """
    Shared layout. The legend sits BELOW the panels: at the top it collides with
    the first row's panel titles, which are annotations just above the same
    paper coordinate.
    """
    for ann in fig.layout.annotations:
        if ann.font.size is None:
            ann.font = dict(size=BASE_FONT - 2)
    fig.update_layout(
        autosize=False, font=dict(size=BASE_FONT),
        width=unit_width * n_cols + 200, height=unit_height * 2 + 300,
        title_text=title,
        legend=dict(orientation="h", yanchor="top", y=-0.13,
                    xanchor="center", x=0.5, font=dict(size=BASE_FONT - 2)),
        margin=dict(t=110, b=130),
        title=dict(y=0.98, yanchor="top"),
    )


def plot_curves(curves, meta_df, designs, metric, ylabel, img_path,
                unit_width=520, unit_height=300):
    """
    The battery's own figure, averaged over scenarios.

    ``curves[metric]`` is (n_scenarios, n_designs, n_cohorts). Within each
    stratum the curves are averaged ACROSS SCENARIOS and the band is a
    bootstrap over scenarios -- the same resampling unit as every other number
    in this study. Rows are well-specified / misspecified, columns are the
    strata, so nothing is visually merged that should not be.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from doseescalation.evaluate import (DEFAULT_COLORS, PAPER_COLORS,
                                         _add_ci_band, _bootstrap_band)

    arr = curves[metric]
    n_cohorts = arr.shape[2]
    spec_rows = SPEC_ROWS
    strata = _strata(meta_df)

    fig = make_subplots(
        rows=2, cols=len(strata),
        subplot_titles=_panel_titles(meta_df, spec_rows, strata),
        shared_yaxes=True, vertical_spacing=0.16, horizontal_spacing=0.035,
    )
    for r, (mis, _) in enumerate(spec_rows, start=1):
        spec_mask = (meta_df.misspecified == mis).to_numpy()
        for c, (_, mask) in enumerate(strata, start=1):
            sel = spec_mask & mask
            if not sel.any():
                continue
            for i, d in enumerate(designs):
                block = arr[sel, i, :]                     # (scenarios, cohorts)
                block = block[~np.isnan(block).all(axis=1)]
                if len(block) == 0:
                    continue
                color = PAPER_COLORS.get(d, DEFAULT_COLORS[i % 10])
                if len(block) > 1:
                    lo, hi = _bootstrap_band(block)
                    _add_ci_band(fig, range(n_cohorts), lo, hi, color, r, c)
                fig.add_trace(
                    go.Scatter(x=list(range(n_cohorts)), y=list(block.mean(axis=0)),
                               mode="lines", marker_color=color, name=display_name(d),
                               legendgroup=d, showlegend=(r == 1 and c == 1)),
                    row=r, col=c,
                )
        fig.update_yaxes(title_text=AXIS_LABELS.get(metric, ylabel),
                         row=r, col=1)
    for c in range(1, len(strata) + 1):
        fig.update_xaxes(title_text="Cohort number", row=2, col=c)
    _finish(fig, len(strata),
            f"Random scenarios: {ylabel} (mean over scenarios, bootstrap 95% CI)",
            unit_width, unit_height)
    Path(img_path).parent.mkdir(parents=True, exist_ok=True)
    fig.write_image(str(img_path))
    return img_path


def plot_offsets(offsets, meta_df, designs, metric, ylabel, img_path,
                 unit_width=520, unit_height=300):
    """
    Dose distribution relative to k*, averaged over scenarios -- the poolable
    analogue of the battery's per-dose recommendation/allocation figure. A bar
    chart is the right form here: the x-axis is a discrete dose offset, not time.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from doseescalation.evaluate import (DEFAULT_COLORS, PAPER_COLORS,
                                         _bootstrap_band)

    arr = offsets[metric]
    spec_rows = SPEC_ROWS
    strata = _strata(meta_df)
    xs = [("k*" if o == 0 else f"k*{o:+d}") for o in OFFSET_RANGE]

    fig = make_subplots(
        rows=2, cols=len(strata),
        subplot_titles=_panel_titles(meta_df, spec_rows, strata),
        shared_yaxes=True, vertical_spacing=0.16, horizontal_spacing=0.035,
    )
    for r, (mis, _) in enumerate(spec_rows, start=1):
        spec_mask = (meta_df.misspecified == mis).to_numpy()
        for c, (_, mask) in enumerate(strata, start=1):
            sel = spec_mask & mask
            if not sel.any():
                continue
            for i, d in enumerate(designs):
                block = arr[sel, i, :]
                mean = block.mean(axis=0)
                err = None
                if len(block) > 1:
                    lo, hi = _bootstrap_band(block)
                    err = dict(type="data", symmetric=False,
                               array=list(np.asarray(hi) - mean),
                               arrayminus=list(mean - np.asarray(lo)),
                               thickness=1, width=2, color="#444")
                fig.add_trace(
                    go.Bar(x=xs, y=list(mean), name=display_name(d), error_y=err,
                           marker_color=PAPER_COLORS.get(d, DEFAULT_COLORS[i % 10]),
                           legendgroup=d, showlegend=(r == 1 and c == 1)),
                    row=r, col=c,
                )
        fig.update_yaxes(title_text=AXIS_LABELS.get(metric, "%"),
                         row=r, col=1)
    _finish(fig, len(strata),
            f"Random scenarios: {ylabel} (mean over scenarios, bootstrap 95% CI)",
            unit_width, unit_height)
    fig.update_layout(barmode="group")
    Path(img_path).parent.mkdir(parents=True, exist_ok=True)
    fig.write_image(str(img_path))
    return img_path


def _boot_ci(x, rng):
    """95% CI for a mean, resampling SCENARIOS with replacement."""
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return float(x.mean()), float(x.mean())
    w = rng.multinomial(len(x), np.full(len(x), 1 / len(x)), size=N_BOOT)
    b = (w @ x) / len(x)
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def summarize(df: pd.DataFrame, by: list, metric: str) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    out = []
    for keys, g in df.groupby(by + ["design"], sort=False):
        # Type I/II are NaN for designs with no safety model (3 + 3); drop
        # those rather than propagating NaN through the mean.
        vals = g[metric].to_numpy(dtype=float)
        vals = vals[~np.isnan(vals)]
        if len(vals) == 0:
            continue
        lo, hi = _boot_ci(vals, rng)
        rec = dict(zip(by + ["design"], keys if isinstance(keys, tuple) else (keys,)))
        rec.update(n_scenarios=len(vals), mean=vals.mean(),
                   sd_across_scenarios=vals.std(ddof=1) if len(vals) > 1 else 0.0,
                   ci_lo=lo, ci_hi=hi)
        out.append(rec)
    return pd.DataFrame(out)


def draw_figures(root: Path, out: Path = None):
    """
    Rebuild every figure from a finished run's persisted arrays.

    The study takes hours; changing a label or a scale must not cost another
    run. ``--redraw <run dir>`` reads back the .npz and the metadata CSV and
    overwrites ``Figures/``, exactly as bootstrap.py does for the battery.

    ``out`` sends the figures and tables somewhere else instead, leaving the run
    being read from untouched.
    """
    root = Path(root)
    dest = Path(out) if out else root
    (dest / "Results" / "Figures").mkdir(parents=True, exist_ok=True)
    raw = root / "Data" / "random_scenarios_raw.csv"
    if raw.exists():
        p = plot_accuracy_vs_gap(pd.read_csv(raw), dest)
        print(f"  figure -> {Path(p).relative_to(dest)}", flush=True)
    meta_df = pd.read_csv(root / "Data" / "random_scenarios_meta.csv")
    combined = root / "Data" / "random_scenarios_curves.npz"
    if combined.exists():
        z = np.load(combined, allow_pickle=True)
        designs = [str(d) for d in z["designs"]]
        curves = {m: z[f"curve_{m}"] for m in CURVE_METRICS}
        offsets = {m: z[f"offset_{m}"] for m in OFFSET_METRICS}
    else:
        # The run did not reach its final write -- rebuild from the per-scenario
        # partials, which is exactly why they are kept.
        parts = sorted((root / "Data" / "Partials").glob("scenario_*.npz"))
        if not parts:
            raise SystemExit(f"no curves and no partials under {root}")
        print(f"  no combined .npz; rebuilding from {len(parts)} partials")
        meta_df = meta_df.iloc[:len(parts)]
        designs = ALGOS
        loaded = [np.load(p) for p in parts]
        curves = {m: np.stack([d[f"curve_{m}"] for d in loaded])
                  for m in CURVE_METRICS}
        offsets = {m: np.stack([d[f"offset_{m}"] for d in loaded])
                   for m in OFFSET_METRICS}
    # The example-curves figure needs the scenarios themselves, and the metadata
    # CSV stores only their summary (k*, MTD, gap). They are cheap and seeded, so
    # replay the generator: sample_scenario is the only consumer of that RNG, and
    # the draw order is fixed, so this reproduces the run's own scenarios exactly.
    rng = np.random.default_rng(SEED)
    sampled = []
    for i in range(len(meta_df)):
        sc, meta = sample_scenario(rng, misspecified=(i % 2 == 1))
        if sc is None:
            continue
        meta["scenario_id"] = i
        sampled.append((sc, meta))
    for p_ex in plot_example_curves(
            sampled, N_CURVE_EXAMPLES,
            dest / "Results" / "Figures" / "example_scenario_curves.png",
            np.random.default_rng(SEED)):
        print(f"  figure -> Figures/{p_ex.name}", flush=True)

    export_latex_tables(offsets, meta_df, dest)
    for metric, ylabel in CURVE_METRICS.items():
        plot_curves(curves, meta_df, designs, metric, ylabel,
                    dest / "Results" / "Figures" / f"curve_{metric}.png")
        print(f"  figure -> Figures/curve_{metric}.png", flush=True)
    for metric, ylabel in OFFSET_METRICS.items():
        plot_offsets(offsets, meta_df, designs, metric, ylabel,
                     dest / "Results" / "Figures" / f"{metric}.png")
        print(f"  figure -> Figures/{metric}.png", flush=True)


# The CSVs carry every ordered pair; this only picks which slice gets PRINTED.
REFERENCE_DESIGN = "SEEDA Plateau"     # the published rule -- the thing to beat


def paired_differences(df: pd.DataFrame, metric: str, by: list,
                       reference: str = None) -> pd.DataFrame:
    """
    Compare designs scenario by scenario instead of on their separate averages.

    Every design runs on the identical scenarios, so scenario difficulty is a
    nuisance term shared by both sides of a comparison. Subtracting within a
    scenario cancels it, and the bootstrap then resamples the differences --
    giving the uncertainty of the GAP rather than of two separate averages.

    Applies to every design equally. By default it emits ALL ordered pairs, so
    no design is privileged as the baseline; pass `reference` to keep only the
    comparisons against one design, which is what the printed summary uses.

    `frac_better` accompanies the mean because they answer different questions:
    the mean is how large the edge is, the fraction is how often it holds. A
    large mean with `frac_better` near 0.5 is a few extreme scenarios rather
    than a general result.
    """
    rng = np.random.default_rng(SEED)
    out = []
    for keys, g in df.groupby(by, sort=False):
        wide = g.pivot_table(index="scenario_id", columns="design", values=metric)
        cols = list(wide.columns)
        for ref in ([reference] if reference is not None else cols):
            if ref not in cols:
                continue
            for d in cols:
                if d == ref:
                    continue
                pair = wide[[d, ref]].dropna()
                if len(pair) < 2:
                    continue
                diff = (pair[d] - pair[ref]).to_numpy(dtype=float)
                lo, hi = _boot_ci(diff, rng)
                rec = dict(zip(by, keys if isinstance(keys, tuple) else (keys,)))
                rec.update(design=d, reference=ref, n_scenarios=len(diff),
                           mean_diff=diff.mean(), ci_lo=lo, ci_hi=hi,
                           frac_better=float((diff > 0).mean()),
                           significant=bool(lo > 0 or hi < 0))
                out.append(rec)
    return pd.DataFrame(out)


MTD_GREEN = "C8E6C9"      # same shade build_tables.py uses for the optimal dose


def _tex_escape(s):
    return str(s).replace("&", r"\&").replace("_", r"\_").replace("%", r"\%")


def latex_table2(offsets, meta_df, misspecified, path, designs=None):
    """
    The paper's Table 2, adapted to random scenarios. Presented exactly as
    build_tables.export_table2_latex does it: Recommended | Allocated side by
    side separated by a vertical rule, cells showing "mean (std)", the optimal
    dose column shaded green with a bold header, and the majority (most
    recommended / most allocated) cell bolded per design per half.

    THE COLUMNS ARE DOSES RELATIVE TO k*, not absolute doses. Absolute doses
    cannot be averaged across scenarios -- dose 3 is the optimum in one draw and
    unsafe in another -- whereas "one dose below k*" means the same thing in
    every scenario. The shaded column is therefore the k* offset itself, which
    plays the role the optimal dose column plays in the battery's table.

    The std is taken ACROSS SCENARIOS, the resampling unit used throughout this
    study; the battery's table uses across-trial std because its scenario is
    fixed. The paper's toxicity / efficacy probability header rows are dropped:
    those differ from scenario to scenario, so there is no single row to show.
    """
    designs = designs or ALGOS
    sel = (meta_df.misspecified == misspecified).to_numpy()
    n_off = len(OFFSET_RANGE)
    zero = OFFSET_RANGE.index(0)
    heads = []
    for j, o in enumerate(OFFSET_RANGE):
        # \textbf around math does not bold the symbols; \mathbf does.
        h = ("$\\mathbf{k^\\star}$" if j == zero
             else f"$k^\\star{o:+d}$")
        heads.append(h)

    halves = [("Recommended", offsets["rec_offset"][sel]),
              ("Allocated", offsets["alloc_offset"][sel])]
    lines = [
        # No float wrapper, no caption: the document \inputs this inside its own
        # table environment and supplies both. \caption is only legal inside a
        # float, so a file carrying its own \begin{table} cannot be captioned.
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{l|" + "c" * n_off + "|" + "c" * n_off + "}",
        r"\toprule",
        r"& \multicolumn{" + str(n_off) + r"}{c|}{Recommended} & \multicolumn{"
        + str(n_off) + r"}{c}{Allocated} \\",
        r"\cmidrule(lr){2-" + str(1 + n_off) + r"}\cmidrule(lr){"
        + str(2 + n_off) + "-" + str(1 + 2 * n_off) + r"}",
        "Design & " + " & ".join(heads + heads) + r" \\",
        r"\midrule",
    ]
    for i, d in enumerate(designs):
        cells = []
        for _, arr in halves:
            mean = arr[:, i, :].mean(axis=0)
            std = arr[:, i, :].std(axis=0, ddof=1)
            majority = int(np.argmax(mean))
            for j in range(n_off):
                c = (r"\makecell{" + f"{mean[j]:.2f}" + r" \\ ("
                     + f"{std[j]:.2f}" + ")}")
                if j == majority:
                    c = r"\bfseries " + c
                if j == zero:
                    c = r"\cellcolor[HTML]{" + MTD_GREEN + "}" + c
                cells.append(c)
        lines.append(_tex_escape(display_name(d)) + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}}"]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n")
    return path


PAIRED_COMPARISONS = [
    ("SEEDA Plateau (fixed)", "SEEDA Plateau"),
    ("SEEDA Plateau (ours)", "SEEDA Plateau"),
    ("SEEDA Plateau (fixed)", "SEEDA"),
    ("SEEDA Plateau (ours)", "SEEDA"),
    ("SEEDA Plateau (ours)", "SEEDA Plateau (fixed)"),
]


def latex_paired(paired, path, comparisons=None):
    """
    The paired head-to-head differences, for the comparisons the write-up
    actually claims.

    `paired` is one of the paired_pooled_<metric>.csv frames: one row per
    ORDERED pair, so (A, B) and (B, A) both exist and are negatives of each
    other. Only the ordered pairs in `comparisons` are printed, which is what
    keeps the table to the handful of claims made rather than all 42.

    Columns are the mean difference in percentage points, its bootstrap
    interval, and the share of scenarios in which the difference ran the same
    way as the mean. The last is the one that distinguishes a general edge from
    a few extreme scenarios, so it belongs beside the interval rather than in
    the text.
    """
    comparisons = comparisons or PAIRED_COMPARISONS
    lines = [
        # As latex_table2: no float wrapper and no caption, since the document
        # \inputs this inside its own table environment and supplies both.
        r"\begin{tabular}{llrrr}",
        r"\toprule",
        r"Design & against & Difference & $95\%$ interval & Fraction \\",
        r"\midrule",
    ]
    blocks = [(False, "Well-specified"), (True, "Misspecified")]
    for bi, (mis, label) in enumerate(blocks):
        if bi:
            lines.append(r"\midrule")
        lines.append(r"\multicolumn{5}{l}{\emph{" + label + r"}} \\")
        for d, ref in comparisons:
            row = paired[(paired.misspecified == mis)
                         & (paired.design == d)
                         & (paired.reference == ref)]
            if row.empty:
                continue
            r = row.iloc[0]
            # Bold the interval when it excludes zero, so the eye finds the
            # claims that stand without reading every bracket.
            body = f"[{r.ci_lo:+.2f},\\ {r.ci_hi:+.2f}]"
            # \bfseries does not reach inside math mode; \mathbf does.
            iv = (f"$\\mathbf{{{body}}}$" if r.ci_lo > 0 or r.ci_hi < 0
                  else f"${body}$")
            lines.append(
                f"{_tex_escape(display_name(d))} & {_tex_escape(display_name(ref))} "
                f"& ${r.mean_diff:+.2f}$ & {iv} & ${r.frac_better:.2f}$ \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n")
    return path


def plot_accuracy_vs_gap(df, root, metric="rec_at_kstar",
                         ylabel="Recommendation accuracy at k* (%)",
                         max_gap=3, designs=None, img_path=None):
    """
    THE SUMMARY FIGURE: accuracy against MTD - k*, one line per design.

    The other figures put cohort number on the x-axis and split the strata into
    separate panels, which forces the reader to compare across panels to see
    that the ranking changes. Putting the stratifier itself on the x-axis makes
    that the shape of the plot: each design becomes one curve, and where the
    curves cross IS the result.

    Strata above `max_gap` are dropped -- they rest on a handful of scenarios
    (n = 1 and 3 at gap 4) and would add visual noise, not information. The
    scenario count for each retained stratum is printed on its tick label.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from doseescalation.evaluate import (DEFAULT_COLORS, PAPER_COLORS,
                                         _add_ci_band)

    designs = designs or ALGOS
    st = summarize(df, ["misspecified", "gap"], metric)
    st = st[st.gap <= max_gap]
    gaps = sorted(st.gap.unique())
    spec_rows = [(False, "Well-specified toxicity"), (True, "Misspecified toxicity")]

    fig = make_subplots(rows=1, cols=2, subplot_titles=[s for _, s in spec_rows],
                        shared_yaxes=True, horizontal_spacing=0.06)
    for c, (mis, _) in enumerate(spec_rows, start=1):
        block = st[st.misspecified == mis]
        ticks = []
        for g in gaps:
            n = block[block.gap == g].n_scenarios.max()
            ticks.append(f"{int(g)}<br><span style='font-size:14px'>n={int(n)}</span>")
        for i, d in enumerate(designs):
            s = block[block.design == d].sort_values("gap")
            if s.empty:
                continue
            color = PAPER_COLORS.get(d, DEFAULT_COLORS[i % 10])
            _add_ci_band(fig, s.gap, s.ci_lo, s.ci_hi, color, 1, c, alpha=0.12)
            fig.add_trace(
                go.Scatter(x=list(s.gap), y=list(s["mean"]), mode="lines+markers",
                           marker=dict(size=7), line=dict(width=2.5),
                           marker_color=color, name=display_name(d), legendgroup=d,
                           showlegend=(c == 1)),
                row=1, col=c,
            )
        fig.update_xaxes(title_text="MTD - k*  (doses between the optimum and the MTD)",
                         tickmode="array", tickvals=gaps, ticktext=ticks,
                         row=1, col=c)
    fig.update_yaxes(title_text=ylabel, range=[0, 100], row=1, col=1)
    for ann in fig.layout.annotations:
        if ann.font.size is None:
            ann.font = dict(size=BASE_FONT)
    fig.update_layout(
        autosize=False, font=dict(size=BASE_FONT), width=1450, height=680,
        title_text=f"{ylabel} by distance between the optimal dose and the MTD"
                   "<br><span style='font-size:15px'>mean over random scenarios, "
                   "bootstrap 95% CI</span>",
        legend=dict(orientation="h", yanchor="top", y=-0.24,
                    xanchor="center", x=0.5, font=dict(size=BASE_FONT - 2)),
        margin=dict(t=110, b=190),
        title=dict(y=0.97, yanchor="top"),
    )
    img_path = img_path or (root / "Results" / "Figures" / f"summary_{metric}_by_gap.png")
    Path(img_path).parent.mkdir(parents=True, exist_ok=True)
    fig.write_image(str(img_path))
    return img_path


def export_latex_tables(offsets, meta_df, root):
    """The Table 2 equivalent, one per specification class."""
    out = root / "Results" / "Tables"
    for mis in (False, True):
        p = latex_table2(offsets, meta_df, mis,
                         out / f"table2_{'mis' if mis else 'well'}.tex")
        print(f"  latex table -> {p}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-scenarios", type=int, default=N_SCENARIOS)
    ap.add_argument("--n-trials", type=int, default=N_TRIALS)
    ap.add_argument("--n-cohorts", type=int, default=N_COHORTS)
    ap.add_argument("--metric", default="rec_at_kstar")
    ap.add_argument("--n-curve-examples", type=int, default=N_CURVE_EXAMPLES,
                    help="how many sampled scenarios to draw curves for (0 = none)")
    ap.add_argument("--redraw", metavar="RUN_DIR",
                    help="redraw the figures of a finished run, no simulation")
    ap.add_argument("--out", metavar="DIR",
                    help="write them here instead, leaving the run untouched")
    args = ap.parse_args()

    if args.redraw:
        draw_figures(Path(args.redraw), args.out)
        return

    root = Path("experiments/plots") / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    print(f"random-scenario study -> {root}")
    print(f"{args.n_scenarios} scenarios x {args.n_trials} trials x "
          f"{args.n_cohorts} cohorts x {len(ALGOS)} designs\n")

    # EVERY scenario is written to disk the moment it finishes -- the raw rows
    # and the metadata appended to their CSVs, the curves dropped into
    # partials/ as their own .npz. The run takes hours, so a crash (or a closed
    # laptop) at scenario 180 must cost one scenario, not the night: whatever
    # completed is on disk and `--redraw` rebuilds every figure from it.
    rows, sampled, t0 = [], [], datetime.now()
    data_dir = root / "Data"
    data_dir.mkdir(parents=True, exist_ok=True)
    raw_path = data_dir / "random_scenarios_raw.csv"
    meta_path = data_dir / "random_scenarios_meta.csv"
    partials = data_dir / "Partials"
    partials.mkdir(exist_ok=True)
    all_curves = {m: [] for m in CURVE_METRICS}
    all_offsets = {m: [] for m in OFFSET_METRICS}
    for i in range(args.n_scenarios):
        sc, meta = sample_scenario(rng, misspecified=(i % 2 == 1))
        if sc is None:
            continue
        meta["scenario_id"] = i
        sampled.append((sc, meta))
        r, curves, offsets = evaluate(sc, meta, args.n_trials, args.n_cohorts)
        rows.extend(r)
        for m in CURVE_METRICS:
            all_curves[m].append(curves[m])
        for m in OFFSET_METRICS:
            all_offsets[m].append(offsets[m])

        pd.DataFrame(r).to_csv(raw_path, mode="a",
                               header=not raw_path.exists(), index=False)
        pd.DataFrame([meta]).to_csv(meta_path, mode="a",
                                    header=not meta_path.exists(), index=False)
        np.savez_compressed(
            partials / f"scenario_{i:05d}.npz",
            **{f"curve_{m}": curves[m] for m in CURVE_METRICS},
            **{f"offset_{m}": offsets[m] for m in OFFSET_METRICS},
        )

        if (i + 1) % 10 == 0:
            el = (datetime.now() - t0).total_seconds()
            eta = el / (i + 1) * (args.n_scenarios - i - 1)
            print(f"  {i+1}/{args.n_scenarios}   elapsed {el/60:.1f} min   "
                  f"eta {eta/60:.1f} min", flush=True)

    df = pd.DataFrame(rows)

    # (n_scenarios, n_designs, n_points), aligned row-for-row with meta_df.
    all_curves = {m: np.stack(v) for m, v in all_curves.items()}
    all_offsets = {m: np.stack(v) for m, v in all_offsets.items()}
    meta_df = pd.DataFrame([meta for _, meta in sampled])
    # Persisted so the figures can be redrawn without re-running the study --
    # the same reason bootstrap.py works off the battery's CSVs.
    np.savez_compressed(
        data_dir / "random_scenarios_curves.npz",
        designs=np.array(ALGOS, dtype=object),
        offset_range=np.array(OFFSET_RANGE),
        **{f"curve_{m}": a for m, a in all_curves.items()},
        **{f"offset_{m}": a for m, a in all_offsets.items()},
    )

    # Summary TABLES for every scalar metric (the figures below are the curves).
    summ_dir = root / "Summaries"
    summ_dir.mkdir(parents=True, exist_ok=True)
    for metric, ylabel in METRICS:
        po = summarize(df, ["misspecified"], metric)
        st = summarize(df, ["misspecified", "gap"], metric)
        po.to_csv(summ_dir / f"summary_pooled_{metric}.csv", index=False)
        st.to_csv(summ_dir / f"summary_stratified_{metric}.csv", index=False)
        pd_po = paired_differences(df, metric, ["misspecified"])
        pd_st = paired_differences(df, metric, ["misspecified", "gap"])
        pd_po.to_csv(summ_dir / f"paired_pooled_{metric}.csv", index=False)
        pd_st.to_csv(summ_dir / f"paired_stratified_{metric}.csv", index=False)
    export_latex_tables(all_offsets, meta_df, root)
    plot_accuracy_vs_gap(df, root)
    print("  figure -> Results/Figures/summary_rec_at_kstar_by_gap.png", flush=True)

    # Headline FIGURES: the battery's own curves, averaged over scenarios.
    for metric, ylabel in CURVE_METRICS.items():
        plot_curves(all_curves, meta_df, ALGOS, metric, ylabel,
                    root / "Results" / "Figures" / f"curve_{metric}.png")
        print(f"  figure -> Figures/curve_{metric}.png", flush=True)
    for metric, ylabel in OFFSET_METRICS.items():
        plot_offsets(all_offsets, meta_df, ALGOS, metric, ylabel,
                     root / "Results" / "Figures" / f"{metric}.png")
        print(f"  figure -> Figures/{metric}.png", flush=True)

    m = args.metric
    pooled = summarize(df, ["misspecified"], m)
    strat = summarize(df, ["misspecified", "gap"], m)

    for mis in (False, True):
        tag = "MISSPECIFIED" if mis else "WELL-SPECIFIED"
        p = pooled[pooled.misspecified == mis]
        print(f"\n=== {tag}: pooled {m} (mean over scenarios) ===")
        print(f"{'design':<24}{'mean':>8}{'sd':>8}   95% CI")
        for a in ALGOS:
            r = p[p.design == a]
            if r.empty:
                continue
            r = r.iloc[0]
            print(f"{a:<24}{r['mean']:>8.1f}{r.sd_across_scenarios:>8.1f}"
                  f"   [{r.ci_lo:.1f}, {r.ci_hi:.1f}]")

        s = strat[strat.misspecified == mis]
        gaps = sorted(s.gap.unique())
        print(f"\n--- {tag}: by MTD - k* ---")
        print(f"{'design':<24}" + "".join(f"{'gap '+str(g):>12}" for g in gaps))
        for a in ALGOS:
            cells = []
            for g in gaps:
                r = s[(s.design == a) & (s.gap == g)]
                cells.append(f"{r['mean'].iloc[0]:>12.1f}" if not r.empty else f"{'-':>12}")
            print(f"{a:<24}" + "".join(cells))
        print(f"{'(n scenarios)':<24}"
              + "".join(f"{int(s[s.gap==g].n_scenarios.iloc[0]):>12}" for g in gaps))

    paired = paired_differences(df, m, ["misspecified"], reference=REFERENCE_DESIGN)
    for mis in (False, True):
        tag = "MISSPECIFIED" if mis else "WELL-SPECIFIED"
        p = paired[paired.misspecified == mis]
        if p.empty:
            continue
        print(f"\n--- {tag}: paired difference in {m} vs {REFERENCE_DESIGN} ---")
        print(f"{'design':<24}{'diff':>8}{'95% CI':>20}{'better in':>11}")
        for _, r in p.iterrows():
            star = " *" if r.significant else "  "
            print(f"{r.design:<24}{r.mean_diff:>+8.1f}"
                  f"{f'[{r.ci_lo:+.1f}, {r.ci_hi:+.1f}]':>20}"
                  f"{r.frac_better * 100:>9.0f}%{star}")
        print("  * interval excludes zero")

    if args.n_curve_examples:
        paths = plot_example_curves(
            sampled, args.n_curve_examples,
            root / "Results" / "Figures" / "example_scenario_curves.png",
            np.random.default_rng(SEED))
        if paths:
            print()
        for p in paths:
            print(f"example curves -> {p}")

    print(f"\nraw        -> {root/'random_scenarios_raw.csv'}")
    print(f"pooled     -> {root/f'summary_pooled_{m}.csv'}")
    print(f"stratified -> {root/f'summary_stratified_{m}.csv'}")


if __name__ == "__main__":
    main()
