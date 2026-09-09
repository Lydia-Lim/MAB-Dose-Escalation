"""
Sample complexity across RANDOM scenarios -- the slow companion to
random_scenarios.py, kept separate so it can be launched overnight.

Same question as the battery's Figure 3 (how many patients does a design need
to reach a given recommendation accuracy?), but averaged over randomly sampled
scenarios and stratified the same way as the main random-scenario study:
by MTD - k*, and by well-specified vs misspecified toxicity.

SCENARIOS ARE THE SAME DRAWS as random_scenarios.py. Both use the same seeded
generator and the same misspecified-on-odd-index rule, so scenario_id i here is
scenario_id i there and the two studies can be read side by side. Just keep
--n-scenarios <= the value used for the main study.

WHY THIS IS A SEPARATE, REDUCED RUN. Sample complexity re-simulates the whole
trial at every candidate horizon, for every accuracy target, for every design.
At the battery's settings (rounds=20, iters=50, step=3, n_max=300) one scenario
costs on the order of an hour, so 200 scenarios is several hundred hours --
not an option. The defaults below cut the cost to roughly 2 minutes per
scenario. What was cut, and why it is defensible:

  ITERS IS NOT CUT (50, as in the battery). iters controls BIAS: the horizon
  search stops at the first candidate clearing the target, so a noisy inner
  estimate stops early on a lucky batch, and averaging more rounds never
  removes that. Cutting iters would bias every design's horizon downward.

  ROUNDS IS CUT HARD (20 -> 3). rounds only controls VARIANCE of one scenario's
  estimate, and we are averaging over many scenarios anyway: between-scenario
  spread dominates, so per-scenario precision is the cheap thing to give up.
  This is the same "many scenarios beat many replicates" logic as the main
  study.

  STEP is coarsened (3 -> 6 cohorts) and N_MAX shortened (300 -> 150), which
  bounds the search. Horizons above the ceiling are reported censored, exactly
  as in the battery, and a design that never reaches a target is censored at
  N_MAX rather than dropped -- so a shorter ceiling compresses the top of the
  curve. Read the high-accuracy end with that in mind.

Run (defaults to the main study's 200 scenarios, ~7 h -- an overnight job):
    python experiments/random_scenarios_sample_complexity.py
    python experiments/random_scenarios_sample_complexity.py --n-scenarios 50   # ~2 h look
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

from random_scenarios import (
    COHORT_SIZE,
    SEED,
    _boot_ci,
    sample_scenario,
)
from scenario_battery import ALGOS, make_config
from doseescalation.run_simulation import sample_complexity

N_SCENARIOS = 200   # matches random_scenarios.py, so scenario_id lines up 1:1
SC_ROUNDS = 3       # cut from the battery's 20 -- variance only, see above
SC_ITERS = 50       # NOT cut -- controls bias
SC_STEP = 6
SC_N_MAX = 150      # cohorts; horizons beyond this are censored
# MTD-only designs are excluded: sample complexity is defined against k*.
SC_ALGOS = [a for a in ALGOS if a not in ("3 + 3", "CRM")]


def plot_sample_complexity_random(summary, img_path, n_max, cohort_size,
                                  unit_width=520, unit_height=320,
                                  title_text=None):
    """
    Sample complexity averaged over scenarios, in the same panel layout as the
    main random-scenario figures: rows are well-specified / misspecified,
    columns are the pooled set followed by each MTD - k* stratum.

    x is the accuracy target and y the mean patients needed to reach it, with
    the band being the bootstrap over SCENARIOS already computed into the
    summary frame.

    Open markers flag censoring. A scenario counts as censored at a given
    (design, target) when AT LEAST ONE of its searches failed to reach the
    target within n_max -- that is what run_simulation flags -- and the marker
    is drawn where more than half the scenarios were censored. Those searches
    record the budget rather than the horizon they would eventually have needed,
    so the plotted mean understates the truth: read it as a lower bound.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from doseescalation.evaluate import (DEFAULT_COLORS, PAPER_COLORS,
                                         display_name,
                                         _add_ci_band)

    ceiling = n_max * cohort_size
    spec_rows = [(False, "Well-specified toxicity"), (True, "Misspecified toxicity")]
    gaps = sorted(summary.loc[summary.stratified, "gap"].dropna().unique())
    cols = [("all scenarios", None)] + [(f"MTD - k* = {int(g)}", g) for g in gaps]

    titles = []
    for _, spec in spec_rows:
        for lab, _ in cols:
            titles.append(f"{spec}<br>{lab}")

    fig = make_subplots(rows=2, cols=len(cols), subplot_titles=titles,
                        shared_yaxes=True, vertical_spacing=0.16,
                        horizontal_spacing=0.035)

    for r, (mis, _) in enumerate(spec_rows, start=1):
        for c, (_, g) in enumerate(cols, start=1):
            sel = ((summary.misspecified == mis)
                   & (summary.stratified if g is not None else ~summary.stratified))
            if g is not None:
                sel &= summary.gap == g
            block = summary[sel]
            for i, d in enumerate(SC_ALGOS):
                s = block[block.design == d].sort_values("accuracy")
                if s.empty:
                    continue
                color = PAPER_COLORS.get(d, DEFAULT_COLORS[i % 10])
                _add_ci_band(fig, s.accuracy, s.ci_lo, s.ci_hi, color, r, c)
                fig.add_trace(
                    go.Scatter(x=list(s.accuracy), y=list(s.mean_patients),
                               mode="lines", marker_color=color,
                               name=display_name(d), legendgroup=d, showlegend=(r == 1 and c == 1)),
                    row=r, col=c,
                )
                cens = s[s.censored_frac > 0.5]
                if not cens.empty:
                    fig.add_trace(
                        go.Scatter(x=list(cens.accuracy), y=list(cens.mean_patients),
                                   mode="markers", marker=dict(
                                       symbol="circle-open", size=7, color=color,
                                       line=dict(width=2)),
                                   legendgroup=d, showlegend=False,
                                   hovertext="mostly censored -- lower bound"),
                        row=r, col=c,
                    )
            fig.add_hline(y=ceiling, line=dict(color="#888", width=1, dash="dot"),
                          row=r, col=c)
        fig.update_yaxes(title_text="Patients", row=r, col=1)
    for c in range(1, len(cols) + 1):
        fig.update_xaxes(title_text="Recommendation accuracy target", row=2, col=c)
    for ann in fig.layout.annotations:
        if ann.font.size is None:
            ann.font = dict(size=11)

    fig.update_layout(
        autosize=False,
        width=unit_width * len(cols) + 160, height=unit_height * 2 + 250,
        # Plotly does not wrap titles, so a narrow canvas needs a short one and
        # puts the rest in the LaTeX caption.
        title_text=title_text or (
            "Random scenarios: patients to reach a recommendation-accuracy "
            f"target (mean over scenarios, bootstrap 95% CI; dotted line = "
            f"{ceiling}-patient budget, open markers = budget reached in "
            "over half the scenarios, so a lower bound)"),
        legend=dict(orientation="h", yanchor="top", y=-0.13,
                    xanchor="center", x=0.5, font=dict(size=11)),
        margin=dict(t=110, b=130),
        title=dict(y=0.98, yanchor="top"),
    )
    Path(img_path).parent.mkdir(parents=True, exist_ok=True)
    fig.write_image(str(img_path))
    return img_path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-scenarios", type=int, default=N_SCENARIOS)
    ap.add_argument("--rounds", type=int, default=SC_ROUNDS)
    ap.add_argument("--iters", type=int, default=SC_ITERS)
    ap.add_argument("--n-max", type=int, default=SC_N_MAX)
    ap.add_argument("--step", type=int, default=SC_STEP)
    ap.add_argument("--into", metavar="RUN_DIR",
                    help="write into <RUN_DIR>/SampleComplexity, alongside the "
                         "matching random-scenario run")
    ap.add_argument("--redraw", metavar="RUN_DIR",
                    help="redraw the figure of a finished run, no simulation")
    args = ap.parse_args()

    if args.redraw:
        d = Path(args.redraw)
        summary = pd.read_csv(d / "Summaries" / "random_sample_complexity_summary.csv")
        p = plot_sample_complexity_random(
            summary, d / "Results" / "Figures" / "random_sample_complexity.png",
            args.n_max, COHORT_SIZE)
        print(f"figure -> {p}")
        return

    # Lives INSIDE the matching random-scenario run when one is given, since the
    # two studies share a scenario set and belong together; otherwise it makes
    # its own timestamped folder.
    if args.into:
        root = Path(args.into) / "SampleComplexity"
    else:
        root = Path("experiments/plots") / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    root.mkdir(parents=True, exist_ok=True)

    print(f"random-scenario sample complexity -> {root}")
    print(f"{args.n_scenarios} scenarios, rounds={args.rounds}, iters={args.iters}, "
          f"n_max={args.n_max}, step={args.step}\n")

    # This is an overnight run, so each scenario is appended to the raw CSV as
    # soon as it finishes: a crash at hour six then costs one scenario, not all
    # of them, and the summary below can be rebuilt from whatever completed.
    data_dir = root / "Data"
    data_dir.mkdir(parents=True, exist_ok=True)
    raw_path = data_dir / "random_sample_complexity_raw.csv"
    rng = np.random.default_rng(SEED)      # same stream as random_scenarios.py
    frames, t0 = [], datetime.now()
    for i in range(args.n_scenarios):
        sc, meta = sample_scenario(rng, misspecified=(i % 2 == 1))
        if sc is None:
            continue
        cfg = replace(make_config(sc, "random sc"), n_trials=1)
        df = sample_complexity(
            cfg, optimal_dose=meta["kstar"], algos=SC_ALGOS,
            n_max=args.n_max, step=args.step,
            rounds=args.rounds, iters=args.iters,
            scenario_name=f"random-{i}", progress=False,
        )
        df["scenario_id"] = i
        df["gap"] = meta["gap"]
        df["misspecified"] = meta["misspecified"]
        df["kstar"] = meta["kstar"]
        df["mtd"] = meta["mtd"]
        frames.append(df)
        df.to_csv(raw_path, mode="a", header=not raw_path.exists(), index=False)
        if (i + 1) % 5 == 0:
            el = (datetime.now() - t0).total_seconds()
            eta = el / (i + 1) * (args.n_scenarios - i - 1)
            print(f"  {i+1}/{args.n_scenarios}   elapsed {el/60:.1f} min   "
                  f"eta {eta/60:.1f} min", flush=True)

    raw = pd.concat(frames, ignore_index=True)

    # Average the horizon across scenarios at each accuracy target, resampling
    # SCENARIOS for the CI -- the same unit as the main random-scenario study.
    boot_rng = np.random.default_rng(SEED)
    out = []
    for by in (["misspecified"], ["misspecified", "gap"]):
        for keys, g in raw.groupby(by + ["design", "accuracy"], sort=False):
            # One value per scenario at this (design, accuracy) point.
            vals = g.groupby("scenario_id")["n_patients"].mean().to_numpy(dtype=float)
            vals = vals[~np.isnan(vals)]
            if len(vals) == 0:
                continue
            lo, hi = _boot_ci(vals, boot_rng)
            rec = dict(zip(by + ["design", "accuracy"], keys))
            rec.update(stratified=(len(by) > 1), n_scenarios=len(vals),
                       mean_patients=vals.mean(),
                       sd_across_scenarios=vals.std(ddof=1) if len(vals) > 1 else 0.0,
                       ci_lo=lo, ci_hi=hi,
                       censored_frac=float(g["censored"].mean()))
            out.append(rec)
    summary = pd.DataFrame(out)
    (root / "Summaries").mkdir(parents=True, exist_ok=True)
    summary.to_csv(root / "Summaries" / "random_sample_complexity_summary.csv", index=False)

    fig_path = plot_sample_complexity_random(
        summary, root / "Results" / "Figures" / "random_sample_complexity.png",
        args.n_max, COHORT_SIZE)
    print(f"\nfigure  -> {fig_path}")

    pooled = summary[~summary.stratified]
    for mis in (False, True):
        tag = "MISSPECIFIED" if mis else "WELL-SPECIFIED"
        p = pooled[pooled.misspecified == mis]
        if p.empty:
            continue
        accs = sorted(a for a in p.accuracy.unique() if a in (0.3, 0.5, 0.7, 0.9))
        print(f"\n=== {tag}: mean patients to reach accuracy (pooled) ===")
        print(f"{'design':<24}" + "".join(f"{'acc '+str(a):>12}" for a in accs))
        for a in SC_ALGOS:
            cells = []
            for acc in accs:
                r = p[(p.design == a) & (p.accuracy == acc)]
                cells.append(f"{r.mean_patients.iloc[0]:>12.0f}" if not r.empty
                             else f"{'-':>12}")
            print(f"{a:<24}" + "".join(cells))

    print(f"\nraw     -> {root/'random_sample_complexity_raw.csv'}")
    print(f"summary -> {root/'random_sample_complexity_summary.csv'}")
    print("\nNOTE horizons are censored at "
          f"{args.n_max} cohorts ({args.n_max * COHORT_SIZE} patients); "
          "the `censored_frac` column says how often that bound bit.")


if __name__ == "__main__":
    main()
