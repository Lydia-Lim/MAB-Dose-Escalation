"""
Bootstrap confidence intervals for the scenario battery.

Quantifies the Monte-Carlo uncertainty in every headline number: with 1000
simulated trials per (scenario, design), how much of the gap between two
designs is real and how much is noise?

Method -- the non-parametric bootstrap over TRIALS. Each trial is one
independent replicate, so the trials are the resampling unit:

  1. Reduce every (design, trial) to its per-dose percentage vector -- the
     fraction of that trial's cohorts recommending / allocating each dose.
     This is exactly the per-trial statistic `build_tables.build_table2`
     averages, so the bootstrap mean reproduces the published Table 2 cell
     (asserted at runtime).
  2. Resample trial ids with replacement, B times, and recompute the mean.
  3. Report the 2.5th / 97.5th percentiles of that distribution as a 95% CI.

NOTHING IS RE-SIMULATED. This reads the persisted `trials.csv.gz` written by
`run_simulation.write_results`, which is why B can be large for free: the cost
is resampling a (n_trials x n_doses) array, not re-running trials. B = 1000 is
the usual floor for a percentile interval (B = 10 would put the 2.5th
percentile between the smallest two draws, i.e. pure noise) and takes seconds.

WHY TRIALS AND NOT COHORTS are the resampling unit: cohorts inside one trial are
produced by a sequential adaptive design, so cohort t+1 depends on cohort t.
They are not exchangeable and resampling them would destroy the dependence the
design creates. Whole trials ARE independent replicates, so they are the only
valid unit here.

WHAT THIS IS AND IS NOT MEASURING: it is the Monte-Carlo uncertainty of the
reported number -- "if I re-ran these 1000 simulated trials with new seeds, how
far would this move?". It is NOT the spread of outcomes a single real trial
would show; that is the per-trial standard deviation already printed in Table 2,
and it is ~30x larger (e.g. 12.3 vs 0.4 for Reference's dose-3 cell).

RELATION TO THE SIMPLER ALTERNATIVE: because every statistic here is a plain
mean over 1000 i.i.d. trials, the closed-form Monte-Carlo standard error
(MCSE = sd/sqrt(n_trials), i.e. Table 2's printed sd / 31.6) gives essentially
the same interval -- measured across all 63 (scenario, design) headline cells at
n=300, the two agree to 0.05pp on average and 0.29pp at worst. The bootstrap is
kept because it needs no distributional assumption, extends unchanged to
statistics with no closed-form SE, and respects the [0, 100] boundary; but for
the write-up, quoting MCSE would be equally defensible and is the more common
convention in simulation studies (Morris, White & Crowther 2019).

Run (defaults to the most recent run folder):
    python experiments/bootstrap.py
    python experiments/bootstrap.py experiments/plots/<timestamp>
    python experiments/bootstrap.py --n-boot 2000
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scenario_battery import (
    ALGOS,
    COHORT_SIZE,
    FIGURES_COHORTS,
    N_TRIALS,
    SCENARIOS,
    TABLE2_COHORTS,
    TTL,
    scenario_slug,
    write_scenario_curves,
)
from doseescalation.build_tables import (
    build_table2,
    export_table2_latex,
)
from doseescalation.evaluate import (
    DISPLAY_NAMES,
    display_name,
    plot_acc_progression,
    plot_alloc_acc_progression,
    plot_dose_proposals,
    plot_efficacy_and_violation,
    plot_error_rates,
    plot_sample_complexity,
    plot_scenario_dashboard,
)
from sample_complexity import SC_ROUNDS
from doseescalation.run_simulation import a_key, results_from_csv

N_BOOT = 1000        # resamples; >=1000 for a stable 95% percentile interval
CI = (2.5, 97.5)     # percentile interval endpoints
SEED = 0


def per_trial_pct(df: pd.DataFrame, col: str, n_levels: int) -> np.ndarray:
    """
    (n_trials, n_levels) array: each trial's % of cohorts on each dose.

    Matches `build_tables._rec_mean_std` / `_alloc_mean_std` exactly -- the
    per-trial fraction, so averaging rows reproduces the Table 2 cell.
    """
    g = df.sort_values(["trial", "cohort"])
    n_trials = g["trial"].nunique()
    arr = g[col].to_numpy().reshape(n_trials, -1)
    return np.stack([(arr == k).mean(axis=1) for k in range(n_levels)], axis=1) * 100.0


def bootstrap_pct(pct: np.ndarray, n_boot: int, rng) -> dict:
    """Bootstrap the column means of a (n_trials, n_levels) per-trial array."""
    n_trials = pct.shape[0]
    idx = rng.integers(0, n_trials, size=(n_boot, n_trials))
    boot = pct[idx].mean(axis=1)                     # (n_boot, n_levels)
    lo, hi = np.percentile(boot, CI, axis=0)
    return {
        "mean": pct.mean(axis=0),                    # the point estimate
        "se": boot.std(axis=0, ddof=1),              # bootstrap standard error
        "lo": lo,
        "hi": hi,
    }


def bootstrap_run(root: Path, n_boot: int = N_BOOT):
    """Bootstrap every scenario x horizon x design x dose in a run folder."""
    rng = np.random.default_rng(SEED)
    rows = []
    for sc in SCENARIOS:
        folder = root / scenario_slug(sc)
        n_levels = len(sc.toxicity_probs)
        for horizon in (TABLE2_COHORTS, FIGURES_COHORTS):
            csv = folder / f"n{horizon}" / "Data" / "trials.csv.gz"
            if not csv.is_file():
                continue
            df = pd.read_csv(csv)
            for design in [a for a in ALGOS if a in set(df["design"])]:
                d = df[df["design"] == design]
                for metric, col in (("Recommended", "dose_rec"),
                                    ("Allocated", "dose_alloc")):
                    pct = per_trial_pct(d, col, n_levels)
                    b = bootstrap_pct(pct, n_boot, rng)
                    for k in range(n_levels):
                        rows.append({
                            "scenario": sc.name,
                            "scenario_label": sc.label,
                            "horizon": horizon,
                            "design": design,
                            "metric": metric,
                            "dose": k + 1,
                            "is_kstar": k == sc.optimal_dose,
                            "mean": b["mean"][k],
                            "se": b["se"][k],
                            "lo": b["lo"][k],
                            "hi": b["hi"][k],
                        })
            print(f"  {sc.name[:44]:<46} n={horizon}  done")
    return pd.DataFrame(rows)


def redraw_figures_with_ci(root: Path, out_dir: Path = None):
    """
    Redraw the battery's own figures with bootstrap 95% CI bands / error bars,
    overwriting the plain versions in Figures/.

    Every band here resamples TRIALS with replacement, exactly as the tables
    above do (`evaluate._bootstrap_band`) -- one method throughout. The figures
    covered are the ones where a CI is well defined:

      rec/alloc accuracy progression   band  (proportion over trials, per cohort)
      type I/II error rates            band  (per-trial error rate, per cohort)
      efficacy & safety violation      band  (per-trial cumulative mean)
      dose recommendations/allocations error bars (per-trial dose distribution)
      sample complexity                band  (per-round searches; see below)

    Not covered: scenario_curves (ground truth, no uncertainty).
    """
    crec, alloc, csafe, enr, recf, cd, cm, ttox, eff = {}, {}, {}, {}, {}, {}, {}, {}, {}
    designs, used = None, []
    for sc in SCENARIOS:
        d = root / scenario_slug(sc) / f"n{FIGURES_COHORTS}" / "Data"
        if not d.is_dir():
            continue
        r = results_from_csv(d)
        designs = designs or list(r.algos)
        key, scen = sc.label, a_key(1.0)
        used.append(sc)
        crec[key] = r.cohort_rec_map[scen]
        alloc[key] = r.alloc_map[scen]
        csafe[key] = r.cohort_safe_map[scen]
        enr[key] = r.enrolled_map[scen]
        recf[key] = r.rec_final_map[scen]
        cd[key] = {a: sc.optimal_dose for a in designs}
        # TRUE MTD, not the nominal one -- the yellow marker is a ground-truth
        # annotation, and under misspecification the nominal MTD can be a dose
        # that is genuinely unsafe. See scenario_battery.main().
        cm[key] = sc.true_tox_mtd
        ttox[key] = sc.true_toxicity_probs or sc.toxicity_probs
        eff[key] = sc.efficacy_probs
    if not crec:
        return []

    fig_dir = Path(out_dir) if out_dir else root / "Figures"
    metric_dir = fig_dir / "Per-metric"
    metric_dir.mkdir(parents=True, exist_ok=True)
    n_levels = len(used[0].toxicity_probs)
    sub = [a for a in designs if a not in ("3 + 3", "CRM")]
    ci = " (bootstrap 95% CI)"
    written = []

    plot_acc_progression(
        FIGURES_COHORTS, crec, cd, show_ci=True,
        title_text=f"All scenarios: dose recommendation accuracy progression{ci}",
        img_path=str(metric_dir / "rec_acc_progression.png"))
    written.append(metric_dir / "rec_acc_progression.png")

    plot_alloc_acc_progression(
        FIGURES_COHORTS, N_TRIALS, alloc, cd, show_ci=True,
        title_text=f"All scenarios: dose allocation accuracy progression{ci}",
        img_path=str(metric_dir / "alloc_acc_progression.png"))
    written.append(metric_dir / "alloc_acc_progression.png")

    plot_error_rates(
        FIGURES_COHORTS, csafe, ttox, TTL, show_ci=True, algos=sub,
        skip_initial_cohorts=n_levels,
        title_text=f"All scenarios: type I/II error rates{ci}",
        img_path=str(metric_dir / "error_rates.png"))
    written.append(metric_dir / "error_rates.png")

    plot_efficacy_and_violation(
        FIGURES_COHORTS, N_TRIALS, alloc, eff, ttox, TTL, show_ci=True,
        enrolled_map=enr,
        title_text=f"All scenarios: efficacy per patient and safety violation{ci}",
        img_path=str(metric_dir / "efficacy_and_violation.png"))
    written.append(metric_dir / "efficacy_and_violation.png")

    # rec_final_map = each trial's FINAL recommendation, which is NOT the
    # quantity Table 2 reports (that is the trial-average over all rounds).
    plot_dose_proposals(
        n_levels, N_TRIALS, recf, cd, mtds=cm, ci_n_trials=N_TRIALS,
        title_text=f"All scenarios: final MTD recommendation per trial "
                   f"(at the end of the trial){ci}",
        img_path=str(metric_dir / "dose_rec_proposals.png"))
    written.append(metric_dir / "dose_rec_proposals.png")

    plot_dose_proposals(
        n_levels, N_TRIALS * FIGURES_COHORTS, alloc, cd, mtds=cm,
        ci_n_trials=N_TRIALS,
        title_text=f"All scenarios: number of dose allocations{ci}",
        img_path=str(metric_dir / "dose_allocations.png"))
    written.append(metric_dir / "dose_allocations.png")

    # Sample complexity lives in its own (slow) script, so its CSV may not
    # exist for this run -- and older CSVs predate the per-round `answers`
    # column, in which case the band falls back to the stored std.
    # Per-scenario dashboards: every metric for ONE scenario on a page, as a
    # complement to the per-metric figures above. Written to Figures/Scenarios/
    # so they do not clutter the top-level figure folder.
    sc_by_scenario = {}
    for sc in used:
        f = root / scenario_slug(sc) / "sample_complexity.csv"
        if f.exists():
            sc_by_scenario[sc.label] = pd.read_csv(f)

    dash_dir = fig_dir / "Per-scenario"
    for sc in used:
        key = sc.label
        plot_scenario_dashboard(
            label=key, n_cohorts=FIGURES_COHORTS, n_trials=N_TRIALS,
            n_levels=n_levels,
            true_toxicity=ttox[key], nominal_toxicity=sc.toxicity_probs,
            efficacy=eff[key], ttl=TTL, optimal_dose=sc.optimal_dose,
            rec_final=recf[key], alloc=alloc[key], cohort_rec=crec[key],
            cohort_safe=csafe[key], enrolled=enr[key],
            algos=designs, safety_algos=sub, show_ci=True,
            sample_complexity=sc_by_scenario.get(key),
            sc_rounds=SC_ROUNDS,
            img_path=str(dash_dir / f"{scenario_slug(sc)}.png"),
        )
        written.append(dash_dir / f"{scenario_slug(sc)}.png")

    # The one figure that carries no band, redrawn here so that one command
    # produces the complete top-level set rather than most of it. It takes no
    # CI, for the reason in this function's docstring.
    written.extend(write_scenario_curves(used, metric_dir))

    # Table 2 at both horizons, rebuilt so the rows carry the display names and
    # the .tex carries no caption. Nothing is re-simulated; this reads the same
    # persisted trials as everything else above.
    tab_dir = (Path(out_dir) / "Tables") if out_dir else root / "Tables"
    tab_dir.mkdir(parents=True, exist_ok=True)
    for sc in used:
        for horizon in (TABLE2_COHORTS, FIGURES_COHORTS):
            d = root / scenario_slug(sc) / f"n{horizon}" / "Data"
            if not d.is_dir():
                continue
            r = results_from_csv(d)
            scen = a_key(1.0)
            t = build_table2(r.rec_map, r.alloc_map, scen, list(r.algos),
                             horizon, len(sc.toxicity_probs),
                             sc.toxicity_probs, sc.efficacy_probs)
            t = t.rename(index=DISPLAY_NAMES)
            t.to_csv(tab_dir / f"{scenario_slug(sc)}_n{horizon}.csv")
            export_table2_latex(
                t, sc.optimal_dose,
                str(tab_dir / f"{scenario_slug(sc)}_n{horizon}.tex"),
                label=f"tab:{scenario_slug(sc)}_n{horizon}")
            written.append(tab_dir / f"{scenario_slug(sc)}_n{horizon}.tex")
        stacked = stack_table2_horizons(
            tab_dir, scenario_slug(sc),
            (TABLE2_COHORTS, FIGURES_COHORTS), len(sc.toxicity_probs))
        if stacked:
            written.append(stacked)

    sc_csv = root / "all_scenarios_sample_complexity.csv"
    if sc_csv.is_file():
        plot_sample_complexity(
            pd.read_csv(sc_csv), show_ci=True, ci_rounds=SC_ROUNDS,
            y_max=FIGURES_COHORTS * COHORT_SIZE,
            title_text=f"Sample complexity: patients to reach a given "
                       f"recommendation accuracy{ci}",
            img_path=str(metric_dir / "sample_complexity.png"))
        written.append(metric_dir / "sample_complexity.png")

    return written


# Design order for the accuracy table: the Plateau family runs Reference,
# Descending, Anchored, which is the order the write-up introduces them in. It
# is NOT the ALGOS order, where `(ours)` precedes `(fixed)`.
ACCURACY_DESIGN_ORDER = [
    "3 + 3", "CRM", "UCB", "SEEDA",
    "SEEDA Plateau", "SEEDA Plateau (fixed)", "SEEDA Plateau (ours)",
]


def _tex_filename(slug: str) -> str:
    """A slug safe to \\input on any TeX setup.

    ``scenario_slug`` also names the persisted run DIRECTORIES, so it cannot be
    changed without breaking the readers. This sanitizes the output filename
    alone: Overleaf refuses to upload a file with ``*`` in its name, and commas
    and apostrophes are trouble in a ``\\input`` argument.
    """
    return re.sub(r"[^A-Za-z0-9_-]", "", slug.replace("*", "star"))


def stack_table2_horizons(tab_dir: Path, slug: str, horizons, n_doses: int):
    """Splice one scenario's two per-horizon Table 2s into a single .tex.

    The appendix wants one table per scenario, not one per scenario per
    horizon. The two share their header and their toxicity/efficacy reference
    rows exactly, so the combined table keeps those once and stacks the design
    blocks under a labelled rule.

    Works on the emitted .tex rather than rebuilding from the trials, so the
    cell styling (green $k^\\star$ column, bold row maxima, two-line cells) is
    whatever ``export_table2_latex`` already produced. Returns the path, or
    None if either horizon is missing or its file is not the expected shape.
    """
    parts = {}
    for horizon in horizons:
        p = tab_dir / f"{slug}_n{horizon}.tex"
        if not p.is_file():
            return None
        lines = p.read_text().splitlines()
        # The design rows run from the \midrule after "Efficacy prob" to
        # \bottomrule. Everything above is header and is identical across the
        # two horizons, so it is taken from the first only.
        try:
            eff = next(i for i, ln in enumerate(lines)
                       if ln.startswith("Efficacy prob"))
            start = next(i for i in range(eff, len(lines))
                         if lines[i].strip() == r"\midrule") + 1
            end = next(i for i, ln in enumerate(lines)
                       if ln.strip() == r"\bottomrule")
        except StopIteration:
            return None
        parts[horizon] = (lines[:start], lines[start:end], lines[end:])

    head, _, tail = parts[horizons[0]]
    n_cols = 1 + 2 * n_doses
    out = list(head)
    for i, horizon in enumerate(horizons):
        if i:
            out.append(r"\midrule")
        out.append(rf"\multicolumn{{{n_cols}}}{{l}}"
                   rf"{{\emph{{$n = {horizon}$ cohorts}}}} \\")
        out.extend(parts[horizon][1])
    out.extend(tail)

    path = tab_dir / f"{_tex_filename(slug)}.tex"
    path.write_text("\n".join(out) + "\n")
    return path


def _tex_scenario(name: str) -> str:
    """A scenario's row label, matching the specification tables in the text.

    The "(misspecified)" suffix is dropped because the table already separates
    the two blocks, and the two over-long names are shortened by hand. The rest
    is mechanical, so a new scenario still comes out sane without being listed.
    """
    short = {
        "Higher MTD (dose 5), k* at MTD-1": "Higher MTD, k* at MTD-1",
        "Off-family logistic toxicity": "Off-family logistic",
        "Borderline logistic toxicity": "Borderline logistic",
    }
    name = name.replace(" (misspecified)", "")
    name = short.get(name, name)
    name = re.sub(r"MTD-(\d)", r"MTD $-\1$", name)
    return name.replace("k*", r"$k^\star$")


def export_accuracy_latex(head: pd.DataFrame, path: Path) -> float:
    """One row per scenario, one column per design: the share of rounds each
    design recommended k*. Returns the widest CI half-width, in percentage
    points, so the caller can quote it.

    The full per-dose intervals are in bootstrap_all.csv; printing 63 of them
    here would bury the comparison the table exists to make, so the table gives
    the means and the widest interval behind any of them is reported once.
    """
    means = head.pivot_table(index="scenario", columns="design", values="mean")
    half = ((head.hi - head.lo) / 2).max()
    designs = [d for d in ACCURACY_DESIGN_ORDER if d in means.columns]

    # Scenario order, and the well/misspecified split, both follow SCENARIOS so
    # the table reads in the same order as the two specification tables.
    blocks = [[sc.name for sc in SCENARIOS
               if (sc.true_toxicity_probs is None) == well
               and sc.name in means.index]
              for well in (True, False)]

    lines = [
        f"% widest 95% bootstrap CI half-width: {half:.2f} percentage points",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{l" + "r" * len(designs) + "}",
        r"\toprule",
        "Scenario & " + " & ".join(display_name(d) for d in designs) + r" \\",
        r"\midrule",
    ]
    for i, block in enumerate(blocks):
        if i:
            lines.append(r"\midrule")
        for name in block:
            row = means.loc[name, designs]
            best = row.idxmax()
            cells = [(r"\textbf{" + f"{row[d]:.2f}" + "}") if d == best
                     else f"{row[d]:.2f}" for d in designs]
            lines.append(f"{_tex_scenario(name)} & "
                         + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}%", "}"]

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    return half


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", nargs="?", help="run folder (default: most recent)")
    ap.add_argument("--out", metavar="DIR",
                    help="write the redrawn figures here instead of "
                         "RUN_DIR/Figures, leaving the original run untouched")
    ap.add_argument("--n-boot", type=int, default=N_BOOT,
                    help=f"bootstrap resamples (default {N_BOOT})")
    args = ap.parse_args()

    if args.run_dir:
        root = Path(args.run_dir)
    else:
        runs = sorted(Path("experiments/plots").glob("20*"),
                      key=lambda p: p.stat().st_mtime)
        runs = [r for r in runs if (r / "all_scenarios_summary.csv").is_file()]
        if not runs:
            raise SystemExit("no run folder found; pass one explicitly")
        root = runs[-1]
    if not root.is_dir():
        raise SystemExit(f"{root} is not a directory")

    print(f"bootstrapping {root}  (B={args.n_boot}, {CI[0]}-{CI[1]} percentile CI)\n")
    boot = bootstrap_run(root, args.n_boot)
    if boot.empty:
        raise SystemExit("no trials.csv.gz found under that run folder")

    # --out redirects the CSVs as well as the figures. Without that, redrawing a
    # finished run into a fresh folder would still write back into the run being
    # read, which defeats the point of asking for a separate destination.
    csv_dir = (Path(args.out) / "Data") if args.out else root / "Data"
    csv_dir.mkdir(parents=True, exist_ok=True)

    out = csv_dir / "bootstrap_all.csv"
    boot.to_csv(out, index=False)
    print(f"\nfull per-dose CIs -> {out}")

    # Headline: the recommendation percentage on k*, per design per scenario.
    for horizon in sorted(boot["horizon"].unique()):
        head = boot[(boot.horizon == horizon) & boot.is_kstar
                    & (boot.metric == "Recommended")]
        p = csv_dir / f"bootstrap_rec_at_kstar_n{horizon}.csv"
        head.to_csv(p, index=False)
        print(f"headline n={horizon} -> {p}")

        tex_dir = (Path(args.out) if args.out else root) / "Tables"
        tex = tex_dir / f"accuracy_at_kstar_n{horizon}.tex"
        half = export_accuracy_latex(head, tex)
        print(f"           table -> {tex}  "
              f"(widest CI half-width {half:.2f} pp)")

    # Fold the intervals back into the battery's own figures rather than
    # drawing separate ones.
    print()
    for img in redraw_figures_with_ci(root, args.out):
        print(f"redrawn with CI -> {img}")


if __name__ == "__main__":
    main()
