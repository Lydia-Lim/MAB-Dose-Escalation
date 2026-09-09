"""
Scenario battery: does Anchored's MTD-anchored recommendation generalize, and
how do all seven designs handle toxicity misspecification.

Nine scenarios, each run as its own SimulationConfig + run_designs() call (own
CSV, own run_config.json), all sharing the paper's main scenario's algorithm
hyperparameters -- only toxicity_probs / efficacy_probs / true_toxicity_probs
differ per scenario.

0. Paper's main scenario (reference) -- unchanged, included for side-by-side comparison.
1. Small efficacy gaps -- same optimal dose (dose 3, MTD-1) as the main scenario,
   but the pre-plateau rise is 0.05 steps instead of 0.25: stresses whether small
   real gaps are correctly resolved rather than mistaken for noise.
2. Optimal dose at MTD-2 -- outside the range {MTD-1, MTD} Reference's L1 rule
   can structurally ever return.
3. Optimal dose at MTD-3 -- extreme case of the same.
4. Step-function TRUE toxicity (misspecification case 1) -- escalators keep
   assuming the standard power-of-tanh model (NOMINAL toxicity_probs = TOX_MAIN
   unchanged, so the nominal grid still puts the MTD at dose 4); the environment draws
   from a two-level step, [.02,.02,.30,.30,.30,.30] (boundary at dose2/dose3,
   exactly where EFF_MAIN's plateau begins), under which EVERY dose is
   genuinely safe (0.30 <= theta) and k*=dose3, identical to the main
   scenario's answer. True vs the nominal a=20 curve CROSSES (higher at doses
   1,3,4; lower at 2,5,6) -- guaranteed unfittable by any single a, since a
   single a can only rescale the whole curve up or down together, never invert
   the relative ordering at individual doses.
5. Off-family logistic TRUE toxicity (misspecification case 2) -- standard
   2-parameter logistic p(d) = exp(b0 + b1*d) / (1 + exp(b0 + b1*d)) evaluated
   at the SAME nominal grid as the paper's main scenario. b0=-6.0, b1=4.0 keeps
   the same safe set / MTD / k* as the main scenario (dose4 / dose3), isolating
   a pure SHAPE misspecification: the best achievable single-a tanh fit is still
   off by ~8 percentage points at the worst dose.

Scenarios 6-8 all move the MTD up to dose 5 with k* at dose 4, holding k*=MTD-1
and the efficacy cliff below k* (0.25) IDENTICAL to the paper's main scenario,
so the only thing that varies is that 3 doses now sit below k* instead of 2. In
scenarios 0-5 the cliff size and the number of doses below k* always move
together, so they cannot be told apart; these separate them.

6. Higher MTD (dose 5), k* at MTD-1 -- well-specified CONTROL for the above.
7. Step-function with an UNSAFE upper tail (misspecification case 3) -- the one
   direction the other misspecified scenarios miss: toxicity UNDER-estimated.
   Scenarios 4 and 5 both err safe (4 makes every dose genuinely safe; 5 keeps
   the safe set exact). Here doses 5-6 are truly 0.50 > theta, so the TRUE MTD
   is dose 4, while the nominal grid places it at dose 5. The true curve is
   still monotone with a clean safe/unsafe split (p1..p4 <= theta < p5,p6), so
   it satisfies the paper's structural assumption and breaks only the
   one-parameter tanh FORM -- best single-a fit is still ~18pp off.
   NOTE this scenario deliberately breaks the k*=MTD-1 pattern that scenarios 6
   and 8 hold: k* = dose 4 = the true MTD, so here the optimal dose IS the MTD.
   It is therefore NOT part of that controlled comparison. The risk it probes is
   that a design admits dose 5 (the nominal grid makes it look safe) and both
   doses patients there and anchors its recommendation on it -- a direct test of
   the per-dose toxicity margin. Whether that happens depends on how far the
   observed DLEs pull a_hat down, not on any fixed belief.
8. Borderline logistic TRUE toxicity (misspecification case 4) -- b0=-2.0,
   b1=1.0: nearly FLAT where the assumed tanh is steep, 19.0pp off the best
   single-a fit (vs 7.8pp for scenario 5), with all five safe doses inside
   0.21-0.34 and dose 5 sitting 0.014 under theta, so every admissibility
   decision is genuinely borderline rather than trivially safe.

Run:
    python experiments/scenario_battery.py
"""

import sys
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from doseescalation.build_tables import build_table2, export_table2_latex
from doseescalation.evaluate import (
    plot_acc_progression,
    plot_alloc_acc_progression,
    plot_dose_proposals,
    plot_efficacy_and_violation,
    plot_error_rates,
    plot_scenario_curves,
)
from doseescalation.run_simulation import (
    SimulationConfig,
    a_key,
    inv_dose_toxic,
    logistic_toxic_curve,
    run_designs,
    write_results,
)


ALGOS = ["3 + 3", "CRM", "UCB", "SEEDA", "SEEDA Plateau", "SEEDA Plateau (ours)",
         "SEEDA Plateau (fixed)"]
TTL = 0.35
COHORT_SIZE = 3
N_TRIALS = 1000
TABLE2_COHORTS = 100
FIGURES_COHORTS = 300

# Fig 3 (sample complexity) lives in its own script, experiments/sample_complexity.py
# -- it re-simulates from scratch at every candidate horizon and so costs far
# more than everything here put together. Run it separately, against a
# completed battery run.

# Shared algorithm hyperparameters -- identical to the paper's main scenario
# run. Only toxicity_probs / efficacy_probs / true_toxicity_probs vary per
# scenario below.
SHARED = dict(
    ttl=TTL,
    n_trials=N_TRIALS,
    cohort_size=COHORT_SIZE,
    algos=ALGOS,
    ucb_coefficient=2.0,
    eta=2,
    seeda_c=1.0,
    plateau_c=1.0,
    l1_coefficient=0.16,
    a_seeda=20.0,
    a_plateau=20.0,
    a_max_seeda=20.0,
    a_max_plateau=20.0,
    a_max_ours=None,
    p_smoothing=0.5,
    tox_margin_c=0.01
)

# The a0 every design's internal dose grid is built from (SHARED's a_seeda /
# a_plateau). Scenarios whose TRUE toxicity is given as a logistic evaluate it
# on THIS grid, so the assumed and the true model sit at the same dose positions.
A_NOMINAL = 20.0

TOX_MAIN = [0.01, 0.05, 0.15, 0.2, 0.45, 0.6]
EFF_MAIN = [0.1, 0.35, 0.6, 0.6, 0.6, 0.6]

# Shifted-up pair for scenarios 6-8: MTD at dose 5 (rather than dose 4) and k*
# at dose 4, so k* is still MTD-1 and the efficacy cliff below k* is still 0.25
# -- both identical to the paper's main scenario. The ONLY thing that changes is
# that there are now 3 doses below k* instead of 2. That isolation is the point:
# in scenarios 0-5 the cliff size and the number of doses below k* always move
# together, so "accuracy is driven by the distance up to the MTD" and "accuracy
# is driven by the room to leak downward below k*" cannot be told apart.
TOX_MTD5 = [0.01, 0.05, 0.10, 0.15, 0.25, 0.60]  # safe 1-5 -> MTD = dose 5
EFF_K4 = [0.1, 0.2, 0.35, 0.6, 0.6, 0.6]         # plateau from dose 4 -> k* = dose 4


@dataclass
class Scenario:
    name: str
    toxicity_probs: Sequence[float]  # NOMINAL -- feeds the escalators' own grid
    efficacy_probs: Sequence[float]
    optimal_dose: int  # k*, 0-indexed
    tox_mtd: int  # MTD, 0-indexed, implied by the NOMINAL curve. That curve is
    # never given to the designs: it only positions the dose grid, via
    # d_k = atanh(2*p_k**(1/a0) - 1), so the model reproduces it at a = a0. The
    # designs estimate a from observed DLEs and derive their own safe set. Under
    # misspecification the TRUE MTD differs -- see `true_tox_mtd`.
    true_toxicity_probs: Optional[Sequence[float]] = None  # env override
    # Alternative to true_toxicity_probs: give the TRUE toxicity as a logistic
    # (beta_0, beta_1) and let it be evaluated on the nominal dose grid, rather
    # than hard-coding the per-dose probabilities it happens to produce.
    true_toxicity_logistic: Optional[Tuple[float, float]] = None

    def __post_init__(self):
        if self.true_toxicity_logistic is None:
            return
        if self.true_toxicity_probs is not None:
            raise ValueError(
                f"{self.name}: give either true_toxicity_probs or "
                "true_toxicity_logistic, not both"
            )
        beta_0, beta_1 = self.true_toxicity_logistic
        dose_levels = [inv_dose_toxic(v, A_NOMINAL) for v in self.toxicity_probs]
        self.true_toxicity_probs = [
            float(p) for p in logistic_toxic_curve(dose_levels, beta_0, beta_1)
        ]

    @property
    def true_tox_mtd(self) -> int:
        """Highest GENUINELY safe dose (0-indexed), from the curve the
        environment actually draws from. This is THE MTD of the scenario -- the
        ground truth every safety metric is scored against. Under
        misspecification it need not equal ``tox_mtd``, which is only what the
        nominal dose grid implies."""
        env = list(self.true_toxicity_probs or self.toxicity_probs)
        return max(k for k in range(len(env)) if env[k] <= TTL)

    @property
    def label(self) -> str:
        """Figure label: the scenario name plus its ground truth, so every plot
        is readable without cross-referencing the scenario table.

        The MTD quoted is always the TRUE one -- the highest genuinely safe
        dose. Where the nominal grid implies a different one (the misspecified
        scenarios) that is noted second, since the gap between them is the
        point of those scenarios. It is only a property of the dose grid: the
        designs are never told either curve, and infer their own safe set from
        the DLEs they observe."""
        mtd = (f"MTD={self.true_tox_mtd + 1}" if self.true_tox_mtd == self.tox_mtd
               else f"MTD={self.true_tox_mtd + 1}, nominal grid {self.tox_mtd + 1}")
        return f"{self.name} (k*={self.optimal_dose + 1}, {mtd})"


# Order: the four recommendation-generalization scenarios (same nominal
# toxicity curve, varying efficacy / k* position relative to MTD) first, then
# the two toxicity-misspecification scenarios -- kept as one flat list/one
# set of outputs so every scenario can be compared side by side.
SCENARIOS = [
    Scenario(
        "Paper's main scenario",
        TOX_MAIN, EFF_MAIN, optimal_dose=2, tox_mtd=3
    ),
    Scenario(
        "Small efficacy gaps",
        TOX_MAIN, [0.50, 0.55, 0.60, 0.60, 0.60, 0.60],
        optimal_dose=2, tox_mtd=3
    ),
    Scenario(
        "Optimal dose at MTD-2",
        TOX_MAIN, [0.1, 0.6, 0.6, 0.6, 0.6, 0.6],
        optimal_dose=1, tox_mtd=3
    ),
    Scenario(
        "Optimal dose at MTD-3",
        TOX_MAIN, [0.6, 0.6, 0.6, 0.6, 0.6, 0.6],
        optimal_dose=0, tox_mtd=3
    ),
    Scenario(
        # EFF_MAIN unchanged (isolates the toxicity-shape change as the only
        # variable). The step sits at the dose2/dose3 boundary, exactly where
        # EFF_MAIN's plateau begins, so k* lands on dose3 -- identical to the
        # main scenario's answer. True vs nominal a=20 crosses (higher at
        # doses 1,3,4; lower at 2,5,6), which GUARANTEES no single a can fit
        # it: a single a can only rescale the whole curve up or down together,
        # never invert the relative ordering at individual doses.
        "Step-function toxicity (misspecified)",
        TOX_MAIN, EFF_MAIN,
        optimal_dose=2, tox_mtd=3,
        true_toxicity_probs=[0.02, 0.02, 0.30, 0.30, 0.30, 0.30]
    ),
    Scenario(
        # beta_0=-6, beta_1=4 keeps the same safe set / MTD / k* as the main
        # scenario (dose4 / dose3), isolating a pure SHAPE misspecification --
        # the best achievable single-a tanh fit is still ~7.8pp off at the
        # worst dose.
        "Off-family logistic toxicity (misspecified)",
        TOX_MAIN, EFF_MAIN, optimal_dose=2, tox_mtd=3,
        true_toxicity_logistic=(-6.0, 4.0)
    ),
    Scenario(
        # CONTROL for the mechanism question -- well-specified, and matched to
        # the paper's main scenario on BOTH k*=MTD-1 and cliff=0.25. Only the
        # number of doses below k* differs (3 vs 2). If recommendation accuracy
        # is governed by the distance up to the MTD it should land near the main
        # scenario's ~58.8%; if it is governed by the room to leak downward
        # below k*, it should land below that.
        "Higher MTD (dose 5), k* at MTD-1",
        TOX_MTD5, EFF_K4, optimal_dose=3, tox_mtd=4
    ),
    Scenario(
        # The one direction of misspecification the battery does not otherwise
        # cover: toxicity UNDER-estimated. The step and logistic scenarios above
        # both err safe (the step makes every dose genuinely safe; the logistic
        # preserves the safe set exactly). Here doses 5-6 are truly 0.50 > theta,
        # so the TRUE MTD is dose 4 while the nominal grid places it at dose 5.
        # ⚠ k* = dose 4 = the true MTD, so unlike scenarios 6 and 8 the optimal
        # dose IS the MTD -- this scenario is NOT part of their controlled
        # comparison, and it favours MTD-targeting designs (measured: UCB 94.4%,
        # 3+3 81.3%, ours 54.8% at n=300). Its real value is on the safety side:
        # every design allocates ~31% of patients to the genuinely unsafe doses
        # (CRM 77%), because the nominal grid makes dose 5 look admissible.
        # Unfittable by any single a (best fit a = 16.7, ~18.5pp error), but the
        # curve is still monotone with a clean split, so the paper's structural
        # assumption holds and only the tanh FORM is violated.
        "Step-function, unsafe tail (misspecified)",
        TOX_MTD5, EFF_K4, optimal_dose=3, tox_mtd=4,
        true_toxicity_probs=[0.02, 0.02, 0.02, 0.02, 0.50, 0.50]
    ),
    Scenario(
        # Chosen to maximise the mismatch subject to keeping the TRUE MTD at
        # dose 5 AND keeping several doses in the hard band just under theta.
        # Nearly FLAT where the assumed tanh is steep: 19.0pp off the best
        # single-a fit (vs only 7.8pp for the logistic above), with all five
        # safe doses inside 0.21-0.34 and dose 5 sitting 0.014 under theta, so
        # every admissibility decision is genuinely borderline.
        "Borderline logistic toxicity (misspecified)",
        TOX_MTD5, EFF_K4, optimal_dose=3, tox_mtd=4,
        true_toxicity_logistic=(-2.0, 1.0)
    )
]


def make_config(sc: Scenario, label: str) -> SimulationConfig:
    return SimulationConfig(
        a_star=1.0,
        toxicity_probs=sc.toxicity_probs,
        efficacy_probs=sc.efficacy_probs,
        true_toxicity_probs=sc.true_toxicity_probs,
        label=f"scenario battery: {sc.name} ({label})",
        **SHARED
    )


def summarize(results, n_levels) -> pd.DataFrame:
    df = results.trials
    rows = []
    for algo in ALGOS:
        d = df[df.design == algo]
        en = d[d.enrolled]
        rec = [100 * (d.dose_rec == k).mean() for k in range(n_levels)]
        alloc = [100 * (en.dose_alloc == k).mean() for k in range(n_levels)]
        rows.append({"design": algo,
                     **{f"rec_d{k+1}": rec[k] for k in range(n_levels)},
                     **{f"alloc_d{k+1}": alloc[k] for k in range(n_levels)}})
    return pd.DataFrame(rows)


def curve_plot_entry(sc: Scenario) -> dict:
    # TRUE curve = what the environment actually draws DLEs from. NOMINAL curve
    # = what the dose grid was calibrated to; passed so the plot can overlay it
    # dashed on the misspecified scenarios, where the gap between the two IS the
    # misspecification. It is dropped automatically where the two coincide.
    return dict(
        name=sc.label,
        toxicity_probs=(
            sc.true_toxicity_probs if sc.true_toxicity_probs is not None
            else sc.toxicity_probs
        ),
        nominal_toxicity_probs=sc.toxicity_probs,
        efficacy_probs=sc.efficacy_probs,
        ttl=TTL,
        optimal_dose=sc.optimal_dose,
    )


# Canvas geometry for the dose-curve figures. Legibility in print is the ratio
# of the font size to the canvas WIDTH, since LaTeX scales the image to the text
# width, so these are deliberately smaller than the plot defaults rather than
# larger. See plot_scenario_curves.
CURVE_UNIT_WIDTH = 350
CURVE_UNIT_HEIGHT = 220


def write_scenario_curves(scenarios: Sequence[Scenario], out_dir: Path) -> list:
    """Write the dose-curve figure as two files, well-specified and misspecified.

    A single figure for all nine scenarios is around three times taller than it
    is wide, so LaTeX scales it down to fit the page height and it ends up using
    barely half the text width, with every label shrinking to match. Split by
    specification, which is also how the thesis discusses them, each figure is
    close to square, keeps the full text width, and stays readable on the page.
    """
    groups = [
        ("scenario_curves_well", "well-specified",
         [sc for sc in scenarios if sc.true_toxicity_probs is None]),
        ("scenario_curves_misspecified", "misspecified",
         [sc for sc in scenarios if sc.true_toxicity_probs is not None]),
    ]
    written = []
    for stem, label, group in groups:
        if not group:
            continue
        path = out_dir / f"{stem}.png"
        plot_scenario_curves(
            [curve_plot_entry(sc) for sc in group],
            unit_width=CURVE_UNIT_WIDTH,
            unit_height=CURVE_UNIT_HEIGHT,
            title_text=("Dose-toxicity and dose-efficacy curves: "
                        f"{label} scenarios"),
            img_path=str(path),
        )
        written.append(path)
    return written


def scenario_slug(sc: Scenario) -> str:
    return sc.name.replace(" ", "_").replace("(", "").replace(")", "")


def build_table2_for_horizon(sc: Scenario, results, n_cohorts: int, folder: Path, suffix: str):
    """
    Compute one horizon's Table 2 (CSV + LaTeX) for one scenario, writing
    `table2_{suffix}.csv` / `.tex` into `folder`. `suffix` is e.g. "n100" or
    "n300", matching TABLE2_COHORTS / FIGURES_COHORTS.
    """
    scen_key = a_key(1.0)
    n_levels = len(sc.toxicity_probs)
    table2 = build_table2(
        results.rec_map, results.alloc_map, scen_key, ALGOS,
        n_cohorts, n_levels, sc.toxicity_probs, sc.efficacy_probs
    )
    table2.to_csv(folder / f"table2_{suffix}.csv")
    export_table2_latex(
        table2, sc.optimal_dose, str(folder / f"table2_{suffix}.tex"),
        label=f"tab:{scenario_slug(sc)}_table2_{suffix}"
    )
    return table2


def build_table2_outputs(sc: Scenario, results_table2, results_figs, folder: Path):
    """
    Compute Table 2 at both horizons (n=TABLE2_COHORTS and n=FIGURES_COHORTS,
    for comparing how the distribution settles with more cohorts) for one
    scenario, writing into `folder`. Returns (table2_n100, table2_n300,
    summary) -- all three also fed into the combined, all-scenarios-stacked
    tables built in main() after the loop.
    """
    table2_n100 = build_table2_for_horizon(
        sc, results_table2, TABLE2_COHORTS, folder, f"n{TABLE2_COHORTS}"
    )
    table2_n300 = build_table2_for_horizon(
        sc, results_figs, FIGURES_COHORTS, folder, f"n{FIGURES_COHORTS}"
    )

    summary = summarize(results_table2, len(sc.toxicity_probs))
    summary.insert(0, "scenario", sc.name)
    return table2_n100, table2_n300, summary


def main():
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    root = Path("experiments/plots") / stamp
    # Per-metric/, matching where bootstrap.py later redraws these with
    # their CI bands; otherwise a fresh run leaves band-less duplicates
    # one level up.
    figures_dir = root / "Figures" / "Per-metric"
    print(f"scenario battery -> {root}")
    print(f"n_trials={N_TRIALS}, table2 horizon={TABLE2_COHORTS}, "
          f"figures horizon={FIGURES_COHORTS}\n")
    figures_dir.mkdir(parents=True, exist_ok=True)

    for curves_path in write_scenario_curves(SCENARIOS, figures_dir):
        print(f"scenario curves -> {curves_path}")
    print()

    all_table2_n100 = {}
    all_table2_n300 = {}
    # Accumulated across scenarios then handed to each plot function ONCE at the
    # end, so every figure shows all scenarios side by side rather than writing
    # one plot file per scenario. Keyed by sc.label (name + k*/MTD) since those
    # keys become the row/column titles; the CSV/LaTeX outputs keep sc.name.
    scen_key = a_key(1.0)
    global_rec_final_map, global_alloc_map = {}, {}
    global_cohort_rec_map, global_cohort_safe_map = {}, {}
    global_enrolled_map = {}
    correct_doses, correct_mtds = {}, {}
    true_tox_by_scenario, efficacy_by_scenario = {}, {}

    for sc in SCENARIOS:
        folder = root / scenario_slug(sc)
        print(f"=== {sc.name} ===")
        if sc.true_toxicity_probs is not None:
            print(f"    NOMINAL tox (escalators' grid): {sc.toxicity_probs}")
            print(f"    TRUE tox (environment draws):   {sc.true_toxicity_probs}")
        else:
            print(f"    toxicity: {sc.toxicity_probs}")
        print(f"    efficacy: {sc.efficacy_probs}")
        print(f"    k*=dose{sc.optimal_dose + 1}, MTD=dose{sc.tox_mtd + 1}")

        cfg_table2 = make_config(sc, "table2")
        results_table2 = run_designs(cfg_table2, n_cohorts=TABLE2_COHORTS)
        write_results(results_table2, cfg_table2, folder / f"n{TABLE2_COHORTS}")

        cfg_figs = make_config(sc, "figures")
        results_figs = run_designs(cfg_figs, n_cohorts=FIGURES_COHORTS)
        write_results(results_figs, cfg_figs, folder / f"n{FIGURES_COHORTS}")

        table2_n100, table2_n300, summary = build_table2_outputs(
            sc, results_table2, results_figs, folder
        )
        all_table2_n100[sc.name] = table2_n100
        all_table2_n300[sc.name] = table2_n300
        # Appended per scenario, not just rebuilt at the end: the per-scenario
        # folders (trials.csv.gz, table2_*.csv/.tex) are already written inside
        # this loop, so this is the last thing that would have been lost if the
        # run died partway through.
        combined_path = root / "all_scenarios_summary.csv"
        summary.to_csv(combined_path, mode="a",
                       header=not combined_path.exists(), index=False)
        rec_col = f"rec_d{sc.optimal_dose + 1}"
        print(f"    rec % at k* by design: "
              + ", ".join(f"{r.design}={r[rec_col]:.1f}" for _, r in summary.iterrows()))
        print()

        key = sc.label
        global_rec_final_map[key] = results_figs.rec_final_map[scen_key]
        global_alloc_map[key] = results_figs.alloc_map[scen_key]
        global_cohort_rec_map[key] = results_figs.cohort_rec_map[scen_key]
        global_cohort_safe_map[key] = results_figs.cohort_safe_map[scen_key]
        global_enrolled_map[key] = results_figs.enrolled_map[scen_key]
        correct_doses[key] = {algo: sc.optimal_dose for algo in ALGOS}
        # The yellow "Toxicity MTD" marker is a GROUND-TRUTH annotation, so it
        # must use the TRUE MTD, not the nominal one. Under misspecification the
        # two differ, and marking the nominal one would put the label on a dose
        # that is genuinely unsafe (the unsafe-tail scenario: nominal dose 5,
        # true toxicity 0.50 > theta). When the true MTD coincides with k*,
        # plot_dose_proposals draws no separate marker, which is correct.
        correct_mtds[key] = sc.true_tox_mtd
        true_tox_by_scenario[key] = (
            sc.true_toxicity_probs if sc.true_toxicity_probs is not None
            else sc.toxicity_probs
        )
        efficacy_by_scenario[key] = sc.efficacy_probs

    print(f"combined summary -> {root / 'all_scenarios_summary.csv'}")

    # Table 2 at both horizons, all scenarios stacked one below each other
    # (outer index level = scenario name), as both CSV and one concatenated
    # LaTeX document (each scenario's own table, in order -- every
    # export_table2_latex call above wrote a bare tabular, with no float
    # wrapper, for the document to \input inside its own table environment).
    for suffix, all_table2 in (
        (f"n{TABLE2_COHORTS}", all_table2_n100),
        (f"n{FIGURES_COHORTS}", all_table2_n300)
    ):
        combined_table2 = pd.concat(all_table2, names=["Scenario"])
        combined_table2.to_csv(root / f"table2_all_scenarios_{suffix}.csv")
        tex_parts = [
            (root / scenario_slug(sc) / f"table2_{suffix}.tex").read_text()
            for sc in SCENARIOS
        ]
        (root / f"table2_all_scenarios_{suffix}.tex").write_text("\n\n".join(tex_parts))
        print(f"table2 ({suffix}, all scenarios) -> "
              f"{root / f'table2_all_scenarios_{suffix}.csv'}, "
              f"{root / f'table2_all_scenarios_{suffix}.tex'}")

    n_levels = len(SCENARIOS[0].toxicity_probs)
    _battery_figures(figures_dir, n_levels, global_rec_final_map,
                     global_alloc_map, global_enrolled_map,
                     global_cohort_rec_map, global_cohort_safe_map,
                     correct_doses, correct_mtds,
                     true_tox_by_scenario, efficacy_by_scenario)


def _battery_figures(figures_dir, n_levels, global_rec_final_map,
                     global_alloc_map, global_enrolled_map,
                     global_cohort_rec_map, global_cohort_safe_map,
                     correct_doses, correct_mtds,
                     true_tox_by_scenario, efficacy_by_scenario):
    """The figures directly in ``Figures/``, shared by a fresh run and a redraw."""
    # NOTE this plots rec_final_map -- each trial's FINAL recommendation, one
    # value per trial. Table 2 reports a different quantity: the fraction of ALL
    # rounds recommending each dose, trial-averaged (the MATLAB k_rec1
    # convention, which is what reproduces the paper). A slowly-converging design
    # scores much lower on the table than on this plot, because the table
    # includes every early round. The titles say which is which.
    plot_dose_proposals(
        n_levels, N_TRIALS,
        global_rec_final_map, correct_doses, mtds=correct_mtds,
        title_text="All scenarios: final MTD recommendation per trial "
                   "(at the end of the trial)",
        img_path=str(figures_dir / "dose_rec_proposals.png")
    )
    plot_dose_proposals(
        n_levels, N_TRIALS * FIGURES_COHORTS,
        global_alloc_map, correct_doses, mtds=correct_mtds,
        title_text="All scenarios: number of dose allocations",
        img_path=str(figures_dir / "dose_allocations.png")
    )
    plot_acc_progression(
        FIGURES_COHORTS, global_cohort_rec_map, correct_doses,
        title_text="All scenarios: dose recommendation accuracy progression",
        img_path=str(figures_dir / "rec_acc_progression.png")
    )
    plot_alloc_acc_progression(
        FIGURES_COHORTS, N_TRIALS, global_alloc_map, correct_doses,
        title_text="All scenarios: dose allocation accuracy progression",
        img_path=str(figures_dir / "alloc_acc_progression.png")
    )
    plot_efficacy_and_violation(
        FIGURES_COHORTS, N_TRIALS, global_alloc_map,
        efficacy_by_scenario, true_tox_by_scenario, TTL,
        enrolled_map=global_enrolled_map,
        title_text="All scenarios: efficacy per patient and safety violation",
        img_path=str(figures_dir / "efficacy_and_violation.png")
    )
    plot_error_rates(
        FIGURES_COHORTS, global_cohort_safe_map, true_tox_by_scenario, TTL,
        algos=[a for a in ALGOS if a not in ("3 + 3", "CRM")],
        skip_initial_cohorts=n_levels,
        title_text="All scenarios: type I/II error rates",
        img_path=str(figures_dir / "error_rates.png")
    )
    print(f"combined plots -> {figures_dir}")


if __name__ == "__main__":
    main()
