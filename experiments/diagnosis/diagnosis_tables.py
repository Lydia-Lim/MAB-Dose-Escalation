"""The diagnosis run: what the paper reports, what we reproduce, and why they differ.

One entry point, one timestamped folder under ``experiments/diagnosis/runs``,
holding ``Data/`` and ``Results/{Tables,Figures}`` as the main experiments do.

Three tables:

* ``table_published``  -- the relevant rows of Table 2 of Shen et al. (2020),
  transcribed from the paper so the two tables can be set side by side.
* ``table_reproduction`` -- the same layout and the same definitions computed
  here: the four baselines, then the three renderings of SEEDA-Plateau --
  Algorithm 2 as printed, the MATLAB port, and Reference. The corrected variants
  are deliberately absent: they belong to the contribution, not the reproduction.
* ``table_sweep`` -- the recommendation row alone, over the choices neither the
  paper nor the shared code pins down. See ``build_sweep``.

And two figures, from ``diagnosis_figures``: the L1 flat test against the true
efficacy gaps, and what ``min(L1, L2)`` resolves to.

Every row of the reproduction comes from ONE run, so the table is internally
consistent. It is a different random stream from the scenario battery and from
the random-scenario study, which are reported as their own runs rather than
folded in here -- adding two more renderings to those tables would crowd them
for no gain. The algorithm scripts are the same, so differences under about
3 percentage points against the evaluation tables are sampling.

Recommendation percentages follow the convention used throughout the thesis:
the fraction of all rounds recommending each dose, averaged within a trial and
then across trials, with the across-trial standard deviation.

Usage:
    python experiments/diagnosis/diagnosis_tables.py
        [--trials 1000] [--skip-sweep] [--skip-figures]
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "experiments"))
sys.path.insert(0, str(HERE))

from doseescalation.build_tables import (
    build_table2, export_table2_latex,
)
from doseescalation.run_simulation import (
    a_key, dose_toxic_curve, run_designs, run_simulations,
)
from scenario_battery import SCENARIOS, make_config, TABLE2_COHORTS

from _seeda_plateau_matlab import SEEDAPlateauMatlabDoseEscalator
from _seeda_plateau_naive import SEEDAPlateauNaiveDoseEscalator
from diagnosis_figures import (
    build_figures, summarize as summarize_figures,
    NAME_MATLAB, NAME_PRINTED, NAME_REFERENCE,
)
from diagnosis_sensitivity import run_sensitivity

# The printed pseudocode is a specification, never a variant, so it keeps its
# own label. Reference is our reproduction of the shipped design. Both labels
# are imported from the figures module so a row and a bar always agree; the
# prose says "Algorithm 2 as printed" where the distinction has to be spelled
# out, but the tables cannot spare the width.
DISPLAY = {"SEEDA Plateau": NAME_REFERENCE}
BATTERY_ALGOS = ["3 + 3", "CRM", "UCB", "SEEDA", "SEEDA Plateau"]
ROW_ORDER = ["3 + 3", "CRM", "UCB", "SEEDA",
             NAME_PRINTED, NAME_MATLAB, NAME_REFERENCE]
DOSE_COLS = [f"dose_{k + 1}" for k in range(6)]

# Table 2 of Shen et al. (2020), transcribed: for each design, the recommended
# means, their standard deviations, then the same two for the allocation. The
# paper prints a deviation beside every mean without saying what it varies over;
# they are carried here so the two tables can be compared cell for cell.
PUBLISHED = {
    "3+3": ([0.00, 2.40, 12.00, 17.60, 45.20, 22.80],
            [0.00, 0.41, 4.31, 5.31, 7.15, 4.35],
            [16.04, 17.82, 20.19, 18.12, 16.81, 5.82],
            [5.12, 4.23, 10.25, 9.15, 8.15, 4.12]),
    "CRM": ([0.00, 0.00, 0.00, 33.80, 65.80, 0.40],
            [0.00, 0.00, 0.00, 8.26, 10.63, 0.40],
            [0.12, 0.35, 2.62, 33.90, 57.69, 5.33],
            [0.11, 0.25, 0.32, 10.21, 11.24, 0.23]),
    "UCB": ([0.00, 2.40, 54.00, 40.40, 3.20, 0.00],
            [0.00, 2.15, 9.92, 9.05, 3.09, 0.00],
            [5.45, 9.50, 22.13, 20.93, 20.43, 24.57],
            [0.49, 1.16, 2.11, 2.24, 2.15, 2.11]),
    "SEEDA": ([0.00, 1.00, 47.20, 47.40, 4.40, 0.00],
              [0.00, 0.71, 3.40, 3.41, 2.46, 0.00],
              [11.18, 9.18, 30.76, 31.71, 12.06, 5.11],
              [0.58, 1.99, 7.76, 7.69, 3.45, 0.62]),
    "SEEDA-Plateau": ([0.80, 2.20, 86.60, 10.40, 0.00, 0.00],
                      [0.32, 1.96, 8.58, 3.65, 0.00, 0.00],
                      [7.83, 8.98, 30.12, 37.17, 14.91, 1.00],
                      [1.61, 4.21, 6.01, 7.54, 3.02, 0.61]),
}


def published_table(sc) -> pd.DataFrame:
    """Shen et al.'s own rows, in the same column layout as ours.

    Cells are pre-formatted as ``"mean\\n(std)"`` to two decimals, the same shape
    ``build_table2`` produces, so the transcription and the reproduction can be
    read against each other cell for cell. ``build_table2`` hands
    ``export_table2_latex`` strings, so leaving raw floats here would let pandas
    print its own six-decimal default and this table alone would not match the
    others.
    """
    cols = pd.MultiIndex.from_product(
        [["Recommended", "Allocated"], [f"Dose {k + 1}" for k in range(6)]]
    )
    head = pd.DataFrame(
        [[f"{p:.2f}" for p in list(sc.toxicity_probs) * 2],
         [f"{p:.2f}" for p in list(sc.efficacy_probs) * 2]],
        index=["Toxicity prob", "Efficacy prob"], columns=cols,
    )
    body = pd.DataFrame(
        {name: [f"{m:.2f}\n({s:.2f})"
                for m, s in zip(rec + alloc, rec_sd + alloc_sd)]
         for name, (rec, rec_sd, alloc, alloc_sd) in PUBLISHED.items()},
        index=cols,
    ).T
    return pd.concat([head, body])


def run_diagnosis_designs(cfg, n_cohorts, results):
    """Run the two diagnosis designs and extend ``results``' maps in place."""
    scen = a_key(cfg.a_star)
    env_levels = cfg.dose_levels[cfg.a_star]
    efficacy_env = cfg.efficacy_env()
    grid = cfg.dose_levels_plateau        # the a0 = 20 grid, as the battery uses

    def make_printed(n):
        return SEEDAPlateauNaiveDoseEscalator(
            grid, cfg.ttl, dose_toxic_curve,
            ucb_coefficient=cfg.plateau_c, delta_1=1.0 / n, eta=cfg.eta,
            a0=cfg.a_plateau, use_log_bonus=True, is_training=True, no_skip=True,
        )

    def make_matlab(n):
        return SEEDAPlateauMatlabDoseEscalator(
            grid, cfg.ttl, dose_toxic_curve, n_cohorts=n,
            a0=cfg.a_plateau, a_max=cfg.a_max_plateau,
            ucb_coefficient=cfg.plateau_c,
            eta=cfg.eta, l1_coefficient=0.2, l1_mode="min",
            is_training=True,
        )

    for algo, factory in ((NAME_PRINTED, make_printed),
                          (NAME_MATLAB, make_matlab)):
        results.rec_map[scen][algo] = []
        results.alloc_map[scen][algo] = []
        for _trial in range(cfg.n_trials):
            allocations, recommendations, _n_dles, _safe, _enrolled = \
                run_simulations(
                    factory(n_cohorts), env_levels,
                    lambda dose: dose_toxic_curve(dose, cfg.a_star),
                    cfg.cohort_size, n_cohorts=n_cohorts,
                    efficacy_env=efficacy_env,
                )
            results.rec_map[scen][algo].extend(recommendations)
            results.alloc_map[scen][algo].extend(allocations)
    return results


def simulate_recommendations(factory, cfg, n_cohorts, seed):
    """Both readouts of one design: final-round %, and trial-averaged %."""
    rng = np.random.default_rng(seed)
    K = cfg.n_levels
    tox = list(cfg.toxicity_probs)
    eff = list(cfg.efficacy_probs)
    final = np.zeros(K)
    averaged = np.zeros(K)

    for _trial in range(cfg.n_trials):
        escalator = factory(n_cohorts)
        per_round = np.zeros(K)
        for _cohort in range(n_cohorts):
            escalator.train(True)
            allocated = escalator.propose()
            # What the design would recommend if the trial stopped here.
            escalator.train(False)
            per_round[escalator.propose()] += 1
            escalator.train(True)
            escalator.update(
                allocated, cfg.cohort_size,
                int(rng.binomial(cfg.cohort_size, tox[allocated])),
                int(rng.binomial(cfg.cohort_size, eff[allocated])),
            )
        escalator.train(False)
        final[escalator.propose()] += 1
        averaged += per_round / n_cohorts

    return final / cfg.n_trials * 100, averaged / cfg.n_trials * 100


def build_sweep(cfg, seed: int = 0, horizons=(100, 300)) -> pd.DataFrame:
    """Sweep the choices neither the paper nor the shared code pins down.

    Neither faithful implementation reproduces the published recommendation row,
    and the sweep shows why the question has no single answer. The paper never
    states the flat test's form or its coefficient, and the shared RealWorld
    ``Safe_UCB_Plateau.m`` contains TWO versions of it: a ``min(0.2*b_i, 0.2*b_j)``
    test that is active, and a ``0.4*b_i + 0.4*b_j`` test that is commented out.
    This is NOT the difference between the two shipped folders -- those differ in
    hyperparameters and a cohort-size port, not in structure.

    Varied here, everything else held at the RealWorld settings:

    * ``l1_mode``        -- ``min`` (the active test) or ``sum`` (the commented one);
    * ``l1_coefficient`` -- 0.2 or 0.4, crossed with the form so the two can be
      told apart;
    * readout            -- the recommendation at the final round, or averaged
      over the per-round recommendations within each trial;
    * the horizon        -- Table 2 is read at 100 cohorts (supplement section K)
      while section 5.1 describes 300, and the main text states neither.

    The printed Algorithm 2 has none of these knobs, so it is swept only over its
    own unstated one: whether the allocation bonus carries the ``log t`` factor.

    Nothing in the shared code identifies which combination produced the paper.
    """
    grid = cfg.dose_levels_plateau
    rows = [dict(zip(DOSE_COLS, PUBLISHED["SEEDA-Plateau"][0]),
                 implementation="Shen et al. (2020), Table 2",
                 n_cohorts="", l1_mode="", l1_coefficient="", readout="")]

    for n_cohorts in horizons:
        for use_log_bonus in (True, False):
            def make_printed(n, use_log_bonus=use_log_bonus):
                return SEEDAPlateauNaiveDoseEscalator(
                    grid, cfg.ttl, dose_toxic_curve,
                    ucb_coefficient=cfg.plateau_c, delta_1=1.0 / n, eta=cfg.eta,
                    a0=cfg.a_plateau, use_log_bonus=use_log_bonus,
                    is_training=True, no_skip=True,
                )

            label = (NAME_PRINTED if use_log_bonus
                     else f"{NAME_PRINTED} (no-log bonus)")
            final, averaged = simulate_recommendations(
                make_printed, cfg, n_cohorts, seed)
            for readout, values in (("final round", final),
                                    ("trial-averaged", averaged)):
                rows.append(dict(zip(DOSE_COLS, values), implementation=label,
                                 n_cohorts=n_cohorts, l1_mode="",
                                 l1_coefficient="", readout=readout))

    for n_cohorts in horizons:
        for l1_mode in ("min", "sum"):
            for l1_coefficient in (0.2, 0.4):
                def make_port(n, l1_mode=l1_mode, l1_coefficient=l1_coefficient):
                    return SEEDAPlateauMatlabDoseEscalator(
                        grid, cfg.ttl, dose_toxic_curve, n_cohorts=n,
                        a0=cfg.a_plateau, a_max=cfg.a_max_plateau,
                        ucb_coefficient=cfg.plateau_c, eta=cfg.eta,
                        l1_mode=l1_mode, l1_coefficient=l1_coefficient,
                        is_training=True,
                    )

                final, averaged = simulate_recommendations(
                    make_port, cfg, n_cohorts, seed)
                for readout, values in (("final round", final),
                                        ("trial-averaged", averaged)):
                    rows.append(dict(zip(DOSE_COLS, values),
                                     implementation=NAME_MATLAB,
                                     n_cohorts=n_cohorts, l1_mode=l1_mode,
                                     l1_coefficient=l1_coefficient,
                                     readout=readout))

    return pd.DataFrame(rows)[
        ["implementation", "n_cohorts", "l1_mode", "l1_coefficient", "readout"]
        + DOSE_COLS
    ]


GREEN = r"{\cellcolor[HTML]{C8E6C9}} "


def export_sweep_latex(frame: pd.DataFrame, path: Path,
                       optimal_dose: int) -> None:
    """The sweep in the same house style as every other table in the thesis.

    Not a Table-2-shaped frame, so ``export_table2_latex`` does not apply; the
    conventions it enforces are reproduced here by hand -- ``[H]`` placement,
    booktabs rules, ``\\resizebox`` to the text width, a green optimal-dose
    column and a bold majority cell per row, and no caption.
    """
    head = ["Implementation", "$n$", "L1 form", "L1 coef.", "Readout"]
    head += [(r"\bfseries " if k == optimal_dose else "") + f"Dose {k + 1}"
             for k in range(len(DOSE_COLS))]

    lines = []
    for _, row in frame.iterrows():
        values = [float(row[c]) for c in DOSE_COLS]
        majority = int(np.argmax(values))
        cells = [str(row["implementation"]), str(row["n_cohorts"]),
                 str(row["l1_mode"]), str(row["l1_coefficient"]),
                 str(row["readout"])]
        for k, value in enumerate(values):
            cell = (r"\bfseries " if k == majority else "") + f"{value:.2f}"
            cells.append((GREEN if k == optimal_dose else "") + cell)
        lines.append(" & ".join(cells) + r" \\")

    # No float wrapper, no caption, no label: the document \inputs this inside
    # its own table environment and supplies all three. See export_table2_latex.
    path.write_text(
        "\\resizebox{\\textwidth}{!}{%\n"
        "\\begin{tabular}{lllll|cccccc}\n\\toprule\n"
        + " & ".join(head) + r" \\" + "\n\\midrule\n"
        + "\n".join(lines)
        + "\n\\bottomrule\n\\end{tabular}\n}%\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=None,
                        help="override the battery's n_trials (default: use it)")
    parser.add_argument("--skip-sweep", action="store_true")
    parser.add_argument("--skip-figures", action="store_true")
    parser.add_argument("--skip-sensitivity", action="store_true")
    args = parser.parse_args()

    stamp = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out = HERE / "runs" / stamp
    (out / "Data").mkdir(parents=True, exist_ok=True)
    (out / "Results" / "Tables").mkdir(parents=True, exist_ok=True)
    (out / "Results" / "Figures").mkdir(parents=True, exist_ok=True)

    sc = SCENARIOS[0]
    assert sc.name == "Paper's main scenario", sc.name
    cfg = make_config(sc, label=f"diagnosis-tables {stamp}")
    cfg.algos = BATTERY_ALGOS          # no corrected variants in the reproduction
    if args.trials is not None:
        cfg.n_trials = args.trials
    n = TABLE2_COHORTS
    print(f"scenario={sc.name}  n_cohorts={n}  trials={cfg.n_trials}  "
          f"cohort={cfg.cohort_size}  a_plateau={cfg.a_plateau}")

    results = run_designs(cfg, n_cohorts=n)
    results = run_diagnosis_designs(cfg, n, results)

    algos = BATTERY_ALGOS + [NAME_PRINTED, NAME_MATLAB]
    table = build_table2(results.rec_map, results.alloc_map, a_key(cfg.a_star),
                         algos, n, len(sc.toxicity_probs),
                         sc.toxicity_probs, sc.efficacy_probs)
    table = table.rename(index=DISPLAY)
    table = table.loc[["Toxicity prob", "Efficacy prob"] + ROW_ORDER]

    published = published_table(sc)

    table.to_csv(out / "Data" / "table_reproduction.csv")
    published.to_csv(out / "Data" / "table_published.csv")

    export_table2_latex(
        table, sc.optimal_dose,
        str(out / "Results" / "Tables" / "table_reproduction.tex"),
        label="tab:diagnosis_reproduction",
    )
    export_table2_latex(
        published, sc.optimal_dose,
        str(out / "Results" / "Tables" / "table_published.tex"),
        label="tab:diagnosis_published",
    )

    with pd.option_context("display.width", 220, "display.max_columns", None):
        print("\n=== reproduction ===")
        print(table.to_string())

    if not args.skip_sweep:
        print("\nsweeping the unpinned choices ...", flush=True)
        sweep = build_sweep(cfg)
        sweep.to_csv(out / "Data" / "table_sweep.csv", index=False)
        export_sweep_latex(sweep, out / "Results" / "Tables" / "table_sweep.tex",
                           sc.optimal_dose)
        with pd.option_context("display.width", 220, "display.max_columns", None):
            print("\n=== sweep ===")
            print(sweep.to_string(index=False,
                                  float_format=lambda v: f"{v:.2f}"))

    if not args.skip_figures:
        print("\nbuilding the figures ...", flush=True)
        traces = build_figures(cfg, n, out / "Results" / "Figures", sc.tox_mtd,
                               data_out=out / "Data")
        summarize_figures(traces, cfg, sc.tox_mtd)

    if not args.skip_sensitivity:
        print("\nsweeping the toxicity-estimate clamp ...", flush=True)
        run_sensitivity(cfg, n, out / "Data", out / "Results" / "Figures")

    print(f"\nWrote Data/, Results/Tables/ and Results/Figures/ into {out}")


if __name__ == "__main__":
    main()
