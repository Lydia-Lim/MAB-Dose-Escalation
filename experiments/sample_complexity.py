"""
Sample complexity for the scenario battery.

Kept OUT of scenario_battery.py and run separately -- as in the authors'
MATLAB, where the sample-complexity sweep is its own script. The reason is
cost: every other figure is read off a single simulation run, whereas this one
re-simulates the whole trial from scratch at every candidate horizon, for every
target accuracy, for every design. It is
comfortably slower than the entire rest of the battery put together, so you
want to launch it deliberately rather than have it ride along on every run.

Scenario definitions, designs and hyperparameters are imported from
scenario_battery, so the two stay in lockstep by construction.

Run (writes into an existing battery run's folder):
    python experiments/sample_complexity.py experiments/plots/<timestamp>

Run (standalone, into a fresh timestamped folder):
    python experiments/sample_complexity.py
"""

import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scenario_battery import (
    ALGOS,
    COHORT_SIZE,
    FIGURES_COHORTS,
    SCENARIOS,
    make_config,
    scenario_slug,
)
from doseescalation.evaluate import plot_sample_complexity
from doseescalation.run_simulation import sample_complexity


# iters controls BIAS, rounds controls VARIANCE. The search stops at the first
# horizon clearing delta, so a noisy inner estimate stops early on a lucky batch
# and no amount of rounds undoes it -- at SC_ITERS=5 SEEDA appears to reach 0.9
# accuracy despite a 0.51 asymptote. Cut SC_ROUNDS for speed, never SC_ITERS.
SC_ROUNDS = 20
SC_ITERS = 50

# The MTD-only designs are excluded: sample complexity is defined against k*,
# which 3 + 3 and CRM do not target.
SC_ALGOS = [a for a in ALGOS if a not in ("3 + 3", "CRM")]


def main():
    if len(sys.argv) > 1:
        root = Path(sys.argv[1])
        if not root.is_dir():
            raise SystemExit(f"{root} is not a directory")
    else:
        root = Path("experiments/plots") / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # Per-metric/, matching scenario_battery.py and bootstrap.py.
    figures_dir = root / "Figures" / "Per-metric"
    figures_dir.mkdir(parents=True, exist_ok=True)

    print(f"sample complexity -> {root}")
    print(f"{len(SCENARIOS)} scenarios, rounds={SC_ROUNDS}, iters={SC_ITERS}, "
          f"n_max={FIGURES_COHORTS} cohorts\n")

    frames = []
    for sc in SCENARIOS:
        print(f"=== {sc.name} (k*=dose{sc.optimal_dose + 1}) ===")
        folder = root / scenario_slug(sc)
        folder.mkdir(parents=True, exist_ok=True)
        df = sample_complexity(
            make_config(sc, "sample complexity"),
            optimal_dose=sc.optimal_dose,
            algos=SC_ALGOS,
            n_max=FIGURES_COHORTS,
            step=3,
            rounds=SC_ROUNDS,
            iters=SC_ITERS,
            # Label with sc.label so the per-scenario panel titles carry k*/MTD,
            # matching every other figure in the battery.
            scenario_name=sc.label,
            progress=True,
            checkpoint_path=folder / "sample_complexity.csv",
        )
        frames.append(df)
        print()

    combined = pd.concat(frames, ignore_index=True)
    combined_path = root / "all_scenarios_sample_complexity.csv"
    combined.to_csv(combined_path, index=False)
    plot_sample_complexity(
        combined, y_max=FIGURES_COHORTS * COHORT_SIZE,
        img_path=str(figures_dir / "sample_complexity.png"),
    )
    print(f"combined -> {combined_path}")
    print(f"figure   -> {figures_dir / 'sample_complexity.png'}")


if __name__ == "__main__":
    main()
