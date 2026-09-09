# Safe Bandit Dose Finding Beyond the MTD: Diagnosing and Improving Plateau Identification

Code for the MSc Machine Learning thesis of the same name (University College London, 2026), written with UCB as industry partner.

The thesis reproduces the SEEDA-Plateau design of Shen et al. (ICML 2020), diagnoses why the recommendation rule implemented in the code the authors shared can return only the estimated MTD or the dose immediately below it, and proposes new two corrected variants. It then evaluates every design across 200 randomly sampled scenarios, stratified by the distance between the optimal biological dose and the MTD.

## Attribution

The project began from a codebase provided by UCB, comprising the package structure, the toxicity estimators, the baseline dose escalators, the simulated environment, and their tests. That is the first commit in this repository; everything after it is the author's own work. Much of the original code has since been rewritten, though some of its structure and interfaces remain.

## Installation

There is no packaging step. Install the dependencies and put `src/` on the import path:

    pip install numpy==1.26.4 pandas==2.3.2 scipy==1.15.3 plotly==6.3.1 \
                tensorflow==2.21.0 tensorflow-probability==0.24.0
    export PYTHONPATH=src

The versions are the ones the reported results were produced with. TensorFlow
Probability is version-sensitive: it is pulled in by the toxicity estimators, so
an incompatible pair, or NumPy 2.x, will stop every script from importing.

Every command in `experiments/README.md` assumes both.

## Layout

- `src/doseescalation/` — the designs, the simulated environment, and the evaluation and plotting code.
- `experiments/` — the three studies. See [`experiments/README.md`](experiments/README.md) for the commands, the order to run them in, and approximate runtimes.
- `experiments/diagnosis/` — the separate diagnosis run behind Chapter 5. See [`experiments/diagnosis/README.md`](experiments/diagnosis/README.md).

## Design names

The thesis renames the plateau designs; the code keeps the older internal names.

| Thesis | Internal name | Class |
|---|---|---|
| Reference | `SEEDA Plateau` | `SEEDAPlateauTwoSidedDecoupledDoseEscalator` |
| Descending | `SEEDA Plateau (fixed)` | `SEEDAPlateauFixedDoseEscalator` |
| Anchored | `SEEDA Plateau (ours)` | `SEEDAPlateauOursDoseEscalator` |

Persisted CSVs carry the internal names. The mapping is applied at plotting time in `DISPLAY_NAMES` (`src/doseescalation/evaluate.py`), so figures and tables can be redrawn without re-running any simulation.

## Reproducing the results

Every figure and table in the thesis regenerates from persisted intermediate results without re-simulating. Re-running the simulations from scratch takes several hours. Both paths are documented in [`experiments/README.md`](experiments/README.md).

All runtimes quoted here and in `experiments/README.md` were measured on a MacBook Air (Apple M2, 8 cores, 16 GB) running macOS Tahoe 26.3.1.
