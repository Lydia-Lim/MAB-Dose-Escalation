# Diagnosis: can SEEDA-Plateau's published Table 2 be reproduced?

This directory holds the two faithful implementations of SEEDA-Plateau —
Algorithm 2 as printed and the MATLAB port — together with the experiment that
scores them, and Reference, against the published numbers. None of the three
recovers Table 2. The printed algorithm collapses onto the lowest dose on the
grid, the MATLAB port settles on the MTD, and Reference lands nearest the
published figure without locating the plateau at all.

## Contents

| File | What it is |
| --- | --- |
| `_seeda_plateau_naive.py` | Algorithm 2 exactly as printed in Shen et al. (2020), including the bottom-up first-flat-pair scan for the turning point `L1`. Standalone. |
| `_seeda_plateau_matlab.py` | Clean-room, line-by-line port of `RealWorldData/Safe_UCB_Plateau.m` from the MATLAB the authors shared. Its ascending-overwrite loop collapses to "MTD−1 if the top admissible pair is flat, else MTD". Standalone. |
| `diagnosis_tables.py` | The entry point. Runs the baselines, all three renderings and the flat-test sweep, then calls the figures and the clamp sweep. |
| `diagnosis_figures.py` | The four diagnosis and variant figures. Importable, and runnable alone. |
| `diagnosis_sensitivity.py` | The sweep over the ceiling on $\hat{a}$. Importable, and runnable alone; also redraws from a stored summary without sweeping. |

Both diagnosis escalators are deliberately standalone: they inline what they
need rather than subclassing the package's `SEEDADoseEscalator`, so that they
stay pinned to their respective sources while the package's designs continue to
change. The third rendering, **Reference**, is the package's
`SEEDAPlateauTwoSidedDecoupledDoseEscalator` — the reproduction the evaluation
chapters use — and is included so the diagnosis and the evaluation can be read
against each other.

## Running it

One step, unlike the battery's two — everything the diagnosis needs is
simulated in the same call.

```bash
python experiments/diagnosis/diagnosis_tables.py
```

Roughly an hour at the default 1000 trials. It runs four stages in order, each
skippable:

| Stage | Produces | Skip with |
| --- | --- | --- |
| baselines and the three renderings | `table_published`, `table_reproduction` | — |
| the flat-test sweep | `table_sweep` | `--skip-sweep` |
| the diagnosis and variant figures | the four `fig_*` files | `--skip-figures` |
| the clamp sweep | `table_sensitivity`, `a_max_*` | `--skip-sensitivity` |

`--trials N` shortens all of them for a quick look.

Everything lands in one timestamped folder under `runs/`, laid out as the main
experiments are:

```
runs/<timestamp>/Data/              table_published.csv, table_reproduction.csv,
                                    table_sweep.csv, table_sensitivity.csv
runs/<timestamp>/Results/Tables/    the first three as .tex, caption-free
runs/<timestamp>/Results/Figures/   fig_flat_test.png, fig_l1_l2.png,
                                    fig_variants_resolving_power.png,
                                    fig_variants_l1_l2.png,
                                    a_max_sensitivity.png, a_max_allocation.png
```

PNG only, as the battery and the random-scenario study do. `runs/` is
gitignored: every file in it is regenerable from this one command.

The two component scripts can also be run alone, each writing its own run
folder — `diagnosis_figures.py` for the figures, `diagnosis_sensitivity.py` for
the clamp sweep. The latter also takes a summary CSV positionally and redraws
its figures from it without sweeping again.

Unlike the battery, this study re-simulates on every run: the clamp sweep is
eight different designs, so there is nothing to reuse between them. It is cheap
enough that this does not matter.

The diagnosis runs on its own random stream. The scenario battery and the
random-scenario study are reported from separate runs and leave out Algorithm 2
as printed and the MATLAB port, which only the diagnosis needs. The designs are
the same code either way, so a difference of a point or two against the
evaluation tables is sampling noise.

## What it shows

Chapters 4 and 5 of the thesis report and interpret these results in full.
