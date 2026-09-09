# Running the experiments

Three studies. Two are documented here; the diagnosis behind Chapter 5 is
separate and documented in [`diagnosis/README.md`](diagnosis/README.md).

Each run writes its own timestamped folder under `experiments/plots/`, which is
gitignored. The one run the thesis reports is committed instead, at
[`2026-09-01_15-44-27/`](2026-09-01_15-44-27), collected by study rather than
laid out as a raw run folder:

```
2026-09-01_15-44-27/Battery/           the scenario battery
2026-09-01_15-44-27/RandomScenarios/   the random-scenario study
2026-09-01_15-44-27/Diagnosis/         the diagnosis run
```

Every figure and table in the thesis regenerates from the CSVs in there without
re-simulating anything.

Every command below assumes the dependencies are installed and `src/` is on the
import path, as the [root README](../README.md) describes:

```bash
export PYTHONPATH=src
```

---

## 1. The scenario battery — a two-step pipeline

Nine hand-built scenarios, seven designs, both horizons. **The figures are only
complete after both steps.** Step 1 simulates; step 2 adds the confidence
intervals and the per-scenario dashboards, reading back what step 1 persisted.

### Step 1 — simulate

```bash
python experiments/scenario_battery.py            # ~13 h
python experiments/sample_complexity.py <run dir> # slow; re-simulates per horizon
```

`scenario_battery.py` creates the run folder and writes, per scenario, the
per-trial records and the recommendation--allocation tables, plus a first pass of
the per-metric figures **without** confidence bands. `sample_complexity.py` takes
an existing run folder and adds the sample-complexity CSVs and figure.

### Step 2 — bootstrap, bands and dashboards

```bash
python experiments/bootstrap.py experiments/plots/<run>
```

Re-simulates nothing. It reads the persisted `trials.csv.gz`, computes the
bootstrap intervals, overwrites the per-metric figures with banded versions, and
writes the per-scenario dashboards. Run it again after any change to a figure's
appearance.

### Outputs

```
<run>/<scenario>/n100/Data/trials.csv.gz     per-trial records, table horizon
<run>/<scenario>/n300/Data/trials.csv.gz     per-trial records, figures horizon
<run>/<scenario>/table2_n{100,300}.{csv,tex} recommendation--allocation tables
<run>/<scenario>/sample_complexity.csv       step 1b
<run>/table2_all_scenarios_n{100,300}.{csv,tex}
<run>/all_scenarios_summary.csv
<run>/Figures/Per-metric/                    one figure per metric, all scenarios
<run>/Figures/Per-scenario/                  one dashboard per scenario  (step 2)
<run>/Data/                                  bootstrap CSVs              (step 2)
```

The thesis uses the dashboards in `Per-scenario/`. The per-metric figures compare
one metric across scenarios and are secondary.

---

## 2. The random-scenario study

Two hundred scenarios sampled from a generator rather than chosen by hand. One
step, plus an optional slow companion.

```bash
python experiments/random_scenarios.py                    # ~8 h 15 m
python experiments/random_scenarios_sample_complexity.py --into experiments/plots/<run>   # ~5 h
```

The two share a seeded generator, so `scenario_id` matches one to one between
them. ⚠ Editing the generator desynchronizes them silently; both must then be
re-run together.

Redrawing needs no simulation:

```bash
python experiments/random_scenarios.py --redraw experiments/plots/<run> [--out <dir>]
```

### Outputs

```
<run>/Data/random_scenarios_raw.csv      one row per scenario x design
<run>/Data/random_scenarios_meta.csv     each scenario's curves, k*, MTD, gap
<run>/Data/random_scenarios_curves.npz   the per-cohort curves and dose offsets
<run>/Data/Partials/                     written as each scenario completes
<run>/Summaries/                         paired differences, all ordered pairs
<run>/Results/Figures/                   the study's figures
<run>/Results/Tables/                    table2_well.tex, table2_mis.tex
<run>/SampleComplexity/                  the companion script, same layout
```

`summary_rec_at_kstar_by_gap.png` is the thesis's headline figure.

The partials are why `--redraw` works on an interrupted run: if the combined
`.npz` is missing it rebuilds from whatever scenarios finished.

---

## Conventions

These live in shared code, so any run picks them up without being asked.

**Design names.** `display_name()` in `src/doseescalation/evaluate.py` maps the
internal names to what a reader sees: `SEEDA Plateau` → **Reference**,
`SEEDA Plateau (fixed)` → **Descending**, `SEEDA Plateau (ours)` → **Anchored**.
The mapping is applied when drawing, so the persisted CSVs keep the internal
names and nothing has to be re-simulated after a rename.

**Figure text.** `BASE_FONT` in the same module. Every figure keys its title,
legend, axis and tick sizes off it.

**Captions.** Generated `.tex` tables carry no `\caption{}`; the wording is added
in the LaTeX source rather than by the code that writes the tables.

**Nothing is overwritten.** `--out`, on `bootstrap.py` and on
`random_scenarios.py --redraw`, sends figures, tables and CSVs to a fresh
destination and leaves the source run as it was.
