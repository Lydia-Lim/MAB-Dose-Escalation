"""
a_max sensitivity sweep:

Sweeps the toxicity-estimate clamp with a0 fixed at 20, so the internal dose grid
is held constant and only the clamp moves. (Sweeping a0 would also change the
grid, since DOSE_LEVELS_SEEDA = inv_dose_toxic(v, a0) -- that confounded sweep is
deliberately not done here.)

What it is for: a_max = a0 = 20 clamps a_hat at exactly the value that reproduces
the true toxicity curve, so the model can only ever over-estimate toxicity, never
under-estimate it. That is why our Type II sits at 0 and our dose-5 allocation is
1.00% against the paper's 14.91%. This sweep measures how much of the remaining
mismatch is that parameter choice rather than the algorithm.
"""

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "experiments"))

from doseescalation.run_simulation import (
    a_key, dose_toxic_curve, run_designs,
)
from scenario_battery import SCENARIOS, make_config, TABLE2_COHORTS

# The scenario and every hyperparameter come from the battery, so the sweep
# cannot drift from the rest of the thesis.
SCENARIO = SCENARIOS[0]
TOXICITY_PROBS = list(SCENARIO.toxicity_probs)
TTL = 0.35
ALGOS = ["3 + 3", "CRM", "UCB", "SEEDA", "SEEDA Plateau"]
DISPLAY = {"SEEDA Plateau": "Reference"}
SWEPT = ("SEEDA", "SEEDA Plateau")  # the only designs that take an a_max

A_MAX_VALUES = [5, 10, 15, 20, 25, 30, 40, None]

# Paper Table 2, for reference in the summary.
PAPER = {
    "SEEDA": dict(rec=[0, 1, 47.20, 47.40, 4.40, 0],
                  alloc=[11.18, 9.18, 30.76, 31.71, 12.06, 5.11]),
    "SEEDA Plateau": dict(rec=[0.80, 2.20, 86.60, 10.40, 0, 0],
                          alloc=[7.83, 8.98, 30.12, 37.17, 14.91, 1.00]),
}


def admissible_at(a_max, a0):
    """Doses the model calls safe when a_hat is pinned at the clamp.

    a_hat can only reach the clamp, so this is the most permissive admissible set
    the design can hold. Predicts when dose 5 (true tox 0.45) becomes allocatable.
    """
    if a_max is None:
        return "unbounded"
    grid = [np.arctanh(2 * p ** (1 / a0) - 1) for p in TOXICITY_PROBS]
    tox = dose_toxic_curve(np.array(grid), a_max)
    return [k + 1 for k in range(len(tox)) if tox[k] <= TTL]


def summarize(results, n_levels):
    """Per-design allocation %, recommendation %, and Type I / II error rates."""
    df = results.trials
    tox = np.array(TOXICITY_PROBS)
    safe_true = tox <= TTL
    rows = []
    for algo in ALGOS:
        d = df[df["design"] == algo]
        enrolled = d[d["enrolled"]]
        alloc = [100 * (enrolled["dose_alloc"] == k).mean() for k in range(n_levels)]
        rec = [100 * (d["dose_rec"] == k).mean() for k in range(n_levels)]

        # Type I / II from the per-cohort safety classification, skipping the
        # warm-up round-robin where nothing is classified yet.
        post = d[d["cohort"] >= n_levels]
        safe_cols = [f"safe_{k + 1}" for k in range(n_levels)]
        flags = post[safe_cols]
        if flags.isna().all(axis=None):
            type1 = type2 = np.nan  # 3 + 3 has no safety model
        else:
            declared = flags.to_numpy(dtype=float)
            type1 = 100 * np.nanmean(
                (1 - declared[:, safe_true]).mean(axis=1))
            type2 = 100 * np.nanmean(
                declared[:, ~safe_true].mean(axis=1))
        rows.append(dict(design=algo, alloc=alloc, rec=rec,
                         type1=type1, type2=type2))
    return rows


def _a_max_label(v):
    """Normalize an a_max cell to a label.

    Handles both call paths: in-memory from main() the column holds strings
    (including the literal "None"), while read back from CSV pandas has parsed
    it to float64 with NaN.
    """
    if v is None or v == "None" or (isinstance(v, float) and pd.isna(v)):
        return "None"
    return str(int(float(v)))


def _order_labels(values):
    """a_max labels left to right, numeric ascending with None last."""
    # Sort numerically, not lexicographically -- as strings "5" lands after "40".
    nums = sorted((v for v in values if v != "None"), key=float)
    return [str(int(float(v))) for v in nums] + (["None"] if "None" in values else [])


def plot_summary(flat, out_dir):
    """Four panels vs a_max: Type I, Type II, recommendation accuracy, unsafe allocation.

    Only SEEDA and SEEDA-Plateau are drawn -- the other designs never receive an
    a_max, so their rows are identical across configs (a useful invariance check
    on the sweep itself).
    """
    flat = flat.copy()
    flat["a_max"] = [
        _a_max_label(v) for v in flat["a_max"]
    ]
    order = _order_labels(set(flat["a_max"]))
    flat["unsafe_alloc"] = flat["alloc_d5"] + flat["alloc_d6"]

    panels = [
        ("type1", "Type I — safe doses called unsafe (%)", None),
        ("type2", "Type II — unsafe doses called safe (%)", None),
        ("rec_d3", "Recommends dose 3 = k* (%)",
         {"SEEDA": 47.20, "SEEDA Plateau": 86.60}),
        ("unsafe_alloc", "Allocation to unsafe doses 5+6 (%)",
         {"SEEDA": 17.17, "SEEDA Plateau": 15.91}),
    ]
    fig = make_subplots(rows=2, cols=2,
                        subplot_titles=[p[1] for p in panels],
                        shared_xaxes=False)

    for idx, (col, _title, paper_ref) in enumerate(panels):
        row, cc = idx // 2 + 1, idx % 2 + 1
        for design, color in (("SEEDA", "red"), ("SEEDA Plateau", "#2ca02c")):
            d = flat[flat["design"] == design].set_index("a_max").reindex(order)
            fig.add_trace(
                go.Scatter(x=order, y=d[col], name=DISPLAY.get(design, design),
                           marker_color=color,
                           mode="lines+markers", legendgroup=design,
                           showlegend=(idx == 0)),
                row=row, col=cc,
            )
            if paper_ref:
                fig.add_hline(y=paper_ref[design], line_dash="dot",
                              line_color=color, opacity=0.5, row=row, col=cc)

    fig.update_layout(
        autosize=False, width=1150, height=760, template="plotly_white",
        title=dict(text=("Ceiling on a_hat, with the dose grid held fixed. "
                         "Dotted lines: the paper's Table 2. "
                         "Only SEEDA and Reference take a ceiling."),
                   font=dict(size=17)),
        legend=dict(title_text="Design", font=dict(size=14)),
    )
    fig.update_xaxes(title_text="Ceiling on a_hat", type="category",
                     title_font_size=15, tickfont_size=13)
    fig.update_yaxes(rangemode="tozero", tickfont_size=13)
    for annotation in fig.layout.annotations[:4]:
        annotation.font.size = 15

    path = Path(out_dir) / "a_max_sensitivity.png"
    fig.write_image(str(path))
    return path


def plot_allocation(flat, out_dir):
    """Stacked per-dose allocation vs a_max, one panel per SEEDA-family design.

    Shows which doses gain and lose as the clamp moves: the collapse onto doses
    1-2 below 15, and dose 5 opening up at 30.
    """
    flat = flat.copy()
    flat["a_max"] = [
        _a_max_label(v) for v in flat["a_max"]
    ]
    order = _order_labels(set(flat["a_max"]))
    designs = SWEPT
    # Green (safe) through red (unsafe); doses 5-6 are above theta = 0.35.
    dose_colors = ["#1b7837", "#7fbf7b", "#d9f0d3", "#fddbc7", "#ef8a62", "#b2182b"]

    fig = make_subplots(rows=1, cols=len(designs), shared_yaxes=True,
                        subplot_titles=[DISPLAY.get(d, d) for d in designs])
    for ci, design in enumerate(designs, start=1):
        d = flat[flat["design"] == design].set_index("a_max").reindex(order)
        for k in range(len(TOXICITY_PROBS)):
            unsafe = TOXICITY_PROBS[k] > TTL
            fig.add_trace(
                go.Bar(x=order, y=d[f"alloc_d{k+1}"],
                       name=f"Dose {k+1}" + (" (unsafe)" if unsafe else ""),
                       marker_color=dose_colors[k],
                       legendgroup=f"d{k+1}", showlegend=(ci == 1)),
                row=1, col=ci,
            )
    fig.update_layout(
        barmode="stack", autosize=False, width=1150, height=520,
        title_text=("Where patients go as a_max moves (a0 = 20 fixed). "
                    "Doses 5-6 are above theta = 0.35."),
        legend_title_text="Allocated dose",
    )
    fig.update_xaxes(title_text="a_max", type="category")
    fig.update_yaxes(title_text="Allocation (%)", row=1, col=1)

    path = Path(out_dir) / "a_max_allocation.png"
    fig.write_image(str(path))
    return path


def run_sensitivity(base, n_cohorts, data_dir, fig_dir):
    """Sweep the clamp, writing one summary CSV and two figures.

    Unlike the other diagnosis outputs this one re-simulates: each value of the
    clamp is a different design, so there is nothing to reuse between them. Eight
    configurations at the table horizon, a few minutes in total.

    The per-configuration trial dumps the standalone version used to write are
    deliberately not produced here. They run to well over a megabyte apiece, the
    run folder is under version control, and every number the thesis quotes is in
    the summary.
    """
    data_dir, fig_dir = Path(data_dir), Path(fig_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    print(f"a0 = {base.a_seeda} fixed, n_trials = {base.n_trials}, "
          f"n_cohorts = {n_cohorts}\n")

    all_rows = []
    for a_max in A_MAX_VALUES:
        label = "None" if a_max is None else str(a_max)
        cfg = replace(
            base,
            a_max_seeda=None if a_max is None else float(a_max),
            a_max_plateau=None if a_max is None else float(a_max),
            label=f"a_max sweep: {label}",
        )
        print(f"--- a_max = {label:>4}  (model calls safe: "
              f"{admissible_at(a_max, base.a_seeda)}) ---", flush=True)
        results = run_designs(cfg, n_cohorts=n_cohorts)

        for row in summarize(results, cfg.n_levels):
            row["a_max"] = label
            all_rows.append(row)
            if row["design"] in SWEPT:
                print(f"    {DISPLAY.get(row['design'], row['design']):16} "
                      f"alloc d5={row['alloc'][4]:5.2f}%  "
                      f"rec d3={row['rec'][2]:5.2f}%  "
                      f"TypeI={row['type1']:5.2f}%  TypeII={row['type2']:5.2f}%",
                      flush=True)
        print()

    flat = pd.DataFrame([
        dict(a_max=r["a_max"], design=r["design"],
             **{f"alloc_d{k+1}": r["alloc"][k] for k in range(len(TOXICITY_PROBS))},
             **{f"rec_d{k+1}": r["rec"][k] for k in range(len(TOXICITY_PROBS))},
             type1=r["type1"], type2=r["type2"])
        for r in all_rows
    ])
    summary_path = data_dir / "table_sensitivity.csv"
    flat.to_csv(summary_path, index=False)

    print("=" * 78)
    print("dose-5 allocation % and Type II % vs a_max "
          "(paper: SEEDA 12.06 / Plateau 14.91)")
    print("=" * 78)
    for design in SWEPT:
        sub = flat[flat["design"] == design]
        print(f"\n{DISPLAY.get(design, design)}")
        print(f"  {'a_max':>6} {'alloc d5':>9} {'alloc d6':>9} "
              f"{'rec d3':>8} {'TypeI':>7} {'TypeII':>7}")
        for _, r in sub.iterrows():
            print(f"  {r['a_max']:>6} {r['alloc_d5']:9.2f} {r['alloc_d6']:9.2f} "
                  f"{r['rec_d3']:8.2f} {r['type1']:7.2f} {r['type2']:7.2f}")
        p = PAPER[design]
        print(f"  {'paper':>6} {p['alloc'][4]:9.2f} {p['alloc'][5]:9.2f} "
              f"{p['rec'][2]:8.2f} {'-':>7} {'-':>7}")

    print(f"\nsummary -> {summary_path}")
    print(f"panels  -> {plot_summary(flat, fig_dir)}")
    print(f"alloc   -> {plot_allocation(flat, fig_dir)}")
    return flat


def main():
    """Standalone entry point. Normally this runs from ``diagnosis_tables.py``."""
    import datetime as dt

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "summary_csv", nargs="?",
        help="redraw the figures from an existing summary CSV instead of sweeping",
    )
    parser.add_argument("--trials", type=int, default=None)
    parser.add_argument(
        "--n-cohorts", type=int, default=TABLE2_COHORTS,
        help=f"trial horizon (default {TABLE2_COHORTS}, the table horizon; "
             f"use 300 for the figures horizon)",
    )
    args = parser.parse_args()

    if args.summary_csv:
        csv = Path(args.summary_csv)
        print("panels ->", plot_summary(pd.read_csv(csv), csv.parent))
        print("alloc  ->", plot_allocation(pd.read_csv(csv), csv.parent))
        return

    stamp = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out = HERE / "runs" / stamp
    cfg = make_config(SCENARIO, label=f"diagnosis-sensitivity {stamp}")
    cfg.algos = ALGOS
    if args.trials is not None:
        cfg.n_trials = args.trials
    run_sensitivity(cfg, args.n_cohorts, out / "Data",
                    out / "Results" / "Figures")


if __name__ == "__main__":
    main()
