import os
import re

import numpy as np
import pandas as pd


def build_results_table(rec_map, alloc_map, correct_mtds, n_levels, algos,
                        opt_doses=None, efficacy_algos=()):
    records = []
    for scenario in rec_map:
        for algo in algos:
            # Efficacy-aware methods are scored against the efficacy-optimal dose:
            if opt_doses is not None and algo in efficacy_algos:
                correct = opt_doses[scenario]
            # Toxicity-only methods are scored against the toxicity MTD:
            else:
                correct = correct_mtds[scenario]
            recs = np.asarray(rec_map[scenario][algo])
            allocs = np.asarray(alloc_map[scenario][algo])
            for dose in range(n_levels):
                records.append({
                    "Scenario": scenario,
                    "Algorithm": algo,
                    "Dose": dose,
                    # Whether this dose is the algorithm's correct (target) dose:
                    "Is correct": dose == correct,
                    "Rec (in %)": round(100 * np.mean(recs == dose), 2) if recs.size else np.nan,
                    "Alloc (in %)": round(100 * np.mean(allocs == dose), 2) if allocs.size else np.nan
                })
    return pd.DataFrame.from_records(records)


def build_correct_dose_summary(results_table, correct_mtds):
    """
    Per-(scenario, algorithm) summary: the row's correct dose (1-indexed, paper
    numbering), whether that's the toxicity MTD, and the % of trials
    recommending / allocating it. Derived from `build_results_table`'s
    long-format output, filtered to each row's correct-dose record.
    """
    correct_only = results_table[results_table["Is correct"]].copy()
    correct_only["Is MTD"] = [
        dose == correct_mtds[scenario]
        for scenario, dose in zip(correct_only["Scenario"], correct_only["Dose"])
    ]
    summary = correct_only.set_index(["Scenario", "Algorithm"])[
        ["Dose", "Is MTD", "Rec (in %)", "Alloc (in %)"]
    ].rename(columns={
        "Dose": "Correct dose",
        "Rec (in %)": "Correct dose rec %",
        "Alloc (in %)": "Correct dose alloc %",
    })
    # Display doses as 1-6 (paper's numbering):
    summary["Correct dose"] = summary["Correct dose"] + 1
    return summary


def _rec_mean_std(recs, n_levels, n_batches):
    """
    Recommendation %: each trial yields a single final dose, so the per-trial
    value is degenerate (0/100). Mean over all trials = the reported %. The std
    is the Monte-Carlo std of that proportion, estimated by splitting the trials
    into `n_batches` batches and taking the std of the batch percentages.
    """
    recs = np.asarray(recs)
    onehot = np.zeros((recs.size, n_levels))
    if recs.size:
        onehot[np.arange(recs.size), recs] = 100.0
    mean = onehot.mean(axis=0)
    nb = max(2, min(n_batches, recs.size))
    batch_means = np.array([b.mean(axis=0) for b in np.array_split(onehot, nb)])
    return mean, batch_means.std(axis=0, ddof=1)


def _alloc_mean_std(allocs_flat, n_cohorts, n_levels):
    """
    Allocation %: each trial yields a full allocation distribution, so the
    per-trial value (fraction of that trial's cohorts at each dose) is
    meaningful. Mean and std are taken across the per-trial percentages.
    """
    arr = np.asarray(allocs_flat).reshape(-1, n_cohorts)  # (n_trials, n_cohorts)
    frac = np.stack([(arr == k).mean(axis=1) for k in range(n_levels)], axis=1) * 100.0
    return frac.mean(axis=0), frac.std(axis=0, ddof=1)


def build_table2(rec_map, alloc_map, scenario_key, algos, n_cohorts, n_levels,
                 tox_probs, eff_probs, n_batches=5):
    """
    Reproduce Table 2 of the SEEDA paper: Recommended | Allocated side by side,
    one column per dose level, with toxicity / efficacy probability header rows
    and each cell showing "mean\\n(std)" over the trials.

    NOTE on std: the paper states "mean over 1000 repetitions, (standard
    deviation)" but does not define the std precisely, and its two halves have
    different magnitudes. We use the natural definition for each: a Monte-Carlo
    batch std for recommendation (a single trial gives only 0/100), and the
    across-trial std for allocation. Both reproduce the paper's magnitudes;
    `n_batches` is the one knob for the recommendation std.
    """
    doses = [f"Dose {k + 1}" for k in range(n_levels)]
    cols = pd.MultiIndex.from_product([["Recommended", "Allocated"], doses])
    index = ["Toxicity prob", "Efficacy prob"] + list(algos)
    table = pd.DataFrame(index=index, columns=cols, dtype=object)

    for half in ["Recommended", "Allocated"]:
        for k in range(n_levels):
            table.loc["Toxicity prob", (half, doses[k])] = f"{tox_probs[k]:g}"
            table.loc["Efficacy prob", (half, doses[k])] = f"{eff_probs[k]:g}"

    for algo in algos:
        r_mean, r_std = _rec_mean_std(rec_map[scenario_key][algo], n_levels, n_batches)
        a_mean, a_std = _alloc_mean_std(alloc_map[scenario_key][algo], n_cohorts, n_levels)
        for k in range(n_levels):
            table.loc[algo, ("Recommended", doses[k])] = f"{r_mean[k]:.2f}\n({r_std[k]:.2f})"
            table.loc[algo, ("Allocated", doses[k])] = f"{a_mean[k]:.2f}\n({a_std[k]:.2f})"
    return table


def _opt_col_header_style(opt_col):
    # Bold the optimal-dose column header (the paper bolds Dose 3):
    def header_style(vals):
        return ['font-weight: bold' if v == opt_col else '' for v in vals]
    return header_style


def _parse_cell_value(cell):
    """Extract the numeric mean from a 'value\\n(std)' or LaTeX
    '\\makecell{value \\ (std)}' cell string."""
    if isinstance(cell, str) and r"\makecell" in cell:
        try:
            inner = cell[len(r"\makecell{"):-1]          # "value \\ (std)"
            return float(inner.split(r" \\ ")[0].strip())
        except (ValueError, IndexError):
            return -np.inf
    if isinstance(cell, str) and "\n" in cell:
        try:
            return float(cell.split("\n")[0].strip())
        except ValueError:
            return -np.inf
    try:
        return float(str(cell))
    except (ValueError, TypeError):
        return -np.inf


def _table2_body_styles(table, opt_col):
    """
    Combined on-screen body styling for Table 2:
      - green background on the optimal-dose column (opt_col), every row.
      - bold on the majority (most-recommended / most-allocated) cell per
        algorithm row, computed separately within each half (Recommended /
        Allocated). Mirrors export_table2_latex's majority-cell logic, which
        this on-screen table was previously missing.
    """
    prob_rows = {"Toxicity prob", "Efficacy prob"}
    styles = pd.DataFrame('', index=table.index, columns=table.columns)
    for half in ["Recommended", "Allocated"]:
        half_cols = [c for c in table.columns if c[0] == half]
        mtd_key = (half, opt_col)
        if mtd_key in styles.columns:
            styles[mtd_key] += 'background-color: #2ca02c82; '
        for row in table.index:
            if row in prob_rows:
                continue
            vals = {c: _parse_cell_value(table.loc[row, c]) for c in half_cols}
            if vals:
                majority = max(vals, key=vals.get)
                styles.loc[row, majority] += 'font-weight: bold; '
    return styles


def style_table2(table, opt_dose):
    """On-screen styler: optimal-dose column highlighted green, majority
    (most-recommended / most-allocated) cell per row in bold, two-line
    mean/(std) cells, everything centred."""
    opt_col = f"Dose {opt_dose + 1}"
    styles = _table2_body_styles(table, opt_col)
    return (
        table.style
        .apply(lambda _: styles, axis=None)
        .apply_index(_opt_col_header_style(opt_col), axis="columns", level=1)
        .set_properties(**{"white-space": "pre-wrap", "text-align": "center"})
    )


def export_table2_latex(table, opt_dose, path, caption=None, label=None):
    """
    Export the Table-2-format DataFrame to LaTeX with:
      - Green shading on the MTD (optimal-dose) column only.
      - Bold on the majority (most-recommended / most-allocated) cell per
        algorithm row, separately for the Recommended and Allocated halves.
      - Centered \\multicolumn headers with \\cmidrule underlines.
      - Vertical rule separating the two halves (l|cccccc|cccccc).
      - \\resizebox to keep the wide table within the text width.
      - Extra \\midrule after the Efficacy prob reference row.

    Requires in the document preamble:
        \\usepackage[table]{xcolor}   % \\cellcolor
        \\usepackage{booktabs}        % booktabs rules + \\cmidrule
        \\usepackage{makecell}        % two-line \\makecell cells
        \\usepackage{float}           % [H] placement
        \\usepackage{graphicx}        % \\resizebox
    """
    opt_col = f"Dose {opt_dose + 1}"
    prob_rows = {"Toxicity prob", "Efficacy prob"}

    # Turn "mean\n(std)" into \makecell{mean \\ (std)}:
    latex_tbl = table.map(
        lambda v: r"\makecell{" + v.replace("\n", r" \\ ") + "}"
        if isinstance(v, str) and "\n" in v else v
    )

    # Build per-cell style DataFrame:
    #   - green background  → MTD column (opt_col) for every row
    #   - bold              → argmax cell per algorithm row per half
    styles = pd.DataFrame('', index=latex_tbl.index, columns=latex_tbl.columns)

    for half in ["Recommended", "Allocated"]:
        half_cols = [c for c in latex_tbl.columns if c[0] == half]
        # Green on MTD column:
        mtd_key = (half, opt_col)
        if mtd_key in styles.columns:
            styles[mtd_key] += 'background-color: #c8e6c9; '
        # Bold on majority cell (algorithm rows only):
        for row in latex_tbl.index:
            if row in prob_rows:
                continue
            vals = {c: _parse_cell_value(latex_tbl.loc[row, c]) for c in half_cols}
            if vals:
                majority = max(vals, key=vals.get)
                styles.loc[row, majority] += 'font-weight: bold; '

    styler = (
        latex_tbl.style
        .apply(lambda _: styles, axis=None)
        .apply_index(_opt_col_header_style(opt_col), axis="columns", level=1)
    )

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    # Generate raw LaTeX string:
    raw = styler.to_latex(
        convert_css=True,    # background-color -> \cellcolor, font-weight -> \bfseries
        hrules=True,         # booktabs rules
        caption=caption,
        label=label,
        position_float="centering",
        position="H",
    )

    # ── Post-process the generated LaTeX ──────────────────────────────────────

    # 1. Fix column spec: replace any all-l spec with l|cccccc|cccccc
    raw = re.sub(r'\\begin\{tabular\}\{[^}]+\}',
                 r'\\begin{tabular}{l|cccccc|cccccc}', raw)

    # 2. Center the section headers and add a vertical divider after Recommended:
    raw = raw.replace(r'\multicolumn{6}{r}{Recommended}',
                      r'\multicolumn{6}{c|}{Recommended}')
    raw = raw.replace(r'\multicolumn{6}{r}{Allocated}',
                      r'\multicolumn{6}{c}{Allocated}')

    # 3. Insert \cmidrule lines below the Recommended / Allocated header row:
    raw = raw.replace(
        r'\multicolumn{6}{c}{Allocated} \\',
        r'\multicolumn{6}{c}{Allocated} \\' + '\n'
        + r'\cmidrule(lr){2-7}\cmidrule(lr){8-13}'
    )

    # 4. Add \midrule after the Efficacy prob row to separate it from algorithms:
    lines = raw.split('\n')
    new_lines = []
    for line in lines:
        new_lines.append(line)
        if 'Efficacy prob' in line:
            new_lines.append(r'\midrule')
    raw = '\n'.join(new_lines)

    # 5. Wrap the tabular in \resizebox so the wide table fits the text width:
    raw = raw.replace(
        r'\begin{tabular}',
        r'\resizebox{\textwidth}{!}{%' + '\n' + r'\begin{tabular}'
    )
    raw = raw.replace(
        r'\end{tabular}',
        r'\end{tabular}' + '\n' + r'}%'
    )

    with open(path, 'w') as f:
        f.write(raw)
    print(f"Saved LaTeX table to {path}")
    return path
