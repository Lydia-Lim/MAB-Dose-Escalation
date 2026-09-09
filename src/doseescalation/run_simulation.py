from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .dose_escalator import (
    CRMDoseEscalator,
    DoseEscalatorBase,
    SEEDADoseEscalator,
    SEEDAPlateauFixedDoseEscalator,
    SEEDAPlateauOursDoseEscalator,
    SEEDAPlateauTwoSidedDecoupledDoseEscalator,
    ThreePlusThreeDoseEscalator,
    UCBDoseEscalator,
)
from .estimator import CRMEstimator
from .evaluate import simulate
from .simulated_env import SimulatedEnv


def dose_toxic_curve(dose_levels, a_hat):
    return np.power((np.tanh(dose_levels) + 1) / 2, a_hat)


def inv_dose_toxic(p_dle, a):
    return np.arctanh(2 * np.power(p_dle, 1 / a) - 1)


def logistic_toxic_curve(dose_levels, beta_0, beta_1):
    """
    Standard 2-parameter logistic dose-toxicity model,

        p(d) = exp(beta_0 + beta_1 * d) / (1 + exp(beta_0 + beta_1 * d))

    equivalently ``logit p(d) = beta_0 + beta_1 * d``. Counterpart to
    ``dose_toxic_curve`` (the power-of-tanh model the designs assume), for
    generating TRUE toxicities that lie outside that family -- pass the result
    as ``SimulationConfig.true_toxicity_probs``. ``dose_levels`` is the
    internal dose grid (i.e. ``inv_dose_toxic(p, a0)``), NOT the 1..K dose
    index, so both models are evaluated at the same dose positions.
    """
    odds = np.exp(beta_0 + beta_1 * np.asarray(dose_levels, dtype=float))
    return odds / (1.0 + odds)


# Canonical design names. `SimulationConfig.algos` selects from these (any
# subset, in any order) and they key every result map, so they must stay in
# sync with `build_escalators` and with evaluate.PAPER_COLORS.
NAME_3P3 = "3 + 3"
NAME_CRM = "CRM"
NAME_UCB = "UCB"
NAME_SEEDA = "SEEDA"
NAME_PLATEAU = "SEEDA Plateau"
NAME_OURS = "SEEDA Plateau (ours)"
NAME_FIXED = "SEEDA Plateau (fixed)"


def a_key(a):
    return f"a = {a:.1f}"


def cohort_key(cohort):
    return f"Cohort {cohort + 1}"


# Container for the settings a study script defines. The values live in the
# caller; this only fixes their shape so the runner takes one argument and
# the whole config can be dumped next to the results.
@dataclass
class SimulationConfig:
    a_star: float
    toxicity_probs: Sequence[float]
    efficacy_probs: Sequence[float]
    ttl: float
    n_trials: int
    cohort_size: int
    algos: Sequence[str]
    ucb_coefficient: float
    eta: int
    seeda_c: float
    plateau_c: float
    l1_coefficient: float
    a_seeda: float
    a_plateau: float
    a_max_seeda: Optional[float] = None
    a_max_plateau: Optional[float] = None
    # Our variant reuses the Plateau's a0 for its dose grid but deliberately runs
    # WITHOUT a clamp -- its Jeffreys smoothing is meant to supply the safety the
    # clamp was providing, without needing to know the true a.
    a_max_ours: Optional[float] = None
    p_smoothing: float = 0.5
    tox_margin_c: float = 0.01
    # Toxicity scenarios keyed by their `a`; defaults to the single a_star one.
    p_dle_levels: Optional[Dict[float, Sequence[float]]] = None
    # Free-text provenance label recorded in run_config.json, e.g. "baseline",
    # "a_max = None", "MTD-anchored L1".
    label: str = ""
    # Misspecification hook: when set, the ENVIRONMENT draws DLTs from this
    # per-dose array instead of dose_toxic_curve(dose, a) -- while the
    # escalators' own grid and internal model stay built from toxicity_probs
    # (the NOMINAL curve) as usual, so they keep assuming the standard
    # power-of-tanh model even though the truth no longer follows it. Needs no
    # escalator changes: reuses the same index-keyed SimulatedEnv already used
    # for efficacy_env(). Length must equal n_levels. None = well-specified
    # (environment == the escalators' own assumed curve, as today).
    true_toxicity_probs: Optional[Sequence[float]] = None

    def __post_init__(self):
        if self.p_dle_levels is None:
            self.p_dle_levels = {self.a_star: list(self.toxicity_probs)}

    @property
    def n_levels(self):
        return len(self.toxicity_probs)

    @property
    def dose_levels(self):
        return {
            a: [inv_dose_toxic(v, a) for v in vs]
            for a, vs in self.p_dle_levels.items()
        }

    @property
    def dose_levels_seeda(self):
        return [inv_dose_toxic(v, self.a_seeda) for v in self.toxicity_probs]

    @property
    def dose_levels_plateau(self):
        return [inv_dose_toxic(v, self.a_plateau) for v in self.toxicity_probs]

    # Efficacy env: reuse SimulatedEnv with "dose levels" = indices so the curve
    # maps index -> probability, drawing responders ~ Binomial(cohort, q_idx).
    def efficacy_env(self):
        eff = list(self.efficacy_probs)
        return SimulatedEnv(list(range(self.n_levels)), lambda i: eff[int(i)])


# Field names map onto the old ones (rec_map = a_algo_rec_map, etc.); the plotting
# and table helpers consume these unchanged.
@dataclass
class SimulationResults:
    n_cohorts: int
    algos: Sequence[str]
    rec_map: Dict = field(default_factory=dict)
    rec_final_map: Dict = field(default_factory=dict)
    cohort_rec_map: Dict = field(default_factory=dict)
    cohort_safe_map: Dict = field(default_factory=dict)
    alloc_map: Dict = field(default_factory=dict)
    enrolled_map: Dict = field(default_factory=dict)
    n_dle_map: Dict = field(default_factory=dict)
    # Long-format raw record, one row per design x trial x cohort.
    trials: Optional[pd.DataFrame] = None


def _empty_maps(cfg: SimulationConfig, n_cohorts: int) -> SimulationResults:
    scenarios = [a_key(a) for a in cfg.dose_levels]

    def flat():
        return {s: {algo: [] for algo in cfg.algos} for s in scenarios}

    def per_cohort():
        return {
            s: {cohort_key(c): {algo: [] for algo in cfg.algos}
                for c in range(n_cohorts)}
            for s in scenarios
        }

    return SimulationResults(
        n_cohorts=n_cohorts,
        algos=list(cfg.algos),
        rec_map=flat(),
        rec_final_map=flat(),
        cohort_rec_map=per_cohort(),
        cohort_safe_map=per_cohort(),
        alloc_map=flat(),
        enrolled_map=flat(),
        n_dle_map=flat(),
    )


def build_escalators(cfg: SimulationConfig, dose_levels, n_cohorts):
    """
    Instantiate one escalator per design named in ``cfg.algos``, in that order.

    Every design known to the runner is built below under its canonical name;
    the result is then filtered to the ones ``cfg.algos`` actually asks for. So
    a config listing only a subset (e.g. a run that omits the Descending
    variant) works unchanged, and callers that index
    this list positionally against ``cfg.algos`` -- ``sample_complexity`` does
    -- still line up.
    """
    delta_1 = 1.0 / n_cohorts
    ALL = [
        (NAME_3P3, ThreePlusThreeDoseEscalator(
            dose_levels=dose_levels
        )),

        # CRM (one-parameter power model, prior a ~ Exp(mean 0.5); MATLAB CRM.m).
        # Greedy nearest-to-theta allocation, no no-skip constraint:
        (NAME_CRM, CRMDoseEscalator(
            dose_levels=dose_levels,
            target_toxicity_level=cfg.ttl,
            estimator=CRMEstimator(),
            conservative=False,
            no_skip=False,
        )),

        # UCB-1 on efficacy (paper baseline; MATLAB UCB.m). Consumes efficacy
        # feedback (via the efficacy env), no dose-toxicity model:
        (NAME_UCB, UCBDoseEscalator(
            dose_levels=dose_levels,
            target_toxicity_level=cfg.ttl,
            ucb_coefficient=cfg.ucb_coefficient,
        )),

        # SEEDA (a0 = 20, a_max = 20 -- the shipped Safe_UCB.m clamp):
        (NAME_SEEDA, SEEDADoseEscalator(
            dose_levels=cfg.dose_levels_seeda,
            target_toxicity_level=cfg.ttl,
            dose_toxicity_curve=dose_toxic_curve,
            ucb_coefficient=cfg.seeda_c,
            delta_1=delta_1,
            a_init=cfg.a_seeda,
            a_max=cfg.a_max_seeda,
            seed=0,
            no_skip=True,
        )),

        # SEEDA Plateau (Two-sided-decoupled; faithful to the paper's Algorithm 2):
        (NAME_PLATEAU, SEEDAPlateauTwoSidedDecoupledDoseEscalator(
            dose_levels=cfg.dose_levels_plateau,
            target_toxicity_level=cfg.ttl,
            dose_toxicity_curve=dose_toxic_curve,
            ucb_coefficient=cfg.plateau_c,
            delta_1=delta_1,
            l1_coefficient=cfg.l1_coefficient,
            a_init=cfg.a_plateau,
            a_max=cfg.a_max_plateau,
            seed=0,
            eta=cfg.eta,
            no_skip=True,
        )),

        # Our variant: leader restricted to the admissible set (Algorithm 2 step 4),
        # recommendation = lowest admissible dose indistinguishable from the MTD
        # scanned top-down, and Jeffreys smoothing of p_hat in place of the clamp.
        (NAME_OURS, SEEDAPlateauOursDoseEscalator(
            dose_levels=cfg.dose_levels_plateau,
            target_toxicity_level=cfg.ttl,
            dose_toxicity_curve=dose_toxic_curve,
            ucb_coefficient=cfg.plateau_c,
            delta_1=delta_1,
            l1_coefficient=cfg.l1_coefficient,
            a_init=cfg.a_plateau,
            a_max=cfg.a_max_ours,
            p_smoothing=cfg.p_smoothing,
            tox_margin_c=cfg.tox_margin_c,
            seed=0,
            eta=cfg.eta,
            no_skip=True,
        )),

        # Reference with the two paper-vs-MATLAB deviations put back: leader
        # restricted to the admissible set (Algorithm 2 step 4) and an L1 that
        # descends from the MTD and breaks, so it can reach a plateau onset
        # below MTD-1. Every other setting is identical to Reference above, so
        # the pair isolates exactly those two changes.
        (NAME_FIXED, SEEDAPlateauFixedDoseEscalator(
            dose_levels=cfg.dose_levels_plateau,
            target_toxicity_level=cfg.ttl,
            dose_toxicity_curve=dose_toxic_curve,
            ucb_coefficient=cfg.plateau_c,
            delta_1=delta_1,
            l1_coefficient=cfg.l1_coefficient,
            a_init=cfg.a_plateau,
            a_max=cfg.a_max_plateau,
            seed=0,
            eta=cfg.eta,
            no_skip=True,
        )),
    ]
    wanted = set(cfg.algos)
    by_name = {name: esc for name, esc in ALL if name in wanted}
    missing = wanted - by_name.keys()
    if missing:
        raise ValueError(f"unknown design(s) in cfg.algos: {sorted(missing)}")
    return [(name, by_name[name]) for name in cfg.algos]


def run_simulations(
    dose_escalator: DoseEscalatorBase,
    dose_levels: Sequence[float],
    dose_toxic_curve_fn: Callable,
    cohort_size: int,
    n_cohorts: int,
    efficacy_env=None,
):
    env = SimulatedEnv(dose_levels, dose_toxic_curve_fn)
    return simulate(
        cohort_sizes=[cohort_size] * n_cohorts,
        dose_escalator=dose_escalator,
        env=env,
        efficacy_env=efficacy_env,
    )


def run_designs(cfg: SimulationConfig, n_cohorts: int) -> SimulationResults:
    results = _empty_maps(cfg, n_cohorts)
    efficacy_env = cfg.efficacy_env()
    rows = []

    # Misspecification: draw the environment's DLTs from an arbitrary per-dose
    # array (index-keyed, same trick as efficacy_env) instead of the escalators'
    # own assumed curve. dose_levels below stays the escalators' NOMINAL grid
    # either way -- only which array the environment reads from changes.
    if cfg.true_toxicity_probs is not None:
        true_tox = list(cfg.true_toxicity_probs)
        tox_domain = list(range(cfg.n_levels))
        tox_curve = lambda dose: true_tox[int(dose)]
    else:
        tox_domain = None  # per-scenario dose_levels, set in the loop below
        tox_curve = None

    for a, dose_levels in cfg.dose_levels.items():
        scen = a_key(a)
        for trial in range(cfg.n_trials):
            for algo, escalator in build_escalators(cfg, dose_levels, n_cohorts):
                allocations, recommendations, n_dles, safe_sets, enrolled = \
                    run_simulations(
                        escalator,
                        tox_domain if tox_domain is not None else dose_levels,
                        tox_curve if tox_curve is not None
                        else (lambda dose: dose_toxic_curve(dose, a)),
                        cfg.cohort_size,
                        n_cohorts=n_cohorts,
                        efficacy_env=efficacy_env,
                    )

                # Table 2 recommendation = trial-average of per-round picks
                # (RealWorld Safe_UCB_Plateau.m: k_rec1 accumulates the
                # recommended dose every round, then /t). Pool every per-round
                # rec across trials; build_table2 / the results summary then
                # average them into the per-dose "Recommended %".
                results.rec_map[scen][algo].extend(recommendations)
                # Final declared MTD (one per trial) for the recommendation-count plot:
                results.rec_final_map[scen][algo].append(recommendations[-1])
                # Determine every cohort's allocated dose pooled across trials:
                results.alloc_map[scen][algo].extend(allocations)
                # Per-cohort enrolment flags, pooled to match alloc_map:
                results.enrolled_map[scen][algo].extend(enrolled)
                # Determine the total toxicities during this trial:
                results.n_dle_map[scen][algo].append(sum(n_dles))
                # Determine the recommendation at each cohort:
                for cohort, rec in enumerate(recommendations):
                    results.cohort_rec_map[scen][cohort_key(cohort)][algo].append(rec)
                # Determine the per-dose safety classification at each cohort:
                for cohort, safe in enumerate(safe_sets):
                    results.cohort_safe_map[scen][cohort_key(cohort)][algo].append(safe)

                # One row per cohort, carrying `trial` so a bootstrap is
                # "resample trial ids with replacement and recompute".
                for c in range(n_cohorts):
                    safe = safe_sets[c]
                    rows.append((
                        scen, algo, trial, c,
                        allocations[c], recommendations[c],
                        n_dles[c], bool(enrolled[c]),
                    ) + (tuple(bool(x) for x in safe) if safe is not None
                         else (None,) * cfg.n_levels))

    # Exactly one row per (scenario, design, trial, cohort): n_cohorts rows per
    # trial per design, so len(df) = n_scenarios * n_designs * n_trials * n_cohorts.
    # At the paper's settings (1 scenario, 5 designs, 1000 trials) that is 50,000
    # rows for the realistic run (10 cohorts), 500,000 for Table 2 (100) and
    # 1,500,000 for the figures (300).
    #
    #   scenario    which toxicity scenario, as `a_key(a)`, e.g. "a = 1.0"
    #   design      algorithm name, from cfg.algos
    #   trial       0-based repetition index -- THE BOOTSTRAP RESAMPLING UNIT
    #   cohort      0-based cohort index within the trial
    #   dose_alloc  0-based dose actually given to this cohort (allocation policy)
    #   dose_rec    0-based dose the design would declare if it stopped here
    #   n_dle       dose-limiting events observed at dose_alloc this cohort
    #   enrolled    False once the design has terminated (see below)
    #   safe_1..K   the design's per-dose safety classification this cohort;
    #               all NaN for designs with no safety model (3 + 3)
    #
    # `enrolled` is False when the design has already stopped and dosed nobody
    # this cohort. dose_alloc / dose_rec are still filled in -- held at their last
    # value so every trial has the same length -- and n_dle is forced to 0, so
    # those rows describe cohorts that never happened. Only 3 + 3 terminates (it
    # declares an MTD after ~5 cohorts); the model-based designs run the whole
    # horizon and are always True. So False rows are expected and correct, but
    # they dominate 3 + 3 at long horizons: ~51% of its rows at n = 10 and ~95% at
    # n = 100. Any aggregation over allocations must filter on `enrolled` or it
    # will credit 3 + 3 with hundreds of phantom cohorts at its final dose.
    cols = ["scenario", "design", "trial", "cohort",
            "dose_alloc", "dose_rec", "n_dle", "enrolled"]
    cols += [f"safe_{k + 1}" for k in range(cfg.n_levels)]
    results.trials = pd.DataFrame(rows, columns=cols)
    return results


# Persistence. write_results dumps the raw records + the config that produced them
# results_from_csv rebuilds the maps so figures and tables can be regenerated
# or bootstrapped without re-simulating.
def _git_commit():
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        return None


def write_results(results: SimulationResults, cfg: SimulationConfig, out_dir):
    data_dir = Path(out_dir) / "Data"
    data_dir.mkdir(parents=True, exist_ok=True)

    csv_path = data_dir / "trials.csv.gz"
    results.trials.to_csv(csv_path, index=False, compression="gzip")

    meta = asdict(cfg)
    meta["p_dle_levels"] = {str(k): list(v) for k, v in cfg.p_dle_levels.items()}
    meta.update({
        "n_cohorts": results.n_cohorts,
        "n_levels": cfg.n_levels,
        "git_commit": _git_commit(),
    })
    (data_dir / "run_config.json").write_text(json.dumps(meta, indent=2))

    return csv_path


def results_from_csv(path):
    path = Path(path)
    data_dir = path if path.is_dir() else path.parent
    df = pd.read_csv(data_dir / "trials.csv.gz")
    meta = json.loads((data_dir / "run_config.json").read_text())

    n_cohorts = int(meta["n_cohorts"])
    n_levels = int(meta["n_levels"])
    algos = list(meta["algos"])
    scenarios = sorted(df["scenario"].unique())

    def flat():
        return {s: {algo: [] for algo in algos} for s in scenarios}

    def per_cohort():
        return {
            s: {cohort_key(c): {algo: [] for algo in algos}
                for c in range(n_cohorts)}
            for s in scenarios
        }

    out = SimulationResults(
        n_cohorts=n_cohorts, algos=algos,
        rec_map=flat(), rec_final_map=flat(),
        cohort_rec_map=per_cohort(), cohort_safe_map=per_cohort(),
        alloc_map=flat(), enrolled_map=flat(), n_dle_map=flat(),
        trials=df,
    )

    safe_cols = [f"safe_{k + 1}" for k in range(n_levels)]
    for (scen, algo), g in df.groupby(["scenario", "design"], sort=False):
        g = g.sort_values(["trial", "cohort"])
        out.rec_map[scen][algo] = g["dose_rec"].tolist()
        out.alloc_map[scen][algo] = g["dose_alloc"].tolist()
        out.enrolled_map[scen][algo] = g["enrolled"].astype(bool).tolist()
        for _, tg in g.groupby("trial", sort=True):
            out.rec_final_map[scen][algo].append(int(tg["dose_rec"].iloc[-1]))
            out.n_dle_map[scen][algo].append(int(tg["n_dle"].sum()))
        for cohort, cg in g.groupby("cohort", sort=True):
            key = cohort_key(int(cohort))
            out.cohort_rec_map[scen][key][algo] = cg["dose_rec"].tolist()
            safe = cg[safe_cols]
            out.cohort_safe_map[scen][key][algo] = (
                [None] * len(cg) if safe.isna().all(axis=None)
                else safe.astype(bool).values.tolist()
            )
    return out


# Sample complexity, matching the authors' samplecomplexity.m + SamComp_*.m.
#
# The MATLAB does NOT invert a single long run. For each error target delta it
# searches upward over the horizon, and at every candidate horizon n it runs
# `iters` FRESH trials of length n and reads only their FINAL recommendation:
#
#     n = n_init
#     while n <= n_max and not done:
#         err = mean over `iters` fresh runs of length n of 1{final rec != k*}
#         if err <= delta: done          # answer is this n
#         else:            n += step
#
# That whole search is repeated `rounds` times and the answers averaged, and the
# deltas are walked from loosest to strictest with each search warm-started at
# the previous delta's answer (sample complexity is monotone in delta).
#
# Fresh runs per horizon are what make this stable: our old plot inverted one
# 300-cohort trajectory, so near a design's asymptote the answer was decided by
# wherever noise first crossed the target. It also matters for correctness here,
# because a design's behavior depends on its horizon (delta_1 = 1/n_cohorts, and
# the L1 width uses log n) -- cohort n of a 300-cohort run is NOT the same as a
# fresh n-cohort run.
#
# Cost: roughly n_designs * n_deltas * rounds * (horizons tried) * iters trials.
# The MATLAB's rounds=300, iters=100 is far too slow in Python; start small.
def _final_recommendation(cfg, algo_idx, dose_levels, n_cohorts, efficacy_env, a):
    _, escalator = build_escalators(cfg, dose_levels, n_cohorts)[algo_idx]
    _, recommendations, _, _, _ = run_simulations(
        escalator,
        dose_levels,
        lambda dose: dose_toxic_curve(dose, a),
        cfg.cohort_size,
        n_cohorts=n_cohorts,
        efficacy_env=efficacy_env,
    )
    return recommendations[-1]


def sample_complexity(
    cfg: SimulationConfig,
    optimal_dose: int,
    algos=None,
    deltas=None,
    n_max: int = 300,
    step: int = 3,
    rounds: int = 20,
    iters: int = 50,
    scenario_a=None,
    scenario_name=None,
    progress: bool = False,
    checkpoint_path=None,
) -> pd.DataFrame:
    """Minimum horizon reaching each recommendation-error target.

    Returns one row per (design, delta) with the mean/std horizon over `rounds`
    independent searches, in cohorts and in patients. `accuracy` = 1 - delta is
    the x-axis of the paper's Figure 3. `censored` flags targets no search met
    within n_max, where the reported value is a lower bound. A `scenario`
    column (defaults to `a_key(a)`, override with `scenario_name` for a
    human-readable label) is always included, so results from several calls
    can be concatenated and handed to `plot_sample_complexity` as one frame --
    it grids by `scenario` whenever more than one is present.

    Pass `checkpoint_path` to have the frame rewritten after every completed
    point; recommended for anything above the default rounds/iters, where one
    point can take hours.
    """
    if deltas is None:
        deltas = [round(0.1 * i, 1) for i in range(1, 11)]
    algos = list(algos) if algos is not None else list(cfg.algos)
    a = cfg.a_star if scenario_a is None else scenario_a
    scen = scenario_name if scenario_name is not None else a_key(a)
    dose_levels = cfg.dose_levels[a]
    efficacy_env = cfg.efficacy_env()

    records = []
    for algo in algos:
        algo_idx = list(cfg.algos).index(algo)
        # Loosest target first, warm-starting each search at the previous answer.
        warm_start = cfg.n_levels
        exhausted = False
        for delta in sorted(deltas, reverse=True):
            # Sample complexity is monotone in delta, so once a target cannot be
            # met within n_max no stricter one can be either. The MATLAB still
            # walks those searches to n_max; skipping them changes no reported
            # value and saves the bulk of the runtime, because a design sitting
            # below the target accuracy (SEEDA and UCB plateau near 0.51) would
            # otherwise re-walk the whole horizon grid on every round.
            # The value a walk exits on when it never clears the target: the
            # first n > n_max reached by stepping from warm_start. Not simply
            # n_max + step, which only coincides when warm_start is congruent
            # to n_max mod step -- and the difference is visible now that
            # ceiling points are plotted. Takes the round's phase offset so the
            # early exit below reports the same value the walk would have
            # exited on.
            def ceiling_at(phase):
                start = warm_start + phase
                return start + step * (((n_max - start) // step) + 1)

            ceiling = ceiling_at(0)

            def error_rate(n):
                errors = sum(
                    _final_recommendation(
                        cfg, algo_idx, dose_levels, n, efficacy_env, a
                    ) != optimal_dose
                    for _ in range(iters)
                )
                return errors / iters

            if exhausted:
                answers = [ceiling] * rounds
            else:
                answers = []
                for r in range(rounds):
                    # PHASE OFFSET. The Plateau family allocates on a fixed
                    # (eta + 1) = 3 round cycle -- exploit the leader, then
                    # explore its neighbors twice -- so recommendation accuracy
                    # carries a real period-3 ripple (measured ~1.4pp
                    # peak-to-peak) that averaging never removes, because every
                    # trial starts the cycle at the same point after the
                    # K-cohort warm-up. Walking n = warm_start, +step, +2*step
                    # with warm_start = K = 6 and step 3 or 6 makes EVERY
                    # candidate congruent mod 3, so the search would always
                    # land on the same beat and report a horizon biased by
                    # whichever beat that is (worth ~11 cohorts on the flat
                    # tail). Rotating the grid by r % step across rounds covers
                    # every residue, so the mean over rounds averages the phase
                    # out instead of freezing it.
                    phase = r % step
                    # Test the full budget FIRST. Sample complexity is monotone
                    # in the horizon, so a target the design cannot reach with
                    # n_max cohorts it cannot reach with fewer -- one evaluation
                    # replaces a walk of up to (n_max - warm_start) / step
                    # candidates. It also stops the walk crediting a lucky early
                    # batch for an unreachable target: at iters = 100 a design
                    # sitting at error 0.5 clears a 0.4 target ~2.8% of the time,
                    # and a walk gets ~47 attempts at it.
                    if error_rate(n_max) > delta:
                        answers.append(ceiling_at(phase))
                        continue
                    n = warm_start + phase
                    while n <= n_max:
                        if error_rate(n) <= delta:
                            break
                        n += step
                    answers.append(n)

            censored = bool(np.mean([x > n_max for x in answers]) > 0)
            exhausted = exhausted or all(x > n_max for x in answers)
            mean_n = float(np.mean(answers))
            records.append({
                "scenario": scen,
                "design": algo,
                "delta": delta,
                "accuracy": round(1 - delta, 10),
                "n_cohorts": mean_n,
                "n_cohorts_std": float(np.std(answers, ddof=1)) if rounds > 1 else 0.0,
                # The individual per-round answers, so a CI can be bootstrapped
                # from them later without re-running this (very slow) search.
                "n_rounds": rounds,
                "answers": ";".join(str(int(x)) for x in answers),
                "n_patients": mean_n * cfg.cohort_size,
                "censored": censored,
            })
            warm_start = max(cfg.n_levels, int(np.floor(mean_n)))
            if progress:
                flag = "  (censored)" if censored else ""
                print(f"  {algo:16} delta={delta:.1f} -> "
                      f"n={mean_n:.1f} cohorts{flag}")
            if checkpoint_path is not None:
                # Rewritten after every completed point. A single point can cost
                # hours at large rounds/iters, so an interrupted run keeps
                # everything it has already measured.
                Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(records).to_csv(checkpoint_path, index=False)
    return pd.DataFrame(records)
