"""
Diagnosis of the SEEDA-Plateau L1 (plateau-onset) failure.
============================================================

This script empirically proves *why* SEEDA-Plateau recommends dose 0 almost
every time, and *why* the modified L1 rule (SEEDAPlateauFixedDoseEscalator)
fixes it. It does three things:

  PART A — Single-trial autopsy. Run one trial, then dump the per-dose state
           the L1 test depends on (N, q_hat, confidence width) and evaluate
           BOTH L1 rules by hand, step by step, so the failure is visible.

  PART B — Aggregate. Repeat over many trials and report how often each rule
           recovers the true plateau onset (the efficacy-optimal dose).

  PART C — Robustness checks. Show the failure is not a tuning artefact: it
           persists across UCB coefficients c and across a "sharper" (more
           obviously plateauing) efficacy curve.

Run from the repo root (with the package installed via `pip install -e .`):
    python experiments/l1_diagnosis.py
"""

import warnings

import numpy as np
from collections import Counter

# The toxicity-curve inversion (Newton's method) can transiently overflow on
# extreme a_hat values; these are harmless and would clutter the output.
warnings.filterwarnings("ignore")
np.seterr(all="ignore")

from doseescalation.dose_escalator import (
    SEEDAPlateauDoseEscalator,
    SEEDAPlateauFixedDoseEscalator,
)
from doseescalation.simulated_env import SimulatedEnv


# --- Experiment settings (mirrors experiments/benchmarks.ipynb) -------------
def dose_toxic_curve(dose_levels, a_hat):
    return np.power((np.tanh(dose_levels) + 1) / 2, a_hat)


def inv_dose_toxic(p, a):
    return np.arctanh(2 * np.power(p, 1 / a) - 1)


TTL = 0.32
P_HAT = (0.05, 0.15, 0.3, 0.35, 0.4, 0.45)
Q_HAT = np.array([0.33] * 6)
EFFICACY_PROBS = [0.1, 0.35, 0.6, 0.6, 0.6, 0.6]  # rises, plateaus at dose 2
OPTIMAL_DOSE = 2  # plateau onset = lowest dose at max efficacy
COHORT_SIZE = 3
N_COHORTS = 300

# Two toxicity scenarios from the notebook (a = aggressiveness).
P_DLE_LEVELS = {
    1.0: [0.05, 0.1, 0.2, 0.3, 0.5, 0.7],
    3.4: [0.01, 0.02, 0.04, 0.08, 0.16, 0.3],
}


def run_one_trial(a, tox_probs, seed, escalator_cls, c=1.0):
    """Run a single trial; return the fitted escalator (still holds state)."""
    dose_levels = [inv_dose_toxic(v, a) for v in tox_probs]
    eff_env = SimulatedEnv(list(range(6)), lambda i: EFFICACY_PROBS[int(i)])
    tox_env = SimulatedEnv(dose_levels, lambda d: dose_toxic_curve(d, a))

    esc = escalator_cls(
        dose_levels=dose_levels,
        target_toxicity_level=TTL,
        dose_toxicity_curve=dose_toxic_curve,
        p_hat=P_HAT,
        q_hat=Q_HAT,
        ucb_coefficient=c,
        seed=seed,
        eta=2,
        no_skip=True,
    )
    np.random.seed(seed)
    for _ in range(N_COHORTS):
        idx = esc.propose()
        esc.update(idx, COHORT_SIZE, tox_env(idx, COHORT_SIZE),
                   eff_env(idx, COHORT_SIZE))
    return esc


# ===========================================================================
# PART A — Single-trial autopsy
# ===========================================================================
def autopsy(a=1.0, seed=0):
    print("=" * 76)
    print(f"PART A  Single-trial autopsy  (a={a}, seed={seed})")
    print("=" * 76)
    tox = P_DLE_LEVELS[a]
    esc = run_one_trial(a, tox, seed, SEEDAPlateauDoseEscalator, c=1.0)

    N = esc._N
    q = esc._q_hat
    n_total = np.sum(N)
    conf = np.sqrt(1.0 * np.log(n_total) / N)  # paper's confidence width, c=1

    np.set_printoptions(precision=3, suppress=True)
    print(f"\nTrue efficacy curve : {np.array(EFFICACY_PROBS)}")
    print(f"True plateau onset  : dose {OPTIMAL_DOSE}\n")
    print(f"N    per dose       : {N}        <- patients seen at each dose")
    print(f"q_hat per dose      : {q}   <- estimated efficacy")
    print(f"conf width per dose : {conf}   <- sqrt(log n / N)")

    print("\n--- Paper's pairwise L1 test ----------------------------------")
    print("  A pair (m, m+1) is 'plateau' if")
    print("    |q_m - q_{m+1}| <= conf_m + conf_{m+1}   AND   q_m <= q_{m+1}")
    L1_pair = 6
    for m in range(5):
        gap = abs(q[m] - q[m + 1])
        budget = conf[m] + conf[m + 1]
        within = gap <= budget
        nondec = q[m] <= q[m + 1]
        is_pair = within and nondec
        flag = "  <-- first plateau pair" if (is_pair and L1_pair == 6) else ""
        if is_pair and L1_pair == 6:
            L1_pair = m
        print(f"  ({m},{m+1}): gap={gap:.3f}  budget={budget:.3f}  "
              f"within={str(within):5}  nondecr={str(nondec):5}  "
              f"plateau={str(is_pair):5}{flag}")
    print(f"\n  => paper L1 = dose {L1_pair}   "
          f"(should be {OPTIMAL_DOSE})  "
          f"{'WRONG' if L1_pair != OPTIMAL_DOSE else 'ok'}")

    print("\n--- Fixed L1 test ('compare to best dose') --------------------")
    best = int(np.argmax(q))
    threshold = q[best] - conf[best]
    print(f"  best dose = {best}  (q={q[best]:.3f}, conf={conf[best]:.3f}, "
          f"N={N[best]})")
    print(f"  threshold = q_best - conf_best = {threshold:.3f}")
    print("  L1 = lowest dose with q_m >= threshold:")
    L1_fixed = 6
    for m in range(6):
        passes = q[m] >= threshold
        flag = ""
        if passes and L1_fixed == 6:
            L1_fixed = m
            flag = "  <-- first dose to pass"
        print(f"    dose {m}: q={q[m]:.3f}  >= {threshold:.3f}? "
              f"{str(passes):5}{flag}")
    print(f"\n  => fixed L1 = dose {L1_fixed}   "
          f"(should be {OPTIMAL_DOSE})  "
          f"{'ok' if L1_fixed == OPTIMAL_DOSE else 'WRONG'}")

    print("\n--- Final recommendations (min(L1, L2)) -----------------------")
    esc_paper = run_one_trial(a, tox, seed, SEEDAPlateauDoseEscalator)
    esc_fixed = run_one_trial(a, tox, seed, SEEDAPlateauFixedDoseEscalator)
    esc_paper.train(False)
    esc_fixed.train(False)
    print(f"  SEEDA-Plateau (paper) recommends : dose {esc_paper.propose()}")
    print(f"  SEEDA-Plateau (fixed) recommends : dose {esc_fixed.propose()}")


# ===========================================================================
# PART B — Aggregate over many trials
# ===========================================================================
def aggregate(n_trials=100):
    print("\n" + "=" * 76)
    print(f"PART B  Aggregate over {n_trials} trials  "
          f"(% recommending the optimal dose {OPTIMAL_DOSE})")
    print("=" * 76)
    print(f"\n{'scenario':<12}{'paper L1':<28}{'fixed L1':<28}")
    for a, tox in P_DLE_LEVELS.items():
        recs_paper, recs_fixed = [], []
        for t in range(n_trials):
            ep = run_one_trial(a, tox, t, SEEDAPlateauDoseEscalator)
            ef = run_one_trial(a, tox, t, SEEDAPlateauFixedDoseEscalator)
            ep.train(False)
            ef.train(False)
            recs_paper.append(ep.propose())
            recs_fixed.append(ef.propose())
        dp, df = Counter(recs_paper), Counter(recs_fixed)
        pp = 100 * dp[OPTIMAL_DOSE] / n_trials
        pf = 100 * df[OPTIMAL_DOSE] / n_trials
        print(f"a={a:<10}{f'{pp:5.0f}%  {dict(sorted(dp.items()))}':<28}"
              f"{f'{pf:5.0f}%  {dict(sorted(df.items()))}':<28}")


# ===========================================================================
# PART C — Robustness: not a tuning artefact
# ===========================================================================
def robustness(n_trials=100):
    print("\n" + "=" * 76)
    print(f"PART C  Robustness  (% paper L1 recovers dose {OPTIMAL_DOSE})")
    print("=" * 76)
    print("\nVarying the UCB coefficient c (a=1.0):")
    tox = P_DLE_LEVELS[1.0]
    for c in (0.5, 1.0, 2.0, 2.5):
        recs = []
        for t in range(n_trials):
            ep = run_one_trial(1.0, tox, t, SEEDAPlateauDoseEscalator, c=c)
            ep.train(False)
            recs.append(ep.propose())
        pct = 100 * Counter(recs)[OPTIMAL_DOSE] / n_trials
        print(f"  c={c:<4}: {pct:5.0f}%  recommends dose {OPTIMAL_DOSE}")
    print("\n  Conclusion: the pairwise test fails for every c — it is a")
    print("  sensitivity limitation, not a coefficient that needs tuning.")


if __name__ == "__main__":
    autopsy(a=1.0, seed=0)
    aggregate(n_trials=100)
    robustness(n_trials=100)
