from typing import Callable, Sequence

import numpy as np

from ._seeda import SEEDADoseEscalator


class SEEDAPlateauFixedDoseEscalator(SEEDADoseEscalator):
    """
    SEEDA-Plateau with the two places where the authors' MATLAB departs from the
    paper's printed Algorithm 2 put back the way the paper specifies. A
    **standalone copy** of ``SEEDAPlateauTwoSidedDecoupledDoseEscalator``
    (deliberately not a subclass, so the two can be compared and neither drifts
    when the other is edited) with exactly two changes:

    **Fix 1 -- allocation: the leader is restricted to the admissible set.**
    Algorithm 2 step 4 reads ``L(t) = argmax_{d_k in D_1(t)} q_hat_k``, i.e. the
    arg-max runs over the admissible set only. The MATLAB takes a *global*
    arg-max (``[~,L] = max(q_hat)``), which lets a non-admissible dose whose
    ``q_hat`` is frozen at a 3-patient warm-up estimate stay leader
    indefinitely -- measured at 73.75% of post-warm-up rounds. Since exploration
    is confined to ``{L-1, L, L+1}``, that pins allocation on the top admissible
    dose regardless of efficacy. Restricting the leader moved allocation on
    doses 3/4 from 24.07/66.44 to 43.84/45.53 in the paper's main scenario.

    **Fix 2 -- recommendation: L1 descends from the MTD and breaks.** The
    MATLAB's ascending overwrite keeps the *largest* start index whose whole
    tail is flat, which (because "all pairs flat from i to MTD" is monotone in
    i) collapses to just ``MTD-1 if the top pair is flat, else MTD`` -- it can
    never return anything lower, so it structurally cannot find a plateau onset
    below MTD-1. Here the scan instead walks *down* from the MTD and stops at
    the first non-flat pair, returning the lowest dose of the contiguous flat
    run ending at the MTD. That is the paper's plateau onset, restricted to
    contiguity (the paper's literal ``min{m}`` allows an isolated flat pair
    anywhere below, which measured ~86-95% collapse to dose 1).

    Everything else is unchanged from Reference, and still follows the
    MATLAB: Part A only in the flat test (no ``q_hat_m <= q_hat_{m+1}``), a
    decoupled ``l1_coefficient``, an admissible set built on ``a_hat`` alone
    (no alpha margin), ``L1 = K`` during warm-up, and ``d_hat = min(L1, L2)``.

    ⚠ **Expected trade-off, not a strict upgrade.** Fix 1 is a clear win. Fix 2
    unlocks k* below MTD-1 (where Reference scores ~2%) but exposes the
    pairwise chain's accumulation blind spot: K-1 gaps each individually inside
    the confidence width sum to a large gap the chain never sees, so it can walk
    past k*. Predicted from the measured allocations at n=100 cohorts, this
    leaks to dose 1 on the paper's main scenario and never resolves the
    0.05-step "small efficacy gaps" scenario at any horizon. Whether that
    prediction holds is exactly what running this design in the battery tests.
    """

    def __init__(
        self,
        dose_levels: Sequence[float],
        target_toxicity_level: float,
        dose_toxicity_curve: Callable[[Sequence[float], float],
                                      Sequence[float]],
        ucb_coefficient: float = 1,
        gamma_1: float = 3/2,
        delta_1: float = 0.05,
        eta: int = 2,
        l1_coefficient: float = 0.16,
        is_training: bool = True,
        seed: float = 0,
        no_skip: bool = True,
        a_init=None,
        a_max=None,
    ):
        # No p_hat / q_hat: like the MATLAB, this design uses no prior and samples
        # each dose once (handled by the base when the priors are omitted).
        super().__init__(
            dose_levels,
            target_toxicity_level,
            dose_toxicity_curve,
            ucb_coefficient=ucb_coefficient,
            gamma_1=gamma_1,
            delta_1=delta_1,
            is_training=is_training,
            seed=seed,
            no_skip=no_skip,
            a_init=a_init,
            a_max=a_max,
        )
        self._eta = eta
        self._update_counts = 0
        self._l = np.array([0] * self._K)
        # Confidence coefficient used ONLY in the L1 plateau test, decoupled from
        # the allocation UCB coefficient (self._c):
        self._l1_coefficient = l1_coefficient

    def _calc_model_params(self):
        # MATLAB's Safe_UCB_Plateau.m builds the admissible set WITHOUT the alpha
        # confidence margin -- D = find(toxicity(d, a_hat) <= thre) -- unlike
        # SEEDA which uses a_hat + alpha. Recompute the admissible set on a_hat
        # alone (reusing the most-sampled a_hat and the F bonus from the base).
        a_hat, _, F, _ = super()._calc_model_params()
        admissible_set = self._dose_toxicity_curve(
            self._dose_levels, a_hat
        ) <= self._ttl
        F_filtered = F[admissible_set]
        return a_hat, admissible_set, F, F_filtered

    def flat_threshold(self, m: int) -> float:
        """The L1 tolerance for the pair (m, m + 1), in efficacy-probability units.

        Identical in form and coefficient to Reference's -- this variant
        changes the SCAN, not the test. Public so the diagnosis figures can set
        it against the true adjacent efficacy gaps.
        """
        def conf_width(k: int) -> float:
            return np.sqrt(
                self._l1_coefficient * np.log(np.sum(self._N)) / self._N[k]
            )

        return float(conf_width(m) + conf_width(m + 1))

    def propose(self) -> int:
        # Warm-up: the ALLOCATION samples each dose once (round-robin). The
        # RECOMMENDATION is still produced every round -- the MATLAB does the same,
        # its k_rec1 accumulator sitting outside the warm-up branch -- so we don't
        # short-circuit it to the round-robin dose. What the MATLAB does NOT do
        # during warm-up is run the flat test: it leaves the admissible set D empty
        # until every dose has been sampled, so L1 = K and the recommendation
        # reduces to the toxicity MTD (L2). We match that in the L1 block below.
        # Before the first cohort (sum(N) = 0) there is no data -> fall back to 0.
        if self._is_training and self._init_idx < self._K:
            return self._init_idx
        if not self._is_training and np.sum(self._N) == 0:
            return 0

        a_hat, admissible_set, F, F_filtered = self._calc_model_params()

        # === FIX 1 =========================================================
        # Leader L(t) = argmax q_hat over the ADMISSIBLE SET (paper Algorithm 2
        # step 4), not over all doses as the MATLAB does. Ties resolve to the
        # lowest index, i.e. the first dose on the efficacy plateau. If nothing
        # is admissible yet, fall back to the lowest dose.
        admissible_where = np.flatnonzero(admissible_set)
        if len(admissible_where) == 0:
            L = 0
        else:
            L = int(admissible_where[np.argmax(self._q_hat[admissible_where])])
        # ===================================================================
        if np.sum(self._l) < self._update_counts:
            self._l[L] += 1

        if self._is_training:
            if len(F_filtered) == 0:
                self._I = 0
                return self._I
            # f = efficacy UCB bonus, zeroed on non-admissible doses (MATLAB f).
            f = np.where(admissible_set, F, 0.0)
            # Exploit the leader every (eta+1) visits; otherwise explore ONLY its
            # neighborhood {L-1, L, L+1} (MATLAB Safe_UCB_Plateau.m), picking the
            # largest f; if that whole window is 0, fall back to the global argmax.
            if (self._l[L] - 1) % (self._eta + 1) == 0 and f[L] != 0:
                self._I = L
            else:
                window = list(range(max(L - 1, 0), min(L + 1, self._K - 1) + 1))
                if sum(f[j] for j in window) == 0:
                    self._I = int(np.argmax(f))
                else:
                    self._I = max(window, key=lambda j: f[j])
            return self._I

        # Recommendation: min(L1, L2) with the two-sided, decoupled L1.
        def is_flat(m: int) -> bool:
            # Part A only (two-sided): |q_m - q_{m+1}| within the combined width.
            # (Part B, q_hat_m <= q_hat_{m+1}, deliberately dropped, as in the
            # MATLAB and consistent with the paper's Theorem 4 proof.)
            return (
                np.abs(self._q_hat[m] - self._q_hat[m + 1])
                <= self.flat_threshold(m)
            )

        admissible_idx = [
            idx for idx in range(self._K)
            if admissible_set[idx] and self._validator.validate(idx)
        ]
        if self._init_idx < self._K:
            # Warm-up: the MATLAB initializes D = [] and only reassigns it once
            # t > K, so its flat-test loop never runs and Ri = K. Match that.
            # Evaluating the test here would divide by N_m = 0 on the not-yet-
            # sampled doses and make every pair trivially "flat".
            L_1 = self._K
        elif admissible_idx:
            # === FIX 2 =====================================================
            # Walk DOWN from the MTD and BREAK at the first non-flat pair, so
            # L1 is the lowest dose of the CONTIGUOUS flat run ending at the
            # MTD -- the plateau onset. The MATLAB's ascending overwrite keeps
            # the largest qualifying start index instead, which collapses to
            # "MTD-1 if the top pair is flat, else MTD" and can never reach
            # lower. Contiguity (the break) is what keeps an isolated noisy low
            # dose unreachable; the paper's literal min{m} has no such guard.
            mtd = admissible_idx[-1]
            L_1 = mtd
            for m in range(mtd - 1, admissible_idx[0] - 1, -1):
                if not is_flat(m):
                    break
                L_1 = m
            # ===============================================================
        else:
            L_1 = self._K

        # L2: greedy toxicity MTD (highest safe dose by model toxicity).
        L_2 = 0
        p_dle = self._dose_toxicity_curve(self._dose_levels, a_hat)
        for idx in range(len(p_dle)):
            if (self._validator.validate(idx) and
                    p_dle[idx] <= self._ttl and
                    p_dle[idx] >= p_dle[L_2]):
                L_2 = idx

        # Observation only, for the diagnosis figures: which of the two terms
        # min(L1, L2) actually binds. Nothing downstream of propose reads these.
        self._last_l1, self._last_l2 = int(L_1), int(L_2)
        return min(L_1, L_2)
