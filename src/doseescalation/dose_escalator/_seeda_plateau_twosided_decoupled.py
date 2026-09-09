from typing import Callable, Sequence

import numpy as np

from ._seeda import SEEDADoseEscalator


class SEEDAPlateauTwoSidedDecoupledDoseEscalator(SEEDADoseEscalator):
    """
    SEEDA-Plateau (self-contained) with a corrected turning-point (L1)
    recommendation. This is the variant that most closely reproduces the paper's
    SEEDA-Plateau results, so it is kept as a standalone base rather than layered
    on the other Plateau scripts: it inherits only the SEEDA admissible-set /
    a-estimation machinery from ``SEEDADoseEscalator`` and implements the
    SEEDA-Plateau allocation (leader + eta-neighborhood exploration) and the
    recommendation here.

    Allocation (training): follow OSUB-style selection — compute the leader
    ``L(t) = argmax_{admissible} q_hat``, and exploit it when
    ``(l_L - 1) mod (eta + 1) == 0``, otherwise explore via the efficacy UCB
    bonus ``F = q_hat + sqrt(c*log(n)/N)`` (paper Eq. 4).

    Recommendation: ``d_hat(n) = min(L1, L2)`` with L2 the toxicity MTD. The L1 rule
    reproduces the authors' MATLAB (``Safe_UCB_Plateau.m``), which itself deviates
    from the paper's printed Algorithm 2 step 12 in three ways:

    - **Direction = MTD-1, not the plateau onset.** The paper takes ``L1 = min{m}``
      (the *lowest* flat pair). The MATLAB tests only adjacent *admissible* pairs up
      to the MTD and, via an ascending overwrite, keeps the *largest* start index
      whose whole tail is flat -- i.e. ``MTD-1`` if the top admissible pair is flat,
      else ``MTD``. This matches the paper's *numbers* but not its equations; it
      returns k* here only because the plateau onset equals MTD-1 in this scenario.
    - **"Two-sided" = no Part B.** The paper flags an adjacent pair (m, m+1) as flat
      only if ``|q_hat_m - q_hat_{m+1}| <= beta_m + beta_{m+1}`` (Part A, two-sided)
      AND ``q_hat_m <= q_hat_{m+1}`` (Part B). The MATLAB keeps only Part A. Dropping
      Part B is consistent with the paper's Theorem 4 proof, which bounds the error
      using Part A alone.
    - **"Decoupled" = separate L1 coefficient.** The plateau test uses its own
      ``l1_coefficient`` (default 0.16, i.e. MATLAB's ``0.4 * sqrt(log n / N)``)
      instead of the allocation UCB coefficient ``ucb_coefficient``. The two want
      opposite magnitudes: exploration wants a large coefficient, the plateau test a
      small one so the true efficacy gaps are resolved instead of being swamped by
      the confidence width. With the large allocation ``c`` the L1 test collapses.

    See ``SEEDAPlateauTwoSidedDecoupledNoLogDoseEscalator`` for the no-log
    exploration variant.
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
        # SEEDA-Plateau allocation state (previously inherited from the UCB Plateau):
        self._eta = eta
        self._update_counts = 0
        self._l = np.array([0] * self._K)
        # Confidence coefficient used ONLY in the L1 plateau test, decoupled from
        # the allocation UCB coefficient (self._c):
        self._l1_coefficient = l1_coefficient

    def _calc_model_params(self):
        # #6: MATLAB's Safe_UCB_Plateau.m builds the admissible set WITHOUT the
        # alpha confidence margin -- D = find(toxicity(d, a_hat) <= thre) -- unlike
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

        Public so the diagnosis figures can set it against the true adjacent
        efficacy gaps. ``l1_coefficient`` sits INSIDE the square root, so the
        battery's 0.16 is the MATLAB's commented-out 0.4 per term.
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

        # Leader L(t) = argmax q_hat over ALL doses (MATLAB [~,L]=max(q_hat)); ties
        # resolve to the lowest index, i.e. the first dose on the efficacy plateau.
        L = int(np.argmax(self._q_hat))
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
            # (Part B, q_hat_m <= q_hat_{m+1}, deliberately dropped.)
            return (
                np.abs(self._q_hat[m] - self._q_hat[m + 1])
                <= self.flat_threshold(m)
            )

        # L1: faithful reproduction of the authors' MATLAB (Safe_UCB_Plateau.m).
        # NOTE this DEVIATES from the paper's printed Algorithm 2 step 12, which
        # takes the *lowest* flat-and-non-decreasing pair (the plateau onset). The
        # MATLAB instead: (a) tests only adjacent *admissible* pairs up to the MTD,
        # Part A only; (b) via an ascending overwrite keeps the *largest* start
        # index whose entire tail (to the MTD) is flat -- which reduces to
        # "MTD-1 if the top admissible pair is flat, else MTD". It matches the
        # paper's numbers, not its equations.
        admissible_idx = [
            idx for idx in range(self._K)
            if admissible_set[idx] and self._validator.validate(idx)
        ]
        if self._init_idx < self._K:
            # Warm-up: the MATLAB initializes D = [] and only reassigns it once
            # t > K, so its flat-test loop never runs and Ri = K. Match that.
            # Evaluating the test here would divide by N_m = 0 on the not-yet-
            # sampled doses, make every pair trivially "flat", and drag L1 down to
            # MTD-1 on rounds the MATLAB scores as the MTD -- which biases the
            # trial-averaged Table 2 recommendation toward the lower dose.
            L_1 = self._K
        elif admissible_idx:
            mtd = admissible_idx[-1]  # highest admissible dose
            temp_flat = {k: is_flat(k) for k in range(admissible_idx[0], mtd)}
            L_1 = mtd
            for i in range(admissible_idx[0], mtd):  # ascending -> largest wins
                if all(temp_flat[k] for k in range(i, mtd)):
                    L_1 = i
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

    def update(self,
               dose_level_index: int,
               cohort_size: int,
               n_dle: int,
               n_efficate: int):
        # Count only leader-eligible cohorts (i.e. after the initial sampling
        # phase), matching MATLAB's l(L) increment which runs only outside the
        # sampling round-robin.
        if self._init_idx >= self._K:
            self._update_counts += 1
        super().update(dose_level_index, cohort_size, n_dle, n_efficate)
