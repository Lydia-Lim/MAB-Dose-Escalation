from typing import Callable, Sequence

import numpy as np

from doseescalation.dose_escalator._base import DoseEscalatorBase
from doseescalation.dose_escalator._validate import (
    NoOpValidator,
    NoSkipValidator,
    Validator,
)


class SEEDAPlateauNaiveDoseEscalator(DoseEscalatorBase):
    """
    SEEDA-Plateau exactly as printed in Algorithm 2 of Shen et al. (2020).

    Standalone by design: this class inlines everything it needs from SEEDA's
    Algorithm 1 rather than subclassing the package's ``SEEDADoseEscalator``, so
    that it stays pinned to the paper while the package's SEEDA continues to
    evolve. Its counterpart in this directory,
    ``SEEDAPlateauMatlabDoseEscalator``, is standalone for the same reason.

    What "as printed" means here:

    - **Zero initialization, then round-robin.** ``N_k = 0``, ``q_hat_k = 0``,
      ``p_hat_k = 0``, and each dose is sampled once before any model-based
      selection. The paper supplies no efficacy prior to Algorithm 2.
    - **Admissible set with the alpha margin** (Equation 3):
      ``D_1(t) = {k : p_k(a_hat + alpha_t) <= theta}``.
    - **Leader and neighborhood** (steps 4-5): ``L(t) = argmax`` of ``q_hat``
      *over the admissible set*, exploited every ``eta + 1`` visits, otherwise the
      UCB index is maximized over ``{L-1, L, L+1}``.
    - **The turning point ``L1``** (step 12) scanned as written: the *lowest*
      admissible dose whose adjacent pair is statistically flat. This is the
      bottom-up first-flat-pair reading, and is what makes the printed rule
      behave very differently from the authors' MATLAB, whose ascending-overwrite
      loop collapses to "MTD-1 if the top admissible pair is flat, else MTD".
    - **Recommendation** ``min(L1, L2)`` with ``L2`` the model-toxicity MTD.

    ``use_log_bonus`` selects the exploration bonus. The default, ``True``, is the
    paper's Equation (4), ``F = q_hat + sqrt(c * log(t) / N)``. Setting it to
    ``False`` recovers ``F = q_hat + sqrt(c * sum(N) / N)``, the no-log bonus the
    earlier archived implementation inherited from our original SEEDA class; it
    explores under-sampled doses far more aggressively.
    """

    def __init__(
        self,
        dose_levels: Sequence[float],
        target_toxicity_level: float,
        dose_toxicity_curve: Callable[[Sequence[float], float],
                                      Sequence[float]],
        ucb_coefficient: float = 1.0,
        gamma_1: float = 3 / 2,
        delta_1: float = 0.05,
        eta: int = 2,
        a0: float = 20.0,
        use_log_bonus: bool = True,
        is_training: bool = True,
        seed: float = 0,
        no_skip: bool = True,
    ):
        self._dose_levels = np.asarray(dose_levels, dtype=float)
        self._K = len(self._dose_levels)
        self._ttl = target_toxicity_level
        self._dose_toxicity_curve = dose_toxicity_curve

        self._c = ucb_coefficient
        self._gamma_1 = gamma_1
        self._delta_1 = delta_1
        self._eta = eta
        self._use_log_bonus = use_log_bonus
        self._is_training = is_training

        # C_1 as computed in the authors' MATLAB; the paper leaves it unspecified.
        self._C_1 = ((np.min(np.abs(
            -np.log((np.tanh(self._dose_levels) + 1) / 2)
        ))) ** (-1 / self._gamma_1)) / 30

        # Algorithm 2's Initialize block does not say what a_hat_k starts at:
        # step 8 only sets it after a dose has been observed. We start every
        # dose at a0, the same value the dose grid is back-solved at, so the
        # model reproduces the nominal toxicities before any data arrives. The
        # paper imposes no clamp, so a_hat is left free to overshoot.
        self._a0 = float(a0)
        self._rng = np.random.RandomState(seed)
        self._a_hat_doses = np.full(self._K, self._a0, dtype=float)

        # Algorithm 2 initialization.
        self._q_hat = np.zeros(self._K)
        self._p_hat = np.zeros(self._K)
        self._N = np.zeros(self._K, dtype=int)
        self._l = np.zeros(self._K, dtype=int)
        self._I = self._K - 1
        self._init_idx = 0  # next dose in the initial round-robin

        self._validator: Validator = (
            NoSkipValidator(self._K) if no_skip else NoOpValidator()
        )

    def train(self, is_training: bool):
        self._is_training = is_training

    def _calc_model_params(self):
        total = int(np.sum(self._N))
        w = self._N / total
        a_hat = float(np.dot(w, self._a_hat_doses))

        # Paper eq. (3): the exponent is gamma_bar_1 / 2, and supplementary
        # Proposition 1 gives gamma_bar_1 = 1 / gamma_1. Supplement section B
        # suggests gamma_1 = 3/2, so the exponent is 1 / (2 * gamma_1) = 1/3.
        # ``gamma_1`` here is the paper's gamma_1, NOT gamma_bar_1.
        alpha_t = self._C_1 * self._K * (
            np.log(2 * self._K / self._delta_1) / (2 * total)
        ) ** (1.0 / (self._gamma_1 * 2.0))

        admissible_set = self._dose_toxicity_curve(
            self._dose_levels, a_hat + alpha_t
        ) <= self._ttl

        if self._use_log_bonus:
            bonus = np.sqrt(self._c * np.log(total) / self._N)
        else:
            bonus = np.sqrt(self._c * total / self._N)
        return a_hat, admissible_set, self._q_hat + bonus

    def safe_doses(self):
        """
        Per-dose mask of the doses currently declared safe. No dose is reported
        safe during the initial round-robin, when ``sum(N)`` is still zero.
        """
        if self._init_idx < self._K:
            return [False] * self._K
        _, admissible_set, _ = self._calc_model_params()
        return [bool(x) for x in admissible_set]

    def _leader(self, admissible_set):
        """Step 4: the admissible, validatable dose with the highest q_hat."""
        leader = None
        for idx, admissible in enumerate(admissible_set):
            if admissible and self._validator.validate(idx):
                if leader is None or self._q_hat[idx] > self._q_hat[leader]:
                    leader = idx
        return leader

    def propose(self) -> int:
        # Sample each dose once, in ascending order, before model-based selection.
        if self._init_idx < self._K:
            return self._init_idx

        a_hat, admissible_set, F = self._calc_model_params()

        if self._is_training:
            leader = self._leader(admissible_set)
            if leader is None:
                self._I = 0
                return self._I

            self._l[leader] += 1

            # Step 5: exploit the leader every eta + 1 visits, else explore.
            if (self._l[leader] - 1) % (self._eta + 1) == 0:
                self._I = leader
            else:
                best = None
                for idx in (leader - 1, leader, leader + 1):
                    if (0 <= idx < self._K
                            and admissible_set[idx]
                            and self._validator.validate(idx)):
                        if best is None or F[idx] > F[best]:
                            best = idx
                self._I = best if best is not None else leader
            return self._I

        # Step 12, as printed.
        def is_plateau_pair(m: int) -> bool:
            gap = abs(self._q_hat[m] - self._q_hat[m + 1])
            return (gap <= self.flat_threshold(m)
                    and self._q_hat[m] <= self._q_hat[m + 1])

        # L1: the lowest admissible dose at which the plateau begins.
        L_1 = self._K
        for k in (idx for idx in range(self._K) if admissible_set[idx]):
            for m in range(k, self._K - 1):
                if is_plateau_pair(m):
                    L_1 = min(L_1, m)
                    break

        # L2: the highest safe dose under the fitted toxicity model.
        p_dle = self._dose_toxicity_curve(self._dose_levels, a_hat)
        L_2 = 0
        for idx in range(self._K):
            if (self._validator.validate(idx)
                    and p_dle[idx] <= self._ttl
                    and p_dle[idx] >= p_dle[L_2]):
                L_2 = idx

        # Observation only, for the diagnosis figures: which of the two terms
        # min(L1, L2) actually binds. Nothing downstream of propose reads these.
        self._last_l1, self._last_l2 = int(L_1), int(L_2)
        return int(min(L_1, L_2))

    def flat_threshold(self, m: int) -> float:
        """Step 12's tolerance for the pair (m, m + 1): the sum of the two
        per-dose confidence widths, in the units of an efficacy probability.

        Public so the diagnosis figures can set it against the true adjacent
        efficacy gaps, which is what decides whether a real rise reads as flat.
        ``N`` counts PATIENTS and the log is over the patients seen so far, both
        as printed -- the MATLAB counts cohorts and logs the fixed horizon.
        """
        total = int(np.sum(self._N))
        return float(
            np.sqrt(self._c * np.log(total) / self._N[m])
            + np.sqrt(self._c * np.log(total) / self._N[m + 1])
        )

    def update(self, dose_level_index: int, cohort_size: int,
               n_dle: int, n_efficate: int):
        if not self._is_training:
            raise RuntimeError("Cannot update in non-training mode")

        i = dose_level_index
        self._validator.visit(i)
        n = self._N[i]
        self._q_hat[i] = (self._q_hat[i] * n + n_efficate) / (n + cohort_size)
        self._p_hat[i] = (self._p_hat[i] * n + n_dle) / (n + cohort_size)
        self._N[i] = n + cohort_size

        # Invert p_k(a) = base^a in closed form: a = log(p_hat) / log(base).
        base = self._dose_toxicity_curve(
            dose_levels=self._dose_levels[i], a_hat=1
        )
        target = float(np.clip(self._p_hat[i], 1e-6, 1 - 1e-6))
        self._a_hat_doses[i] = np.log(target) / np.log(base)

        if self._init_idx < self._K:
            self._init_idx += 1
