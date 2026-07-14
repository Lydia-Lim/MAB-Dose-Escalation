from typing import Callable, Sequence

import numpy as np

from ._seeda_plateau_original import SEEDAPlateauDoseEscalator


class SEEDAPlateauTwoSidedDecoupledDoseEscalator(SEEDAPlateauDoseEscalator):
    """
    SEEDA-Plateau with a corrected turning-point (L1) recommendation, built on top
    of UCB's Plateau (``SEEDAPlateauDoseEscalator`` — the entire-upper-tail-flat
    variant). Only the non-training *recommendation* differs from UCB's Plateau; the
    allocation policy is inherited unchanged.

    The two words in the name are the two changes vs. UCB's Plateau, both from the
    L1 investigation (see ``SEEDA_PLATEAU_L1_INVESTIGATION.md``):

    - **"Two-sided" = no Part B.** UCB's Plateau flags an adjacent pair (m, m+1) as
      flat only if ``|q_hat_m - q_hat_{m+1}| <= beta_m + beta_{m+1}`` (Part A,
      two-sided) AND ``q_hat_m <= q_hat_{m+1}`` (Part B). Part B is a coin-flip on
      the truly-equal plateau pairs, so requiring it on *every* tail pair makes L1
      almost never fire and the recommendation collapses onto the toxicity MTD (L2).
      This variant keeps only Part A (the two-sided test). Dropping Part B is
      consistent with the paper's Theorem 4 proof, which bounds the error using
      Part A alone.
    - **"Decoupled" = separate L1 coefficient.** The plateau test uses its own
      ``l1_coefficient`` (default 0.1) instead of the allocation UCB coefficient
      ``ucb_coefficient``. The paper reuses the symbol ``c`` for both but treats the
      allocation index as swappable (Eq. 4, "can be replaced by ... KL-UCB"), and
      the two want opposite magnitudes: exploration wants a large coefficient, the
      plateau test a small one so the true efficacy gaps are resolved instead of
      being swamped by the confidence width. With the large allocation ``c`` the L1
      test collapses onto dose 1.

    Exploration uses the inherited (log) bonus ``F = q_hat + sqrt(c*log(n)/N)``
    (paper Eq. 4). See ``SEEDAPlateauTwoSidedDecoupledNoLogDoseEscalator`` for the
    no-log exploration variant.
    """

    def __init__(
        self,
        dose_levels: Sequence[float],
        target_toxicity_level: float,
        dose_toxicity_curve: Callable[[Sequence[float], float],
                                      Sequence[float]],
        p_hat: Sequence[float],
        q_hat: Sequence[float],
        ucb_coefficient: float = 1,
        gamma_1: float = 3/2,
        delta_1: float = 0.05,
        eta: int = 2,
        l1_coefficient: float = 0.1,
        is_training: bool = True,
        seed: float = 0,
        no_skip: bool = True,
    ):
        super().__init__(
            dose_levels,
            target_toxicity_level,
            dose_toxicity_curve,
            p_hat,
            q_hat,
            ucb_coefficient,
            gamma_1,
            delta_1,
            eta,
            is_training,
            seed,
            no_skip,
        )
        # Confidence coefficient used ONLY in the L1 plateau test, decoupled from
        # the allocation UCB coefficient (self._c):
        self._l1_coefficient = l1_coefficient

    def propose(self) -> int:
        # Allocation (training) is inherited unchanged from UCB's Plateau:
        if self._is_training:
            return super().propose()

        # Recommendation: entire-upper-tail-flat L1, but two-sided only (no Part B)
        # and using the decoupled L1 coefficient:
        a_hat, admissible_set, F, _ = self._calc_model_params()

        def conf_width(m: int) -> float:
            return np.sqrt(
                self._l1_coefficient * np.log(np.sum(self._N)) / self._N[m]
            )

        def is_flat(m: int) -> bool:
            # Part A only: the two estimates are within noise of each other:
            # (Part B, q_hat_m <= q_hat_{m+1}, deliberately dropped.)
            return (
                np.abs(self._q_hat[m] - self._q_hat[m + 1])
                <= conf_width(m) + conf_width(m + 1)
            )

        # L1: lowest admissible dose from which the entire upper tail is flat:
        L_1 = len(self._dose_levels)
        for idx, admissible in enumerate(admissible_set):
            if admissible and self._validator.validate(idx) and all(
                is_flat(m) for m in range(idx, self._K - 1)
            ):
                L_1 = idx
                break

        # L2: greedy toxicity MTD (unchanged from UCB's Plateau):
        L_2 = 0
        p_dle = self._dose_toxicity_curve(self._dose_levels, a_hat)
        for idx in range(len(p_dle)):
            if (self._validator.validate(idx) and
                    p_dle[idx] <= self._ttl and
                    p_dle[idx] >= p_dle[L_2]):
                L_2 = idx

        return min(L_1, L_2)
