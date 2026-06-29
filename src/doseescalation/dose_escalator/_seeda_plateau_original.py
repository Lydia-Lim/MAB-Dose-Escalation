from typing import Callable, Sequence

import numpy as np

from ._seeda import SEEDADoseEscalator


class SEEDAPlateauDoseEscalator(SEEDADoseEscalator):
    """
    SEEDA-Plateau dose escalator.
    This class uses the SEEDA-Plateau Method proposed in
    "Learning for Dose Allocation in Adaptive Clinical
    Trials with Safety Constraints", which is more apt at
    dealing with pleateauing efficacy as the drug dosage
    levels increase.
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
        eta: float = 2,
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
            is_training,
            seed,
            no_skip,
        )

        # Check intuition about eta parameter
        self._eta = eta
        self._update_counts = 0
        self._l = np.array([0] * self._K)

    def propose(self) -> int:
        a_hat, admissible_set, F, F_filtered = self._calc_model_params()

        L = 0
        for idx, admissible in enumerate(admissible_set):
            if (admissible and
                    self._validator.validate(idx) and
                    self._q_hat[idx] >= self._q_hat[L]):
                L = idx

        if np.sum(self._l) < self._update_counts:
            self._l[L] += 1

        if self._is_training:
            if len(F_filtered) == 0:
                self._I = 0
            else:
                if (self._l[L] - 1) % self._eta == 0:
                    self._I = L
                else:
                    for idx, admissible in enumerate(admissible_set):
                        if (admissible and
                                self._validator.validate(idx) and
                                F[idx] >= F[self._I]):
                            self._I = idx
            return self._I
        else:
            def log_inv_frac(m: int) -> float:
                return np.sqrt(
                    self._c * np.log(np.sum(self._N) / self._N[m])
                )

            def is_diff_valid(m: int) -> bool:
                diff = np.abs(self._q_hat[m] - self._q_hat[m + 1])
                return (
                    diff <= log_inv_frac(m) + log_inv_frac(m + 1)
                    and self._q_hat[m] <= self._q_hat[m + 1]
                )

            # For each arm in the admissible set, ensure all higher doses
            # satisfy the plateau condition.
            L_1 = len(self._dose_levels)
            for idx, admissible in enumerate(admissible_set):
                if admissible and self._validator.validate(idx) and all(
                    is_diff_valid(m) for m in range(idx, self._K - 1)
                ):
                    L_1 = idx
                    break

            # L2 condition is simply the greedy toxicity policy
            L_2 = 0
            p_dle = self._dose_toxicity_curve(self._dose_levels, a_hat)
            for idx in range(len(p_dle)):
                if (self._validator.validate(idx) and
                        p_dle[idx] <= self._ttl and
                        p_dle[idx] >= p_dle[L_2]):
                    L_2 = idx

            return min(L_1, L_2)

    def update(self,
               dose_level_index: int,
               cohort_size: int,
               n_dle: int,
               n_efficate: int):
        """
        Update the escalator with the environment feedback
        """
        self._update_counts += 1
        super().update(dose_level_index, cohort_size, n_dle, n_efficate)