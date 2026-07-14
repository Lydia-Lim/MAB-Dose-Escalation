from typing import Callable, Sequence

import numpy as np

from ._seeda import SEEDADoseEscalator


class SEEDAPlateauNaiveDoseEscalator(SEEDADoseEscalator):
    """
    SEEDA-Plateau dose escalator (naive variant).
    This class uses the SEEDA-Plateau Method proposed in
    "Learning for Dose Allocation in Adaptive Clinical
    Trials with Safety Constraints", which is more apt at
    dealing with pleateauing efficacy as the drug dosage
    levels increase.

    This version is a faithful implementation of Algorithm 2
    described in the paper.
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
        self._l = np.array([0] * self._K)

        # Paper Algorithm 2 initialization: N_k = 0, q_hat_k = 0, p_hat_k = 0,
        # then "sample each dose once" before any leader/UCB selection.
        # The base SEEDADoseEscalator instead seeds N_k = 1 and q_hat_k = q_hat
        # (the increasing efficacy prior), which makes the leader
        # L(t) = argmax_k q_hat_k jump to the top dose from the very
        # first cohort and never revisit the low doses.
        self._q_hat = np.zeros(self._K)
        self._p_hat = np.zeros(self._K)
        self._N = np.array([0] * self._K)
        self._init_idx = 0  # next dose to sample during the initial round-robin

    def _leader(self, admissible_set) -> int:
        """
        Finds L(t): the admissible, validatable dose with the highest estimated efficacy.
        Returns None if no such dose exists.
        """
        leader = None
        for idx, admissible in enumerate(admissible_set):
            if admissible and self._validator.validate(idx):
                if leader is None or self._q_hat[idx] > self._q_hat[leader]:
                    leader = idx
        return leader

    def propose(self) -> int:
        # Initial phase (Algorithm 2): sample each dose once, in ascending
        # order, before any model-based selection. Returned for both the
        # allocation and the interim recommendation so we never call
        # _calc_model_params (which divides by sum(N)) while all N = 0.
        if self._init_idx < self._K:
            return self._init_idx

        a_hat, admissible_set, F, _ = self._calc_model_params()
        if self._is_training:
            leader = self._leader(admissible_set)

            # No admissible / validatable dose:
            if leader is None:
                # Fall back to the lowest dose (matches SEEDA):
                self._I = 0
                return self._I
            
            self._l[leader] += 1

            # Select the leader when (l_L - 1) / (eta + 1) is an integer:
            if (self._l[leader] - 1) % (self._eta + 1) == 0:
                self._I = leader
            # Otherwise explore the neighborhood:
            else:
                neighbors = [leader - 1, leader, leader + 1]
                best_idx = None
                for idx in neighbors:
                    if (0 <= idx < self._K and
                        admissible_set[idx] and
                        self._validator.validate(idx)):
                        if best_idx is None or F[idx] > F[best_idx]:
                            best_idx = idx
                self._I = best_idx if best_idx is not None else leader

            return self._I
        
        else:
 
            def conf_width(m: int) -> float:
                # sqrt(c * log(n) / N_m), per Algorithm 2's L1 test:
                return np.sqrt(
                    self._c * np.log(np.sum(self._N)) / self._N[m]
                )
 
            def is_plateau_pair(m: int) -> bool:
                # Doses m and m+1 are "statistically equal" (plateau)
                # when their efficacy gap is within the combined
                # confidence width and is non-decreasing:
                diff = np.abs(self._q_hat[m] - self._q_hat[m + 1])
                return (
                    diff <= conf_width(m) + conf_width(m + 1)
                    and self._q_hat[m] <= self._q_hat[m + 1]
                )
 
            # L1 (paper Alg. 2): the lowest admissible dose where the efficacy
            # plateau begins, i.e. the first dose whose pair (idx, idx+1) is
            # statistically flat.
            admissible_indices = [
                idx
                for idx in range(self._K)
                if admissible_set[idx]
            ]

            L_1 = self._K

            for k in admissible_indices:
                for m in range(k, self._K - 1):
                    if is_plateau_pair(m):
                        L_1 = min(L_1, m)
                        break

            # L2: highest safe dose by model toxicity (greedy toxicity rule).
            p_dle = self._dose_toxicity_curve(self._dose_levels, a_hat)
            L_2 = 0
            for idx in range(len(p_dle)):
                if (self._validator.validate(idx)
                        and p_dle[idx] <= self._ttl
                        and p_dle[idx] >= p_dle[L_2]):
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
        super().update(dose_level_index, cohort_size, n_dle, n_efficate)
        # Advance the initial round-robin once the sampled dose has been
        # observed, so the next cohort samples the next dose.
        if self._init_idx < self._K:
            self._init_idx += 1
