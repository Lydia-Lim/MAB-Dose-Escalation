import math
from typing import Sequence

import numpy as np

from ._base import DoseEscalatorBase
from ._validate import Validator, NoOpValidator, NoSkipValidator


class UCBDoseEscalator(DoseEscalatorBase):
    """
    Efficacy UCB-1 dose escalator, matching the SEEDA paper's UCB-1 baseline
    (supplementary Section J; authors' MATLAB ``UCB.m``).

    Allocation follows UCB-1 (Auer et al., 2002) on the *efficacy* estimate,
    ignoring the safety constraint during exploration: after sampling each dose
    once, it allocates the dose with the largest

        F_k = q_hat_k + c * sqrt(log(t) / N_k),

    where ``q_hat_k`` is the running mean efficacy at dose k, ``N_k`` the number
    of cohorts allocated there, ``t`` the cohort index and ``c`` the UCB-1
    coefficient (the paper uses c = 2). The final recommendation *does* respect
    safety: it is the highest-efficacy dose among those whose estimated
    toxicity is within the MTD threshold,

        d_hat(n) = argmax_{k : p_hat_k <= theta} q_hat_k.

    Both toxicity (``p_hat``) and efficacy (``q_hat``) are tracked as running
    means of the observed per-cohort rates; no dose-toxicity model or prior is
    used (estimates start at 0, then each dose is sampled once).

    Difference from the previous UCB implementation
    -----------------------------------------------
    An earlier version of this class was a *toxicity-gated dose-preference*
    bandit that never observed efficacy: it scored each dose by its (scaled)
    dose level, discounted when the estimated toxicity exceeded the MTD, plus a
    visit-count UCB bonus, and allocated the argmax. That rewards higher doses
    directly and only uses toxicity, so it is a different algorithm from the
    paper's baseline. This version is the paper's genuine UCB-1: the index is
    built on the *efficacy* estimate ``q_hat`` (not the dose level), safety is
    ignored during allocation and enforced only in the recommendation, and it
    therefore requires efficacy feedback (``n_efficate`` in ``update``).
    """

    def __init__(
        self,
        dose_levels: Sequence[float],
        target_toxicity_level: float,
        ucb_coefficient: float = 2.0,
        is_training: bool = True,
        no_skip: bool = False,
    ):
        self._dose_levels = dose_levels
        self._ttl = target_toxicity_level
        self._K = len(dose_levels)
        self._c = ucb_coefficient
        self._is_training = is_training

        # Running per-dose efficacy / toxicity estimates and visit counts:
        self._q_hat = np.zeros(self._K)
        self._p_hat = np.zeros(self._K)
        self._n_choose = np.zeros(self._K)
        self._t = 0                 # cohorts observed so far (for log(t) bonus)
        self._init_idx = 0          # next dose to sample in the initial round-robin

        self._validator: Validator = (
            NoSkipValidator(len(dose_levels)) if no_skip
            else NoOpValidator()
        )

    def train(self, is_training: bool):
        self._is_training = is_training

    def propose(self) -> int:
        # Initial phase (UCB-1): sample each dose once, in ascending order,
        # before any index-based selection.
        if self._init_idx < self._K:
            return self._init_idx

        if self._is_training:
            # Allocation: UCB-1 on efficacy, ignoring safety (paper baseline).
            t = max(self._t, 1)
            ucbs = [
                self._q_hat[idx] + self._c * math.sqrt(
                    math.log(t) / self._n_choose[idx]
                ) if self._validator.validate(idx) else -math.inf
                for idx in range(self._K)
            ]
            return int(np.argmax(ucbs))

        # Recommendation: highest-efficacy dose among the estimated-safe ones.
        max_idx = 0
        for idx in range(self._K):
            if (self._validator.validate(idx) and
                    self._p_hat[idx] <= self._ttl and
                    self._q_hat[idx] >= self._q_hat[max_idx]):
                max_idx = idx
        return max_idx

    def safe_doses(self):
        """Per-dose boolean mask of doses whose estimated toxicity is <= TTL."""
        return [bool(self._p_hat[idx] <= self._ttl) for idx in range(self._K)]

    def update(
        self,
        dose_level_index: int,
        cohort_size: int,
        n_dle: int,
        n_efficate: int,
    ) -> None:
        if not self._is_training:
            raise RuntimeError("Cannot update in non-training mode")

        self._validator.visit(dose_level_index)
        n = self._n_choose[dose_level_index]
        p_rate = n_dle / cohort_size
        q_rate = n_efficate / cohort_size
        self._q_hat[dose_level_index] = (
            self._q_hat[dose_level_index] * n + q_rate
        ) / (n + 1)
        self._p_hat[dose_level_index] = (
            self._p_hat[dose_level_index] * n + p_rate
        ) / (n + 1)
        self._n_choose[dose_level_index] = n + 1
        self._t += 1

        # Advance the initial round-robin once this dose has been sampled.
        if self._init_idx < self._K:
            self._init_idx += 1
