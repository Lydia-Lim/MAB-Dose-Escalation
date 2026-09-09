from typing import Callable, Sequence

import numpy as np

from doseescalation.dose_escalator._base import DoseEscalatorBase


class SEEDAPlateauMatlabDoseEscalator(DoseEscalatorBase):
    """
    Clean-room Python port of the authors' RealWorld ``Safe_UCB_Plateau.m`` (the
    version that ships with a complete ``main.m``). Written to mirror the MATLAB
    line-by-line rather than reuse ``SEEDADoseEscalator`` -- because the MATLAB's
    behavior depends on details our escalator abstraction does not express:

    - **Selection counts, not patient counts.** ``n_choose`` increments by 1 per
      cohort (a "selection"), independent of cohort size; the UCB bonus and the L1
      confidence width use it directly.
    - **Fixed horizon ``n``.** The L1 width uses ``log(n)`` (n = total cohorts) and
      ``delta = 1/n``, not ``log`` of the patients seen so far.
    - **Init quirk.** After sampling each dose once it stores ``q_hat = sum(X)`` /
      ``p_hat = sum(Y)`` (0..cohort), not the mean -- faithfully reproduced.
    - **Leader over ALL doses.** ``L = argmax_k q_hat_k`` over every dose (ties ->
      lowest index), then exploit / explore the neighborhood ``{L-1, L, L+1}``.
    - **a_hat = ak_hat of the most-sampled dose**; admissible set uses NO alpha
      margin (``toxicity(d, a_hat) <= theta``).

    Recommendation: ``min(Ii, Ri)`` where ``Ii`` = toxicity MTD and ``Ri`` is the
    ascending-overwrite turning point (= MTD-1 when the top admissible pair is
    "flat"). The flat test is parameterized: ``l1_mode='min'`` reproduces the
    RealWorld active test ``min(0.2*sqrt(log n/N_i), 0.2*sqrt(log n/N_{i+1}))``;
    ``l1_mode='sum'`` reproduces the synthetic active test
    ``0.4*sqrt(log n/N_i) + 0.4*sqrt(log n/N_{i+1})``.

    ``n_cohorts`` is the trial horizon (used for ``log(n)`` / ``delta``). The
    per-cohort recommendation is available via ``propose`` in non-training mode, so
    the caller can read Table 2 at any cohort (the paper reports it at n = 100).
    """

    def __init__(
        self,
        dose_levels: Sequence[float],
        target_toxicity_level: float,
        dose_toxicity_curve: Callable[[Sequence[float], float],
                                      Sequence[float]],
        n_cohorts: int,
        a0: float = 20.0,
        a_max: float = 20.0,
        ucb_coefficient: float = 1.0,
        eta: int = 2,
        l1_coefficient: float = 0.2,
        l1_mode: str = "min",
        gamma_1: float = 3 / 2,
        delta_1: float = None,
        is_training: bool = True,
        seed: float = 0,
    ):
        self._dose_levels = np.asarray(dose_levels, dtype=float)
        self._K = len(self._dose_levels)
        self._ttl = target_toxicity_level
        self._curve = dose_toxicity_curve
        self._n = n_cohorts
        self._a0 = float(a0)
        # a_max = None -> no clamp (lets a_hat overshoot early, as the paper's
        # figures do); a finite value clamps like the shipped MATLAB (a_max=20).
        self._a_max = a_max if a_max is None else float(a_max)
        self._c = ucb_coefficient
        self._eta = eta
        self._l1_coef = l1_coefficient
        self._l1_mode = l1_mode
        self._gamma_1 = gamma_1
        self._delta = delta_1 if delta_1 is not None else 1.0 / n_cohorts
        self._is_training = is_training

        # State (all 0-indexed; MATLAB is 1-indexed):
        self._n_choose = np.zeros(self._K)   # selection counts (MATLAB n_choose)
        self._p_hat = np.zeros(self._K)
        self._q_hat = np.zeros(self._K)
        self._ak_hat = np.full(self._K, self._a0)
        self._l = np.zeros(self._K)          # OSUB leader-visit counter
        self._L = 0                          # current leader (MATLAB L = 1)
        self._t = 0                          # cohorts observed so far
        self._init_idx = 0                   # sample-each-dose-once round-robin
        self._D = []                         # last admissible set (for the L1)

    def train(self, is_training: bool):
        self._is_training = is_training

    def _a_hat(self) -> float:
        # MATLAB: [~,I_est] = max(n_choose); a_hat = ak_hat(I_est).
        return self._ak_hat[int(np.argmax(self._n_choose))]

    def _admissible(self, a_hat: float):
        # MATLAB Plateau: D = find(toxicity(d, a_hat) <= thre) -- NO alpha margin.
        tox = self._curve(self._dose_levels, a_hat)
        return [k for k in range(self._K) if tox[k] <= self._ttl]

    def flat_threshold(self, i: int) -> float:
        """The L1 tolerance for the pair (i, i + 1), in efficacy-probability units.

        Public so the diagnosis figures can set it against the true adjacent
        efficacy gaps. ``n_choose`` counts COHORTS and the log is over the fixed
        horizon, both as the MATLAB has them -- the printed Algorithm 2 counts
        patients and logs the patients seen so far.
        """
        bi = self._l1_coef * np.sqrt(np.log(self._n) / self._n_choose[i])
        bj = self._l1_coef * np.sqrt(np.log(self._n) / self._n_choose[i + 1])
        return float(min(bi, bj) if self._l1_mode == "min" else bi + bj)

    def _flat(self, i: int) -> bool:
        gap = abs(self._q_hat[i] - self._q_hat[i + 1])
        return gap <= self.flat_threshold(i)

    def propose(self) -> int:
        # Sample each dose once, in ascending order, before any model selection:
        if self._init_idx < self._K:
            return self._init_idx

        a_hat = self._a_hat()
        self._D = self._admissible(a_hat)

        if self._is_training:
            t = self._t + 1  # current cohort index (1-based) for log(t) in F
            f = np.zeros(self._K)
            for k in self._D:
                f[k] = self._q_hat[k] + self._c * np.sqrt(
                    np.log(t) / self._n_choose[k]
                )
            L = self._L
            # Exploit the leader every (eta + 1) visits, else explore {L-1,L,L+1}:
            if (self._l[L] - 1) % (self._eta + 1) == 0 and f[L] != 0:
                return L
            window = [j for j in (L - 1, L, L + 1) if 0 <= j < self._K]
            if sum(f[j] for j in window) == 0:
                return int(np.argmax(f))
            return max(window, key=lambda j: f[j])

        # Recommendation: min(Ii, Ri).
        a_hat = self._a_hat()
        p_out = self._curve(self._dose_levels, a_hat)
        safe = p_out <= self._ttl
        Ii = int(np.argmax(p_out * safe))  # toxicity MTD (= L2)
        if self._D:
            temp = np.zeros(self._K)
            for idx in range(len(self._D) - 1):
                i = self._D[idx]
                if self._flat(i):
                    temp[i] = 1
            dend = self._D[-1]
            Ri = dend
            for i in range(dend):  # ascending overwrite -> largest qualifying i
                if np.prod(temp[i:dend]) == 1:
                    Ri = i
        else:
            Ri = self._K - 1
        # Observation only, for the diagnosis figures: which of the two terms
        # min(Ii, Ri) actually binds. Nothing downstream of propose reads these.
        self._last_l1, self._last_l2 = int(Ri), int(Ii)
        return min(Ii, Ri)

    def update(self, dose_level_index: int, cohort_size: int,
               n_dle: int, n_efficate: int):
        i = dose_level_index
        if self._init_idx < self._K:
            # Sample-once init (MATLAB quirk: store the SUM, and set n_choose = 1):
            self._q_hat[i] = n_efficate
            self._p_hat[i] = n_dle
            self._n_choose[i] = 1
            self._init_idx += 1
        else:
            nc = self._n_choose[i]
            self._q_hat[i] = (self._q_hat[i] * nc + n_efficate / cohort_size) / (nc + 1)
            self._p_hat[i] = (self._p_hat[i] * nc + n_dle / cohort_size) / (nc + 1)
            self._n_choose[i] += 1
            # Leader over ALL doses (MATLAB [~,L]=max(q_hat)); increment its counter:
            self._L = int(np.argmax(self._q_hat))
            self._l[self._L] += 1

        # Per-dose a-estimate (closed form) with the a_max clamp:
        base = (np.tanh(self._dose_levels[i]) + 1) / 2
        target = float(np.clip(self._p_hat[i], 1e-12, 1 - 1e-12))
        val = np.log(target) / np.log(base)
        self._ak_hat[i] = val if self._a_max is None else min(val, self._a_max)
        self._t += 1

    def safe_doses(self):
        # No safety classification until every dose has been sampled once:
        if self._init_idx < self._K:
            return [False] * self._K
        tox = self._curve(self._dose_levels, self._a_hat())
        return [bool(x) for x in (tox <= self._ttl)]
