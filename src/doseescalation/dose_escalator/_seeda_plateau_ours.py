from typing import Callable, Sequence

import numpy as np

from ._seeda import SEEDADoseEscalator


class SEEDAPlateauOursDoseEscalator(SEEDADoseEscalator):
    """
    Our SEEDA-Plateau variant.

    A standalone copy of ``SEEDAPlateauTwoSidedDecoupledDoseEscalator`` with three
    changes. It deliberately does NOT subclass that class, so the two can be run
    side by side and neither drifts when the other is edited.

    **1. Recommendation: compare every dose to the MTD, not to its neighbor.**
    Reference chains a pairwise flat test down from the MTD, which has an
    accumulation blind spot: K-1 gaps each individually inside the noise can sum
    to a large total gap the chain never sees (efficacy [0.5, 0.55, 0.6, 0.6] has
    local gaps 0.05/0.05/0, so the chain walks to dose 1 despite a true cumulative
    gap of 0.1). Here L1 is the lowest admissible dose whose efficacy is
    statistically indistinguishable from the MTD's, which sees the cumulative gap
    in a single test. This is the direct empirical analogue of the paper's own
    definition of the optimal dose, §2.2: ``k* = min{k : q_k = max_{l: p_l<=theta} q_l}``.

    The threshold is ONE-SIDED -- ``q_hat_mtd - conf_width(mtd)``, using only the
    reference's own margin. Adding the candidate's ``conf_width`` back would let a
    starved low dose pass on the strength of its own uncertainty, which is exactly
    the failure that sinks the paper's printed pairwise rule.

    L2 is redundant here and is not computed: the MTD trivially passes its own
    test, so L1 <= MTD always, and ``min(L1, L2)`` reduces to L1.

    **2. Admissibility uses a pessimistic bound on a.** SEEDA tests
    ``p_k(a_hat + alpha) <= theta``. Since ``p = base^a`` with ``base < 1``, a
    larger ``a`` means LOWER modelled toxicity, so ``a_hat + alpha`` is the most
    OPTIMISTIC end of the confidence interval -- the paper's Lemma 1 controls false
    alarms (excluding a safe dose), not miss-detection. We test
    ``p_k(a_hat - alpha) <= theta`` instead: admit a dose only when confident it is
    safe. Motivation: the sweep in ``experiments/sensitivity_a_max.py`` showed the
    shipped design's Type II is identically zero only because ``a_max = a0`` clamps
    the estimate at the value that reproduces the true toxicity curve, so the model
    is structurally unable to under-estimate toxicity. No real trial can set that
    clamp. A pessimistic bound is meant to keep the safety property without needing
    to know the answer -- so this variant is intended to be run with a LOOSE or
    absent ``a_max``.

    **3. The leader is restricted to the admissible set** (paper Algorithm 2 step 4,
    ``L(t) = argmax_{d_k in D_1(t)} q_hat_k``). Reference takes a global
    argmax, following the MATLAB, which lets a non-admissible dose whose ``q_hat``
    is frozen at a 3-patient warm-up estimate stay leader indefinitely -- measured
    at 73.75% of post-warm-up rounds, dragging allocation to 68% on dose 4 against
    the paper's 37%. This is a fidelity fix rather than a contribution.

    ``pessimistic_admissible`` and ``restrict_leader`` are exposed so the three
    changes can be ablated independently.
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
        pessimistic_admissible: bool = False,
        restrict_leader: bool = True,
        shrinkage_m: float = 0.0,
        a_prior: float = 1.0,
        p_smoothing: float = 0.5,
        tox_margin_c: float = 0.01,
    ):
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
        # Confidence coefficient used ONLY in the recommendation's equivalence
        # test, decoupled from the allocation UCB coefficient (self._c):
        self._l1_coefficient = l1_coefficient
        self._pessimistic_admissible = pessimistic_admissible
        self._restrict_leader = restrict_leader
        # Shrinkage of the per-dose a-estimate toward a pessimistic prior, with
        # weight m / (N_k + m). Set shrinkage_m = 0 to disable.
        self._shrinkage_m = shrinkage_m
        self._a_prior = a_prior
        # Jeffreys-style smoothing of the observed toxicity rate: replaces the
        # base class's clip of p_hat at 1e-6. See update().
        self._p_smoothing = p_smoothing
        # Per-dose safety margin added to the modelled toxicity before the
        # admissibility test, scaled by that dose's own patient count. 0 disables.
        self._tox_margin_c = tox_margin_c

    def _alpha_t(self) -> float:
        # Same alpha(t) as the base class (paper eq. 3); recomputed here because
        # _calc_model_params does not return it.
        return self._C_1 * self._K * (
            np.log(2 * self._K / self._delta_1) / (2 * sum(self._N))
        ) ** (1.0 / (self._gamma_1 * 2.0))

    def _calc_model_params(self):
        a_hat, _, F, _ = super()._calc_model_params()

        # Shrink the a-estimate toward a pessimistic prior, weighted by how much
        # data backs it. The base picks a_hat = the MOST-SAMPLED dose's estimate,
        # so that is what gets shrunk. This targets the actual failure: right
        # after warm-up the most-sampled dose is the lowest one, which has almost
        # certainly seen zero DLTs, so p_hat clips to 1e-6 and a_hat jumps to ~60
        # -- the model declares the drug near-harmless and every dose admissible.
        # Unlike alpha(t), which is pooled over all doses and identical for each,
        # this scales with the per-dose count N_k, which is where the ignorance
        # actually is.
        if self._shrinkage_m > 0:
            k = int(np.argmax(self._N))
            n_k = self._N[k]
            a_hat = ((n_k * a_hat + self._shrinkage_m * self._a_prior)
                     / (n_k + self._shrinkage_m))

        if self._pessimistic_admissible:
            # Lower confidence bound on a = upper bound on toxicity. Floored at a
            # small positive value: alpha exceeds a_hat only in pathological cases
            # (tiny a_init with almost no data), where a negative exponent would
            # give toxicity > 1 and empty the admissible set anyway.
            a_eff = max(a_hat - self._alpha_t(), 1e-6)
        else:
            a_eff = a_hat
        p_model = self._dose_toxicity_curve(self._dose_levels, a_eff)

        # Per-dose safety margin, same UCB-style form as F and the flatness width
        # (sqrt(c * log n / N)), added to toxicity instead of efficacy. alpha(t)
        # is pooled -- it uses the TOTAL patient count, so a dose with no patients
        # gets exactly the same margin as one with 200, which is why it cannot
        # express per-dose ignorance. This scales with N_k instead: a
        # barely-sampled dose must clear theta by a wide margin, a well-sampled
        # one by almost nothing. It directly counteracts an over-optimistic
        # a_hat on the doses where that optimism is unearned.
        if self._tox_margin_c > 0:
            n_total = max(np.sum(self._N), 2)
            p_model = p_model + np.sqrt(
                self._tox_margin_c * np.log(n_total) / (self._N + 1)
            )

        admissible_set = p_model <= self._ttl
        F_filtered = F[admissible_set]
        return a_hat, admissible_set, F, F_filtered

    def flat_threshold(self, mtd: int) -> float:
        """This variant's equivalence tolerance, in efficacy-probability units.

        Deliberately NOT the same object as the pairwise designs' threshold, and
        the difference is the point. Those add two per-dose widths and apply the
        sum to an ADJACENT pair; this uses the MTD's own width alone and applies
        it to the gap between the MTD and the candidate, however far below it
        sits. One width rather than two roughly halves the tolerance, and
        measuring against the MTD rather than the neighbor means a candidate
        several doses down is judged on the cumulative gap.

        Public so the diagnosis figures can set it against the true gaps. Takes
        the MTD index, since that is what it is anchored to.
        """
        return float(np.sqrt(
            self._l1_coefficient * np.log(np.sum(self._N)) / self._N[mtd]
        ))

    def propose(self) -> int:
        # Warm-up: the ALLOCATION samples each dose once (round-robin), while the
        # RECOMMENDATION is still produced every round but skips the equivalence
        # test until every dose has been sampled -- see the recommendation block.
        if self._is_training and self._init_idx < self._K:
            return self._init_idx
        if not self._is_training and np.sum(self._N) == 0:
            return 0

        a_hat, admissible_set, F, F_filtered = self._calc_model_params()

        admissible_idx = [
            idx for idx in range(self._K)
            if admissible_set[idx] and self._validator.validate(idx)
        ]

        # Leader: highest q_hat among ADMISSIBLE doses (Algorithm 2 step 4). Ties
        # resolve to the lowest index, i.e. the first dose on the efficacy plateau.
        if self._restrict_leader:
            L = (max(admissible_idx, key=lambda i: self._q_hat[i])
                 if admissible_idx else 0)
        else:
            L = int(np.argmax(self._q_hat))
        if np.sum(self._l) < self._update_counts:
            self._l[L] += 1

        if self._is_training:
            if len(F_filtered) == 0:
                self._I = 0
                return self._I
            # f = efficacy UCB bonus, zeroed on non-admissible doses.
            f = np.where(admissible_set, F, 0.0)
            # Exploit the leader every (eta+1) visits; otherwise explore ONLY its
            # neighborhood {L-1, L, L+1}, picking the largest f; if that whole
            # window is 0, fall back to the global argmax.
            if (self._l[L] - 1) % (self._eta + 1) == 0 and f[L] != 0:
                self._I = L
            else:
                window = list(range(max(L - 1, 0), min(L + 1, self._K - 1) + 1))
                if sum(f[j] for j in window) == 0:
                    self._I = int(np.argmax(f))
                else:
                    self._I = max(window, key=lambda j: f[j])
            return self._I

        # --- Recommendation: lowest admissible dose indistinguishable from the MTD.
        if not admissible_idx:
            return 0
        mtd = admissible_idx[-1]

        # Warm-up: the equivalence test would divide by N_m = 0 on the not-yet-
        # sampled doses and make every dose trivially indistinguishable, dragging
        # the recommendation down on rounds it should score as the MTD.
        if self._init_idx < self._K:
            return mtd

        threshold = self._q_hat[mtd] - self.flat_threshold(mtd)
        # Walk DOWN from the MTD while doses keep passing, stopping at the first
        # failure -- the lowest dose in a CONTIGUOUS passing run ending at the
        # MTD. The break is what matters, not the direction: scanning upward and
        # returning the first pass gives the lowest dose passing ANYWHERE, so a
        # single noisy low dose drags the recommendation down. Requiring
        # contiguity keeps Reference's protection against that while still
        # comparing every dose to the MTD, so cumulative gaps stay visible.
        below = [i for i in admissible_idx if i < mtd]
        L_1 = mtd
        for idx in reversed(below):
            if self._q_hat[idx] < threshold:
                break
            L_1 = idx
        # Observation only, for the diagnosis figures. There is no separate L2
        # here: the scan already runs inside the admissible set, whose top
        # element IS the MTD this variant anchors on.
        self._last_l1, self._last_l2 = int(L_1), int(mtd)
        return L_1

    def update(self,
               dose_level_index: int,
               cohort_size: int,
               n_dle: int,
               n_efficate: int):
        # Count only leader-eligible cohorts (i.e. after the initial sampling
        # phase), so the eta-cycle counter advances once per post-warm-up round.
        if self._init_idx >= self._K:
            self._update_counts += 1
        super().update(dose_level_index, cohort_size, n_dle, n_efficate)

        if self._p_smoothing <= 0:
            return
        # Recompute this dose's a-estimate from a SMOOTHED toxicity rate.
        # The base solves a_hat = log(p_hat) / log(base) and clips p_hat at 1e-6,
        # so a dose that has seen zero DLTs yields a_hat ~= 60 -- "the drug is
        # near-harmless" -- no matter how many patients produced that zero. Since
        # higher a means lower modelled toxicity, that admits every dose. A
        # Jeffreys prior, (DLTs + 0.5) / (N + 1), keeps the estimate finite and
        # lets it relax as real data arrives: 0/3 -> a_hat ~= 9, 0/30 -> ~= 18,
        # converging to the truth rather than to the clip.
        k = dose_level_index
        n_k = self._N[k]
        dlts = self._p_hat[k] * n_k
        p_smooth = ((dlts + self._p_smoothing)
                    / (n_k + 2 * self._p_smoothing))
        log_base = np.log(self._dose_toxicity_curve(
            dose_levels=self._dose_levels[k], a_hat=1))
        a_k = np.log(p_smooth) / log_base
        if self._a_max is not None:
            a_k = min(a_k, self._a_max)
        self._a_hat_doses[k] = a_k
