# SEEDA-Plateau reproduction — investigation notes

Why our SEEDA-Plateau variants did/didn't reproduce the paper's Table 2, and how we
got the recommendation to match. Paper: Shen, Wang, Villar, van der Schaar,
*Learning for Dose Allocation in Adaptive Clinical Trials with Safety Constraints*,
ICML 2020 (`SEEDA.pdf`), synthetic scenario 1.

The failure was first diagnosed in `experiments/archive/l1_diagnosis.py` (kept for
historical reference; its header notes a class-naming caveat now that the
pairwise and entire-tail L1 rules live in separate classes). This document is
the up-to-date investigation and class map.

---

## 1. Experiment settings

- K = 6 doses, cohort size 3, MTD threshold θ = 0.35, 1000 repetitions; "n = 300
  cohorts" in the main text (Supplement K says Table 2 is "n = 100" — unreconciled).
- Toxicity (true): `[0.01, 0.05, 0.15, 0.2, 0.45, 0.6]` → safe doses 1–4 (M = 4).
- Efficacy (true):  `[0.1, 0.35, 0.6, 0.6, 0.6, 0.6]` → plateau onset N = 3.
- **Optimal biological dose k\* = min(M, N) = dose 3.** Toxicity MTD = dose 4.
- Priors: p_hat `[0.02,0.06,0.12,0.20,0.30,0.40]`, q_hat `[0.12,0.20,0.30,0.40,0.50,0.59]`.
- Dose-toxicity model `p_k(a) = ((tanh(d_k)+1)/2)^a`; dose levels built so
  `base_k = (tanh(d_k)+1)/2 = tox_k` at the true `a* = 1`.
- Paper Table 2 SEEDA-Plateau target: Recommended `[0.8, 2.2, 86.6, 10.4, 0, 0]`,
  Allocated `[7.8, 9.0, 30.1, 37.2, 14.9, 1.0]`.

Variants (`src/doseescalation/dose_escalator/`). The two non-plateau escalators
differ only in the exploration bonus `F`; the plateau escalators differ mainly in
the **recommendation (`L1`) rule** (and, for the two-sided-decoupled pair, in the
`L1` coefficient and `F`):

| class (label) | base | allocation | how it differs |
|---|---|---|---|
| `SEEDADoseEscalator` (current SEEDA) | — | pure UCB on `F` over admissible set | `F = q̂ + √(c·log n / N)` (matches Eq. 4). Recommends the highest-efficacy safe dose (no plateau logic). |
| `SEEDAOriginalDoseEscalator` ("SEEDA (UCB)") | — | pure UCB on `F` | Same as current SEEDA **except** `F = q̂ + √(c·n / N)` — **no `log`**, so a much larger exploration bonus (over-samples low-N doses). This is the only difference from current SEEDA. |
| `SEEDAPlateauDoseEscalator` ("Plateau (UCB)") | current SEEDA | leader + global-UCB | Adds plateau logic. `L1` = **entire-upper-tail-flat**: smallest dose where *every* higher adjacent pair passes the flatness test. Output `min(L1, L2)`, `L2` = MTD. |
| `SEEDAPlateauNaiveDoseEscalator` ("Plateau (Paper)") | current SEEDA | leader±1 | `L1` = **first-flat-pair**: the lowest pair that passes the flatness test (pairwise). Also has our added "sample each dose once" init. |
| `SEEDAPlateauFixedDoseEscalator` ("Modified L1") | Plateau (UCB) | inherited | `L1` = lowest dose whose efficacy is within a confidence width of the *best* admissible dose's efficacy (threshold-from-best rule). More robust to noise. **No longer used in the notebook** (kept in the package). |
| `SEEDAPlateauTwoSidedDecoupledDoseEscalator` ("Plateau (Two-sided, decoupled)") | Plateau (UCB) | inherited | UCB's entire-tail-flat `L1`, but **Part B dropped** (Part A only) and using a **separate `l1_coefficient`** (default 0.1) decoupled from the allocation `c`. **Recovers dose 3 (~90%).** |
| `SEEDAPlateauTwoSidedDecoupledNoLogDoseEscalator` ("Plateau (Two-sided, decoupled, NoLog)") | Plateau (Two-sided, decoupled) | inherited, **no-log `F`** | Same `L1` as above, plus SEEDA-UCB's no-log bonus `F = q̂ + √(c·n/N)` (samples low doses more). **Recovers dose 3 (~94%).** |

The naive first-pair and UCB tail rules use the same per-pair test = Part A ∧ Part B.
The two-sided-decoupled variants use Part A only.

`L1` flatness test for a pair (m, m+1), as printed in Algorithm 2 step 12:
- **Part A (two-sided):** `|q̂_m − q̂_{m+1}| ≤ β_m + β_{m+1}`, `β_m = √(c·log n / N_m)`.
- **Part B (clause):** `q̂_m ≤ q̂_{m+1}`.

Bugs fixed just to get faithful behaviour (both applied to code):
- **`a_hat` estimation** — a warm-started Newton solve got stuck at inflated values
  (~4–8 vs true 1) for low-toxicity doses, making unsafe doses look admissible.
  Replaced with the exact closed form `a = log(p̂)/log(base)`. (base `_seeda.py` +
  `_seeda_original.py`; all variants benefit.)
- **Initialization** — restored Algorithm 2's "sample each dose once" warm-up
  instead of seeding with the increasing efficacy prior (`_seeda_plateau_naive.py`).

---

## 2. Issues with Plateau when faithful to the paper

**The naive/paper (pairwise) `L1` collapses to dose 1 (~90% of trials).** Two causes:
1. **Low-dose under-sampling.** Allocation is leader±1 (step 5) and the leader is
   always ≥ dose 3, so dose 1 gets ~3 patients all trial. Its noise bar `β₁` is then
   enormous, so Part A alone reads the *rising* pair (dose 1, dose 2) as "flat" →
   `L1 = 1`.
2. **Part B, read as a hard filter on the noisy `q̂`.** At the true plateau (doses
   3,4 both 0.6), `q̂_3 − q̂_4` is centred at zero, so its sign is a coin-flip;
   Part B rejects the real plateau pair ~50% of the time → a detection ceiling of
   ~50% *independent of sample size*.

**The Plateau (UCB) (entire-tail-flat) `L1` defaults to the MTD (dose 4).** With the
tail rule, Part B must pass on *every* tail pair (≈(1/2)³), so `L1` essentially
never fires and the output falls back to `L2` = MTD. (Confirmed: with Part B on,
100% of tail-rule configs recommend dose 4 regardless of `F` or `c`.)

---

## 3. What we CANNOT correct by following the paper as written

- **The pairwise rule can't be rescued faithfully.** Dropping Part B *alone* makes
  the pairwise rule *worse* (100% dose-1 collapse), because Part B was accidentally
  masking the false (dose1,dose2) plateau. The under-sampling has to be dealt with
  first — and it can't be, via the literal allocation:
- **We can't reproduce the paper's allocation shape.** The paper allocates ~7.8% to
  dose 1 and ~15% to (unsafe) dose 5. No faithful setting we tried — admissible-set
  radius `C₁` from Appendix B, the `a_hat ± α` sign, the exploration coefficient
  `c`, or the init — puts that much budget on dose 1. Under leader±1 the leader
  never sits at dose 1/2, so dose 1 stays ~0.4%. Where the paper's 7.8% comes from
  is not derivable from the algorithm as printed.

(The *recommendation* is nonetheless reproducible without matching the allocation —
see §5.)

---

## 4. What the paper does NOT specify (candidate missing pieces)

1. **Whether allocation and `L1` share the coefficient `c`.** Same symbol `c` in the
   allocation index (Eq. 4) and the `L1` bar (step 12), but the paper never says
   they're equal, and Eq. 4 notes the allocation index "can be replaced by other UCB
   principles, e.g. KL-UCB" (which has no `c`). They want opposite values —
   exploration wants `c` large, `L1` wants `c` small — so whether they're one knob
   or two is decisive. Theorem 4's range `2 < c < 5/2` is asymptotic and makes `L1`
   collapse at finite n.
2. **`C₁` / parameter range `|A|`.** Appendix B derives `C₁` only up to `|A|` (range
   of `a`), never given. Our code used an arbitrary `/30`, ~50× off the Appendix-B
   value (which itself makes the confidence radius vacuous). Underspecified.
3. **Sample size behind Table 2** — main text 300 cohorts, Supplement K says n = 100.
4. **The role of Part B** — in the pseudocode but absent from the Theorem 4 proof
   (which bounds error with Part A only). Unclear if it's an empirical filter or a
   restatement of Assumption 1 (the *true* q is non-decreasing).
5. **Origin of the dose-1 ≈ 7.8% allocation** under a leader±1 policy whose leader
   never sits low.

---

## 5. New findings — the recommendation IS reproducible via Plateau (UCB)'s tail rule

Starting from the **entire-upper-tail-flat** `L1` (Plateau UCB) instead of the
pairwise rule reproduces the paper's recommendation — and it is **robust to the
low-dose under-sampling, so that problem never needs solving** (dose-1 allocation
stays ~0.4% in every winning config).

**Full tail-`L1` grid (n = 300, allocation c = 2.1):**
| Part B | F | c_L1 | dose-3 rec | result |
|---|---|---|---|---|
| on  | log   | coupled (2.1) | 0%    | → 92.5% dose 4 (MTD) |
| on  | log   | 0.1           | 3.8%  | → 94.2% dose 4 |
| on  | nolog | coupled (2.1) | 0%    | → 92.5% dose 4 |
| on  | nolog | 0.1           | 5.8%  | → 92.8% dose 4 |
| off | log   | coupled (2.1) | 0%    | → 100% dose 1 (collapse) |
| off | log   | 0.1           | **77.5%** | ✓ |
| off | nolog | coupled (2.1) | 0%    | → 100% dose 1 (collapse) |
| off | nolog | 0.1           | **80–83%** | ✓ best |
| paper Table 2 |   |             | 86.6% | target |

Reading the grid: **Part B on → dose 4 (MTD) in all four rows** (F and c_L1
irrelevant). **Part B off + coupled (large) c → 100% dose 1 in both rows** (F
irrelevant). Only **Part B off + small c_L1** reaches dose 3; `F` then shifts it a
few points (log 77.5% → nolog 80–83%). Control: the *pairwise* rule with the same
off/nolog/0.1 settings still gives ~96% dose 1 — so it's the **rule**, not the
settings, that mattered.

**Both ingredients are necessary and separable:**
- **Remove Part B** — else the tail rule can't fire and it defaults to the MTD
  (dose 4). Consistent with the Theorem 4 proof (uses Part A only).
- **Small `c_L1` ≈ 0.1, decoupled from the allocation `c`** — else `β` is too large,
  every pair reads flat, and the tail rule picks dose 1. Justified by Eq. 4's
  swappable-index note (§4.1).
- **No-log `F`** (SEEDA-UCB exploration) is *optional*: +~3–6 points (samples dose 2
  more), but it *deviates* from Eq. 4. Drop it for strict fidelity (still ~77%).

**Why the tail rule works where pairwise fails.** The plateau onset is where efficacy
goes flat *and stays flat* (Assumption 1: `q_1 ≤ … ≤ q_N = … = q_K`), so the onset =
smallest dose where *all* higher pairs are flat. A false (dose1,dose2) flat no longer
matters, because dose 1 is rejected on the grounds that (dose2,dose3) is truly
rising — which a small `c_L1` correctly detects. The naive "first flat pair" rule is
the flawed simplification.

**Faithfulness of the working recipe:** tail-`L1` = arguably the correct reading of
the paper's plateau structure; no-Part-B = consistent with the proof; small decoupled
`c_L1` = plausible given Eq. 4; no-log `F` = a deviation (optional). The recommendation
matches (~80% vs 86.6%); the allocation still does not (we under-sample doses 1 & 5).

**Now implemented as two escalator classes** (both subclass UCB's Plateau):
- `SEEDAPlateauTwoSidedDecoupledDoseEscalator` — label "SEEDA Plateau (Two-sided,
  decoupled)". Tail `L1`, Part A only, `l1_coefficient` (default 0.1) decoupled from
  the allocation `c`, log `F`. On the UCB base (global-UCB allocation), recommends
  dose 3 in **~90%** of trials at n=300.
- `SEEDAPlateauTwoSidedDecoupledNoLogDoseEscalator` — label "SEEDA Plateau
  (Two-sided, decoupled, NoLog)". Same, plus SEEDA-UCB's no-log exploration `F`.
  **~94%** at n=300 (the extra exploration helps).

Both are wired into `benchmarks original paper.ipynb`, which now compares the four
plateau variants: Paper (0% dose 3), UCB (0%, → MTD), Two-sided decoupled (~90%),
and Two-sided decoupled NoLog (~94%). The "Modified L1" variant was removed from
that notebook (still in the package). Note the recommendation is recovered *without*
matching the paper's allocation — the tail rule is robust to the low-dose
under-sampling, so that unresolved allocation gap (§3) no longer blocks the result.
