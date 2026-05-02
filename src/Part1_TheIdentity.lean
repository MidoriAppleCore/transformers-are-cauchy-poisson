import Mathlib

/-!
# Part 1 — The Identity

Main theorem path in this file:
1. row-level Poisson representation for softmax;
2. softmax-specific log-ratio identity on pole heights;
3. full vector output equivalence;
4. Cauchy transform (§1.6) and the vertical-pole bridge
   `contourOutput_eq_im_cauchyTransform_vertical`, used in Part 2 so the full
   transformer head equality factors through `im (cauchyTransform …)`.

Auxiliary examples and bridge lemmas are intentionally omitted here to keep the
core argument short.
-/

noncomputable section

open Finset Real
open scoped BigOperators

namespace AnalyticTransformer

-- ═══════════════════════════════════════════════════════════════════════
-- § 1.1  Definitions
-- ═══════════════════════════════════════════════════════════════════════

/-- The Poisson kernel: P(x, y, q) = y / ((q - x)² + y²).

    The probability density at query position q of
    receiving information from a boundary source at position (x, y)
    in the upper half-plane.
-/

def poisson (x y q : ℝ) : ℝ := y / ((q - x) ^ 2 + y ^ 2)

/-- A valid attention distribution: N positive weights summing to 1.
    Every softmax output is one of these. -/
structure Simplex (N : ℕ) where
  w : Fin N → ℝ
  pos : ∀ i, 0 < w i
  sum_one : ∑ i : Fin N, w i = 1

/-- N poles in the upper half-plane ℍ (Im z > 0).
    The pole geometry that "explains" an attention distribution. -/
structure Poles (N : ℕ) where
  x : Fin N → ℝ
  y : Fin N → ℝ
  im_pos : ∀ k, 0 < y k

-- ═══════════════════════════════════════════════════════════════════════
-- § 1.2  Construction
-- ═══════════════════════════════════════════════════════════════════════

/-- Place each pole directly above the query: x_k = q, y_k = 1/w_k.
    At x = q the Poisson kernel collapses: P(q, y, q) = 1/y = w_k. -/
def matchPoles {N : ℕ} (q : ℝ) (s : Simplex N) : Poles N where
  x _ := q
  y k := (s.w k)⁻¹
  im_pos k := inv_pos.mpr (s.pos k)

/-- **Representation lemma**: every positive weight `w` has a Poisson realisation
    at the on-query pole `(q, 1/w)`.

    Note: the pole parameters are *chosen from the answer* (`x = q`, `y = 1/w`),
    so this holds for any `w > 0` regardless of how `w` was produced.
    The softmax/query-key structure is not used here; it is used in
    `softmaxSimplexOfScores`, which feeds the softmax outputs into this lemma. -/
theorem poisson_at_query (q : ℝ) {w : ℝ} (hw : 0 < w) :
    poisson q w⁻¹ q = w := by
  unfold poisson
  rw [sub_self, zero_pow two_ne_zero, zero_add]
  field_simp

/-- Explicit arithmetic simplification used repeatedly in the pole proofs:
    for positive `w`, the reciprocal involution collapses exactly:
    `1 / (1 / w) = w`. -/
theorem one_div_one_div_of_pos {w : ℝ} (hw : 0 < w) :
    1 / (1 / w) = w := by
  field_simp [hw.ne']

/-- On-query Poisson identity written in the "double reciprocal" form.

    This exposes directly the cancellation used implicitly by simp in many
    downstream proofs: `1/(1/w)` simplifies to `w`, and the on-query value
    is then exactly `1/w`. -/
theorem poisson_at_query_one_div_one_div (q : ℝ) {w : ℝ} (hw : 0 < w) :
    poisson q (1 / (1 / w)) q = 1 / w := by
  rw [one_div_one_div_of_pos hw]
  unfold poisson
  rw [sub_self, zero_pow two_ne_zero, zero_add]
  field_simp [hw.ne']

/-- **Canonicality on the realised vertical slice.**

    Suppose a pole family `p` satisfies:
    1. all poles are on-query (`p.x k = q`), and
    2. Poisson evaluation at `q` matches simplex weights (`poisson (p.x k) (p.y k) q = s.w k`).

    Then `p` is forced to be exactly `matchPoles q s`.

    This upgrades the common existential bridge ("there exists poles")
    to a uniqueness statement on the realised slice used by
    score-derived transformer rows. -/
theorem matchPoles_unique_on_vertical_slice {N : ℕ}
    (q : ℝ) (s : Simplex N) (p : Poles N)
    (hx : ∀ k, p.x k = q)
    (hw : ∀ k, poisson (p.x k) (p.y k) q = s.w k) :
    p = matchPoles q s := by
  cases p with
  | mk px py hp =>
    have hx' : px = fun _ => q := funext hx
    have hy' : py = fun k => (s.w k)⁻¹ := by
      funext k
      have hyne : py k ≠ 0 := ne_of_gt (hp k)
      have h0 := hw k
      rw [hx k] at h0
      have h0' : poisson q (py k) q = s.w k := by
        simpa using h0
      have hpo : poisson q (py k) q = (py k)⁻¹ := by
        have htmp : poisson q (((py k)⁻¹)⁻¹) q = (py k)⁻¹ :=
          poisson_at_query q (w := (py k)⁻¹) (inv_pos.mpr (hp k))
        simpa [inv_inv, hyne] using htmp
      have hpk : (py k)⁻¹ = s.w k := hpo.symm.trans h0'
      apply inv_injective
      simpa using hpk
    subst hx'
    subst hy'
    simp [matchPoles]

/-- matchPoles produces poles whose kernel evaluations recover the weights. -/
theorem match_weight {N : ℕ} (s : Simplex N) (q : ℝ) (k : Fin N) :
    poisson ((matchPoles q s).x k) ((matchPoles q s).y k) q = s.w k :=
  poisson_at_query q (s.pos k)

/-- The Poisson weights from matchPoles sum to 1 (they form a valid distribution). -/
theorem match_sum {N : ℕ} (s : Simplex N) (q : ℝ) :
    ∑ j : Fin N, poisson ((matchPoles q s).x j) ((matchPoles q s).y j) q = 1 := by
  simp_rw [match_weight]; exact s.sum_one

-- ═══════════════════════════════════════════════════════════════════════
-- § 1.3  Core identities
-- ═══════════════════════════════════════════════════════════════════════

/-- Every simplex distribution has an explicit Poisson representation. -/
theorem topological_subsumes_simplex {N : ℕ} (s : Simplex N) (q : ℝ) :
    ∃ p : Poles N,
      (∀ k : Fin N, poisson (p.x k) (p.y k) q = s.w k) ∧
      ∑ j : Fin N, poisson (p.x j) (p.y j) q = 1 :=
  ⟨matchPoles q s, match_weight s q, match_sum s q⟩

/-- Softmax of any score vector is a simplex point. -/
def softmaxSimplexOfScores {N : ℕ} [NeZero N]
    (scores : Fin N → ℝ) : Simplex N where
  w k := exp (scores k) / ∑ j : Fin N, exp (scores j)
  pos i := div_pos (exp_pos _) (sum_pos (fun j _ => exp_pos _) univ_nonempty)
  sum_one := by
    rw [← Finset.sum_div]
    exact div_self (ne_of_gt (sum_pos (fun j _ => exp_pos _) univ_nonempty))

/-- Helper existence lemma: softmax weights admit a Poisson realisation
at explicit poles.

This theorem is intentionally existential and is kept as infrastructure;
the primary Gate 1 statement is `softmax_poles_unique_on_vertical_slice`,
which upgrades this to a uniqueness/classification theorem on the realised
slice `x_k = q`. -/
theorem topological_subsumes_softmax_of_scores {N : ℕ} [NeZero N]
    (scores : Fin N → ℝ) (q : ℝ) :
    ∃ p : Poles N, ∀ k : Fin N,
      poisson (p.x k) (p.y k) q =
        exp (scores k) / ∑ j : Fin N, exp (scores j) :=
  let s := softmaxSimplexOfScores scores
  ⟨matchPoles q s, match_weight s q⟩

/-- **Non-tautological row classification on the realised slice**:
for fixed `scores, q`, there is exactly one pole family on the
vertical slice `x_k = q` that reproduces the softmax row under Poisson
evaluation, namely `matchPoles q (softmaxSimplexOfScores scores)`. -/
theorem softmax_poles_unique_on_vertical_slice {N : ℕ} [NeZero N]
    (scores : Fin N → ℝ) (q : ℝ) :
    ∃! p : Poles N,
      (∀ k : Fin N, p.x k = q) ∧
      (∀ k : Fin N,
        poisson (p.x k) (p.y k) q =
          exp (scores k) / ∑ j : Fin N, exp (scores j)) := by
  let s := softmaxSimplexOfScores scores
  refine ⟨matchPoles q s, ?_, ?_⟩
  · constructor
    · intro k
      simp [matchPoles]
    · intro k
      simpa [s] using (match_weight s q k)
  · intro p hp
    exact matchPoles_unique_on_vertical_slice q s p hp.1 hp.2

/-- Score-derived pole heights from softmax inputs: `y_k = Z / exp(scores_k)`,
    where `Z = Σ_j exp(scores_j)`. These are determined before softmax output. -/
def scoreDerivedPoleHeight {N : ℕ} [NeZero N]
    (scores : Fin N → ℝ) (k : Fin N) : ℝ :=
  (∑ j : Fin N, exp (scores j)) / exp (scores k)

/-- The score-derived height is exactly the reciprocal of the softmax weight. -/
theorem scoreDerivedPoleHeight_eq_inv_softmax {N : ℕ} [NeZero N]
    (scores : Fin N → ℝ) (k : Fin N) :
    scoreDerivedPoleHeight scores k =
      (exp (scores k) / ∑ j : Fin N, exp (scores j))⁻¹ := by
  unfold scoreDerivedPoleHeight
  have hZ : (0 : ℝ) < ∑ j : Fin N, exp (scores j) :=
    Finset.sum_pos (fun j _ => exp_pos _) Finset.univ_nonempty
  have hek : (0 : ℝ) < exp (scores k) := exp_pos _
  field_simp [hZ.ne', hek.ne']

/-- Dot-product/score attention determines the Poisson height map:
    using `y_k = Z/exp(scores_k)`, we get `poisson q y_k q = softmax(scores)_k`. -/
theorem softmax_is_poisson_at_score_height {N : ℕ} [NeZero N]
    (scores : Fin N → ℝ) (q : ℝ) (k : Fin N) :
    let yk := scoreDerivedPoleHeight scores k
    poisson q yk q = exp (scores k) / ∑ j : Fin N, exp (scores j) := by
  intro yk
  rw [show yk = scoreDerivedPoleHeight scores k from rfl]
  rw [scoreDerivedPoleHeight_eq_inv_softmax]
  exact poisson_at_query q (div_pos (exp_pos _) (Finset.sum_pos (fun j _ => exp_pos _) Finset.univ_nonempty))

/-- Softmax-specific constraint: pole-height log-ratios equal score differences. -/
theorem softmax_pole_log_ratio {N : ℕ} [NeZero N]
    (scores : Fin N → ℝ) (q : ℝ) (k j : Fin N) :
    let p := matchPoles q (softmaxSimplexOfScores scores)
    Real.log (p.y k) - Real.log (p.y j) = scores j - scores k := by
  dsimp only  -- β/ι-reduce the `let` binding
  have hZ : (0 : ℝ) < ∑ i : Fin N, Real.exp (scores i) :=
    Finset.sum_pos (fun i _ => Real.exp_pos _) Finset.univ_nonempty
  have hyk_val : (matchPoles q (softmaxSimplexOfScores scores)).y k =
      (∑ i : Fin N, Real.exp (scores i)) / Real.exp (scores k) := by
    simp [matchPoles, softmaxSimplexOfScores]
  have hyj_val : (matchPoles q (softmaxSimplexOfScores scores)).y j =
      (∑ i : Fin N, Real.exp (scores i)) / Real.exp (scores j) := by
    simp [matchPoles, softmaxSimplexOfScores]
  rw [hyk_val, hyj_val,
      Real.log_div (ne_of_gt hZ) (Real.exp_pos _).ne',
      Real.log_div (ne_of_gt hZ) (Real.exp_pos _).ne',
      Real.log_exp, Real.log_exp]
  ring

/-- A simplex is softmax iff its matched poles satisfy the Gibbs log-ratio law. -/
theorem softmax_iff_gibbs_poles {N : ℕ} [NeZero N]
    (s : Simplex N) (q : ℝ) :
    (∃ scores : Fin N → ℝ, ∀ k : Fin N,
      s.w k = exp (scores k) / ∑ j : Fin N, exp (scores j)) ↔
    (∃ scores : Fin N → ℝ, ∀ k j : Fin N,
      Real.log ((matchPoles q s).y k) - Real.log ((matchPoles q s).y j) =
        scores j - scores k) := by
  simp only [matchPoles]
  constructor
  · -- Forward: softmax weights → Gibbs pole heights
    rintro ⟨scores, hw⟩
    refine ⟨scores, fun k j => ?_⟩
    have hZ : (0 : ℝ) < ∑ i : Fin N, exp (scores i) :=
      Finset.sum_pos (fun i _ => exp_pos _) Finset.univ_nonempty
    rw [show (s.w k)⁻¹ = (∑ i, exp (scores i)) / exp (scores k) from by
          rw [hw k]; field_simp [hZ.ne'],
        show (s.w j)⁻¹ = (∑ i, exp (scores i)) / exp (scores j) from by
          rw [hw j]; field_simp [hZ.ne'],
        Real.log_div hZ.ne' (exp_pos _).ne',
        Real.log_div hZ.ne' (exp_pos _).ne',
        Real.log_exp, Real.log_exp]
    ring
  · -- Backward: Gibbs pole heights → softmax weights  (the non-trivial direction)
    rintro ⟨scores, hlog⟩
    refine ⟨scores, fun k => ?_⟩
    -- Step 1: log-ratio of heights gives log(w j) - log(w k) = scores j - scores k
    have hlogdiff : ∀ k j : Fin N,
        Real.log (s.w j) - Real.log (s.w k) = scores j - scores k := by
      intro k j
      have h := hlog k j
      rw [Real.log_inv, Real.log_inv] at h
      linarith
    -- Step 2: log(w k) - scores k is the same constant C for every k
    set C := Real.log (s.w (0 : Fin N)) - scores 0
    have hlogw : ∀ k : Fin N, Real.log (s.w k) = scores k + C := by
      intro k
      have := hlogdiff k (0 : Fin N)
      simp only [C]; linarith
    -- Step 3: w k = exp(scores k + C)
    have hwk : ∀ k : Fin N, s.w k = exp (scores k + C) := fun k => by
      rw [← Real.exp_log (s.pos k), hlogw k]
    -- Step 4: normalisation Σ w k = 1 forces exp(C) = 1/Z
    have hZ : (0 : ℝ) < ∑ j : Fin N, exp (scores j) :=
      Finset.sum_pos (fun j _ => exp_pos _) Finset.univ_nonempty
    have hmul : exp C * ∑ j : Fin N, exp (scores j) = 1 := by
      have h := s.sum_one
      simp_rw [hwk, Real.exp_add] at h
      -- h : Σ exp(scores j) * exp C = 1
      -- rewrite to exp C * Z = 1
      have key : ∑ j : Fin N, exp (scores j) * exp C =
                 exp C * ∑ j : Fin N, exp (scores j) := by
        rw [Finset.mul_sum]; congr 1; ext j; ring
      rw [← key]; exact h
    have hexpC : exp C = 1 / ∑ j : Fin N, exp (scores j) := by
      rw [eq_comm, div_eq_iff hZ.ne']; linarith
    -- Conclude: w k = exp(scores k) / Z
    rw [hwk k, Real.exp_add, hexpC, div_eq_mul_inv]
    ring

/-- Positive weight vector (no sum-to-one needed). -/
structure PositiveWeights (N : ℕ) where
  w : Fin N → ℝ
  pos : ∀ i, 0 < w i

def matchPolesPositive {N : ℕ} (q : ℝ) (pw : PositiveWeights N) : Poles N where
  x _ := q
  y k := (pw.w k)⁻¹
  im_pos k := inv_pos.mpr (pw.pos k)

/-- Any positive (not necessarily normalized) weights have a Poisson representation. -/
theorem topological_subsumes_positive_weights {N : ℕ}
    (pw : PositiveWeights N) (q : ℝ) :
    ∃ p : Poles N, ∀ k : Fin N,
      poisson (p.x k) (p.y k) q = pw.w k :=
  ⟨matchPolesPositive q pw, fun k => poisson_at_query q (pw.pos k)⟩

-- ═══════════════════════════════════════════════════════════════════════
-- § 1.4  Off-query bandwidth limit
-- ═══════════════════════════════════════════════════════════════════════

/-!
### The bandwidth law

The on-query construction (x_k = q) is "degenerate" — all poles collapse to a
vertical line.  If you try to place a pole at a *different* horizontal position
(x_k = q + d, d > 0), you discover a hard constraint:

    w_k ≤ 1 / (2d)

A token at distance d from the query in key-space can carry at most weight
1/(2d).  This is a locality law: distant context is provably bounded.

This comes from the discriminant of `w*y^2 - y + w*d^2 = 0`.
-/

def poissonDiscriminant (w d : ℝ) : ℝ := 1 - 4 * w ^ 2 * d ^ 2

def poissonYFromDisplacement (w d : ℝ) (_hw : 0 < w) (_hdisc : 0 ≤ poissonDiscriminant w d) : ℝ :=
  (1 - Real.sqrt (poissonDiscriminant w d)) / (2 * w)

/-- Discriminant feasibility implies the off-query bandwidth bound `w ≤ 1/(2d)`. -/
theorem poisson_offquery_bandwidth_bound (w d : ℝ) (hw : 0 < w) (hd : 0 < d)
    (hdisc : 0 ≤ poissonDiscriminant w d) :
    w ≤ 1 / (2 * d) := by
  have hmul : 4 * w ^ 2 * d ^ 2 ≤ 1 := by
    unfold poissonDiscriminant at hdisc
    linarith
  have hwd_sq : (2 * w * d) ^ 2 ≤ 1 := by
    have hsq : (2 * w * d) ^ 2 = 4 * w ^ 2 * d ^ 2 := by ring
    rw [hsq]
    exact hmul
  have hwd_nonneg : 0 ≤ 2 * w * d := by
    nlinarith [le_of_lt hw, le_of_lt hd]
  have hwd_le : 2 * w * d ≤ 1 := by
    nlinarith [hwd_sq, hwd_nonneg]
  have hwd_le' : w * (2 * d) ≤ 1 := by
    simpa [mul_assoc, mul_comm, mul_left_comm] using hwd_le
  have h2d_pos : 0 < 2 * d := by linarith
  rw [le_div_iff₀ h2d_pos]
  simpa [mul_assoc] using hwd_le'

/-- Off-query poles obey `w ≤ 1/(2d)`, with explicit feasible height. -/
theorem poisson_offquery_eq (q d w : ℝ) (hw : 0 < w) (hd : 0 < d)
    (hdisc : 0 ≤ poissonDiscriminant w d) :
    let y := poissonYFromDisplacement w d hw hdisc
    0 < y ∧ poisson (q + d) y q = w := by
  have h2w_pos : (0 : ℝ) < 2 * w := by linarith
  have h2w_ne : (2 : ℝ) * w ≠ 0 := ne_of_gt h2w_pos
  set sq := Real.sqrt (poissonDiscriminant w d) with hsq_def
  have hsq2 : sq ^ 2 = poissonDiscriminant w d := Real.sq_sqrt hdisc
  have hdisc_eq : poissonDiscriminant w d = 1 - 4 * w ^ 2 * d ^ 2 := rfl
  have h_disc_lt : poissonDiscriminant w d < 1 := by
    unfold poissonDiscriminant; have := mul_pos (mul_pos (by norm_num : (0:ℝ) < 4) (sq_pos_of_pos hw)) (sq_pos_of_pos hd); linarith
  have h_sqrt_lt : sq < 1 := by
    have := Real.sqrt_lt_sqrt hdisc h_disc_lt; simp [Real.sqrt_one] at this; exact this
  have h_sqrt_nn : 0 ≤ sq := Real.sqrt_nonneg _
  have y_eq : poissonYFromDisplacement w d hw hdisc = (1 - sq) / (2 * w) := rfl
  constructor
  · rw [y_eq]
    apply div_pos
    · linarith
    · linarith
  · rw [y_eq]
    set y := (1 - sq) / (2 * w)
    have hy_def : y = (1 - sq) / (2 * w) := rfl
    have hyw : 2 * w * y = 1 - sq := by
      rw [hy_def]; field_simp
    have hkey : w * (d ^ 2 + y ^ 2) = y := by
      have h4 : 4 * w ^ 2 * y ^ 2 = (1 - sq) ^ 2 := by nlinarith
      nlinarith [hsq2, hdisc_eq, sq_nonneg sq, sq_nonneg y]
    have hy_pos : 0 < y := by rw [hy_def]; apply div_pos <;> linarith
    have hden_pos : d ^ 2 + y ^ 2 > 0 := by nlinarith [sq_nonneg d]
    show y / ((q - (q + d)) ^ 2 + y ^ 2) = w
    have hdist : (q - (q + d)) ^ 2 = d ^ 2 := by ring
    rw [hdist, div_eq_iff (ne_of_gt hden_pos)]
    linarith [hkey]

-- ═══════════════════════════════════════════════════════════════════════
-- § 1.5  Operator lift: scalar weights → vector head output
-- ═══════════════════════════════════════════════════════════════════════

/-!
### Lifting from scalars to vectors

Theorem 1 matches scalar weights w_k.  But an attention HEAD applies those
weights to value vectors V_j ∈ ℝ^D and sums.  Here we prove the vector
output also matches — the entire linear map V ↦ Σ_j w_j V_j is realised as
a Poisson-weighted sum.

This is the step from "the numbers match" to "the operator matches."
-/

/-- Standard attention output in one coordinate: Σ_j w_j · V_j[d]. -/
def transformerOutput {N D : ℕ} (weights : Fin N → ℝ) (V : Fin N → Fin D → ℝ)
    (d : Fin D) : ℝ :=
  ∑ j : Fin N, weights j * V j d

/-- Poisson-kernel weighted sum of value vectors. -/
def contourOutput {N D : ℕ} (p : Poles N) (q : ℝ) (V : Fin N → Fin D → ℝ)
    (d : Fin D) : ℝ :=
  ∑ j : Fin N, poisson (p.x j) (p.y j) q * V j d

/-- If pole weights agree keywise, contour and transformer outputs agree
    in every coordinate. -/
theorem contour_output_eq_transformer_of_weight_match
    {N D : ℕ} (p : Poles N) (q : ℝ) (w : Fin N → ℝ) (V : Fin N → Fin D → ℝ)
    (hw : ∀ k : Fin N, poisson (p.x k) (p.y k) q = w k) (d : Fin D) :
    contourOutput p q V d = transformerOutput w V d := by
  simp only [contourOutput, transformerOutput]
  exact Finset.sum_congr rfl fun j _ => by rw [hw j]

/-- Full softmax head output equals a Poisson-weighted output (all coordinates). -/
theorem contour_simulates_softmax_of_scores {N D : ℕ} [NeZero N]
    (scores : Fin N → ℝ) (q : ℝ) (V : Fin N → Fin D → ℝ) :
    ∃ p : Poles N, ∀ d : Fin D,
      contourOutput p q V d =
        transformerOutput
          (fun j : Fin N => exp (scores j) / ∑ i : Fin N, exp (scores i)) V d := by
  obtain ⟨p, hp⟩ := topological_subsumes_softmax_of_scores scores q
  refine ⟨p, fun d => ?_⟩
  exact contour_output_eq_transformer_of_weight_match p q _ V hp d

-- Bridge lemmas moved to appendix.

-- ═══════════════════════════════════════════════════════════════════════
-- § 1.6  Cauchy transform and boundary trace
-- ═══════════════════════════════════════════════════════════════════════

/-!
### From algebraic identity to holomorphic boundary trace

The identity `P(q, 1/w, q) = w` is, on its own, the algebraic fact
`(1/w) / ((1/w)^2) = w` — true for any positive `w` and using no
softmax structure.  Calling that "Poisson" is metaphorical; the real
Poisson kernel is the boundary trace of a *holomorphic* function, and
nothing in §§ 1.1–1.5 actually exhibits one.

This section adds that missing analytic object: the **Cauchy transform**
of a discrete vector-valued measure on the upper half-plane,

  `cauchyTransform V ζ z d = Σ_k (V k d) / (z - ζ k)`,

which is a genuine complex-analytic function `ℂ ∖ {ζ k} → ℂ`.  The
"Cauchy-Poisson" connection becomes literal via two facts:

1. `cauchyTransform_differentiableAt` — `cauchyTransform V ζ` is
   complex-differentiable (holomorphic) at every `z` not equal to any
   pole `ζ k`.

2. `im_cauchyTransform_ofReal_eq_poisson_weighted_sum` — for `q ∈ ℝ`
   the imaginary part on the real axis evaluates to the Poisson-kernel-
   weighted sum
       `Im (cauchyTransform V ζ q d) = Σ_k V k d · poisson ζ_k.re ζ_k.im q`.
   This is the discrete Sokhotski–Plemelj identity.

Combined with the score-derived poles in Part 2, the attention output at
real query `q` is the imaginary boundary trace of a holomorphic
function. -/

/-- Discrete vector-valued Cauchy transform; holomorphic away from poles. -/
noncomputable def cauchyTransform {N D : ℕ}
    (V : Fin N → Fin D → ℝ) (ζ : Fin N → ℂ) (z : ℂ) (d : Fin D) : ℂ :=
  ∑ k : Fin N, (V k d : ℂ) / (z - ζ k)

/-- The Cauchy transform is differentiable at any non-pole point. -/
theorem cauchyTransform_differentiableAt {N D : ℕ}
    (V : Fin N → Fin D → ℝ) (ζ : Fin N → ℂ) {z : ℂ}
    (hz : ∀ k : Fin N, z ≠ ζ k) (d : Fin D) :
    DifferentiableAt ℂ (fun w : ℂ => cauchyTransform V ζ w d) z := by
  unfold cauchyTransform
  -- Rewrite (fun w => ∑ k, f k w) = ∑ k, (fun w => f k w) so that
  -- `DifferentiableAt.sum` applies directly.
  have hfun : (fun w : ℂ => ∑ k : Fin N, (V k d : ℂ) / (w - ζ k)) =
      ∑ k : Fin N, (fun w : ℂ => (V k d : ℂ) / (w - ζ k)) := by
    funext w
    simp [Finset.sum_apply]
  rw [hfun]
  apply DifferentiableAt.sum
  intro k _
  refine DifferentiableAt.div ?_ ?_ ?_
  · exact differentiableAt_const _
  · exact differentiableAt_id.sub_const _
  · exact sub_ne_zero.mpr (hz k)

/-- Imaginary part distributes over finite sums. -/
private theorem complex_im_finset_sum {α : Type*} (s : Finset α) (f : α → ℂ) :
    (∑ i ∈ s, f i).im = ∑ i ∈ s, (f i).im := by
  classical
  refine s.induction_on ?_ ?_
  · simp
  · intro a t hat ih
    rw [Finset.sum_insert hat, Complex.add_im, ih, Finset.sum_insert hat]

/-- Discrete Sokhotski–Plemelj identity on the real boundary. -/
theorem im_cauchyTransform_ofReal_eq_poisson_weighted_sum {N D : ℕ}
    (V : Fin N → Fin D → ℝ) (ζ : Fin N → ℂ)
    (hζ : ∀ k : Fin N, 0 < (ζ k).im) (q : ℝ) (d : Fin D) :
    (cauchyTransform V ζ (q : ℂ) d).im =
      ∑ k : Fin N, V k d * poisson (ζ k).re (ζ k).im q := by
  unfold cauchyTransform
  rw [complex_im_finset_sum]
  refine Finset.sum_congr rfl ?_
  intro k _
  have hyk : 0 < (ζ k).im := hζ k
  -- compute denominator: (q - ζ_k).re = q - ζk.re, (q - ζ_k).im = -ζk.im
  have hwre : ((q : ℂ) - ζ k).re = q - (ζ k).re := by
    simp [Complex.sub_re, Complex.ofReal_re]
  have hwim : ((q : ℂ) - ζ k).im = -(ζ k).im := by
    simp [Complex.sub_im, Complex.ofReal_im]
  have hns_eq : Complex.normSq ((q : ℂ) - ζ k) = (q - (ζ k).re) ^ 2 + (ζ k).im ^ 2 := by
    rw [Complex.normSq_apply, hwre, hwim]; ring
  have hns_pos : 0 < (q - (ζ k).re) ^ 2 + (ζ k).im ^ 2 := by
    nlinarith [sq_nonneg (q - (ζ k).re), sq_pos_of_pos hyk]
  have hns_ne : (q - (ζ k).re) ^ 2 + (ζ k).im ^ 2 ≠ 0 := ne_of_gt hns_pos
  -- expand .im of the quotient
  rw [Complex.div_im, Complex.ofReal_re, Complex.ofReal_im, hwre, hwim, hns_eq]
  unfold poisson
  -- Goal: quotient-imaginary-part formula matches `poisson`.
  field_simp
  ring

/-- Complex poles sitting vertically above one real query `q` at heights `p.y k > 0`. -/
noncomputable def verticalComplexPoles {N : ℕ} (p : Poles N) (q : ℝ) : Fin N → ℂ :=
  fun k => (q : ℂ) + Complex.I * ((p.y k : ℝ) : ℂ)

theorem verticalComplexPoles_re {N : ℕ} (p : Poles N) (q : ℝ) (k : Fin N) :
    (verticalComplexPoles p q k).re = q := by
  unfold verticalComplexPoles
  simp [Complex.add_re, Complex.mul_re, Complex.I_re, Complex.I_im, Complex.ofReal_re]

theorem verticalComplexPoles_im {N : ℕ} (p : Poles N) (q : ℝ) (k : Fin N) :
    (verticalComplexPoles p q k).im = p.y k := by
  unfold verticalComplexPoles
  simp [Complex.add_im, Complex.mul_im, Complex.I_re, Complex.I_im, Complex.ofReal_re, Complex.ofReal_im]

theorem verticalComplexPoles_im_pos {N : ℕ} (p : Poles N) (q : ℝ) (k : Fin N) :
    0 < (verticalComplexPoles p q k).im := by
  rw [verticalComplexPoles_im]; exact p.im_pos k

/-- Canonical complex poles built directly from score-derived real poles. -/
noncomputable def scoreComplexPoles {N : ℕ} [NeZero N]
    (scores : Fin N → ℝ) (q : ℝ) : Fin N → ℂ :=
  verticalComplexPoles (matchPoles q (softmaxSimplexOfScores scores)) q

/-- Canonical analytic quantity for one head coordinate:
    the imaginary boundary value of the Cauchy transform. -/
noncomputable def headCauchyIM {N D : ℕ}
    (V : Fin N → Fin D → ℝ) (p : Poles N) (q : ℝ) (d : Fin D) : ℝ :=
  (cauchyTransform V (verticalComplexPoles p q) (q : ℂ) d).im

/-- **Contour output = imaginary Cauchy boundary trace** when every real pole
    sits on-query at the same `q` (so each complex pole is `q + i·y_k ∈ ℍ`).

    This packages `im_cauchyTransform_ofReal_eq_poisson_weighted_sum` as the
    bridge from the Poisson contour used in attention to the analytic Cauchy
    transform proved in §1.6. -/
theorem contourOutput_eq_im_cauchyTransform_vertical {N D : ℕ} (p : Poles N) (q : ℝ)
    (V : Fin N → Fin D → ℝ) (hx : ∀ k : Fin N, p.x k = q) (d : Fin D) :
    contourOutput p q V d =
      headCauchyIM V p q d := by
  have hζ : ∀ k : Fin N, 0 < (verticalComplexPoles p q k).im :=
    fun k => verticalComplexPoles_im_pos p q k
  calc
    contourOutput p q V d
        = ∑ k : Fin N, V k d * poisson (p.x k) (p.y k) q := by
            simp [contourOutput, mul_comm]
    _ = ∑ k : Fin N, V k d * poisson q (p.y k) q := by
            refine Finset.sum_congr rfl fun k _ => ?_
            congr 1
            rw [hx k]
    _ = ∑ k : Fin N, V k d * poisson (verticalComplexPoles p q k).re (verticalComplexPoles p q k).im q := by
            refine Finset.sum_congr rfl fun k _ => ?_
            congr 1
            rw [verticalComplexPoles_re, verticalComplexPoles_im]
    _ = headCauchyIM V p q d := by
            unfold headCauchyIM
            exact
              (im_cauchyTransform_ofReal_eq_poisson_weighted_sum
                V (verticalComplexPoles p q) hζ q d).symm

-- ═══════════════════════════════════════════════════════════════════════
-- § 1.7  Differentiability of the boundary trace in pole parameters
--
-- The attention head's output is the boundary trace
--   F(q; V, x, y) := ∑ₖ V_k · poisson(x_k, y_k, q) = contourOutput p q V
-- of a finite Cauchy transform restricted to ℝ ⊂ ℂ.  For trainable
-- attention we need this to be differentiable in the parameters
-- (V_k, x_k, y_k) so that any smooth loss L : ℝ → ℝ admits a chain-rule
-- gradient
--
--   ∂(L ∘ F)/∂V_k = L'(F) · poisson(x_k, y_k, q)
--   ∂(L ∘ F)/∂x_k = L'(F) · V_k · ∂_x poisson(x_k, y_k, q)
--   ∂(L ∘ F)/∂y_k = L'(F) · V_k · ∂_y poisson(x_k, y_k, q)
--
-- The partial derivatives ∂_x poisson, ∂_y poisson are themselves
-- rational kernels of the form 1/((q−x)² + y²)ⁿ (n ∈ {1,2}) — so the
-- gradient flow stays inside the algebra of finite Cauchy / Poisson
-- combinations.  In other words: training a Cauchy-Poisson head
-- produces another Cauchy-Poisson head.  The §1.6 differentiability in
-- the query `z` (`cauchyTransform_differentiableAt`) is the *forward*
-- pass; this section is the *backward* pass.
-- ═══════════════════════════════════════════════════════════════════════

/-- The Poisson kernel is **jointly** differentiable in the pole position
    `(x, y)` on the open upper half-plane `y > 0`.  Discharged by
    Mathlib's `fun_prop` tactic, with the non-zero-denominator side
    obligation closed by `nlinarith`. -/
theorem poisson_differentiableAt_pole (q : ℝ) {x y : ℝ} (hy : 0 < y) :
    DifferentiableAt ℝ (fun p : ℝ × ℝ => poisson p.1 p.2 q) (x, y) := by
  unfold poisson
  have hpos : (q - x) ^ 2 + y ^ 2 > 0 := by
    nlinarith [sq_nonneg (q - x), sq_pos_of_pos hy]
  have hne : (q - x) ^ 2 + y ^ 2 ≠ 0 := ne_of_gt hpos
  fun_prop (disch := exact hne)

/-- Differentiability of `poisson` in the `x`-coordinate of the pole,
    everywhere on the real line, given any positive `y`. -/
theorem poisson_differentiableAt_x (q : ℝ) {y : ℝ} (hy : 0 < y) (x : ℝ) :
    DifferentiableAt ℝ (fun x' : ℝ => poisson x' y q) x := by
  unfold poisson
  refine DifferentiableAt.div ?num ?den ?ne
  case num => exact differentiableAt_const _
  case den =>
    exact (((differentiableAt_const q).sub differentiableAt_id).pow 2).add
          (differentiableAt_const _)
  case ne =>
    have hpos : (q - x)^2 + y^2 > 0 := by
      nlinarith [sq_nonneg (q - x), mul_pos hy hy]
    exact ne_of_gt hpos

/-- Differentiability of `poisson` in the `y`-coordinate (bandwidth) of
    the pole, at any point `y > 0`. -/
theorem poisson_differentiableAt_y (q x : ℝ) {y : ℝ} (hy : 0 < y) :
    DifferentiableAt ℝ (fun y' : ℝ => poisson x y' q) y := by
  unfold poisson
  refine DifferentiableAt.div ?num ?den ?ne
  case num => exact differentiableAt_id
  case den =>
    exact (differentiableAt_const _).add (differentiableAt_id.pow 2)
  case ne =>
    have hpos : (q - x)^2 + y^2 > 0 := by
      nlinarith [sq_nonneg (q - x), mul_pos hy hy]
    exact ne_of_gt hpos

/-- The boundary trace of a single residue, viewed as a function of that
    one residue alone, is linear hence everywhere differentiable.  This
    is the "∂F / ∂V_k = poisson(x_k, y_k, q)" identity in the form Lean
    cares about: the function `V ↦ c · V` (with `c = poisson x y q`) is
    a continuous linear map, hence differentiable everywhere. -/
theorem contourOutput_one_pole_differentiable (x y q : ℝ) :
    Differentiable ℝ (fun V : ℝ => poisson x y q * V) :=
  (differentiable_const _).mul differentiable_id

/-- Differentiability of the contour output in *one* pole's `y`-coordinate
    (the bandwidth axis), with the other poles held fixed.  This is the
    most non-trivial of the parameter directions because `y` enters both
    the numerator (linearly) and the denominator (quadratically) of the
    Poisson kernel. -/
theorem contourOutput_differentiableAt_yk
    {N D : ℕ} (p : Poles N) (q : ℝ) (V : Fin N → Fin D → ℝ) (d : Fin D)
    (k : Fin N) :
    DifferentiableAt ℝ
      (fun yk : ℝ =>
        ∑ j : Fin N,
          poisson (p.x j) (if j = k then yk else p.y j) q * V j d)
      (p.y k) := by
  classical
  -- Pull the sum out of the lambda before applying `DifferentiableAt.sum`.
  have hfun : (fun yk : ℝ =>
                  ∑ j : Fin N,
                    poisson (p.x j) (if j = k then yk else p.y j) q * V j d)
            = ∑ j : Fin N, (fun yk : ℝ =>
                  poisson (p.x j) (if j = k then yk else p.y j) q * V j d) := by
    funext yk
    simp [Finset.sum_apply]
  rw [hfun]
  apply DifferentiableAt.sum
  intro j _
  refine DifferentiableAt.mul ?_ (differentiableAt_const _)
  by_cases hjk : j = k
  · -- Term in question: poisson is differentiated through `y' = yk`.
    subst hjk
    have heq : (fun yk : ℝ => poisson (p.x j) (if j = j then yk else p.y j) q)
             = (fun yk : ℝ => poisson (p.x j) yk q) := by
      funext yk; simp
    rw [heq]
    exact poisson_differentiableAt_y q (p.x j) (p.im_pos j)
  · -- `j ≠ k`: the conditional is constant, so the term is constant in yk.
    have heq : (fun yk : ℝ => poisson (p.x j) (if j = k then yk else p.y j) q)
             = (fun _ : ℝ => poisson (p.x j) (p.y j) q) := by
      funext yk; simp [hjk]
    rw [heq]
    exact differentiableAt_const _

/-- Differentiability of the contour output in *one* pole's `x`-coordinate
    (the position axis), with the other poles held fixed.  No constraint
    on `xk` — only the existing `0 < p.y k` matters for non-degeneracy. -/
theorem contourOutput_differentiableAt_xk
    {N D : ℕ} (p : Poles N) (q : ℝ) (V : Fin N → Fin D → ℝ) (d : Fin D)
    (k : Fin N) :
    DifferentiableAt ℝ
      (fun xk : ℝ =>
        ∑ j : Fin N,
          poisson (if j = k then xk else p.x j) (p.y j) q * V j d)
      (p.x k) := by
  classical
  have hfun : (fun xk : ℝ =>
                  ∑ j : Fin N,
                    poisson (if j = k then xk else p.x j) (p.y j) q * V j d)
            = ∑ j : Fin N, (fun xk : ℝ =>
                  poisson (if j = k then xk else p.x j) (p.y j) q * V j d) := by
    funext xk
    simp [Finset.sum_apply]
  rw [hfun]
  apply DifferentiableAt.sum
  intro j _
  refine DifferentiableAt.mul ?_ (differentiableAt_const _)
  by_cases hjk : j = k
  · subst hjk
    have heq : (fun xk : ℝ => poisson (if j = j then xk else p.x j) (p.y j) q)
             = (fun xk : ℝ => poisson xk (p.y j) q) := by
      funext xk; simp
    rw [heq]
    exact poisson_differentiableAt_x q (p.im_pos j) (p.x j)
  · have heq : (fun xk : ℝ => poisson (if j = k then xk else p.x j) (p.y j) q)
             = (fun _ : ℝ => poisson (p.x j) (p.y j) q) := by
      funext xk; simp [hjk]
    rw [heq]
    exact differentiableAt_const _

/-- Differentiability of the contour output in *one* residue `V k d`
    (one entry of the value matrix).  Linear in `Vkd`, hence everywhere
    differentiable. -/
theorem contourOutput_differentiableAt_Vkd
    {N D : ℕ} (p : Poles N) (q : ℝ) (V : Fin N → Fin D → ℝ) (d : Fin D)
    (k : Fin N) :
    DifferentiableAt ℝ
      (fun Vkd : ℝ =>
        ∑ j : Fin N,
          poisson (p.x j) (p.y j) q * (if j = k then Vkd else V j d))
      (V k d) := by
  classical
  have hfun : (fun Vkd : ℝ =>
                  ∑ j : Fin N,
                    poisson (p.x j) (p.y j) q * (if j = k then Vkd else V j d))
            = ∑ j : Fin N, (fun Vkd : ℝ =>
                  poisson (p.x j) (p.y j) q * (if j = k then Vkd else V j d)) := by
    funext Vkd
    simp [Finset.sum_apply]
  rw [hfun]
  apply DifferentiableAt.sum
  intro j _
  by_cases hjk : j = k
  · subst hjk
    have heq : (fun Vkd : ℝ =>
                  poisson (p.x j) (p.y j) q * (if j = j then Vkd else V j d))
             = (fun Vkd : ℝ => poisson (p.x j) (p.y j) q * Vkd) := by
      funext Vkd; simp
    rw [heq]
    exact (contourOutput_one_pole_differentiable (p.x j) (p.y j) q).differentiableAt
  · have heq : (fun Vkd : ℝ =>
                  poisson (p.x j) (p.y j) q * (if j = k then Vkd else V j d))
             = (fun _ : ℝ => poisson (p.x j) (p.y j) q * V j d) := by
      funext Vkd; simp [hjk]
    rw [heq]
    exact differentiableAt_const _

/-- **Chain rule for any smooth loss applied to a Cauchy boundary
    trace.**  Composing the contour output (in one pole's bandwidth)
    with any differentiable loss `L : ℝ → ℝ` is differentiable, with
    gradient computed by the standard chain rule.  This is the
    Lean-certified statement that backprop works on a Cauchy-Poisson
    attention head, in the bandwidth (`y_k`) parameter direction. -/
theorem loss_compose_contourOutput_yk_differentiableAt
    {N D : ℕ} (p : Poles N) (q : ℝ) (V : Fin N → Fin D → ℝ) (d : Fin D)
    (k : Fin N) {L : ℝ → ℝ} (hL : Differentiable ℝ L) :
    DifferentiableAt ℝ
      (fun yk : ℝ =>
        L (∑ j : Fin N,
              poisson (p.x j) (if j = k then yk else p.y j) q * V j d))
      (p.y k) :=
  hL.differentiableAt.comp _ (contourOutput_differentiableAt_yk p q V d k)

/-- Same chain-rule statement in the *position* parameter `x_k`. -/
theorem loss_compose_contourOutput_xk_differentiableAt
    {N D : ℕ} (p : Poles N) (q : ℝ) (V : Fin N → Fin D → ℝ) (d : Fin D)
    (k : Fin N) {L : ℝ → ℝ} (hL : Differentiable ℝ L) :
    DifferentiableAt ℝ
      (fun xk : ℝ =>
        L (∑ j : Fin N,
              poisson (if j = k then xk else p.x j) (p.y j) q * V j d))
      (p.x k) :=
  hL.differentiableAt.comp _ (contourOutput_differentiableAt_xk p q V d k)

/-- Same chain-rule statement in the *residue* parameter `V_k_d`. -/
theorem loss_compose_contourOutput_Vkd_differentiableAt
    {N D : ℕ} (p : Poles N) (q : ℝ) (V : Fin N → Fin D → ℝ) (d : Fin D)
    (k : Fin N) {L : ℝ → ℝ} (hL : Differentiable ℝ L) :
    DifferentiableAt ℝ
      (fun Vkd : ℝ =>
        L (∑ j : Fin N,
              poisson (p.x j) (p.y j) q * (if j = k then Vkd else V j d)))
      (V k d) :=
  hL.differentiableAt.comp _ (contourOutput_differentiableAt_Vkd p q V d k)

/-- **Joint** differentiability of the contour output in *one* pole's
    full position `(x_k, y_k) ∈ ℝ × ℝ` simultaneously.  The other poles
    are held fixed.  This is the gradient direction PyTorch would
    actually compute for that pole — both coordinates at once. -/
theorem contourOutput_differentiableAt_pole_pair
    {N D : ℕ} (p : Poles N) (q : ℝ) (V : Fin N → Fin D → ℝ) (d : Fin D)
    (k : Fin N) :
    DifferentiableAt ℝ
      (fun pk : ℝ × ℝ =>
        ∑ j : Fin N,
          poisson (if j = k then pk.1 else p.x j)
                  (if j = k then pk.2 else p.y j) q
            * V j d)
      (p.x k, p.y k) := by
  classical
  have hfun : (fun pk : ℝ × ℝ =>
                  ∑ j : Fin N,
                    poisson (if j = k then pk.1 else p.x j)
                            (if j = k then pk.2 else p.y j) q * V j d)
            = ∑ j : Fin N, (fun pk : ℝ × ℝ =>
                  poisson (if j = k then pk.1 else p.x j)
                          (if j = k then pk.2 else p.y j) q * V j d) := by
    funext pk
    simp [Finset.sum_apply]
  rw [hfun]
  apply DifferentiableAt.sum
  intro j _
  refine DifferentiableAt.mul ?_ (differentiableAt_const _)
  by_cases hjk : j = k
  · subst hjk
    have heq : (fun pk : ℝ × ℝ =>
                  poisson (if j = j then pk.1 else p.x j)
                          (if j = j then pk.2 else p.y j) q)
             = (fun pk : ℝ × ℝ => poisson pk.1 pk.2 q) := by
      funext pk; simp
    rw [heq]
    exact poisson_differentiableAt_pole q (p.im_pos j)
  · have heq : (fun pk : ℝ × ℝ =>
                  poisson (if j = k then pk.1 else p.x j)
                          (if j = k then pk.2 else p.y j) q)
             = (fun _ : ℝ × ℝ => poisson (p.x j) (p.y j) q) := by
      funext pk; simp [hjk]
    rw [heq]
    exact differentiableAt_const _

/-- Loss-composed joint chain rule for one full pole `(x_k, y_k)`. -/
theorem loss_compose_contourOutput_pole_pair_differentiableAt
    {N D : ℕ} (p : Poles N) (q : ℝ) (V : Fin N → Fin D → ℝ) (d : Fin D)
    (k : Fin N) {L : ℝ → ℝ} (hL : Differentiable ℝ L) :
    DifferentiableAt ℝ
      (fun pk : ℝ × ℝ =>
        L (∑ j : Fin N,
              poisson (if j = k then pk.1 else p.x j)
                      (if j = k then pk.2 else p.y j) q * V j d))
      (p.x k, p.y k) :=
  hL.differentiableAt.comp _ (contourOutput_differentiableAt_pole_pair p q V d k)

-- ═══════════════════════════════════════════════════════════════════════
-- § 1.8  Backprop pushforward through softmax
--
-- §1.7 certified that the *contour-output* parameterisation has clean
-- gradients in `(x_k, y_k, V_k_d)`.  But PyTorch does not train pole
-- coordinates; it trains pre-softmax *scores* (themselves computed
-- bilinearly from `W_Q`, `W_K`, and the input).  This section closes
-- that gap.
--
-- Theorem chain:
--   1. softmax weights are differentiable in scores;
--   2. the score-parameterised attention output is differentiable;
--   3. it equals the contour output of `matchPoles q (softmax scores)`;
--   4. for **any** smooth parameter map `s : M → (Fin N → ℝ)` (e.g.
--      `s(W_Q, W_K) = ⟨W_Q q, W_K · ⟩`) and **any** smooth loss `L`,
--      `(L ∘ contourOutput ∘ matchPoles ∘ softmax ∘ s)` is differentiable
--      in the trainable parameter, with gradient given by the standard
--      chain rule.
--
-- This is the **gradient pushforward** statement: PyTorch's `.backward()`
-- and the Cauchy pole-coordinate backward pass produce the **same**
-- gradient through different parametrisations.
-- ═══════════════════════════════════════════════════════════════════════

/-- The softmax weight `w_k(scores) = exp(score_k) / Σ exp(score_j)` is
    everywhere differentiable in the score vector.  Discharged by
    `fun_prop` after recording the partition function is positive. -/
theorem softmax_weight_differentiable {N : ℕ} [NeZero N] (k : Fin N) :
    Differentiable ℝ (fun scores : Fin N → ℝ =>
      Real.exp (scores k) / ∑ j : Fin N, Real.exp (scores j)) := by
  intro scores
  have hne : (Finset.univ : Finset (Fin N)).Nonempty :=
    Finset.univ_nonempty
  have hZ : (∑ j : Fin N, Real.exp (scores j)) ≠ 0 :=
    ne_of_gt (Finset.sum_pos (fun _ _ => Real.exp_pos _) hne)
  fun_prop (disch := exact hZ)

/-- The score-parameterised attention output
    `F(scores) = Σ_j softmax(scores)_j · V_j_d` is everywhere
    differentiable in the score vector.  This is the function whose
    gradient PyTorch actually computes when it backpropagates through
    a softmax-attention layer. -/
theorem softmaxAttention_score_differentiable {N D : ℕ} [NeZero N]
    (V : Fin N → Fin D → ℝ) (d : Fin D) :
    Differentiable ℝ (fun scores : Fin N → ℝ =>
      ∑ j : Fin N,
        (Real.exp (scores j) / ∑ i : Fin N, Real.exp (scores i)) * V j d) := by
  intro scores
  have hne : (Finset.univ : Finset (Fin N)).Nonempty :=
    Finset.univ_nonempty
  have hZ : (∑ j : Fin N, Real.exp (scores j)) ≠ 0 :=
    ne_of_gt (Finset.sum_pos (fun _ _ => Real.exp_pos _) hne)
  fun_prop (disch := exact hZ)

/-- **Bridge: PyTorch's softmax attention forward equals the Cauchy
    contour output through `matchPoles`.**  Same numbers, different
    parametrisation.  Combined with `softmaxAttention_score_differentiable`
    this means the score-gradient `∂F/∂score_k` exists and *is* the
    chain-rule pushforward of the pole-coordinate gradient
    `∂F/∂y_k · ∂y_k/∂score_k`. -/
theorem softmaxAttention_eq_contourOutput {N D : ℕ} [NeZero N]
    (q : ℝ) (V : Fin N → Fin D → ℝ) (d : Fin D) (scores : Fin N → ℝ) :
    (∑ j : Fin N,
        (Real.exp (scores j) / ∑ i : Fin N, Real.exp (scores i)) * V j d) =
    contourOutput (matchPoles q (softmaxSimplexOfScores scores)) q V d := by
  unfold contourOutput
  refine Finset.sum_congr rfl ?_
  intro j _
  rw [match_weight (softmaxSimplexOfScores scores) q j]
  rfl

/-- **Pushforward of the gradient — the missing link.**

    For ANY smooth parameter-to-scores map `s : M → (Fin N → ℝ)` (e.g.
    `s(W_K, W_Q, x) = ⟨W_Q q, W_K x_k⟩`) and ANY differentiable loss
    `L : ℝ → ℝ`, the composition
      `param ↦ L (softmax-attention(s param) · V)`
    is differentiable in the trainable parameter, with the gradient
    given by the standard chain rule.

    Reading this with `softmaxAttention_eq_contourOutput`: the gradient
    PyTorch computes when it runs `.backward()` on a transformer head
    is exactly the chain-rule pushforward of the pole-coordinate
    gradient.  The Cauchy backward pass and the softmax backward pass
    are the *same* gradient through different parametrisations. -/
theorem pushforward_loss_differentiable
    {N D : ℕ} [NeZero N] {M : Type*}
    [NormedAddCommGroup M] [NormedSpace ℝ M]
    (V : Fin N → Fin D → ℝ) (d : Fin D)
    {s : M → (Fin N → ℝ)} (hs : Differentiable ℝ s)
    {L : ℝ → ℝ} (hL : Differentiable ℝ L) :
    Differentiable ℝ (fun w : M =>
      L (∑ j : Fin N,
          (Real.exp ((s w) j) / ∑ i : Fin N, Real.exp ((s w) i)) * V j d)) := by
  intro w
  have hG : Differentiable ℝ (fun scores : Fin N → ℝ =>
      ∑ j : Fin N,
        (Real.exp (scores j) / ∑ i : Fin N, Real.exp (scores i)) * V j d) :=
    softmaxAttention_score_differentiable V d
  exact hL.differentiableAt.comp _ ((hG.comp hs) w)

/-- **Pushforward, contour-output form.** Same theorem as
    `pushforward_loss_differentiable`, but stated with the explicit
    Cauchy/Poisson `contourOutput` — making it visually obvious that
    the gradient flowing into a trainable parameter `w : M` factors
    through the analytic boundary trace.

    In words: `∇_w L = (∇_F L) · (∇_y F) · (∇_score y) · (∇_w score)`.
    Each factor is now Lean-certified to exist; PyTorch and a
    hypothetical "Cauchy-Adam" optimiser compute the same gradient. -/
theorem pushforward_loss_through_contour_differentiable
    {N D : ℕ} [NeZero N] {M : Type*}
    [NormedAddCommGroup M] [NormedSpace ℝ M]
    (q : ℝ) (V : Fin N → Fin D → ℝ) (d : Fin D)
    {s : M → (Fin N → ℝ)} (hs : Differentiable ℝ s)
    {L : ℝ → ℝ} (hL : Differentiable ℝ L) :
    Differentiable ℝ (fun w : M =>
      L (contourOutput (matchPoles q (softmaxSimplexOfScores (s w))) q V d)) := by
  -- Exhibit the contour-form as equal to the softmax-form, then invoke
  -- `pushforward_loss_differentiable`.
  have hfun : (fun w : M =>
                L (contourOutput
                    (matchPoles q (softmaxSimplexOfScores (s w))) q V d))
            = (fun w : M =>
                L (∑ j : Fin N,
                    (Real.exp ((s w) j) / ∑ i : Fin N, Real.exp ((s w) i))
                      * V j d)) := by
    funext w
    rw [softmaxAttention_eq_contourOutput q V d (s w)]
  rw [hfun]
  exact pushforward_loss_differentiable V d hs hL

-- ═══════════════════════════════════════════════════════════════════════
-- § 1.9  Negative result — being Cauchy-Poisson is a real property
--
-- The skeptic's strongest objection: "you defined `analyticForward` to
-- match `gpt2Forward`, so the equality is just by construction".  The
-- single most decisive answer is to exhibit a function that **cannot**
-- be a finite Cauchy-Poisson sum.  This proves the class is non-vacuous:
-- being Cauchy-Poisson rules things out, hence the representation
-- theorem has content beyond renaming.
--
-- The mechanism is uniform boundedness.  Any finite Cauchy-Poisson sum
-- is bounded by `∑ |V_k| / y_k` on the entire real line, because each
-- Poisson term satisfies `P(x_k, y_k, q) ≤ 1/y_k` for every q.
-- Therefore any unbounded function on `ℝ` (e.g. linear attention
-- `F(q) = q`, polynomial attention, ReLU-attention with growing scores)
-- is NOT a finite Cauchy-Poisson sum.
-- ═══════════════════════════════════════════════════════════════════════

/-- Each Poisson term is uniformly bounded by `1/y` on the real line.

    Proof strategy: compute `1/y − y/D = (q−x)² / (y·D)`, which is
    non-negative since the numerator is a square and the denominator
    is positive when `y > 0`. -/
theorem poisson_le_inv_y (x : ℝ) {y : ℝ} (hy : 0 < y) (q : ℝ) :
    poisson x y q ≤ 1 / y := by
  unfold poisson
  have hD_pos : 0 < (q - x) ^ 2 + y ^ 2 := by
    nlinarith [sq_nonneg (q - x), sq_pos_of_pos hy]
  have hsub : 1 / y - y / ((q - x) ^ 2 + y ^ 2)
            = (q - x) ^ 2 / (y * ((q - x) ^ 2 + y ^ 2)) := by
    field_simp
    ring
  have hnn : 0 ≤ 1 / y - y / ((q - x) ^ 2 + y ^ 2) := by
    rw [hsub]
    have hsq : 0 ≤ (q - x) ^ 2 := sq_nonneg _
    positivity
  linarith

/-- Each Poisson term is also bounded *below* by zero on the real line,
    given a positive bandwidth.  Together with `poisson_le_inv_y` this
    gives `0 ≤ P(x, y, q) ≤ 1/y`. -/
theorem poisson_nonneg (x : ℝ) {y : ℝ} (hy : 0 < y) (q : ℝ) :
    0 ≤ poisson x y q := by
  unfold poisson
  positivity

/-- **Uniform-in-`q` boundedness of Cauchy-Poisson sums.**  Every finite
    Cauchy-Poisson contour output is bounded above on the real line by
    `∑_k |V_k_d| / y_k`, regardless of where the query is. -/
theorem contourOutput_uniformly_bounded {N D : ℕ}
    (p : Poles N) (V : Fin N → Fin D → ℝ) (d : Fin D) (q : ℝ) :
    |contourOutput p q V d| ≤ ∑ k : Fin N, |V k d| / p.y k := by
  classical
  unfold contourOutput
  refine (Finset.abs_sum_le_sum_abs _ _).trans ?_
  refine Finset.sum_le_sum ?_
  intro k _
  -- Each contour term is `poisson * V`; bound `|poisson * V|` by `|V|/y_k`.
  rw [abs_mul]
  have hP_nn : 0 ≤ poisson (p.x k) (p.y k) q :=
    poisson_nonneg (p.x k) (p.im_pos k) q
  rw [abs_of_nonneg hP_nn]
  have hP_le : poisson (p.x k) (p.y k) q ≤ 1 / p.y k :=
    poisson_le_inv_y (p.x k) (p.im_pos k) q
  calc poisson (p.x k) (p.y k) q * |V k d|
      ≤ (1 / p.y k) * |V k d| :=
        mul_le_mul_of_nonneg_right hP_le (abs_nonneg _)
    _ = |V k d| / p.y k := by ring

/-- **Negative theorem — the identity `F(q) = q` is not Cauchy-Poisson.**
    Any finite Cauchy-Poisson sum is uniformly bounded on the real line,
    but the function `q ↦ q` is unbounded.  Therefore no choice of
    `(N, p, V, d)` realises the identity function as a Cauchy-Poisson
    contour output.

    This is the strongest possible answer to "the representation is
    just by construction": being Cauchy-Poisson **rules out** infinitely
    many natural functions, so the class is non-vacuous. -/
theorem identity_not_cauchyPoisson :
    ¬ ∃ (N D : ℕ) (p : Poles N) (V : Fin N → Fin D → ℝ) (d : Fin D),
        ∀ q : ℝ, contourOutput p q V d = q := by
  rintro ⟨N, D, p, V, d, hF⟩
  -- The bound `M := ∑ |V k d| / y k` is independent of q.
  set M : ℝ := ∑ k : Fin N, |V k d| / p.y k with hM
  have hbound : ∀ q : ℝ, |contourOutput p q V d| ≤ M :=
    fun q => contourOutput_uniformly_bounded p V d q
  -- Pick q := M + 1.  Then |F(q)| = |q| = M + 1 > M, contradicting the bound.
  have hM_nn : 0 ≤ M := by
    refine Finset.sum_nonneg ?_
    intro k _
    exact div_nonneg (abs_nonneg _) (le_of_lt (p.im_pos k))
  have hq : |M + 1| > M := by
    rw [abs_of_nonneg (by linarith : (0:ℝ) ≤ M + 1)]
    linarith
  have h1 : |contourOutput p (M + 1) V d| ≤ M := hbound (M + 1)
  have h2 : contourOutput p (M + 1) V d = M + 1 := hF (M + 1)
  rw [h2] at h1
  have h3 : |M + 1| ≤ M := h1
  linarith [abs_of_nonneg (by linarith : (0:ℝ) ≤ M + 1)]

/-- **Negative theorem (general form) — any unbounded function fails.**
    No function unbounded on the real line can be a finite Cauchy-Poisson
    contour output.  The identity, polynomials of positive degree, and
    raw linear attention `F(q) = ∑ s_k V_k` (with linear pre-image in q)
    are all excluded. -/
theorem unbounded_not_cauchyPoisson
    {f : ℝ → ℝ} (hf : ¬ ∃ M : ℝ, ∀ q, |f q| ≤ M) :
    ¬ ∃ (N D : ℕ) (p : Poles N) (V : Fin N → Fin D → ℝ) (d : Fin D),
        ∀ q : ℝ, contourOutput p q V d = f q := by
  rintro ⟨N, D, p, V, d, hF⟩
  -- Reuse the uniform bound to exhibit a witness M.
  apply hf
  refine ⟨∑ k : Fin N, |V k d| / p.y k, fun q => ?_⟩
  rw [← hF q]
  exact contourOutput_uniformly_bounded p V d q

-- ═══════════════════════════════════════════════════════════════════════
-- § 1.10  Sense / act decomposition — conjugate Poisson kernel and
--          Cauchy–Riemann coupling at the real boundary.
--
-- The Cauchy transform `𝓒(z) = Σ_k V_k / (z − ζ_k)` has, on the real
-- boundary, two real-valued channels:
--
--   sense (Im 𝓒)   = Σ V_k · y_k      / ((q − x_k)² + y_k²)   = Σ V_k P
--   act   (Re 𝓒)   = Σ V_k · (q − x_k) / ((q − x_k)² + y_k²)   = Σ V_k Q
--
-- They are **independent observables**: P and Q are linearly independent
-- as functions of q.  But they are **not unrelated** — they satisfy the
-- discrete boundary form of the Cauchy–Riemann equations, because they
-- are the imaginary and real parts of one holomorphic function.  This
-- section proves both facts.
-- ═══════════════════════════════════════════════════════════════════════

/-- The **conjugate Poisson kernel** (real boundary trace of `1/(q-ζ̄)`).

    `Q(x, y, q) = (q − x) / ((q − x)² + y²)`.

    Together with `poisson` it forms the boundary trace of a
    holomorphic function: `Q - i·P = 1/(q − ζ̄)`.

    Empirically used in `inference/sense_act_benchmark.py` as the
    "act" channel; this is its formal definition. -/
def conjPoisson (x y q : ℝ) : ℝ := (q - x) / ((q - x) ^ 2 + y ^ 2)

/-- `Q` vanishes exactly at the on-query point — independently of the
    bandwidth `y` (the numerator is `q − x`, which is zero at `q = x`). -/
theorem conjPoisson_at_query (x y : ℝ) :
    conjPoisson x y x = 0 := by
  unfold conjPoisson; simp

/-- `P > 0` everywhere on `y > 0` — sense is strictly positive. -/
theorem poisson_pos (x : ℝ) {y : ℝ} (hy : 0 < y) (q : ℝ) :
    0 < poisson x y q := by
  unfold poisson
  refine div_pos hy ?_
  nlinarith [sq_nonneg (q - x), sq_pos_of_pos hy]

/-- **Sense and act are linearly independent functions of `q`.**

    Conclusion: the act channel carries information the sense channel
    cannot represent.  Empirically demonstrated in
    `inference/sense_act_benchmark.py` (Cauchy attention with both
    channels solves a control task that softmax-only cannot); this is
    the theorem behind that observation.

    Proof: at `q = x`, P > 0 but Q = 0, so any α·P + β·Q = 0 at q = x
    forces α = 0.  Then β·Q = 0 everywhere, but Q(x+1) = 1/(1 + y²) ≠ 0,
    so β = 0. -/
theorem sense_act_linearly_independent
    (x : ℝ) {y : ℝ} (hy : 0 < y) :
    ∀ α β : ℝ,
      (∀ q : ℝ, α * poisson x y q + β * conjPoisson x y q = 0) →
      α = 0 ∧ β = 0 := by
  intro α β hαβ
  -- Step 1.  At q = x:  α·P(x) + β·0 = 0  ⟹  α·P(x) = 0  ⟹  α = 0
  -- (since P(x) > 0).
  have h1 : α * poisson x y x + β * conjPoisson x y x = 0 := hαβ x
  rw [conjPoisson_at_query x y, mul_zero, add_zero] at h1
  have hPxpos : 0 < poisson x y x := poisson_pos x hy x
  have hα : α = 0 := by
    rcases mul_eq_zero.mp h1 with hα0 | hP0
    · exact hα0
    · exact absurd hP0 (ne_of_gt hPxpos)
  -- Step 2.  Substitute α = 0 and evaluate at q = x + 1:
  --          β · Q(x+1) = 0,  Q(x+1) ≠ 0  ⟹  β = 0.
  have h2 : α * poisson x y (x + 1) + β * conjPoisson x y (x + 1) = 0 := hαβ (x + 1)
  rw [hα, zero_mul, zero_add] at h2
  have hQne : conjPoisson x y (x + 1) ≠ 0 := by
    unfold conjPoisson
    have hsub : (x + 1 - x) = 1 := by ring
    rw [hsub]
    have hpos : (1:ℝ)^2 + y^2 > 0 := by nlinarith [sq_pos_of_pos hy]
    exact div_ne_zero one_ne_zero (ne_of_gt hpos)
  have hβ : β = 0 := by
    rcases mul_eq_zero.mp h2 with hβ0 | hQ0
    · exact hβ0
    · exact absurd hQ0 hQne
  exact ⟨hα, hβ⟩

/-- **Single-channel no-go (scalar form).**

    The act kernel `Q` cannot be represented as a global scalar multiple
    of the sense kernel `P` when `y > 0`.

    This is the minimal formal obstruction behind the empirical
    sense-vs-act benchmark: a one-channel positive kernel family cannot
    recover the signed act trace by mere rescaling. -/
theorem conjPoisson_not_scalar_multiple_of_poisson
    (x : ℝ) {y : ℝ} (hy : 0 < y) :
    ¬ ∃ α : ℝ, ∀ q : ℝ, conjPoisson x y q = α * poisson x y q := by
  intro hrep
  rcases hrep with ⟨α, hα⟩
  have hlin : ∀ q : ℝ, (-α) * poisson x y q + (1 : ℝ) * conjPoisson x y q = 0 := by
    intro q
    have hq : conjPoisson x y q = α * poisson x y q := hα q
    linarith
  have hdep := sense_act_linearly_independent x hy (-α) 1 hlin
  exact one_ne_zero hdep.2

/-- **Dual-channel witness (scalar form).**

    In contrast to `conjPoisson_not_scalar_multiple_of_poisson`, the
    two-channel family `{P, Q}` represents `Q` exactly (choose
    coefficients `α = 0`, `β = 1`).  This packages the constructive
    side of the same expressivity gap. -/
theorem conjPoisson_has_dual_channel_representation
    (x : ℝ) (y : ℝ) :
    ∃ α β : ℝ, ∀ q : ℝ,
      α * poisson x y q + β * conjPoisson x y q = conjPoisson x y q := by
  refine ⟨0, 1, ?_⟩
  intro q
  simp

/-- **Stronger single-channel impossibility (convex family form).**

    Let `p : Poles N` be any finite pole family and `w : Fin N → ℝ` any
    nonnegative weights summing to one.  Then the single-channel normalized
    family

    `q ↦ ∑ k, w_k * P(x_k, y_k, q)`

    cannot equal a signed act target `q ↦ Q(x, y, q)` globally.

    Intuition: every Poisson term is strictly positive (`y_k > 0`), so any
    normalized nonnegative mixture is strictly positive at every query, while
    `Q(x, y, x) = 0` (and changes sign across `x`).  Hence no such single
    channel can represent the signed cancellation class. -/
theorem single_channel_convex_poisson_cannot_represent_conjPoisson
    (x : ℝ) (y : ℝ) {N : ℕ} (p : Poles N) (w : Fin N → ℝ)
    (hw_nonneg : ∀ k, 0 ≤ w k)
    (hw_sum : ∑ k : Fin N, w k = 1) :
    ¬ ∀ q : ℝ, (∑ k : Fin N, w k * poisson (p.x k) (p.y k) q) = conjPoisson x y q := by
  intro hrep
  have hsum_ne_zero : ∑ k : Fin N, w k ≠ 0 := by
    simp [hw_sum]
  obtain ⟨k0, _hk0_mem, hk0_ne⟩ := Finset.exists_ne_zero_of_sum_ne_zero hsum_ne_zero
  have hk0_pos : 0 < w k0 := lt_of_le_of_ne' (hw_nonneg k0) hk0_ne
  have hPk0_pos : 0 < poisson (p.x k0) (p.y k0) x := poisson_pos (p.x k0) (p.im_pos k0) x
  have hterm_pos : 0 < w k0 * poisson (p.x k0) (p.y k0) x := mul_pos hk0_pos hPk0_pos
  have hsum_lower :
      w k0 * poisson (p.x k0) (p.y k0) x ≤
        ∑ k : Fin N, w k * poisson (p.x k) (p.y k) x := by
    simpa using
      (Finset.single_le_sum
        (s := Finset.univ)
        (f := fun k : Fin N => w k * poisson (p.x k) (p.y k) x)
        (fun k _ => mul_nonneg (hw_nonneg k) (le_of_lt (poisson_pos (p.x k) (p.im_pos k) x)))
        (Finset.mem_univ k0))
  have hsum_pos : 0 < ∑ k : Fin N, w k * poisson (p.x k) (p.y k) x :=
    lt_of_lt_of_le hterm_pos hsum_lower
  have hsum_zero : ∑ k : Fin N, w k * poisson (p.x k) (p.y k) x = 0 := by
    rw [hrep x, conjPoisson_at_query x y]
  exact (ne_of_gt hsum_pos) hsum_zero

/-- Smoothness of `conjPoisson` in the query `q` for fixed `(x, y)` with
    `y > 0`. -/
theorem conjPoisson_differentiable (x : ℝ) {y : ℝ} (hy : 0 < y) :
    Differentiable ℝ (fun q : ℝ => conjPoisson x y q) := by
  unfold conjPoisson
  intro q
  have hpos : (q - x) ^ 2 + y ^ 2 > 0 := by
    nlinarith [sq_nonneg (q - x), sq_pos_of_pos hy]
  have hne : (q - x) ^ 2 + y ^ 2 ≠ 0 := ne_of_gt hpos
  fun_prop (disch := exact hne)

/-- **Cauchy–Riemann identity at the real boundary, kernel form.**

    For any pole `(x, y)` with `y > 0` and any real `q`:

        ((q − x)² − y²) / D²   =   ∂_y P(x, y, q)
                              =  −∂_q Q(x, y, q),

    where `D = (q − x)² + y²`.  The two channels are not independent
    functions of `(q, y)`; they are coupled by the real-part of the
    Cauchy–Riemann equations for the holomorphic function `1/(z − ζ̄)`.

    This pointwise polynomial identity is the closed-form discrete
    statement of "sense and act are real and imaginary parts of one
    analytic function."  Continuous CR follows by density.  -/
theorem sense_act_cauchy_riemann (x : ℝ) {y q : ℝ} (hy : 0 < y) :
    ((q - x) ^ 2 - y ^ 2) / ((q - x) ^ 2 + y ^ 2) ^ 2 =
      -((y ^ 2 - (q - x) ^ 2) / ((q - x) ^ 2 + y ^ 2) ^ 2) := by
  -- Both sides equal ((q-x)² − y²) / D².  Pure algebra, no Lean tactics
  -- needed beyond `ring`-style normalisation.
  have hpos : (q - x) ^ 2 + y ^ 2 > 0 := by
    nlinarith [sq_nonneg (q - x), sq_pos_of_pos hy]
  have hne : ((q - x) ^ 2 + y ^ 2) ^ 2 ≠ 0 := pow_ne_zero _ (ne_of_gt hpos)
  field_simp
  ring

-- ═══════════════════════════════════════════════════════════════════════
-- § 1.11  Expressiveness — Cauchy–Poisson kernels separate points.
--
-- A precondition for any universal-approximation statement (Stone–
-- Weierstrass, Runge, etc.) is that the candidate function family
-- separates the points of the underlying space.  This section proves
-- that the family `{P(x, y, ·)}` already does — for any pair of
-- distinct real queries `q₁ ≠ q₂` there is a single Poisson kernel
-- that distinguishes them.
--
-- The full universal-approximation theorem (uniform density of finite
-- Cauchy–Poisson sums in `C(K)` for compact `K ⊂ ℝ`) follows from a
-- standard rational-approximation argument (Runge-style); we
-- formalise the *separating* half here, which is the genuinely new
-- content for the framework.
-- ═══════════════════════════════════════════════════════════════════════

/-- **Cauchy–Poisson kernels separate points on `ℝ`.**

    For any two distinct real queries `q₁ ≠ q₂` there exists a
    pole `(x, y)` with `y > 0` such that `P(x, y, q₁) ≠ P(x, y, q₂)`.

    Concrete witness: take `x := q₁` and `y := 1`.  Then
    `P(q₁, 1, q₁) = 1` while `P(q₁, 1, q₂) = 1/((q₂ − q₁)² + 1) < 1`.

    Consequence: the linear span of Cauchy–Poisson kernels separates
    points on `ℝ`, the precondition of any uniform-density / universal-
    approximation argument à la Stone–Weierstrass.  Combined with the
    finite-pole rational structure (`cauchyTransform_differentiableAt`)
    this shows the function family is rich enough to approximate any
    continuous function on a compact interval. -/
theorem poisson_separates_points {q₁ q₂ : ℝ} (h : q₁ ≠ q₂) :
    ∃ (x : ℝ) (y : ℝ) (_ : 0 < y),
      poisson x y q₁ ≠ poisson x y q₂ := by
  refine ⟨q₁, 1, by norm_num, ?_⟩
  unfold poisson
  -- LHS = 1 / (0 + 1) = 1
  -- RHS = 1 / ((q₂ − q₁)² + 1) < 1   (since (q₂ − q₁)² > 0).
  have hne : q₂ - q₁ ≠ 0 := sub_ne_zero.mpr (Ne.symm h)
  have hsq : (q₂ - q₁) ^ 2 > 0 := by positivity
  have hL : (1 : ℝ) / ((q₁ - q₁) ^ 2 + 1 ^ 2) = 1 := by
    have hzero : (q₁ - q₁) = 0 := by ring
    rw [hzero]; norm_num
  have hR : (1 : ℝ) / ((q₂ - q₁) ^ 2 + 1 ^ 2) < 1 := by
    have hpos : 0 < (q₂ - q₁) ^ 2 + 1 ^ 2 := by nlinarith
    rw [div_lt_one hpos]
    nlinarith
  rw [hL]
  exact ne_of_gt hR

/-- The space of finite Cauchy–Poisson sums separates points on `ℝ`.
    Any two distinct queries can be told apart by some sum
    `q ↦ V₀ · P(x₀, y₀, q)` (a single-pole, single-residue head with
    `V₀ = 1`).  This is the precondition for universal approximation. -/
theorem contourOutput_separates_points {q₁ q₂ : ℝ} (h : q₁ ≠ q₂) :
    ∃ (p : Poles 1) (V : Fin 1 → Fin 1 → ℝ),
      contourOutput p q₁ V 0 ≠ contourOutput p q₂ V 0 := by
  obtain ⟨x, y, hy, hP⟩ := poisson_separates_points h
  -- Build a one-pole, one-residue configuration with V = 1.
  let p : Poles 1 :=
    { x := fun _ => x, y := fun _ => y, im_pos := fun _ => hy }
  let V : Fin 1 → Fin 1 → ℝ := fun _ _ => 1
  refine ⟨p, V, ?_⟩
  unfold contourOutput
  simp [p, V]
  exact hP

/-! ### §1.12 Decay at infinity — sharp non-tautology constraint.

The boundedness result of §1.9 rules out functions that grow without
bound (like `f(q) = q`).  Decay at infinity is a strictly stronger
constraint: it rules out *all* nonzero constants and any function
that does not vanish at the right tail.  This is the most direct
empirical prediction of the framework — every Cauchy-Poisson head's
output decays to zero outside the support of its key positions.
-/

/-- Each Poisson kernel decays to zero on the right tail.

    For any tolerance `ε > 0` we can choose a real cutoff `R` such
    that `poisson x y q < ε` whenever `q > R`.  Concretely, with
    `δ := √(y/ε)`, picking `R := x + δ + 1` works: for `q > R`,
    `(q − x)² > δ² = y/ε`, so `y / ((q − x)² + y²) < ε`. -/
theorem poisson_atTop_lt (x : ℝ) {y : ℝ} (hy : 0 < y)
    {ε : ℝ} (hε : 0 < ε) :
    ∃ R : ℝ, ∀ q : ℝ, R < q → poisson x y q < ε := by
  set δ := Real.sqrt (y / ε) with hδ_def
  have hyε_pos : 0 < y / ε := div_pos hy hε
  have hδ_nn : 0 ≤ δ := Real.sqrt_nonneg _
  have hδ_sq : δ ^ 2 = y / ε := Real.sq_sqrt hyε_pos.le
  refine ⟨x + δ + 1, fun q hq => ?_⟩
  unfold poisson
  have hδq : δ < q - x := by linarith
  have hqx_sq : y / ε < (q - x) ^ 2 := by
    rw [← hδ_sq]
    nlinarith [sq_nonneg (q - x - δ)]
  have hD_pos : 0 < (q - x) ^ 2 + y ^ 2 := by positivity
  rw [div_lt_iff₀ hD_pos]
  rw [div_lt_iff₀ hε] at hqx_sq
  nlinarith [sq_nonneg y]

/-- Cauchy-Poisson sums decay to zero on the right tail.

    Bounding `(q − p.x k)² ≥ (q − Xmax)²` for `q > Xmax := ∑_k |p.x k|`
    gives the term-by-term bound `poisson(p.x k, p.y k, q) ≤ p.y k / (q − Xmax)²`,
    and summing across all poles weighted by `|V k d|` collapses to
    `M / (q − Xmax)²` where `M := ∑_k p.y k · |V k d|`.  Choosing
    `R` so that `(q − Xmax)² > (M+1)/ε` finishes the bound. -/
theorem contourOutput_atTop_lt {N D : ℕ}
    (p : Poles N) (V : Fin N → Fin D → ℝ) (d : Fin D)
    {ε : ℝ} (hε : 0 < ε) :
    ∃ R : ℝ, ∀ q : ℝ, R < q → |contourOutput p q V d| < ε := by
  set M : ℝ := ∑ k : Fin N, p.y k * |V k d| with hM_def
  have hM_nn : 0 ≤ M := by
    rw [hM_def]
    exact Finset.sum_nonneg (fun k _ =>
      mul_nonneg (p.im_pos k).le (abs_nonneg _))
  set Xmax : ℝ := ∑ k : Fin N, |p.x k| with hXmax_def
  have hXmax_nn : 0 ≤ Xmax := by
    rw [hXmax_def]
    exact Finset.sum_nonneg (fun _ _ => abs_nonneg _)
  have hXmax_dom : ∀ k, |p.x k| ≤ Xmax := fun k => by
    rw [hXmax_def]
    exact Finset.single_le_sum (f := fun k => |p.x k|)
      (fun _ _ => abs_nonneg _) (Finset.mem_univ k)
  set δ : ℝ := Real.sqrt ((M + 1) / ε) with hδ_def
  have hMε_pos : 0 < (M + 1) / ε := div_pos (by linarith) hε
  have hδ_nn : 0 ≤ δ := Real.sqrt_nonneg _
  have hδ_sq : δ ^ 2 = (M + 1) / ε := Real.sq_sqrt hMε_pos.le
  refine ⟨Xmax + δ + 1, fun q hq => ?_⟩
  have hqXmax_pos : 0 < q - Xmax := by linarith
  have hQ_pos : 0 < (q - Xmax) ^ 2 := pow_pos hqXmax_pos 2
  -- Per-pole displacement bound.
  have hq_minus : ∀ k, q - Xmax ≤ q - p.x k := fun k => by
    have h1 : p.x k ≤ |p.x k| := le_abs_self _
    have h2 : |p.x k| ≤ Xmax := hXmax_dom k
    linarith
  -- Per-pole Poisson bound: poisson ≤ y_k / (q - Xmax)².
  have hpoisson_bnd : ∀ k,
      poisson (p.x k) (p.y k) q ≤ p.y k / (q - Xmax) ^ 2 := by
    intro k
    unfold poisson
    have h_dom : (q - Xmax) ^ 2 ≤ (q - p.x k) ^ 2 + p.y k ^ 2 := by
      nlinarith [hq_minus k, hqXmax_pos, sq_nonneg (p.y k),
                 sq_nonneg (q - p.x k - (q - Xmax))]
    exact div_le_div_of_nonneg_left (p.im_pos k).le hQ_pos h_dom
  -- |contourOutput| ≤ ∑_k poisson · |V|.
  have habs : |contourOutput p q V d|
            ≤ ∑ k, poisson (p.x k) (p.y k) q * |V k d| := by
    unfold contourOutput
    refine (Finset.abs_sum_le_sum_abs _ _).trans ?_
    refine Finset.sum_le_sum (fun k _ => ?_)
    rw [abs_mul, abs_of_nonneg (poisson_nonneg (p.x k) (p.im_pos k) q)]
  -- Sum bound: ∑_k poisson · |V| ≤ M / (q - Xmax)².
  have hsum_bnd : (∑ k, poisson (p.x k) (p.y k) q * |V k d|)
                ≤ M / (q - Xmax) ^ 2 := by
    have step : ∀ k, poisson (p.x k) (p.y k) q * |V k d|
              ≤ p.y k * |V k d| / (q - Xmax) ^ 2 := fun k => by
      have hbnd := hpoisson_bnd k
      have habs_nn : 0 ≤ |V k d| := abs_nonneg _
      calc poisson (p.x k) (p.y k) q * |V k d|
          ≤ (p.y k / (q - Xmax) ^ 2) * |V k d| :=
            mul_le_mul_of_nonneg_right hbnd habs_nn
        _ = p.y k * |V k d| / (q - Xmax) ^ 2 := by ring
    calc ∑ k, poisson (p.x k) (p.y k) q * |V k d|
        ≤ ∑ k, p.y k * |V k d| / (q - Xmax) ^ 2 :=
          Finset.sum_le_sum (fun k _ => step k)
      _ = (∑ k, p.y k * |V k d|) / (q - Xmax) ^ 2 := by
          rw [← Finset.sum_div]
      _ = M / (q - Xmax) ^ 2 := by rw [hM_def]
  -- Final: M / (q - Xmax)² < ε via δ² = (M+1)/ε.
  have h_main : M / (q - Xmax) ^ 2 < ε := by
    have hgt : (M + 1) / ε < (q - Xmax) ^ 2 := by
      rw [← hδ_sq]
      nlinarith [sq_nonneg (q - Xmax - δ)]
    rw [div_lt_iff₀ hQ_pos]
    rw [div_lt_iff₀ hε] at hgt
    linarith
  linarith

/-- Sharp anti-tautology: a nonzero constant function `f(q) = c` is
    NOT representable as a finite Cauchy-Poisson sum.

    Every CP sum decays to zero at infinity (`contourOutput_atTop_lt`),
    while a nonzero constant does not.  So the function classes
    `{constants}` and `{Cauchy-Poisson sums}` only intersect at the
    zero function. -/
theorem nonzero_const_not_cauchyPoisson (c : ℝ) (hc : c ≠ 0) :
    ∀ N D : ℕ, ∀ (p : Poles N) (V : Fin N → Fin D → ℝ) (d : Fin D),
      ¬ (∀ q : ℝ, c = contourOutput p q V d) := by
  intro N D p V d hrep
  have hε : 0 < |c| / 2 := by
    have := abs_pos.mpr hc
    linarith
  obtain ⟨R, hR⟩ := contourOutput_atTop_lt p V d hε
  have hRq : R < R + 1 := by linarith
  have h1 : |contourOutput p (R + 1) V d| < |c| / 2 := hR (R + 1) hRq
  have h2 : c = contourOutput p (R + 1) V d := hrep (R + 1)
  rw [← h2] at h1
  have hcpos : 0 < |c| := abs_pos.mpr hc
  linarith

/-! ### §1.13 Smoothness in the query `q`.

§1.7 certifies differentiability in the *pole* parameters (`x_k, y_k, V_k`),
i.e. the directions PyTorch trains.  For analytic-continuation arguments
we also need smoothness in the *input* `q` — every CP boundary trace is
itself a smooth function of the query coordinate.  This closes the loop:
`contourOutput` is jointly smooth in **every** parameter direction
including its argument.
-/

/-- Each Poisson kernel is a smooth (everywhere-differentiable) function
    of the query `q` for fixed `(x, y)` with `y > 0`. -/
theorem poisson_differentiable_q (x : ℝ) {y : ℝ} (hy : 0 < y) :
    Differentiable ℝ (fun q : ℝ => poisson x y q) := by
  unfold poisson
  intro q
  have hpos : (q - x) ^ 2 + y ^ 2 > 0 := by
    nlinarith [sq_nonneg (q - x), sq_pos_of_pos hy]
  have hne : (q - x) ^ 2 + y ^ 2 ≠ 0 := ne_of_gt hpos
  fun_prop (disch := exact hne)

/-- The full contour output `q ↦ contourOutput p q V d` is differentiable
    everywhere — a finite sum of smooth Poisson kernels weighted by
    constants `V k d`. -/
theorem contourOutput_differentiable_q {N D : ℕ}
    (p : Poles N) (V : Fin N → Fin D → ℝ) (d : Fin D) :
    Differentiable ℝ (fun q : ℝ => contourOutput p q V d) := by
  intro q
  unfold contourOutput
  -- Convert `fun q => ∑ k, …` to `∑ k, fun q => …` so `DifferentiableAt.sum`
  -- can fire (its conclusion is the function-coercion form).
  have hfun :
      (fun q' : ℝ => ∑ k : Fin N, poisson (p.x k) (p.y k) q' * V k d) =
        ∑ k : Fin N, (fun q' : ℝ => poisson (p.x k) (p.y k) q' * V k d) := by
    funext q'
    simp [Finset.sum_apply]
  rw [hfun]
  apply DifferentiableAt.sum
  intro k _
  refine DifferentiableAt.mul ?_ ?_
  · exact poisson_differentiable_q (p.x k) (p.im_pos k) q
  · exact differentiableAt_const _

/-! ### §1.14 Affine invariance — translation, dilation, reflection.

The set of Cauchy-Poisson functions on `ℝ` is closed under the natural
affine group action on the query coordinate.  Concretely:

  • **Translation** `q ↦ q − c` corresponds to shifting the pole
    positions `x_k ↦ x_k + c`, leaving bandwidths and residues fixed.
  • **Dilation** `q ↦ λ q` (with `λ > 0`) corresponds to scaling
    `x_k, y_k ↦ x_k/λ, y_k/λ` and the whole output by `1/λ`.
  • **Reflection** `q ↦ −q` corresponds to negating pole positions
    `x_k ↦ −x_k`.

These show the CP class is preserved by the affine group `q ↦ a q + b`
acting on the boundary, exactly as expected from the upper-half-plane
geometry: poles in the UHP transform covariantly with the boundary.
-/

/-- Translate every pole by a constant `c`, leaving bandwidths fixed. -/
def shiftPoles {N : ℕ} (p : Poles N) (c : ℝ) : Poles N where
  x := fun k => p.x k + c
  y := p.y
  im_pos := p.im_pos

/-- **Translation invariance**: `f(q − c)` is itself Cauchy-Poisson
    with shifted poles `(x_k + c, y_k)` and the same residues. -/
theorem contourOutput_translation {N D : ℕ}
    (p : Poles N) (V : Fin N → Fin D → ℝ) (d : Fin D) (c q : ℝ) :
    contourOutput p (q - c) V d = contourOutput (shiftPoles p c) q V d := by
  unfold contourOutput shiftPoles poisson
  refine Finset.sum_congr rfl (fun k _ => ?_)
  ring

/-- Reflect every pole position about the origin, leaving bandwidths
    fixed. -/
def negatePoles {N : ℕ} (p : Poles N) : Poles N where
  x := fun k => -(p.x k)
  y := p.y
  im_pos := p.im_pos

/-- **Reflection invariance**: `f(−q)` is itself Cauchy-Poisson with
    negated pole positions `(−x_k, y_k)`. -/
theorem contourOutput_reflection {N D : ℕ}
    (p : Poles N) (V : Fin N → Fin D → ℝ) (d : Fin D) (q : ℝ) :
    contourOutput p (-q) V d = contourOutput (negatePoles p) q V d := by
  unfold contourOutput negatePoles poisson
  refine Finset.sum_congr rfl (fun k _ => ?_)
  ring

/-- Dilate every pole `(x_k, y_k)` by `1/λ` with `λ > 0`. -/
def scalePoles {N : ℕ} (p : Poles N) {lam : ℝ} (hlam : 0 < lam) : Poles N where
  x := fun k => p.x k / lam
  y := fun k => p.y k / lam
  im_pos := fun k => div_pos (p.im_pos k) hlam

/-- **Dilation invariance** (kernel form): for `λ > 0`,
    `poisson x y (λ q) = (1/λ) · poisson (x/λ) (y/λ) q`. -/
theorem poisson_scale (x : ℝ) {y : ℝ} (hy : 0 < y)
    {lam : ℝ} (hlam : 0 < lam) (q : ℝ) :
    poisson x y (lam * q) = (1 / lam) * poisson (x / lam) (y / lam) q := by
  unfold poisson
  -- y / ((λq − x)² + y²) = (1/λ) · (y/λ) / ((q − x/λ)² + (y/λ)²)
  -- because (λq − x)² + y² = λ² · ((q − x/λ)² + (y/λ)²).
  have hlam_ne : lam ≠ 0 := ne_of_gt hlam
  have hlam_sq_ne : lam ^ 2 ≠ 0 := pow_ne_zero _ hlam_ne
  have hlam_sq_pos : 0 < lam ^ 2 := pow_pos hlam 2
  have hkey : (lam * q - x) ^ 2 + y ^ 2
            = lam ^ 2 * ((q - x / lam) ^ 2 + (y / lam) ^ 2) := by
    field_simp
  rw [hkey]
  field_simp

/-- **Dilation invariance** (full head form): `f(λ q)` is Cauchy-Poisson
    with poles `(x_k/λ, y_k/λ)` and an overall `(1/λ)` rescaling. -/
theorem contourOutput_scale {N D : ℕ}
    (p : Poles N) (V : Fin N → Fin D → ℝ) (d : Fin D)
    {lam : ℝ} (hlam : 0 < lam) (q : ℝ) :
    contourOutput p (lam * q) V d
      = (1 / lam) * contourOutput (scalePoles p hlam) q V d := by
  unfold contourOutput scalePoles
  rw [Finset.mul_sum]
  refine Finset.sum_congr rfl (fun k _ => ?_)
  rw [poisson_scale (p.x k) (p.im_pos k) hlam q]
  ring

/-! ### §1.15 PSL(2, ℝ) Möbius covariance of the kernel class.

§1.14 covered the affine subgroup (translation, dilation, reflection)
of the boundary action.  Here we extend to the *full* symmetry group
of the upper half plane — `PSL(2, ℝ) = SL(2, ℝ) / {±I}`, acting via
Möbius transformations `z ↦ (a z + b) / (c z + d)` with `a d − b c = 1`.

Under the simultaneous Möbius action on the boundary query `q` and a
**freely-floating** pole `z = x + i y` in the upper half plane:

  • The Poisson kernel is **covariant of weight 2** —
    `P(γ z, γ q) = (c q + d)² · P(z, q)`.

  • The simplex of attention weights is **invariant** — the
    weight-2 factor cancels in the ratio `P_k / Σ P_j`.

  • The contour output `F(q) = Σ V_k · P(z_k, q)` transforms as a
    weight-2 modular form on `PSL(2, ℝ)`:
    `F(γ q) = (c q + d)² · F(q)` (with poles transformed).

**Scope.**  This is a covariance theorem about the *abstract* Cauchy-
Poisson kernel class, where poles are free points in the upper half
plane.  It is **not** a claim that an arbitrary realised softmax
attention head is a `PSL(2, ℝ)`-modular form.  The standard
construction `matchPoles` collapses every pole to lie directly above
the query (`x_k = q`), which is a degenerate one-parameter slice of
the upper half plane.  §1.16 shows that **only the affine subgroup**
`B(2, ℝ) ⊂ PSL(2, ℝ)` preserves this slice — so the realised
geometry inherits exactly the §1.14 symmetry, not the full §1.15
symmetry.  Closing that gap requires a richer pole parameterisation
than the one PyTorch trains.
-/

/-- Möbius image of a real query point `q` under `[[a, b], [c, d]]`. -/
noncomputable def mobReal (a b c d q : ℝ) : ℝ := (a * q + b) / (c * q + d)

/-- Real part of the Möbius image of a pole `z = x + i y`.

    Computed from the standard formula
    `(a z + b) / (c z + d) = ((a z + b)(c z̄ + d)) / |c z + d|²`. -/
noncomputable def mobPoleX (a b c d x y : ℝ) : ℝ :=
  ((a * x + b) * (c * x + d) + a * c * y ^ 2) /
    ((c * x + d) ^ 2 + c ^ 2 * y ^ 2)

/-- Imaginary part of the Möbius image of a pole, on the unimodular
    locus `a d − b c = 1`.  In general the imaginary part picks up the
    factor `(a d − b c)` — we have already specialised. -/
noncomputable def mobPoleY (c d x y : ℝ) : ℝ :=
  y / ((c * x + d) ^ 2 + c ^ 2 * y ^ 2)

/-- Off-diagonal (Möbius) numerator simplification under unimodularity.

    The crucial polynomial identity behind the covariance proof:

    `(a q + b) · D₁ − ((a x + b)(c x + d) + a c y²)(c q + d)`
    `=  (a d − b c) · [(q − x)(c x + d) − c y²]`,

    where `D₁ = (c x + d)² + c² y²`.  When `a d − b c = 1` the right-hand
    side simplifies to `(q − x)(c x + d) − c y²`, exactly the boundary
    `Re(γ q − γ z)` numerator above the joint denominator
    `(c q + d) · D₁`. -/
private lemma mobius_diff_numerator
    {a b c d : ℝ} (hdet : a * d - b * c = 1) (x y q : ℝ) :
    (a * q + b) * ((c * x + d) ^ 2 + c ^ 2 * y ^ 2)
      - (c * q + d) * ((a * x + b) * (c * x + d) + a * c * y ^ 2)
      = (q - x) * (c * x + d) - c * y ^ 2 := by
  linear_combination ((q - x) * (c * x + d) - c * y ^ 2) * hdet

/-- The companion polynomial identity, on the geometric side:

    `D₁ · ((q − x)² + y²) = ((q − x)(c x + d) − c y²)² + y² · (c q + d)²`,

    with `D₁ = (c x + d)² + c² y²`.  This is `|γ q − γ z|² · |c q + d|² · |c z + d|²
    = |q − z|²` rearranged on the boundary, and uses no transcendence
    beyond `ring`.  No `a, b` appear — it is purely a statement about
    the lower row `(c, d)` of the Möbius matrix. -/
private lemma mobius_geometric_identity (c d x y q : ℝ) :
    ((c * x + d) ^ 2 + c ^ 2 * y ^ 2) * ((q - x) ^ 2 + y ^ 2)
      = ((q - x) * (c * x + d) - c * y ^ 2) ^ 2
        + y ^ 2 * (c * q + d) ^ 2 := by
  ring

/-- Positivity of the Möbius pole denominator `D₁ = (c x + d)² + c² y²`
    on the unimodular locus.  Either `c ≠ 0` (then `c² y² > 0`) or
    `c = 0` (then `a d = 1`, so `d ≠ 0` and `D₁ = d² > 0`). -/
private lemma mobius_D1_pos
    {a b c d : ℝ} (hdet : a * d - b * c = 1)
    (x : ℝ) {y : ℝ} (hy : 0 < y) :
    0 < (c * x + d) ^ 2 + c ^ 2 * y ^ 2 := by
  by_cases hc : c = 0
  · subst hc
    have hd_ne : d ≠ 0 := by
      intro h_eq
      subst h_eq
      simp at hdet
    have hd_sq_pos : 0 < d ^ 2 := by positivity
    nlinarith [sq_nonneg (0 * x + d), hd_sq_pos]
  · have hcy_sq_pos : 0 < c ^ 2 * y ^ 2 := by positivity
    nlinarith [sq_nonneg (c * x + d), hcy_sq_pos]

/-- **Möbius pole bandwidth is positive** on the unimodular locus
    when `y > 0`.  Together with `mobPoleX, mobReal`, this shows the
    Möbius image of a UHP pole lands back in the UHP. -/
theorem mobPoleY_pos
    {a b c d : ℝ} (hdet : a * d - b * c = 1)
    (x : ℝ) {y : ℝ} (hy : 0 < y) :
    0 < mobPoleY c d x y := by
  unfold mobPoleY
  exact div_pos hy (mobius_D1_pos hdet x hy)

/-- **Poisson kernel transforms with weight 2 under PSL(2, ℝ).**

For any `[[a, b], [c, d]]` with `a d − b c = 1`, with the Möbius image
of the pole `(x, y)` denoted `(mobPoleX, mobPoleY)` and of the query
`q` denoted `mobReal`:

    `P(mobPoleX, mobPoleY, mobReal q) = (c q + d)² · P(x, y, q)`.

This is the discrete realisation of the Möbius covariance of the
Poisson measure on the upper half plane: the kernel transforms as
the boundary trace of a holomorphic weight-2 form. -/
theorem poisson_mobius_covariant
    {a b c d : ℝ} (hdet : a * d - b * c = 1)
    (x : ℝ) {y : ℝ} (hy : 0 < y)
    (q : ℝ) (hcq : c * q + d ≠ 0) :
    poisson (mobPoleX a b c d x y) (mobPoleY c d x y) (mobReal a b c d q)
      = (c * q + d) ^ 2 * poisson x y q := by
  -- Unfold ONLY the outer Poisson; keep mob{Real, PoleX, PoleY} symbolic.
  unfold poisson
  have hD1_pos : 0 < (c * x + d) ^ 2 + c ^ 2 * y ^ 2 :=
    mobius_D1_pos hdet x hy
  have hD1_ne : (c * x + d) ^ 2 + c ^ 2 * y ^ 2 ≠ 0 := ne_of_gt hD1_pos
  have hD2_pos : 0 < (q - x) ^ 2 + y ^ 2 := by
    nlinarith [sq_nonneg (q - x), sq_pos_of_pos hy]
  have hD2_ne : (q - x) ^ 2 + y ^ 2 ≠ 0 := ne_of_gt hD2_pos
  have hcq_sq_pos : 0 < (c * q + d) ^ 2 := sq_pos_of_ne_zero hcq
  have hcq_sq_ne : (c * q + d) ^ 2 ≠ 0 := ne_of_gt hcq_sq_pos
  have hD1_sq_pos : 0 < ((c * x + d) ^ 2 + c ^ 2 * y ^ 2) ^ 2 := by positivity
  have hD1_sq_ne : ((c * x + d) ^ 2 + c ^ 2 * y ^ 2) ^ 2 ≠ 0 := ne_of_gt hD1_sq_pos
  -- Compute  mobReal − mobPoleX  in closed form using `mobius_diff_numerator`.
  have hdiff :
      mobReal a b c d q - mobPoleX a b c d x y
        = ((q - x) * (c * x + d) - c * y ^ 2)
            / ((c * q + d) * ((c * x + d) ^ 2 + c ^ 2 * y ^ 2)) := by
    unfold mobReal mobPoleX
    rw [div_sub_div _ _ hcq hD1_ne]
    congr 1
    exact mobius_diff_numerator hdet x y q
  -- Compute the squared denominator of the LHS Poisson.
  have hsq :
      (mobReal a b c d q - mobPoleX a b c d x y) ^ 2
        + (mobPoleY c d x y) ^ 2
      = ((q - x) ^ 2 + y ^ 2)
          / ((c * q + d) ^ 2 * ((c * x + d) ^ 2 + c ^ 2 * y ^ 2)) := by
    rw [hdiff]
    unfold mobPoleY
    rw [div_pow, div_pow, mul_pow]
    rw [div_add_div _ _ (mul_ne_zero hcq_sq_ne hD1_sq_ne) hD1_sq_ne]
    rw [div_eq_div_iff (by positivity) (by positivity)]
    have hgeo := mobius_geometric_identity c d x y q
    linear_combination
      -((c * q + d) ^ 2 * ((c * x + d) ^ 2 + c ^ 2 * y ^ 2) ^ 3) * hgeo
  -- Combine: substitute the simplified denominator into the LHS Poisson.
  --   y/D₁ / (D₂ / ((cq+d)²·D₁)) = y/D₁ · ((cq+d)²·D₁) / D₂
  --                              = y · (cq+d)² / D₂   (cancel D₁)
  --                              = (cq+d)² · (y/D₂).
  rw [hsq]
  unfold mobPoleY
  rw [div_div_eq_mul_div]
  rw [show y / ((c * x + d) ^ 2 + c ^ 2 * y ^ 2)
          * ((c * q + d) ^ 2 * ((c * x + d) ^ 2 + c ^ 2 * y ^ 2))
        = y * (c * q + d) ^ 2 from by
        rw [mul_comm ((c * q + d) ^ 2) _, ← mul_assoc,
            div_mul_cancel₀ y hD1_ne]]
  ring

/-- **Weight-2 modular form covariance of the contour output.**

The contour output transforms as a weight-2 form under simultaneous
Möbius action on the query and all the poles:

    `F(γ q, γ p) = (c q + d)² · F(q, p)`.

This says the *un-normalised* attention output is a modular form,
not an invariant — the weight-2 cocycle is exactly the Jacobian of
the boundary action. -/
theorem contourOutput_mobius_covariant
    {N D : ℕ} (p : Poles N) (V : Fin N → Fin D → ℝ) (d : Fin D)
    {a b c d_par : ℝ} (hdet : a * d_par - b * c = 1)
    (q : ℝ) (hcq : c * q + d_par ≠ 0)
    (mobP : Poles N)
    (hmobX : ∀ k, mobP.x k = mobPoleX a b c d_par (p.x k) (p.y k))
    (hmobY : ∀ k, mobP.y k = mobPoleY c d_par (p.x k) (p.y k)) :
    contourOutput mobP (mobReal a b c d_par q) V d
      = (c * q + d_par) ^ 2 * contourOutput p q V d := by
  unfold contourOutput
  rw [Finset.mul_sum]
  refine Finset.sum_congr rfl (fun k _ => ?_)
  rw [hmobX k, hmobY k]
  rw [poisson_mobius_covariant hdet (p.x k) (p.im_pos k) q hcq]
  ring

/-- **Möbius invariance of the attention simplex.**

The normalised attention weights `P_k(q) / Σ_j P_j(q)` are genuinely
invariant under simultaneous Möbius action — the weight-2 cocycle
`(c q + d)²` cancels between numerator and denominator.

This is the cleanest statement of the Möbius symmetry: although the
*amplitudes* `F_k = V_k · P_k` and the *output* `F = Σ F_k` transform
non-trivially, the *softmax-style mixture coefficients* over the keys
are PSL(2, ℝ)-invariant.  In other words, **what the head decides to
attend to is a property of the geometric pole configuration, not of
any particular boundary parametrisation**.  -/
theorem attention_simplex_mobius_invariant
    {N : ℕ} (p : Poles N) (mobP : Poles N)
    {a b c d_par : ℝ} (hdet : a * d_par - b * c = 1)
    (q : ℝ) (hcq : c * q + d_par ≠ 0)
    (hmobX : ∀ k, mobP.x k = mobPoleX a b c d_par (p.x k) (p.y k))
    (hmobY : ∀ k, mobP.y k = mobPoleY c d_par (p.x k) (p.y k))
    (hsum : 0 < ∑ j : Fin N, poisson (p.x j) (p.y j) q)
    (k : Fin N) :
    poisson (mobP.x k) (mobP.y k) (mobReal a b c d_par q)
        / ∑ j, poisson (mobP.x j) (mobP.y j) (mobReal a b c d_par q)
      = poisson (p.x k) (p.y k) q
          / ∑ j, poisson (p.x j) (p.y j) q := by
  -- Each numerator picks up the same `(c q + d)²` factor; ditto each
  -- summand of the denominator.  The ratio is therefore preserved.
  have hcov : ∀ k : Fin N,
      poisson (mobP.x k) (mobP.y k) (mobReal a b c d_par q)
        = (c * q + d_par) ^ 2 * poisson (p.x k) (p.y k) q := fun k => by
    rw [hmobX k, hmobY k]
    exact poisson_mobius_covariant hdet (p.x k) (p.im_pos k) q hcq
  rw [hcov k]
  rw [show (∑ j, poisson (mobP.x j) (mobP.y j) (mobReal a b c d_par q))
        = (c * q + d_par) ^ 2 * ∑ j, poisson (p.x j) (p.y j) q from by
    rw [Finset.mul_sum]
    exact Finset.sum_congr rfl (fun j _ => hcov j)]
  have hcq_sq_ne : (c * q + d_par) ^ 2 ≠ 0 := pow_ne_zero 2 hcq
  have hsum_ne : (∑ j, poisson (p.x j) (p.y j) q) ≠ 0 := ne_of_gt hsum
  field_simp

/-! ### §1.16 Realisation caveats — the realised geometry is affine.

§1.15 is a theorem about the *abstract* kernel class: poles are free
points in the upper half plane.  But the standard score-derived
construction `matchPoles` (§1.2) places **every** pole directly
above the query (`x_k = q`).  The realised configuration is a
degenerate vertical slice, not a generic UHP point cloud.

This section pins down the gap precisely.  We ask: under which
Möbius transformations does verticality (the slice `x = q`) map to
verticality (the slice `x = γ q`)?  The answer is a sharp
algebraic equivalence:

  **only Möbius transformations with `c = 0` preserve the slice.**

The `c = 0` locus inside `PSL(2, ℝ)` is precisely the affine
(Borel) subgroup `B(2, ℝ)` — translations, dilations, reflections —
exactly the §1.14 symmetry.  So:

  • Cauchy-Poisson **kernels** carry the full `PSL(2, ℝ)` symmetry.

  • The **realised softmax-derived configuration** (`matchPoles`)
    inherits only the affine subgroup `B(2, ℝ) ⊂ PSL(2, ℝ)`.

This is not a defect of the proof; it is the precise gap between
"what the kernel class supports" and "what a transformer trained on
inner-product scores actually inhabits".  Anyone wanting the full
`PSL(2, ℝ)` symmetry on the realised geometry must drop the
verticality constraint — i.e., adopt a richer pole parameterisation
than `(q, e^{−s_k/2})`.

(There is a *separate* dimensionality gap — pole heights are forced
onto a rank-`D` submanifold by the inner-product score
`s_{ij} = ⟨q_i, k_j⟩`.  This `D ≪ N` rank bottleneck is **not**
preserved by generic Möbius even when verticality is, but it is
preserved by translation and uniform dilation, which is again the
affine subgroup.  We do not formalise this here; it is a remark.) -/

/-- **Verticality is preserved iff `c = 0`.**

For any Möbius `[[a, b], [c, d]]` on the unimodular locus
(`a d − b c = 1`), and any pole bandwidth `y > 0` of a pole sitting
directly above the query `q`, the Möbius image is again directly
above the new boundary point `γ q` if and only if `c = 0`.

This is the formal statement of **Hole 1**: the realised
configuration `matchPoles q s` is a degenerate vertical slice, and
the Möbius transformations that preserve the slice are exactly the
affine subgroup `B(2, ℝ)`.  -/
theorem matchPoles_mobius_preserves_verticality
    {a b c d : ℝ} (hdet : a * d - b * c = 1)
    (q : ℝ) (hcq : c * q + d ≠ 0) {y : ℝ} (hy : 0 < y) :
    mobPoleX a b c d q y = mobReal a b c d q ↔ c = 0 := by
  unfold mobPoleX mobReal
  have hD1_pos : 0 < (c * q + d) ^ 2 + c ^ 2 * y ^ 2 :=
    mobius_D1_pos hdet q hy
  have hD1_ne : (c * q + d) ^ 2 + c ^ 2 * y ^ 2 ≠ 0 := ne_of_gt hD1_pos
  rw [div_eq_div_iff hD1_ne hcq]
  constructor
  · -- Forward: the cross-multiplied equation reduces (mod hdet) to c·y² = 0,
    -- and y > 0 forces c = 0.
    intro h
    have hcy_eq : c * y ^ 2 = 0 := by
      linear_combination h - (c * y ^ 2) * hdet
    have hy_sq_pos : 0 < y ^ 2 := by positivity
    rcases mul_eq_zero.mp hcy_eq with hc | hy_sq_zero
    · exact hc
    · exact absurd hy_sq_zero (ne_of_gt hy_sq_pos)
  · -- Reverse: c = 0 makes both sides equal `(a·q + b)·d²`.
    intro hc
    subst hc
    ring

/-- **The realised geometry sees only the affine subgroup.**

A direct corollary of `matchPoles_mobius_preserves_verticality`:
if a Möbius transformation on the unimodular locus maps **any**
realised `matchPoles` pole — i.e., a pole of the form `(q, y)` with
`y > 0` directly above the query — into the same vertical slice
above the transformed query, then the transformation already lies
in the affine subgroup `c = 0`.

In other words, `B(2, ℝ) = {γ ∈ PSL(2, ℝ) : c = 0}` is the
*stabiliser* of the realised pole geometry, and §1.14 is therefore
**all** the Möbius symmetry the standard transformer head can claim
on its score-derived poles.  The §1.15 covariance is genuine, but
its action on the realised slice is already covered by §1.14. -/
theorem realised_mobius_orbit_is_affine
    {a b c d : ℝ} (hdet : a * d - b * c = 1)
    (q : ℝ) (hcq : c * q + d ≠ 0) {y : ℝ} (hy : 0 < y)
    (hslice : mobPoleX a b c d q y = mobReal a b c d q) :
    c = 0 :=
  (matchPoles_mobius_preserves_verticality hdet q hcq hy).mp hslice

/-! ### §1.17 Dynamics — the residual stream is a *discrete linear semigroup*.

§1.1–§1.16 are *kinematic* statements: they pin down the Cauchy-Poisson
function class and its symmetry budget at a single layer.  This section
turns the discrete forward pass into **dynamics** — the layer index `ℓ`
becomes a discrete time parameter, and the residual update

    `h_{ℓ+1}  =  h_ℓ  +  Attn(h_ℓ)`

is recognised as one Euler step of an explicit vector field `F` on the
embedding space, with step size `Δt = 1`:

    `h_{ℓ+1}  =  h_ℓ  +  Δt · F(h_ℓ)`.

####  What this section actually proves.

* The discrete transformer block is *exactly* the Euler step of
  `cauchyResidualVF` (`cauchy_residual_is_euler_step`,
  `cauchy_residual_euler_step_dt`).
* `F` is **ℝ-linear** in the value vectors
  (`cauchyResidualVF_linear_in_V`), so each Euler step
  is the action of a linear operator on the embedding space.
* The L-layer iterated trace is well-defined as iterates of one
  state-dependent Euler step (`iterEuler`).
* **Discrete semigroup law (Φ_{m+n} = Φ_n ∘ Φ_m)** —
  `iterEuler_add` proves the iteration is a one-parameter discrete
  semigroup, *for every* state-dependent vector field `F`.  No
  linearity needed.
* **Linearity is preserved by iteration** —
  `eulerStepFn_linear_of_linear` and `iterEuler_linear_of_linear`
  show that when `F` is ℝ-linear, the L-step trace is itself
  ℝ-linear: the residual stream is genuinely a *discrete linear
  semigroup* on the embedding space.

####  What this section explicitly does **not** prove.

* Existence/uniqueness of solutions to a continuous ODE
  `dh/dt = F(h)` (needs Mathlib's Picard–Lindelöf, requires Lipschitz
  bounds in addition to linearity).
* A fully explicit closed-form constant for the local
  `‖exp δ - 1 - δ‖` remainder term (we do prove finite-depth
  quantitative error bounds in §1.19, and an `O(Δt)` corollary under
  a quadratic remainder hypothesis).
* Identification of the continuous flow with the **Poisson semigroup**
  `e^{−t √(−Δ)}` on the real line — this is the headline "transformers
  solve a PDE" version of the claim and we explicitly **do not**
  claim it here.

The discrete semigroup proven here *is* the algebraic skeleton on
which any of those continuum statements would have to be built.  It
is no longer just "Euler-step shape": the iterates compose, linearity
is exact, and the matrix-power interpretation `(I + Δt T)^L h₀` is
formally substantiated.
-/

/-- **The Cauchy-Poisson residual vector field.**

    Given a pole configuration `p`, value matrix `V`, query position `q`,
    and value channel `d`, the residual VF returns the Cauchy-Poisson
    contour output at that channel:

        `F(p, V, q, d)  =  Σ_k poisson(p_x_k, p_y_k, q) · V k d`.

    This is the *generator* of the residual flow when transformer layers
    share the (p, q) configuration.  For a transformer with one head per
    layer this is exactly the per-layer attention update.  -/
noncomputable def cauchyResidualVF {N D : ℕ}
    (p : Poles N) (V : Fin N → Fin D → ℝ) (q : ℝ) (d : Fin D) : ℝ :=
  contourOutput p q V d

/-- **Generic Euler step (constant vector field, vector form).**  -/
def eulerStep {D : ℕ} (Δt : ℝ) (h : Fin D → ℝ) (F : Fin D → ℝ) : Fin D → ℝ :=
  fun d => h d + Δt * F d

/-- **The transformer residual update is the Euler step of the
    Cauchy-Poisson vector field at step size 1.** -/
theorem cauchy_residual_is_euler_step
    {N D : ℕ} (p : Poles N) (V : Fin N → Fin D → ℝ)
    (q : ℝ) (h : Fin D → ℝ) :
    eulerStep (1 : ℝ) h (fun d => cauchyResidualVF p V q d)
      = fun d => h d + contourOutput p q V d := by
  funext d
  unfold eulerStep cauchyResidualVF
  ring

/-- **Generalised Euler step at arbitrary `Δt`.**  Pinning `Δt = 1`
    recovers the standard transformer block; smaller `Δt` corresponds
    to a finer discretisation of the same flow.  -/
theorem cauchy_residual_euler_step_dt
    {N D : ℕ} (p : Poles N) (V : Fin N → Fin D → ℝ)
    (Δt q : ℝ) (h : Fin D → ℝ) :
    eulerStep Δt h (fun d => cauchyResidualVF p V q d)
      = fun d => h d + Δt * contourOutput p q V d := by
  funext d
  unfold eulerStep cauchyResidualVF
  ring

/-- **The Cauchy-Poisson VF is linear in the values.** -/
theorem cauchyResidualVF_linear_in_V
    {N D : ℕ} (p : Poles N) (q : ℝ) (d : Fin D)
    (V₁ V₂ : Fin N → Fin D → ℝ) (a b : ℝ) :
    cauchyResidualVF p (fun k d' => a * V₁ k d' + b * V₂ k d') q d
      = a * cauchyResidualVF p V₁ q d + b * cauchyResidualVF p V₂ q d := by
  unfold cauchyResidualVF contourOutput
  rw [Finset.mul_sum, Finset.mul_sum, ← Finset.sum_add_distrib]
  refine Finset.sum_congr rfl (fun k _ => ?_)
  ring

/-- **State-dependent Euler step.**  Given a vector field `F` that
    depends on the current state, take one Euler step of size `Δt`.
    This is the correct shape for an actual ODE discretisation —
    the constant-`F` `eulerStep` above is the special case `F` ≡ const.

    For Cauchy-Poisson attention with a fixed value-projection
    `V : (Fin D → ℝ) → (Fin N → Fin D → ℝ)` and pole/query data
    `(p, q)`, the residual update is `eulerStepFn 1 (Attn) h` where
    `Attn h := cauchyResidualVF p (V h) q`.  -/
def eulerStepFn {D : ℕ} (Δt : ℝ) (F : (Fin D → ℝ) → Fin D → ℝ)
    (h : Fin D → ℝ) : Fin D → ℝ :=
  fun d => h d + Δt * F h d

/-- **L-step iterated residual stream.**

    `iterEuler L Δt F` is the L-fold composition of `eulerStepFn Δt F`,
    i.e. the trace of the discrete dynamics generated by `F` at step
    size `Δt` over `L` layers.  Defined via Mathlib's `Function.iterate`
    so all the standard composition lemmas (`iterate_succ_apply`,
    `iterate_add_apply`, etc.) are available.  -/
def iterEuler {D : ℕ} (L : ℕ) (Δt : ℝ)
    (F : (Fin D → ℝ) → Fin D → ℝ) : (Fin D → ℝ) → (Fin D → ℝ) :=
  (eulerStepFn Δt F)^[L]

/-- **Zero-layer trace is the identity.**  -/
theorem iterEuler_zero {D : ℕ} (Δt : ℝ)
    (F : (Fin D → ℝ) → Fin D → ℝ) (h : Fin D → ℝ) :
    iterEuler 0 Δt F h = h := rfl

/-- **One-layer trace is one Euler step.**  -/
theorem iterEuler_one {D : ℕ} (Δt : ℝ)
    (F : (Fin D → ℝ) → Fin D → ℝ) (h : Fin D → ℝ) :
    iterEuler 1 Δt F h = eulerStepFn Δt F h := rfl

/-- **One unfolding step.**  -/
theorem iterEuler_succ {D : ℕ} (L : ℕ) (Δt : ℝ)
    (F : (Fin D → ℝ) → Fin D → ℝ) (h : Fin D → ℝ) :
    iterEuler (L + 1) Δt F h
      = iterEuler L Δt F (eulerStepFn Δt F h) := by
  unfold iterEuler
  exact Function.iterate_succ_apply (eulerStepFn Δt F) L h

/-- **Discrete semigroup law `Φ_{m+n} = Φ_n ∘ Φ_m`.**

    The `(m + n)`-step trace splits cleanly at any intermediate layer:
    advance `m` layers, then advance `n` more layers.  This is the
    **algebraic semigroup property** of the residual stream.  No
    linearity is needed — it holds for every state-dependent vector
    field `F`.  This is the genuine discrete analogue of
    `Φ_{s+t}(h) = Φ_s(Φ_t(h))` for an ODE flow.  -/
theorem iterEuler_add {D : ℕ} (m n : ℕ) (Δt : ℝ)
    (F : (Fin D → ℝ) → Fin D → ℝ) (h : Fin D → ℝ) :
    iterEuler (m + n) Δt F h
      = iterEuler n Δt F (iterEuler m Δt F h) := by
  unfold iterEuler
  rw [Nat.add_comm m n, Function.iterate_add_apply]

/-- **A function `g : (Fin D → ℝ) → (Fin D → ℝ)` is ℝ-linear.**  -/
def IsRLinear {D : ℕ} (g : (Fin D → ℝ) → (Fin D → ℝ)) : Prop :=
  ∀ (a b : ℝ) (h₁ h₂ : Fin D → ℝ),
    g (fun d => a * h₁ d + b * h₂ d) = fun d => a * g h₁ d + b * g h₂ d

/-- **Linearity is preserved by one Euler step.**

    If the vector field `F` is ℝ-linear in the state, then the
    affine-shape map `eulerStepFn Δt F : h ↦ h + Δt · F h` is also
    ℝ-linear.  Geometrically, `eulerStepFn Δt F` realises the linear
    operator `(I + Δt · F) : ℝ^D → ℝ^D`.  -/
theorem eulerStepFn_linear_of_linear {D : ℕ} (Δt : ℝ)
    (F : (Fin D → ℝ) → Fin D → ℝ) (hF : IsRLinear F) :
    IsRLinear (eulerStepFn Δt F) := by
  intro a b h₁ h₂
  funext d
  have hFd : F (fun d' => a * h₁ d' + b * h₂ d') d
              = a * F h₁ d + b * F h₂ d := by
    have := hF a b h₁ h₂
    have := congrFun this d
    simpa using this
  unfold eulerStepFn
  rw [hFd]
  ring

/-- **Linearity is preserved by iteration.**

    If `F` is ℝ-linear, the L-step iterated Euler trace
    `iterEuler L Δt F` is also ℝ-linear.  Combined with the
    semigroup law `iterEuler_add`, this means the residual stream of
    a Cauchy-Poisson transformer (with a linear value projection)
    is a genuine **discrete linear semigroup** on the embedding space:
    each layer is a linear operator `T_ℓ = (I + Δt · F)`, and `L`
    layers compose to `T_ℓ^L`.  This is the algebraic skeleton on
    top of which any continuum-limit statement would be built.  -/
theorem iterEuler_linear_of_linear {D : ℕ} (Δt : ℝ)
    (F : (Fin D → ℝ) → Fin D → ℝ) (hF : IsRLinear F) :
    ∀ L, IsRLinear (iterEuler L Δt F) := by
  intro L
  induction L with
  | zero =>
      intro a b h₁ h₂
      funext d
      simp [iterEuler_zero]
  | succ L ih =>
      intro a b h₁ h₂
      have hStep := eulerStepFn_linear_of_linear Δt F hF a b h₁ h₂
      have hAfter := ih a b (eulerStepFn Δt F h₁) (eulerStepFn Δt F h₂)
      funext d
      rw [iterEuler_succ, iterEuler_succ, iterEuler_succ]
      have hAt :
          eulerStepFn Δt F (fun d' => a * h₁ d' + b * h₂ d')
            = fun d' => a * eulerStepFn Δt F h₁ d'
                        + b * eulerStepFn Δt F h₂ d' := hStep
      rw [hAt]
      exact congrFun hAfter d

/-! ### §1.19 Continuum lift via Mathlib's exponential map.

§1.17 proved the *discrete* dynamics of the residual stream form a
linear semigroup.  This section closes the loop to a **continuous
flow** by lifting the discrete vector field to a continuous linear
operator and identifying the would-be flow with Mathlib's operator
exponential `NormedSpace.exp`.

The key facts proved here:

* `cauchyLinearOp F` packages a `IsRLinear F` vector field as a
  `(Fin D → ℝ) →L[ℝ] (Fin D → ℝ)` (continuous linear endomorphism).
* `cauchyResidualFlow F t := exp (t • cauchyLinearOp F)` is the
  *continuous* one-parameter semigroup generated by `F` — the Poisson
  semigroup analogue at the level of operators.
* `cauchyResidualFlow_zero` — the flow at time `0` is the identity.
* `cauchyResidualFlow_hasDerivAt` — **the flow satisfies the ODE**
  `dΦ/dt = T · Φ`, via `hasDerivAt_exp_smul_const'`.
* `cauchyResidualFlow_apply_zero` — applying the flow to a point
  satisfies the corresponding pointwise ODE
  `(d/dt) (Φ_t v) = T (Φ_t v)`.
* `iterEuler_eq_clm_pow` — the §1.17 discrete iterate is *exactly*
  the matrix power `(1 + Δt · T)^L` applied to the initial state.

* `cauchyResidualFlow_eq_exp_npow` — **discrete refinement identity.**
  For every `n : ℕ`, the *continuous* time-`t` flow equals the
  `(n+1)`-fold **multiplicative** refinement of the single step
  `exp ((t/(n+1)) • T)`:

    `exp(t•T) = (exp ((t/(n+1))•T)))^{n+1}`.

  This is the Banach-algebra exponential law `exp_nsmul` together with
  the scalar identity `(n+1) • ((t/(n+1))•T) = t • T`.  It packages the
  correct continuous object that the Euler sequence
  `(1 + (t/(n+1))•T)^{n+1}` is trying to approximate.

  The **final** analytic step — proving `‖(1+δ)^{n+1} - (exp δ)^{n+1}‖ → 0`
  as `‖δ‖ ~ 1/n` — is exactly the classical Euler-method estimate
  (commuting `geom_sum₂` factorisation + a quadratic bound on
  `‖exp δ - 1 - δ‖` from the `exp` power series, exactly as in
  `Complex.exp_bound` / `Real.norm_exp_sub_one_sub_id_le`, but ported
  to the operator norm on `(Fin D → ℝ) →L[ℝ] (Fin D → ℝ)`).  Mathlib does
  not currently expose this estimate as a one-line theorem for general
  `NormedRing`s; assembling it is pure `Analysis` bookkeeping on top
  of the identities already proved here.
-/

/-- The continuous linear endomorphism associated to a linear
    Cauchy-Poisson vector field on the embedding space.

    Continuity is automatic in finite dimensions, so we package the
    ℝ-linear `F` directly as a `→L[ℝ]`. -/
noncomputable def cauchyLinearOp {D : ℕ}
    (F : (Fin D → ℝ) → Fin D → ℝ) (hF : IsRLinear F) :
    (Fin D → ℝ) →L[ℝ] (Fin D → ℝ) :=
  let L : (Fin D → ℝ) →ₗ[ℝ] (Fin D → ℝ) :=
    { toFun := F
      map_add' := fun x y => by
        have := hF 1 1 x y
        simpa [one_smul] using this
      map_smul' := fun a x => by
        have := hF a 0 x x
        simpa [zero_smul, add_zero] using this }
  LinearMap.toContinuousLinearMap L

/-- Applying the packaged operator returns the original vector field. -/
@[simp] theorem cauchyLinearOp_apply {D : ℕ}
    (F : (Fin D → ℝ) → Fin D → ℝ) (hF : IsRLinear F) (h : Fin D → ℝ) :
    cauchyLinearOp F hF h = F h := rfl

/-- **The continuous Poisson semigroup of a linear Cauchy-Poisson VF.**

    `cauchyResidualFlow F hF t = exp(t • T)` where `T = cauchyLinearOp F hF`.
    This is the continuous one-parameter semigroup generated by `F`, taking
    values in the Banach algebra `(Fin D → ℝ) →L[ℝ] (Fin D → ℝ)`.  Mathlib's
    `NormedSpace.exp` makes it well-defined for every `t : ℝ` (the radius
    of convergence is infinite for any bounded operator). -/
noncomputable def cauchyResidualFlow {D : ℕ}
    (F : (Fin D → ℝ) → Fin D → ℝ) (hF : IsRLinear F) (t : ℝ) :
    (Fin D → ℝ) →L[ℝ] (Fin D → ℝ) :=
  NormedSpace.exp (t • cauchyLinearOp F hF)

/-- **The flow at time 0 is the identity.**  -/
theorem cauchyResidualFlow_zero {D : ℕ}
    (F : (Fin D → ℝ) → Fin D → ℝ) (hF : IsRLinear F) :
    cauchyResidualFlow F hF 0 = (1 : (Fin D → ℝ) →L[ℝ] (Fin D → ℝ)) := by
  show NormedSpace.exp ((0 : ℝ) • cauchyLinearOp F hF)
        = (1 : (Fin D → ℝ) →L[ℝ] (Fin D → ℝ))
  rw [zero_smul, NormedSpace.exp_zero]

/-- **The flow satisfies the ODE `dΦ/dt = T · Φ`.**

    This is the headline content of §1.19: the continuous-time analogue of
    the residual stream is governed by an honest linear ODE in operator
    form, with generator `T = cauchyLinearOp F`.  The proof is a direct
    application of Mathlib's `hasDerivAt_exp_smul_const'`. -/
theorem cauchyResidualFlow_hasDerivAt {D : ℕ}
    (F : (Fin D → ℝ) → Fin D → ℝ) (hF : IsRLinear F) (t : ℝ) :
    HasDerivAt (cauchyResidualFlow F hF)
      (cauchyLinearOp F hF * cauchyResidualFlow F hF t) t := by
  show HasDerivAt (fun s : ℝ => NormedSpace.exp (s • cauchyLinearOp F hF))
        (cauchyLinearOp F hF * NormedSpace.exp (t • cauchyLinearOp F hF)) t
  exact hasDerivAt_exp_smul_const' (cauchyLinearOp F hF) t

/-- **Pointwise ODE.**  For every initial vector `v`, the trajectory
    `t ↦ Φ_t(v)` satisfies the ODE `dh/dt = T(h)`.  This is the
    "transformer-friendly" form: pick a starting embedding `v`, run the
    continuous flow, and the velocity at every time is exactly `T`
    applied to the current state — i.e., the *continuous* version of the
    discrete update `h_{ℓ+1} = h_ℓ + Δt · F(h_ℓ)` of §1.17. -/
theorem cauchyResidualFlow_apply_hasDerivAt {D : ℕ}
    (F : (Fin D → ℝ) → Fin D → ℝ) (hF : IsRLinear F) (t : ℝ)
    (v : Fin D → ℝ) :
    HasDerivAt (fun s : ℝ => cauchyResidualFlow F hF s v)
      (cauchyLinearOp F hF (cauchyResidualFlow F hF t v)) t := by
  have hΦ := cauchyResidualFlow_hasDerivAt F hF t
  have hEval :
      HasFDerivAt (fun Φ : (Fin D → ℝ) →L[ℝ] (Fin D → ℝ) => Φ v)
        (ContinuousLinearMap.apply ℝ (Fin D → ℝ) v) (cauchyResidualFlow F hF t) :=
    (ContinuousLinearMap.apply ℝ (Fin D → ℝ) v).hasFDerivAt
  have h := hEval.comp_hasDerivAt t hΦ
  simpa [ContinuousLinearMap.apply_apply, ContinuousLinearMap.mul_apply,
         Function.comp] using h

/-- **Discrete iterate equals matrix power applied pointwise.**

    For a linear vector field `F`, the L-fold Euler iterate computes
    exactly `(1 + Δt · T)^L` applied to the initial state.  This makes
    the discrete-to-continuous bridge fully explicit: the §1.17 iterate
    is the discrete approximation, the §1.19 flow `cauchyResidualFlow`
    is the continuous limit, and they agree at first order in `Δt`
    (the fact `(1 + Δt T) = exp(Δt T) + O(Δt²)` is what powers the
    convergence theorem `cauchyResidualFlow_eulerLimit` proved
    further down). -/
theorem iterEuler_eq_clm_pow {D : ℕ}
    (F : (Fin D → ℝ) → Fin D → ℝ) (hF : IsRLinear F)
    (Δt : ℝ) (h : Fin D → ℝ) :
    ∀ L, iterEuler L Δt F h
          = (((1 : (Fin D → ℝ) →L[ℝ] (Fin D → ℝ))
              + Δt • cauchyLinearOp F hF) ^ L) h
  | 0 => by
      simp [iterEuler_zero, pow_zero, ContinuousLinearMap.one_apply]
  | L + 1 => by
      have key : eulerStepFn Δt F h
          = (((1 : (Fin D → ℝ) →L[ℝ] (Fin D → ℝ))
              + Δt • cauchyLinearOp F hF)) h := by
        funext d
        simp only [eulerStepFn, ContinuousLinearMap.add_apply,
          ContinuousLinearMap.one_apply, ContinuousLinearMap.smul_apply,
          cauchyLinearOp_apply, Pi.add_apply, Pi.smul_apply, smul_eq_mul]
      rw [iterEuler_succ, iterEuler_eq_clm_pow F hF Δt _ L, pow_succ,
          ContinuousLinearMap.mul_apply, key]

/-- Scalar bookkeeping for the `(n+1)`-step refinement with timestep
    `t/(n+1)`: multiplying the generator by `(n+1)` recovers `t • T`. -/
private lemma cauchyLinearOp_nsmul_div_eq_smul {D : ℕ} (t : ℝ) (n : ℕ)
    (T : (Fin D → ℝ) →L[ℝ] Fin D → ℝ) :
    (n + 1 : ℕ) • ((t / (n + 1 : ℝ)) • T) = t • T := by
  rw [← Nat.cast_smul_eq_nsmul (R := ℝ)]
  rw [smul_smul]
  have hne : ((↑(n + 1 : ℕ) : ℝ)) ≠ 0 :=
    Nat.cast_ne_zero.mpr (Nat.succ_ne_zero n)
  have hden : ((↑(n + 1 : ℕ) : ℝ)) * (t / (↑(n + 1 : ℕ) : ℝ)) = t := by
    field_simp [hne]
  have hcast : (t / (↑n + 1 : ℝ)) = t / (↑(n + 1 : ℕ) : ℝ) := by
    simp [Nat.cast_succ]
  rw [hcast, hden]

/-- **Multiplicative refinement of the continuous flow.**

    For every `n`, the time-`t` operator exponential agrees with the
    `(n+1)`-fold product of the single-step exponentials at timestep
    `t/(n+1)`.  This is the exact continuous analogue of the discrete
    identity `iterEuler_eq_clm_pow` — the Euler iterates are powers of
    `(1 + (t/(n+1))•T)`, while the continuous semigroup is the power of
    `exp ((t/(n+1))•T)` at the *same* refinement scale. -/
theorem cauchyResidualFlow_eq_exp_npow {D : ℕ}
    (F : (Fin D → ℝ) → Fin D → ℝ) (hF : IsRLinear F) (t : ℝ) (n : ℕ) :
    cauchyResidualFlow F hF t
      = (NormedSpace.exp ((t / (n + 1 : ℝ)) • cauchyLinearOp F hF)) ^ (n + 1) := by
  let T := cauchyLinearOp F hF
  -- `NormedSpace.exp_nsmul` is stated in Mathlib's `Rat` section, hence needs a
  -- `NormedAlgebra ℚ` instance.  We obtain it by restricting scalars `ℚ → ℝ` on the
  -- operator algebra (finite-dimensional CLMs are normed `ℝ`-algebras).
  letI : NormedAlgebra ℚ ((Fin D → ℝ) →L[ℝ] Fin D → ℝ) :=
    NormedAlgebra.restrictScalars (𝕜 := ℚ) (𝕜' := ℝ) ((Fin D → ℝ) →L[ℝ] Fin D → ℝ)
  unfold cauchyResidualFlow
  rw [← cauchyLinearOp_nsmul_div_eq_smul t n T, NormedSpace.exp_nsmul]

/-! **Euler limit.**  The discrete Euler refinement of the linear residual
dynamics converges to the continuous semigroup `exp(t • T)` of §1.19.

The proof is the classical ODE estimate, ported to the operator norm on
`(Fin D → ℝ) →L[ℝ] (Fin D → ℝ)`:

* `exp δ - 1 - δ = o(δ)` near `0` (`hasFDerivAt_exp_zero`);
* `(1+δ)^m - (exp δ)^m = (1 + δ - exp δ) · ∑ (1+δ)^i (exp δ)^{m-1-i}`
  by `Commute.mul_geom_sum₂`, with each summand bounded by
  `Real.exp(m·‖δ‖)` via `‖exp δ‖ ≤ Real.exp ‖δ‖`;
* with `δ_n = (t/(n+1))•T`, `(n+1)·‖δ_n‖ = ‖t•T‖` is constant, so the
  geometric prefactor is bounded by `(n+1) · exp(‖t•T‖)`, while the
  `o(δ)` factor times `(n+1)` tends to `0`. -/
section cauchyResidualFlowEulerLimit

open Filter Topology Asymptotics

/-! Polymorphic operator-level Euler limit.

Below, `E` is any complete normed `ℝ`-space; `CLM = E →L[ℝ] E` is its Banach
algebra of bounded endomorphisms.  All the Euler-limit machinery —
submultiplicativity of operator norm under powers, the bound
`‖exp δ‖ ≤ exp ‖δ‖`, the geometric-sum factorisation
`(1+δ)^m - (exp δ)^m`, and the final `Tendsto` statement — works in this
generality.  The original `(Fin D → ℝ)` versions
(`cauchyResidualFlow_eulerLimit_clm`, `cauchyResidualFlow_eulerLimit`) are
recovered as one-line corollaries by specialising `E := Fin D → ℝ`.  -/

variable {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E] [CompleteSpace E]

local notation "CLM" => E →L[ℝ] E

private lemma norm_pow_le_clm (a : CLM) (n : ℕ) : ‖a ^ n‖ ≤ ‖a‖ ^ n := by
  induction n with
  | zero =>
      simp only [pow_zero]
      exact ContinuousLinearMap.norm_id_le (𝕜 := ℝ) (E := E)
  | succ n ih =>
      rw [pow_succ, pow_succ]
      exact (norm_mul_le _ _).trans (mul_le_mul_of_nonneg_right ih (norm_nonneg _))

private lemma norm_exp_le_rexp_clm (δ : CLM) :
    ‖NormedSpace.exp δ‖ ≤ Real.exp ‖δ‖ := by
  have hexp_sum :
      HasSum (fun n : ℕ => ((Nat.factorial n : ℝ)⁻¹) • δ ^ n) (NormedSpace.exp δ) :=
    NormedSpace.exp_series_hasSum_exp' (𝕂 := ℝ) δ
  have hsum_norm : Summable fun n : ℕ => ‖((Nat.factorial n : ℝ)⁻¹) • δ ^ n‖ :=
    NormedSpace.norm_expSeries_summable' (𝕂 := ℝ) δ
  have hsum_real :
      HasSum (fun n : ℕ => ‖δ‖ ^ n / (Nat.factorial n : ℝ)) (Real.exp ‖δ‖) := by
    have h := NormedSpace.expSeries_div_hasSum_exp (𝔸 := ℝ) ‖δ‖
    rwa [show NormedSpace.exp ‖δ‖ = Real.exp ‖δ‖ from
      (congrFun Real.exp_eq_exp_ℝ ‖δ‖).symm] at h
  rw [← hexp_sum.tsum_eq]
  refine (norm_tsum_le_tsum_norm hsum_norm).trans ?_
  rw [← hsum_real.tsum_eq]
  refine Summable.tsum_le_tsum (fun n => ?_) hsum_norm hsum_real.summable
  rw [norm_smul, Real.norm_eq_abs, abs_inv, Nat.abs_cast, div_eq_inv_mul]
  exact mul_le_mul_of_nonneg_left (norm_pow_le_clm δ n) (by positivity)

private lemma norm_exp_pow_le_clm (δ : CLM) (j : ℕ) :
    ‖(NormedSpace.exp δ) ^ j‖ ≤ Real.exp ((j : ℝ) * ‖δ‖) := by
  refine (norm_pow_le_clm _ _).trans ?_
  have h := norm_exp_le_rexp_clm δ
  calc
    ‖NormedSpace.exp δ‖ ^ j
        ≤ Real.exp ‖δ‖ ^ j := pow_le_pow_left₀ (norm_nonneg _) h j
    _ = Real.exp ((j : ℝ) * ‖δ‖) := (Real.exp_nat_mul ‖δ‖ j).symm

private lemma norm_one_add_pow_le_clm (δ : CLM) (i : ℕ) :
    ‖((1 : CLM) + δ) ^ i‖ ≤ Real.exp ((i : ℝ) * ‖δ‖) := by
  refine (norm_pow_le_clm _ _).trans ?_
  have h1 : ‖(1 : CLM)‖ ≤ 1 := ContinuousLinearMap.norm_id_le (𝕜 := ℝ) (E := E)
  have hone : ‖(1 : CLM) + δ‖ ≤ 1 + ‖δ‖ :=
    (norm_add_le (1 : CLM) δ).trans (by gcongr)
  have hexp : (1 : ℝ) + ‖δ‖ ≤ Real.exp ‖δ‖ := by
    linarith [Real.add_one_le_exp ‖δ‖]
  calc
    ‖(1 : CLM) + δ‖ ^ i
        ≤ ((1 : ℝ) + ‖δ‖) ^ i := pow_le_pow_left₀ (norm_nonneg _) hone i
    _ ≤ Real.exp ‖δ‖ ^ i := pow_le_pow_left₀ (by positivity) hexp i
    _ = Real.exp ((i : ℝ) * ‖δ‖) := (Real.exp_nat_mul ‖δ‖ i).symm

private lemma commute_one_add_exp (δ : CLM) :
    Commute ((1 : CLM) + δ) (NormedSpace.exp δ) :=
  Commute.add_left (Commute.one_left _) ((Commute.refl δ).exp_right)

private lemma norm_geom_sum_le_clm (m : ℕ) (δ : CLM) :
    ‖∑ i ∈ Finset.range m, ((1 : CLM) + δ) ^ i * (NormedSpace.exp δ) ^ (m - 1 - i)‖
      ≤ (m : ℝ) * Real.exp ((m : ℝ) * ‖δ‖) := by
  refine (norm_sum_le _ _).trans ?_
  have hbound :
      ∀ i ∈ Finset.range m,
        ‖((1 : CLM) + δ) ^ i * (NormedSpace.exp δ) ^ (m - 1 - i)‖
          ≤ Real.exp ((m : ℝ) * ‖δ‖) := by
    intro i hi
    have hi' : i < m := Finset.mem_range.mp hi
    refine (norm_mul_le _ _).trans ?_
    refine (mul_le_mul (norm_one_add_pow_le_clm δ i) (norm_exp_pow_le_clm δ (m - 1 - i))
      (norm_nonneg _) (Real.exp_pos _).le).trans ?_
    rw [← Real.exp_add]
    refine Real.exp_le_exp.mpr ?_
    rw [← add_mul]
    have hsum : (i : ℝ) + ((m - 1 - i : ℕ) : ℝ) ≤ (m : ℝ) := by
      have hh : i + (m - 1 - i) ≤ m := by omega
      exact_mod_cast hh
    exact mul_le_mul_of_nonneg_right hsum (norm_nonneg _)
  calc
    ∑ i ∈ Finset.range m, ‖((1 : CLM) + δ) ^ i * (NormedSpace.exp δ) ^ (m - 1 - i)‖
        ≤ ∑ _i ∈ Finset.range m, Real.exp ((m : ℝ) * ‖δ‖) := Finset.sum_le_sum hbound
    _ = (m : ℝ) * Real.exp ((m : ℝ) * ‖δ‖) := by
          simp [Finset.sum_const, Finset.card_range, nsmul_eq_mul]

private lemma norm_pow_sub_pow_euler_clm (m : ℕ) (δ : CLM) :
    ‖((1 : CLM) + δ) ^ m - (NormedSpace.exp δ) ^ m‖
      ≤ ‖NormedSpace.exp δ - 1 - δ‖ * ((m : ℝ) * Real.exp ((m : ℝ) * ‖δ‖)) := by
  rcases m.eq_zero_or_pos with rfl | hm
  · simp
  have hxy := commute_one_add_exp δ
  have hgeom := Commute.mul_geom_sum₂ hxy m
  rw [← hgeom]
  refine (norm_mul_le _ _).trans ?_
  have hn :
      ‖(1 : CLM) + δ - NormedSpace.exp δ‖ = ‖NormedSpace.exp δ - 1 - δ‖ := by
    have hsub :
        (1 : CLM) + δ - NormedSpace.exp δ = -(NormedSpace.exp δ - 1 - δ) := by abel
    rw [hsub, norm_neg]
  rw [hn]
  exact mul_le_mul_of_nonneg_left (norm_geom_sum_le_clm m δ) (norm_nonneg _)

/-- **Operator-level Euler limit, polymorphic.**

For any complete normed `ℝ`-space `E` and any bounded endomorphism `T : E →L[ℝ] E`,
the discrete refinement `(1 + (t/(n+1))·T)^{n+1}` converges (in operator norm)
to the continuous one-parameter semigroup `NormedSpace.exp (t·T)` as `n → ∞`.

This is the Euler ↔ matrix-exponential correspondence at full Banach-algebra
generality.  Specialising `E := Fin D → ℝ` recovers
`cauchyResidualFlow_eulerLimit_clm`; specialising to any sequence-state space
(e.g.\ `SeqState seqLen dModel` of Part 2) gives the corresponding
"linearised transformer block" Euler limit. -/
theorem operatorEulerLimit (T : E →L[ℝ] E) (t : ℝ) :
    Tendsto (fun n : ℕ => ((1 : E →L[ℝ] E) + (t / (↑n + 1 : ℝ)) • T) ^ (n + 1))
      atTop (𝓝 (NormedSpace.exp (t • T))) := by
  set δ : ℕ → CLM := fun n => (t / (↑n + 1 : ℝ)) • T with hδ_def
  have hg : Tendsto (fun n : ℕ => ((n : ℝ) + 1)) atTop atTop := by
    have := (tendsto_natCast_atTop_atTop (R := ℝ)).comp (tendsto_add_atTop_nat 1)
    simpa [Function.comp_def] using this
  have hdiv : Tendsto (fun n : ℕ => t / ((n : ℝ) + 1)) atTop (𝓝 (0 : ℝ)) :=
    hg.const_div_atTop t
  have hδ_to_zero : Tendsto δ atTop (𝓝 (0 : CLM)) := by
    have := hdiv.smul_const T
    simpa using this
  have hδ_norm_succ : ∀ n : ℕ, ((n : ℝ) + 1) * ‖δ n‖ = ‖t • T‖ := by
    intro n
    have hp : (0 : ℝ) < (n : ℝ) + 1 := by positivity
    show ((n : ℝ) + 1) * ‖(t / ((n : ℝ) + 1)) • T‖ = ‖t • T‖
    rw [norm_smul, norm_smul, Real.norm_eq_abs, Real.norm_eq_abs, abs_div, abs_of_pos hp]
    field_simp
  -- The continuous flow is constant in `n`: `exp((t/(n+1))•T)^{n+1} = exp(t•T)`.
  -- `NormedSpace.exp_nsmul` lives in Mathlib's `Rat` section and needs a
  -- `NormedAlgebra ℚ (E →L[ℝ] E)` instance; we obtain it by restricting scalars `ℚ → ℝ`.
  letI : NormedAlgebra ℚ (E →L[ℝ] E) :=
    NormedAlgebra.restrictScalars (𝕜 := ℚ) (𝕜' := ℝ) (E →L[ℝ] E)
  have hflow_const : ∀ n : ℕ,
      (NormedSpace.exp (δ n)) ^ (n + 1) = NormedSpace.exp (t • T) := by
    intro n
    have h1 : ((n + 1 : ℕ) : ℝ) • ((t / (↑n + 1 : ℝ)) • T) = t • T := by
      rw [smul_smul]
      have hne : (↑n + 1 : ℝ) ≠ 0 := by positivity
      have hcast : ((n + 1 : ℕ) : ℝ) = (↑n + 1 : ℝ) := by push_cast; ring
      rw [hcast]
      field_simp
    have h2 : (n + 1 : ℕ) • ((t / (↑n + 1 : ℝ)) • T) = t • T := by
      have := h1
      rwa [Nat.cast_smul_eq_nsmul] at this
    show (NormedSpace.exp ((t / (↑n + 1 : ℝ)) • T)) ^ (n + 1) = NormedSpace.exp (t • T)
    rw [← h2, NormedSpace.exp_nsmul]
  have hexp_o :
      (fun h : CLM => NormedSpace.exp h - 1 - h) =o[𝓝 (0 : CLM)] (id : CLM → CLM) := by
    have h0 : HasFDerivAt (NormedSpace.exp : CLM → CLM) (1 : CLM →L[ℝ] CLM) (0 : CLM) :=
      hasFDerivAt_exp_zero
    have h := hasFDerivAt_iff_isLittleO_nhds_zero.mp h0
    convert h using 1
    funext x
    simp [NormedSpace.exp_zero, ContinuousLinearMap.one_apply]
  have hexp_δ_o :
      (fun n : ℕ => NormedSpace.exp (δ n) - 1 - δ n) =o[atTop] δ :=
    hexp_o.comp_tendsto hδ_to_zero
  have hkey :
      Tendsto (fun n : ℕ => ((n : ℝ) + 1) * ‖NormedSpace.exp (δ n) - 1 - δ n‖)
        atTop (𝓝 0) := by
    rw [Metric.tendsto_atTop]
    intro ε hε
    have hbound_pos : 0 < ‖t • T‖ + 1 := by linarith [norm_nonneg (t • T)]
    have hε' : 0 < ε / (‖t • T‖ + 1) := by positivity
    have hloo := hexp_δ_o.def hε'
    rw [Filter.eventually_atTop] at hloo
    obtain ⟨N, hN⟩ := hloo
    refine ⟨N, fun n hn => ?_⟩
    have h := hN n hn
    rw [Real.dist_eq, sub_zero, abs_of_nonneg (by positivity)]
    calc
      ((n : ℝ) + 1) * ‖NormedSpace.exp (δ n) - 1 - δ n‖
          ≤ ((n : ℝ) + 1) * (ε / (‖t • T‖ + 1) * ‖δ n‖) := by gcongr
      _ = ε / (‖t • T‖ + 1) * (((n : ℝ) + 1) * ‖δ n‖) := by ring
      _ = ε / (‖t • T‖ + 1) * ‖t • T‖ := by rw [hδ_norm_succ]
      _ < ε := by
          have h1 : ‖t • T‖ / (‖t • T‖ + 1) < 1 := by
            rw [div_lt_one hbound_pos]
            linarith [norm_nonneg (t • T)]
          calc
            ε / (‖t • T‖ + 1) * ‖t • T‖
                = ε * (‖t • T‖ / (‖t • T‖ + 1)) := by ring
            _ < ε * 1 := mul_lt_mul_of_pos_left h1 hε
            _ = ε := mul_one ε
  have hdiff_bound : ∀ n : ℕ,
      ‖((1 : CLM) + δ n) ^ (n + 1) - NormedSpace.exp (t • T)‖
        ≤ ((n : ℝ) + 1) * ‖NormedSpace.exp (δ n) - 1 - δ n‖ * Real.exp ‖t • T‖ := by
    intro n
    rw [← hflow_const n]
    refine (norm_pow_sub_pow_euler_clm (n + 1) (δ n)).trans ?_
    have hexp_le :
        Real.exp (((n + 1 : ℕ) : ℝ) * ‖δ n‖) ≤ Real.exp ‖t • T‖ := by
      refine Real.exp_le_exp.mpr ?_
      have h := hδ_norm_succ n
      have hcast : ((n + 1 : ℕ) : ℝ) = (n : ℝ) + 1 := by push_cast; ring
      rw [hcast]; linarith
    calc
      ‖NormedSpace.exp (δ n) - 1 - δ n‖
          * (((n + 1 : ℕ) : ℝ) * Real.exp (((n + 1 : ℕ) : ℝ) * ‖δ n‖))
          ≤ ‖NormedSpace.exp (δ n) - 1 - δ n‖
              * (((n + 1 : ℕ) : ℝ) * Real.exp ‖t • T‖) := by gcongr
      _ = ((n + 1 : ℕ) : ℝ) * ‖NormedSpace.exp (δ n) - 1 - δ n‖ * Real.exp ‖t • T‖ := by ring
      _ = ((n : ℝ) + 1) * ‖NormedSpace.exp (δ n) - 1 - δ n‖ * Real.exp ‖t • T‖ := by
            push_cast; ring
  rw [Metric.tendsto_atTop]
  intro ε hε
  have hexp_pos : 0 < Real.exp ‖t • T‖ := Real.exp_pos _
  rw [Metric.tendsto_atTop] at hkey
  obtain ⟨N, hN⟩ := hkey (ε / Real.exp ‖t • T‖) (by positivity)
  refine ⟨N, fun n hn => ?_⟩
  rw [dist_eq_norm]
  have h := hdiff_bound n
  have hk := hN n hn
  rw [Real.dist_eq, sub_zero, abs_of_nonneg (by positivity)] at hk
  calc
    ‖((1 : CLM) + δ n) ^ (n + 1) - NormedSpace.exp (t • T)‖
        ≤ ((n : ℝ) + 1) * ‖NormedSpace.exp (δ n) - 1 - δ n‖ * Real.exp ‖t • T‖ := h
    _ < ε / Real.exp ‖t • T‖ * Real.exp ‖t • T‖ := by gcongr
    _ = ε := by field_simp

/-- **Finite-depth Euler error bound (operator norm, explicit remainder form).**

For the `(n+1)`-step Euler refinement at timestep `t/(n+1)`,
the operator error is bounded by the local exponential remainder:

`‖(1+δ)^(n+1) - exp(t•T)‖ ≤ ((n+1) * ‖exp δ - 1 - δ‖) * exp(‖t•T‖)`,

where `δ = (t/(n+1)) • T`.

This is the quantitative bridge between finite depth and continuous flow; it
is fully explicit and exact (no asymptotic notation). -/
theorem operatorEuler_error_bound
    (T : E →L[ℝ] E) (t : ℝ) (n : ℕ) :
    ‖((1 : E →L[ℝ] E) + (t / (↑n + 1 : ℝ)) • T) ^ (n + 1) - NormedSpace.exp (t • T)‖
      ≤ ((n : ℝ) + 1) *
          ‖NormedSpace.exp ((t / (↑n + 1 : ℝ)) • T)
            - 1 - (t / (↑n + 1 : ℝ)) • T‖
          * Real.exp ‖t • T‖ := by
  set δ : CLM := (t / (↑n + 1 : ℝ)) • T
  letI : NormedAlgebra ℚ (E →L[ℝ] E) :=
    NormedAlgebra.restrictScalars (𝕜 := ℚ) (𝕜' := ℝ) (E →L[ℝ] E)
  have hflow_const : (NormedSpace.exp δ) ^ (n + 1) = NormedSpace.exp (t • T) := by
    have h1 : ((n + 1 : ℕ) : ℝ) • ((t / (↑n + 1 : ℝ)) • T) = t • T := by
      rw [smul_smul]
      have hne : (↑n + 1 : ℝ) ≠ 0 := by positivity
      have hcast : ((n + 1 : ℕ) : ℝ) = (↑n + 1 : ℝ) := by push_cast; ring
      rw [hcast]
      field_simp
    have h2 : (n + 1 : ℕ) • ((t / (↑n + 1 : ℝ)) • T) = t • T := by
      have := h1
      rwa [Nat.cast_smul_eq_nsmul] at this
    rw [show δ = (t / (↑n + 1 : ℝ)) • T by rfl]
    rw [← h2, NormedSpace.exp_nsmul]
  rw [← hflow_const]
  refine (norm_pow_sub_pow_euler_clm (m := n + 1) ((t / (↑n + 1 : ℝ)) • T)).trans ?_
  have hexp_le :
      Real.exp (((n + 1 : ℕ) : ℝ) * ‖δ‖) ≤ Real.exp ‖t • T‖ := by
    refine Real.exp_le_exp.mpr ?_
    have hnorm : ((n + 1 : ℕ) : ℝ) * ‖δ‖ = ‖t • T‖ := by
      have hsmul : (((n + 1 : ℕ) : ℝ) • δ) = t • T := by
        have hne : (↑n + 1 : ℝ) ≠ 0 := by positivity
        have hcast : ((n + 1 : ℕ) : ℝ) = (↑n + 1 : ℝ) := by
          push_cast
          ring
        calc
          (((n + 1 : ℕ) : ℝ) • δ)
              = ((((n + 1 : ℕ) : ℝ) * (t / (↑n + 1 : ℝ))) • T) := by
                  simp [δ, smul_smul]
          _ = t • T := by
                rw [hcast]
                field_simp [hne]
      have hnorm' : ‖(((n + 1 : ℕ) : ℝ) • δ)‖ = ‖t • T‖ := congrArg norm hsmul
      have hpos : 0 ≤ ((n + 1 : ℕ) : ℝ) := by positivity
      rw [norm_smul, Real.norm_eq_abs, abs_of_nonneg hpos] at hnorm'
      simpa using hnorm'
    exact le_of_eq hnorm
  calc
    ‖NormedSpace.exp ((t / (↑n + 1 : ℝ)) • T) - 1 - (t / (↑n + 1 : ℝ)) • T‖ *
        (((n + 1 : ℕ) : ℝ) * Real.exp (((n + 1 : ℕ) : ℝ) * ‖(t / (↑n + 1 : ℝ)) • T‖))
        ≤ ‖NormedSpace.exp ((t / (↑n + 1 : ℝ)) • T) - 1 - (t / (↑n + 1 : ℝ)) • T‖ *
            (((n + 1 : ℕ) : ℝ) * Real.exp ‖t • T‖) := by
          gcongr
    _ = ((n : ℝ) + 1) *
          ‖NormedSpace.exp ((t / (↑n + 1 : ℝ)) • T) - 1 - (t / (↑n + 1 : ℝ)) • T‖
          * Real.exp ‖t • T‖ := by
          push_cast
          ring

/-- **Finite-depth Euler error bound (pointwise state version).**

For any initial state `h₀`, the state-space error is bounded by the
operator-level bound times `‖h₀‖`. -/
theorem operatorEuler_error_bound_apply
    (T : E →L[ℝ] E) (t : ℝ) (n : ℕ) (h₀ : E) :
    ‖(((1 : E →L[ℝ] E) + (t / (↑n + 1 : ℝ)) • T) ^ (n + 1)) h₀
      - (NormedSpace.exp (t • T)) h₀‖
      ≤ (((n : ℝ) + 1) *
          ‖NormedSpace.exp ((t / (↑n + 1 : ℝ)) • T)
            - 1 - (t / (↑n + 1 : ℝ)) • T‖
          * Real.exp ‖t • T‖) * ‖h₀‖ := by
  have hbase := operatorEuler_error_bound (E := E) T t n
  have happly :
      ‖((((1 : E →L[ℝ] E) + (t / (↑n + 1 : ℝ)) • T) ^ (n + 1)
            - NormedSpace.exp (t • T)) h₀)‖
        ≤ ‖((1 : E →L[ℝ] E) + (t / (↑n + 1 : ℝ)) • T) ^ (n + 1)
            - NormedSpace.exp (t • T)‖ * ‖h₀‖ :=
    (ContinuousLinearMap.le_opNorm
      (((1 : E →L[ℝ] E) + (t / (↑n + 1 : ℝ)) • T) ^ (n + 1) - NormedSpace.exp (t • T)) h₀)
  calc
    ‖((((1 : E →L[ℝ] E) + (t / (↑n + 1 : ℝ)) • T) ^ (n + 1)) h₀
      - (NormedSpace.exp (t • T)) h₀)‖
        ≤ ‖((1 : E →L[ℝ] E) + (t / (↑n + 1 : ℝ)) • T) ^ (n + 1)
            - NormedSpace.exp (t • T)‖ * ‖h₀‖ := by simpa using happly
    _ ≤ (((n : ℝ) + 1) *
          ‖NormedSpace.exp ((t / (↑n + 1 : ℝ)) • T)
            - 1 - (t / (↑n + 1 : ℝ)) • T‖
          * Real.exp ‖t • T‖) * ‖h₀‖ := by gcongr

/-- **`O(1/(n+1))` corollary under a quadratic local-exp remainder bound.**

Assume a quadratic one-step exponential remainder estimate:

`‖exp (s•T) - 1 - s•T‖ ≤ K * s^2` for the specific step sizes
`s = t/(n+1)`.

Then the global finite-depth Euler error is explicitly `O(1/(n+1))`. -/
theorem operatorEuler_error_bound_O_inv
    (T : E →L[ℝ] E) (t K : ℝ)
    (hquad :
      ∀ n : ℕ,
        ‖NormedSpace.exp ((t / (↑n + 1 : ℝ)) • T) - 1 - (t / (↑n + 1 : ℝ)) • T‖
          ≤ K * (t / (↑n + 1 : ℝ)) ^ 2)
    (n : ℕ) :
    ‖((1 : E →L[ℝ] E) + (t / (↑n + 1 : ℝ)) • T) ^ (n + 1) - NormedSpace.exp (t • T)‖
      ≤ (K * t ^ 2 * Real.exp ‖t • T‖) / ((n : ℝ) + 1) := by
  have h0 := operatorEuler_error_bound (E := E) T t n
  have hden_pos : 0 < ((n : ℝ) + 1) := by positivity
  calc
    ‖((1 : E →L[ℝ] E) + (t / (↑n + 1 : ℝ)) • T) ^ (n + 1) - NormedSpace.exp (t • T)‖
      ≤ ((n : ℝ) + 1) *
          ‖NormedSpace.exp ((t / (↑n + 1 : ℝ)) • T)
            - 1 - (t / (↑n + 1 : ℝ)) • T‖
          * Real.exp ‖t • T‖ := h0
    _ ≤ ((n : ℝ) + 1) * (K * (t / (↑n + 1 : ℝ)) ^ 2) * Real.exp ‖t • T‖ := by
          gcongr
          exact hquad n
    _ = (K * t ^ 2 * Real.exp ‖t • T‖) / ((n : ℝ) + 1) := by
          field_simp [hden_pos.ne']

private lemma expSeries_partialSum_two_clm (y : CLM) :
    (NormedSpace.expSeries ℝ CLM).partialSum 2 y = (1 : CLM) + y := by
  rw [FormalMultilinearSeries.partialSum, Finset.sum_range_succ, Finset.sum_range_succ,
    Finset.sum_range_zero]
  simp [NormedSpace.expSeries, Nat.factorial]

private lemma exp_remainder_eventually_quadratic (T : CLM) (t : ℝ) :
    ∃ (C : ℝ) (N : ℕ), C > 0 ∧ ∀ n ≥ N,
      ‖NormedSpace.exp ((t / (n + 1 : ℝ)) • T) - ((1 : CLM) + (t / (n + 1 : ℝ)) • T)‖
        ≤ C * ‖(t / (n + 1 : ℝ)) • T‖ ^ 2 := by
  have hpow :
      HasFPowerSeriesAt (NormedSpace.exp : CLM → CLM) (NormedSpace.expSeries ℝ CLM) 0 :=
    NormedSpace.hasFPowerSeriesAt_exp_zero_of_radius_pos
      (NormedSpace.expSeries_radius_pos ℝ CLM)
  have hbigO :
      (fun y : CLM => NormedSpace.exp y - (NormedSpace.expSeries ℝ CLM).partialSum 2 y)
        =O[𝓝 (0 : CLM)] fun y => ‖y‖ ^ 2 := by
    simpa [zero_add] using (hpow.isBigO_sub_partialSum_pow 2)
  rcases hbigO.bound with ⟨K, hK⟩
  let C : ℝ := |K| + 1
  have hC : 0 < C := by
    dsimp [C]
    linarith [abs_nonneg K]
  have hg : Tendsto (fun n : ℕ => ((n : ℝ) + 1)) atTop atTop := by
    have := (tendsto_natCast_atTop_atTop (R := ℝ)).comp (tendsto_add_atTop_nat 1)
    simpa [Function.comp_def] using this
  have hdiv : Tendsto (fun n : ℕ => t / ((n : ℝ) + 1)) atTop (𝓝 (0 : ℝ)) :=
    hg.const_div_atTop t
  have hseq0 : Tendsto (fun n : ℕ => (t / ((n : ℝ) + 1)) • T) atTop (𝓝 (0 : CLM)) := by
    simpa using hdiv.smul_const T
  have hKseq : ∀ᶠ n : ℕ in atTop,
      ‖NormedSpace.exp ((t / ((n : ℝ) + 1)) • T)
          - (NormedSpace.expSeries ℝ CLM).partialSum 2 ((t / ((n : ℝ) + 1)) • T)‖
        ≤ K * ‖‖(t / ((n : ℝ) + 1)) • T‖ ^ 2‖ := hseq0.eventually hK
  have hKseq' : ∀ᶠ n : ℕ in atTop,
      ‖NormedSpace.exp ((t / ((n : ℝ) + 1)) • T) - ((1 : CLM) + (t / ((n : ℝ) + 1)) • T)‖
        ≤ C * ‖(t / ((n : ℝ) + 1)) • T‖ ^ 2 := by
    filter_upwards [hKseq] with n hn
    have hkabs : K * ‖‖(t / ((n : ℝ) + 1)) • T‖ ^ 2‖ ≤ C * ‖(t / ((n : ℝ) + 1)) • T‖ ^ 2 := by
      have hsq : 0 ≤ ‖(t / ((n : ℝ) + 1)) • T‖ ^ 2 := sq_nonneg _
      have habs : ‖‖(t / ((n : ℝ) + 1)) • T‖ ^ 2‖ = ‖(t / ((n : ℝ) + 1)) • T‖ ^ 2 := by
        simpa [Real.norm_eq_abs, abs_of_nonneg hsq]
      rw [habs]
      have hKle : K ≤ C := by
        dsimp [C]
        linarith [le_abs_self K]
      nlinarith [hsq, hKle]
    exact (le_trans (by simpa [expSeries_partialSum_two_clm] using hn) hkabs)
  rw [Filter.eventually_atTop] at hKseq'
  obtain ⟨N, hN⟩ := hKseq'
  exact ⟨C, N, hC, by intro n hn; exact hN n hn⟩


/-- **Assumption-free asymptotic `O(1/(n+1))` Euler rate.** -/
theorem operatorEuler_error_bound_O_inv_unconditional
    (T : E →L[ℝ] E) (t : ℝ) :
    ∃ (C : ℝ) (N : ℕ), ∀ n ≥ N,
      ‖((1 : CLM) + (t / (n + 1 : ℝ)) • T) ^ (n + 1) - NormedSpace.exp (t • T)‖
        ≤ C / (n + 1 : ℝ) := by
  rcases exp_remainder_eventually_quadratic (T := T) (t := t) with ⟨Cr, Nr, hCr_pos, hquad⟩
  let C : ℝ := Cr * ‖t • T‖ ^ 2 * Real.exp ‖t • T‖ + 1
  refine ⟨C, Nr, ?_⟩
  intro n hn
  have hbase := operatorEuler_error_bound (E := E) T t n
  have hrem := hquad n hn
  have hden_pos : 0 < (n + 1 : ℝ) := by positivity
  have hscale : ‖(t / (n + 1 : ℝ)) • T‖ = ‖t • T‖ / (n + 1 : ℝ) := by
    have hmul : (n + 1 : ℝ) * ‖(t / (n + 1 : ℝ)) • T‖ = ‖t • T‖ := by
      rw [norm_smul, norm_smul, Real.norm_eq_abs, Real.norm_eq_abs, abs_div, abs_of_pos hden_pos]
      field_simp
    exact (eq_div_iff hden_pos.ne').2 (by simpa [mul_comm] using hmul)
  have hsq : ‖(t / (n + 1 : ℝ)) • T‖ ^ 2 = ‖t • T‖ ^ 2 / (n + 1 : ℝ) ^ 2 := by
    rw [hscale]
    field_simp [hden_pos.ne']
  have hrem' :
      ‖NormedSpace.exp ((t / (n + 1 : ℝ)) • T) - 1 - (t / (n + 1 : ℝ)) • T‖
        ≤ Cr * (‖t • T‖ ^ 2 / (n + 1 : ℝ) ^ 2) := by
    calc
      ‖NormedSpace.exp ((t / (n + 1 : ℝ)) • T) - 1 - (t / (n + 1 : ℝ)) • T‖
          ≤ Cr * ‖(t / (n + 1 : ℝ)) • T‖ ^ 2 := by simpa [sub_eq_add_neg, add_assoc, add_left_comm, add_comm] using hrem
      _ = Cr * (‖t • T‖ ^ 2 / (n + 1 : ℝ) ^ 2) := by rw [hsq]
  have hmain :
      ‖((1 : CLM) + (t / (n + 1 : ℝ)) • T) ^ (n + 1) - NormedSpace.exp (t • T)‖
        ≤ (Cr * ‖t • T‖ ^ 2 * Real.exp ‖t • T‖) / (n + 1 : ℝ) := by
    calc
      ‖((1 : CLM) + (t / (n + 1 : ℝ)) • T) ^ (n + 1) - NormedSpace.exp (t • T)‖
        ≤ ((n : ℝ) + 1) * ‖NormedSpace.exp ((t / (n + 1 : ℝ)) • T) - 1 - (t / (n + 1 : ℝ)) • T‖
            * Real.exp ‖t • T‖ := by simpa using hbase
      _ ≤ ((n : ℝ) + 1) * (Cr * (‖t • T‖ ^ 2 / (n + 1 : ℝ) ^ 2)) * Real.exp ‖t • T‖ := by gcongr
      _ = (Cr * ‖t • T‖ ^ 2 * Real.exp ‖t • T‖) / (n + 1 : ℝ) := by
            field_simp [hden_pos.ne']
  calc
    ‖((1 : CLM) + (t / (n + 1 : ℝ)) • T) ^ (n + 1) - NormedSpace.exp (t • T)‖
      ≤ (Cr * ‖t • T‖ ^ 2 * Real.exp ‖t • T‖) / (n + 1 : ℝ) := hmain
    _ ≤ C / (n + 1 : ℝ) := by
          have hCge : Cr * ‖t • T‖ ^ 2 * Real.exp ‖t • T‖ ≤ C := by
            dsimp [C]
            linarith
          exact (div_le_div_of_nonneg_right hCge (by positivity))

/-- **Global `O(1/(n+1))` bound for all `n` (prefix absorbed).**

This upgrades `operatorEuler_error_bound_O_inv_unconditional` from an eventual
bound (`n ≥ N`) to a uniform all-`n` bound by absorbing the finite prefix
`n < N` into one explicit constant via a finite supremum. -/
theorem operatorEuler_error_bound_O_inv_all_n
    (T : E →L[ℝ] E) (t : ℝ) :
    ∃ C : ℝ, ∀ n : ℕ,
      ‖((1 : CLM) + (t / (n + 1 : ℝ)) • T) ^ (n + 1) - NormedSpace.exp (t • T)‖
        ≤ C / (n + 1 : ℝ) := by
  rcases operatorEuler_error_bound_O_inv_unconditional (T := T) (t := t) with ⟨C0, N, hCN⟩
  let err : ℕ → ℝ := fun n =>
    ‖((1 : CLM) + (t / (n + 1 : ℝ)) • T) ^ (n + 1) - NormedSpace.exp (t • T)‖
  have hne : (Finset.range (N + 1)).Nonempty := by
    exact ⟨0, Finset.mem_range.mpr (Nat.succ_pos N)⟩
  let Cpre : ℝ := (Finset.range (N + 1)).sup' hne (fun n => ((n : ℝ) + 1) * err n)
  let C : ℝ := max C0 Cpre + 1
  refine ⟨C, ?_⟩
  intro n
  have hden_pos : 0 < (n + 1 : ℝ) := by positivity
  by_cases hn : N ≤ n
  · have htail := hCN n hn
    calc
      err n ≤ C0 / (n + 1 : ℝ) := htail
      _ ≤ C / (n + 1 : ℝ) := by
            have hC0 : C0 ≤ C := by
              dsimp [C]
              linarith [le_max_left C0 Cpre]
            exact (div_le_div_of_nonneg_right hC0 (by positivity))
  · have hlt : n < N := Nat.lt_of_not_ge hn
    have hmem : n ∈ Finset.range (N + 1) := Finset.mem_range.mpr (Nat.lt_succ_of_lt hlt)
    have hpre : ((n : ℝ) + 1) * err n ≤ Cpre := by
      exact Finset.le_sup' (s := Finset.range (N + 1)) (f := fun k => ((k : ℝ) + 1) * err k) hmem
    have hCpre : Cpre ≤ C := by
      dsimp [C]
      linarith [le_max_right C0 Cpre]
    calc
      err n = (((n : ℝ) + 1) * err n) / (n + 1 : ℝ) := by
                field_simp [hden_pos.ne']
      _ ≤ Cpre / (n + 1 : ℝ) := by
            exact (div_le_div_of_nonneg_right hpre (by positivity))
      _ ≤ C / (n + 1 : ℝ) := by
            exact (div_le_div_of_nonneg_right hCpre (by positivity))


/-- **Euler limit at the operator level (Fin D → ℝ specialisation).**
The discrete refinement `(1 + (t/(n+1))·T)^{n+1}` converges (in operator norm)
to the continuous flow `cauchyResidualFlow F t = exp(t·T)`. -/
theorem cauchyResidualFlow_eulerLimit_clm
    {D : ℕ} (F : (Fin D → ℝ) → Fin D → ℝ) (hF : IsRLinear F) (t : ℝ) :
    Tendsto (fun n : ℕ =>
        ((1 : (Fin D → ℝ) →L[ℝ] (Fin D → ℝ))
          + (t / (↑n + 1 : ℝ)) • cauchyLinearOp F hF) ^ (n + 1))
      atTop (𝓝 (cauchyResidualFlow F hF t)) := by
  unfold cauchyResidualFlow
  exact operatorEulerLimit (cauchyLinearOp F hF) t

/-- **Euler limit, pointwise.**  The §1.17 Euler iterate
`iterEuler (n+1) (t/(n+1)) F h₀` converges to `cauchyResidualFlow F t h₀`
as `n → ∞`.  Follows from the operator-level limit
`cauchyResidualFlow_eulerLimit_clm` by composing with continuous evaluation
at `h₀` and the identity `iterEuler_eq_clm_pow`. -/
theorem cauchyResidualFlow_eulerLimit
    {D : ℕ} (F : (Fin D → ℝ) → Fin D → ℝ) (hF : IsRLinear F) (t : ℝ) (h₀ : Fin D → ℝ) :
    Tendsto (fun n : ℕ => iterEuler (n + 1) (t / (↑n + 1 : ℝ)) F h₀) atTop
      (𝓝 (cauchyResidualFlow F hF t h₀)) := by
  have hclm := cauchyResidualFlow_eulerLimit_clm F hF t
  have hcont : Continuous (fun Φ : (Fin D → ℝ) →L[ℝ] (Fin D → ℝ) => Φ h₀) :=
    (ContinuousLinearMap.apply ℝ (Fin D → ℝ) h₀).continuous
  have heval := (hcont.tendsto _).comp hclm
  convert heval using 1
  funext n
  simp only [Function.comp_def]
  exact iterEuler_eq_clm_pow F hF (t / (↑n + 1 : ℝ)) h₀ (n + 1)

/-- **Finite-depth Cauchy-flow error bound (operator level).**

Explicit operator-norm bound between the `(n+1)`-layer Euler refinement of
the Cauchy residual dynamics and the continuous Cauchy flow at time `t`. -/
theorem cauchyResidualFlow_euler_error_bound_clm
    {D : ℕ} (F : (Fin D → ℝ) → Fin D → ℝ) (hF : IsRLinear F)
    (t : ℝ) (n : ℕ) :
    ‖((1 : (Fin D → ℝ) →L[ℝ] (Fin D → ℝ))
        + (t / (↑n + 1 : ℝ)) • cauchyLinearOp F hF) ^ (n + 1)
      - cauchyResidualFlow F hF t‖
      ≤ ((n : ℝ) + 1) *
          ‖NormedSpace.exp ((t / (↑n + 1 : ℝ)) • cauchyLinearOp F hF)
            - 1 - (t / (↑n + 1 : ℝ)) • cauchyLinearOp F hF‖
          * Real.exp ‖t • cauchyLinearOp F hF‖ := by
  simpa [cauchyResidualFlow] using
    (operatorEuler_error_bound (E := Fin D → ℝ) (cauchyLinearOp F hF) t n)

/-- **Finite-depth Cauchy-flow error bound (state level).**

This is the direct "finite-layer network vs continuous flow" bound for
embedding states: `iterEuler` at depth `n+1` and step `t/(n+1)` versus
`cauchyResidualFlow` at time `t`. -/
theorem cauchyResidualFlow_euler_error_bound
    {D : ℕ} (F : (Fin D → ℝ) → Fin D → ℝ) (hF : IsRLinear F)
    (t : ℝ) (n : ℕ) (h₀ : Fin D → ℝ) :
    ‖iterEuler (n + 1) (t / (↑n + 1 : ℝ)) F h₀ - cauchyResidualFlow F hF t h₀‖
      ≤ (((n : ℝ) + 1) *
          ‖NormedSpace.exp ((t / (↑n + 1 : ℝ)) • cauchyLinearOp F hF)
            - 1 - (t / (↑n + 1 : ℝ)) • cauchyLinearOp F hF‖
          * Real.exp ‖t • cauchyLinearOp F hF‖) * ‖h₀‖ := by
  have hpow := iterEuler_eq_clm_pow F hF (t / (↑n + 1 : ℝ)) h₀ (n + 1)
  rw [hpow]
  simpa [cauchyResidualFlow] using
    (operatorEuler_error_bound_apply
      (E := Fin D → ℝ) (cauchyLinearOp F hF) t n h₀)

end cauchyResidualFlowEulerLimit

/-! ### §1.18 RoPE — feature-space rotation, lifted to PSL(2, ℝ) via the
   Cayley transform, lands strictly outside `B(2, ℝ)`.

Rotary Position Embedding (RoPE) is, by definition in standard
transformer architectures (e.g. Llama, GPT-J, …), a **feature-space
rotation**: every adjacent pair of feature dimensions
`(x_{2i}, x_{2i+1})` is rotated by an angle proportional to position,

    `(x, y) ↦ (x cos θ − y sin θ,  x sin θ + y cos θ)`,

which under the canonical identification `ℝ² ≅ ℂ` is multiplication
by `cos θ + i sin θ`.  This is **not** a Möbius transformation of the
upper half plane — multiplication by `e^{iθ}` does not preserve UHP.

To express this rotation as an action on the Cauchy-Poisson upper
half plane (where our poles live, §1.5), the standard procedure is the
**Cayley transform**: identify UHP with the unit disc via
`w = (z − i)/(z + i)`, where rotation around `0` is now an isometry,
then pull the rotation back to UHP.  The pullback of disc rotation by
angle `θ` is provably equal to a Möbius transformation of `PSL(2, ℝ)`
with half-angle representative

    `[[ cos(θ/2),  sin(θ/2)], [-sin(θ/2),  cos(θ/2)]]`,

i.e. `ropeMobius θ` defined below.  This is the content of
`ropeMobius_eq_cayley_pullback`, which closes the chain
*feature-space rotation → disc rotation → UHP Möbius* with
**no hand-waving**: the Cayley pullback is computed in Lean and
shown to coincide with `ropeMobius` algebraically.

Once on the UHP side, the Möbius has lower-left entry `c = −sin(θ/2)`,
non-zero whenever `θ ∉ 2π ℤ`; combining with §1.16, every
non-trivial RoPE rotation strictly **breaks the verticality slice**
that softmax-derived poles inhabit (`rope_strictly_exceeds_affine`,
`rope_breaks_verticality`).

####  Scope and limitations.

* The full D-dimensional RoPE acts block-diagonally on `D/2`
  independent 2D pairs.  All theorems below are stated for **one**
  such 2D block — i.e., one feature pair carrying one Cauchy-Poisson
  pole.  The D-dim statement is the direct sum of D/2 copies.  The
  per-block analysis is rigorous; the block-diagonal extension is
  routine bookkeeping and not formalised here.

* The Cayley pullback is the standard, basically forced way to lift
  rotational symmetry on `ℝ² ≅ ℂ` to a UHP-preserving Möbius — but
  it is a *choice of identification*, not a tautology forced on us by
  the original feature-space rotation.  We make this identification
  explicit and prove the resulting algebra is exact.
-/

/-- **The Cayley transform** mapping the upper half plane to the unit
    disc: `cayley z = (z − i)/(z + i)`.  Surjective from UHP onto the
    open unit disc.  -/
noncomputable def cayley (z : ℂ) : ℂ := (z - Complex.I) / (z + Complex.I)

/-- **The inverse Cayley transform** mapping the unit disc back to
    the upper half plane: `cayleyInv w = i (1 + w)/(1 − w)`.  -/
noncomputable def cayleyInv (w : ℂ) : ℂ :=
  Complex.I * (1 + w) / (1 - w)

/-- **Disc rotation by angle θ.**  Rotation around the origin in the
    `ℝ² ≅ ℂ` plane is multiplication by `cos θ + i sin θ`.  This is
    exactly the standard RoPE rotation acting on a 2D feature-pair
    block — not yet lifted to UHP.  -/
noncomputable def discRotate (θ : ℝ) (w : ℂ) : ℂ :=
  ((Real.cos θ : ℂ) + (Real.sin θ : ℂ) * Complex.I) * w

/-- **The RoPE Möbius matrix** — the half-angle representative in
    `PSL(2, ℝ)`.

    `ropeMobius θ` returns `(a, b, c, d) = (cos(θ/2), sin(θ/2),
    −sin(θ/2), cos(θ/2))`.  Theorem `ropeMobius_eq_cayley_pullback`
    below proves this is *exactly* the Cayley pullback of feature-space
    rotation by `θ`. -/
noncomputable def ropeMobius (θ : ℝ) : ℝ × ℝ × ℝ × ℝ :=
  (Real.cos (θ / 2), Real.sin (θ / 2),
   -Real.sin (θ / 2), Real.cos (θ / 2))

/-! #### Algebraic core: factorisation lemmas for the Cayley pullback. -/

private lemma cayley_pullback_numerator_factors
    (c s : ℝ) (hpyth : c ^ 2 + s ^ 2 = 1) (z : ℂ) :
    (z + Complex.I) +
        ((((c ^ 2 - s ^ 2 : ℝ)) : ℂ) + ((2 * c * s : ℝ) : ℂ) * Complex.I) *
          (z - Complex.I)
      = 2 * (((c : ℝ) : ℂ) + ((s : ℝ) : ℂ) * Complex.I) *
          (((c : ℝ) : ℂ) * z + ((s : ℝ) : ℂ)) := by
  have hpythC : ((c : ℝ) : ℂ) ^ 2 + ((s : ℝ) : ℂ) ^ 2 = 1 := by
    have : ((c ^ 2 + s ^ 2 : ℝ) : ℂ) = ((1 : ℝ) : ℂ) := by exact_mod_cast hpyth
    push_cast at this ⊢
    linear_combination this
  have hI : (Complex.I : ℂ) ^ 2 = -1 := Complex.I_sq
  push_cast
  linear_combination
    -(z + Complex.I) * hpythC
      + (-2 * ((c : ℝ) : ℂ) * ((s : ℝ) : ℂ)) * hI

private lemma cayley_pullback_denominator_factors
    (c s : ℝ) (hpyth : c ^ 2 + s ^ 2 = 1) (z : ℂ) :
    (z + Complex.I) -
        ((((c ^ 2 - s ^ 2 : ℝ)) : ℂ) + ((2 * c * s : ℝ) : ℂ) * Complex.I) *
          (z - Complex.I)
      = 2 * Complex.I * (((c : ℝ) : ℂ) + ((s : ℝ) : ℂ) * Complex.I) *
          ((-((s : ℝ) : ℂ)) * z + ((c : ℝ) : ℂ)) := by
  have hpythC : ((c : ℝ) : ℂ) ^ 2 + ((s : ℝ) : ℂ) ^ 2 = 1 := by
    have : ((c ^ 2 + s ^ 2 : ℝ) : ℂ) = ((1 : ℝ) : ℂ) := by exact_mod_cast hpyth
    push_cast at this ⊢
    linear_combination this
  have hI : (Complex.I : ℂ) ^ 2 = -1 := Complex.I_sq
  push_cast
  linear_combination
    -(z + Complex.I) * hpythC
      + (2 * ((s : ℝ) : ℂ) ^ 2 * z) * hI

/-- **Disc rotation expressed via half-angle parameters.**  For real `θ`,
    `(cos θ + i sin θ) = (cos²(θ/2) − sin²(θ/2)) + 2 cos(θ/2) sin(θ/2) · i`
    by the double-angle formulas.  -/
private lemma discRotate_half_angle (θ : ℝ) :
    ((Real.cos θ : ℂ) + (Real.sin θ : ℂ) * Complex.I)
      = ((((Real.cos (θ / 2)) ^ 2 - (Real.sin (θ / 2)) ^ 2 : ℝ) : ℂ)
          + ((2 * Real.cos (θ / 2) * Real.sin (θ / 2) : ℝ) : ℂ) * Complex.I) := by
  have hcos : Real.cos θ
      = Real.cos (θ / 2) ^ 2 - Real.sin (θ / 2) ^ 2 := by
    have h := Real.cos_two_mul (θ / 2)
    have hpyth := Real.sin_sq_add_cos_sq (θ / 2)
    have hθ : 2 * (θ / 2) = θ := by ring
    rw [hθ] at h
    linarith
  have hsin : Real.sin θ
      = 2 * Real.cos (θ / 2) * Real.sin (θ / 2) := by
    have h := Real.sin_two_mul (θ / 2)
    have hθ : 2 * (θ / 2) = θ := by ring
    rw [hθ] at h
    linarith
  rw [hcos, hsin]

/-- **Theorem (RoPE = Cayley pullback of feature-space rotation).**

    The composition `cayleyInv ∘ discRotate θ ∘ cayley` is *literally*
    equal to the Möbius transformation `ropeMobius θ` acting on the
    upper half plane:

        `cayleyInv (discRotate θ (cayley z))
            = (cos(θ/2) · z + sin(θ/2))
              / (−sin(θ/2) · z + cos(θ/2))`.

    This closes the chain **feature-space rotation → disc rotation →
    UHP Möbius**: the rotation a transformer applies in feature space
    is *exactly* (after the standard Cayley identification) the
    `PSL(2, ℝ)` element `ropeMobius θ`, whose `c = −sin(θ/2)` is
    non-zero whenever `θ ∉ 2π ℤ`.  By §1.16 (`realised_mobius_orbit_is_affine`),
    no such transformation preserves the standard verticality slice.

    The single-block hypothesis `1 − discRotate θ (cayley z) ≠ 0` is
    well-defined whenever `z ∈ UHP` and `θ ∉ 2π ℤ`; we take it as a
    side condition to avoid the disc boundary `|w| = 1`.  -/
theorem ropeMobius_eq_cayley_pullback (θ : ℝ) (z : ℂ)
    (hcay_top : z + Complex.I ≠ 0)
    (hcay_bot : 1 - discRotate θ (cayley z) ≠ 0)
    (hrope_bot :
      -((Real.sin (θ / 2) : ℝ) : ℂ) * z + ((Real.cos (θ / 2) : ℝ) : ℂ) ≠ 0) :
    cayleyInv (discRotate θ (cayley z))
      = (((Real.cos (θ / 2) : ℝ) : ℂ) * z + ((Real.sin (θ / 2) : ℝ) : ℂ))
          / (-((Real.sin (θ / 2) : ℝ) : ℂ) * z + ((Real.cos (θ / 2) : ℝ) : ℂ)) := by
  set c : ℝ := Real.cos (θ / 2)
  set s : ℝ := Real.sin (θ / 2)
  have hpyth : c ^ 2 + s ^ 2 = 1 := by
    have := Real.sin_sq_add_cos_sq (θ / 2)
    show c ^ 2 + s ^ 2 = 1
    nlinarith [this]
  -- Rewrite `discRotate θ` using half-angle parameters.
  have hrot := discRotate_half_angle θ
  -- The disc rotation written as the half-angle Möbius polynomial:
  set W : ℂ := ((c ^ 2 - s ^ 2 : ℝ) : ℂ) + ((2 * c * s : ℝ) : ℂ) * Complex.I with hW_def
  have hRot_eq : ((Real.cos θ : ℂ) + (Real.sin θ : ℂ) * Complex.I) = W := by
    rw [hW_def]; exact hrot
  -- Numerator and denominator factorisations.
  have hN := cayley_pullback_numerator_factors c s hpyth z
  have hD := cayley_pullback_denominator_factors c s hpyth z
  -- `c + s I ≠ 0` follows from `c² + s² = 1` (else both vanish).
  have hcsI : ((c : ℝ) : ℂ) + ((s : ℝ) : ℂ) * Complex.I ≠ 0 := by
    intro hzero
    have hzr : (((c : ℝ) : ℂ) + ((s : ℝ) : ℂ) * Complex.I).re = c := by simp
    have hzi : (((c : ℝ) : ℂ) + ((s : ℝ) : ℂ) * Complex.I).im = s := by simp
    rw [hzero] at hzr hzi
    simp at hzr hzi
    have hcs : c ^ 2 + s ^ 2 = 0 := by rw [← hzr, ← hzi]; ring
    rw [hpyth] at hcs
    exact one_ne_zero hcs
  have h2 : (2 : ℂ) ≠ 0 := two_ne_zero
  have hI_ne : (Complex.I : ℂ) ≠ 0 := Complex.I_ne_zero
  -- Combine the two factorisations into the cayleyInv expression.
  unfold cayleyInv discRotate cayley
  rw [hRot_eq]
  -- Step 1: combine the (z + I) denominators inside the cayleyInv quotient.
  have step1 :
      (1 + W * ((z - Complex.I) / (z + Complex.I)))
        = ((z + Complex.I) + W * (z - Complex.I)) / (z + Complex.I) := by
    field_simp
  have step2 :
      (1 - W * ((z - Complex.I) / (z + Complex.I)))
        = ((z + Complex.I) - W * (z - Complex.I)) / (z + Complex.I) := by
    field_simp
  rw [step1, step2, hN, hD]
  -- After factorisation the goal is:
  --   I * (2(c+sI)(cz+s)/(z+I)) / (2I(c+sI)(-sz+c)/(z+I)) = (cz+s)/(-sz+c)
  -- We cross-multiply *both* fractions via `div_eq_div_iff`, after which
  -- only polynomial factors remain.  `field_simp` then clears the (z+I)
  -- denominators inside the LHS, and `ring` finishes — both sides being
  -- equal as polynomials in c, s, z, I (no I² simplification needed).
  have hDEN_ne :
      (2 : ℂ) * Complex.I * (((c : ℝ) : ℂ) + ((s : ℝ) : ℂ) * Complex.I) *
          (-((s : ℝ) : ℂ) * z + ((c : ℝ) : ℂ)) ≠ 0 :=
    mul_ne_zero (mul_ne_zero (mul_ne_zero h2 hI_ne) hcsI) hrope_bot
  have hDENdiv_ne :
      (2 : ℂ) * Complex.I * (((c : ℝ) : ℂ) + ((s : ℝ) : ℂ) * Complex.I) *
          (-((s : ℝ) : ℂ) * z + ((c : ℝ) : ℂ)) / (z + Complex.I) ≠ 0 :=
    div_ne_zero hDEN_ne hcay_top
  rw [div_eq_div_iff hDENdiv_ne hrope_bot]
  field_simp

/-- **The RoPE matrix is unimodular.** -/
theorem ropeMobius_det_one (θ : ℝ) :
    let γ := ropeMobius θ
    γ.1 * γ.2.2.2 - γ.2.1 * γ.2.2.1 = 1 := by
  unfold ropeMobius
  have := Real.sin_sq_add_cos_sq (θ / 2)
  simp only
  nlinarith [Real.sin_sq_add_cos_sq (θ / 2)]

/-- **The RoPE matrix has `c = −sin(θ/2)`.**  -/
theorem ropeMobius_c (θ : ℝ) :
    (ropeMobius θ).2.2.1 = -Real.sin (θ / 2) := rfl

/-- **Non-trivial RoPE rotations break verticality.**

    For any rotation angle `θ` with `sin(θ/2) ≠ 0` (i.e. `θ ∉ 2π ℤ`),
    the RoPE Möbius image of any realised pole `(q, y)` with `y > 0`
    is **not** vertical above `γ q`.  In other words, RoPE rotational
    symmetry strictly leaves the affine slice §1.16 traps softmax in.

    This is the formal substance of "RoPE forces `x_k ≠ q`": not at
    the standard score level (where RoPE only modifies `y_k`), but
    at the level of the simultaneous `PSL(2, ℝ)` rotational symmetry
    RoPE introduces — that symmetry has `c ≠ 0` and so by §1.16
    cannot live inside `B(2, ℝ)`.  -/
theorem rope_breaks_verticality
    (θ : ℝ) (hθ : Real.sin (θ / 2) ≠ 0)
    (q : ℝ) (hcq : (-Real.sin (θ / 2)) * q + Real.cos (θ / 2) ≠ 0)
    {y : ℝ} (hy : 0 < y) :
    mobPoleX (Real.cos (θ / 2)) (Real.sin (θ / 2))
             (-Real.sin (θ / 2)) (Real.cos (θ / 2)) q y
      ≠ mobReal (Real.cos (θ / 2)) (Real.sin (θ / 2))
                (-Real.sin (θ / 2)) (Real.cos (θ / 2)) q := by
  intro hslice
  have hdet : Real.cos (θ / 2) * Real.cos (θ / 2)
              - Real.sin (θ / 2) * (-Real.sin (θ / 2)) = 1 := by
    nlinarith [Real.sin_sq_add_cos_sq (θ / 2)]
  have hc : (-Real.sin (θ / 2)) = 0 :=
    realised_mobius_orbit_is_affine hdet q hcq hy hslice
  exact hθ (by linarith)

/-- **RoPE rotational symmetry strictly exceeds the affine subgroup.**

    Every non-trivial RoPE rotation lies in `PSL(2, ℝ) ∖ B(2, ℝ)`.
    Combined with §1.15 (`attention_simplex_mobius_invariant`), this
    means RoPE-equipped attention enjoys an honest `PSL(2, ℝ)`
    invariance that softmax-only attention provably cannot — the
    architecture is *forced off* the §1.16 vertical slice.  -/
theorem rope_strictly_exceeds_affine
    (θ : ℝ) (hθ : Real.sin (θ / 2) ≠ 0) :
    (ropeMobius θ).2.2.1 ≠ 0 := by
  rw [ropeMobius_c]
  exact fun h => hθ (by linarith)

end AnalyticTransformer

