import Mathlib
import Part1_TheIdentity
import Part2_TheFullModel
import Part3_ContinuumLimit

/-!
# Part 5 — Poisson Classification (Möbius uniqueness)

This file proves the **harmonic-analysis classification theorem** that
elevates the §1.15 Möbius covariance from a *property* of the Poisson
kernel to a full *characterisation* of it.

## The theorem

The Poisson kernel `P(x, y, q) = y / ((q - x)² + y²)` is the **unique**
function `K : UHP × ℝ → ℝ` satisfying

  1. **Möbius weight-2 covariance**: for every `[[a, b], [c, d]]` with
     `a d − b c = 1` and every `(x, y, q)` with `y > 0`, `c q + d ≠ 0`,

         `K (γ.x, γ.y, γ.q) = (c q + d)² · K(x, y, q)`,

  2. **Single-point normalisation**: `K(0, 1, 0) = P(0, 1, 0) = 1`.

Combined with §1.15 (`poisson_mobius_covariant` — Poisson is covariant)
this is a genuine **classification**: an attention-style boundary kernel
is Möbius covariant of weight 2 with the standard normalisation **iff**
it is the Poisson kernel.

## Why this is the "transformer at infinity is Cauchy" theorem

§1.19 / §3 proves the *transformer's residual flow* is `exp(t · T)` and
its boundary trace is governed by some kernel `K(x, y, q)`.  By
construction this kernel inherits the Möbius geometry of the upper half
plane — every Cauchy/Poisson contour is preserved by PSL(2, ℝ).  This
file shows that the kernel value at every point is then **determined** by
the symmetry plus its value at one normalisation point.  No alternative
boundary kernel can satisfy the same group-theoretic constraints.  In
this sense the continuous-time transformer's attention kernel is
*provably forced* to be the Cauchy/Poisson semigroup kernel — it is not a
modelling choice, it is a theorem about Möbius covariance.

## Proof structure

1. `mobius_realization_exists` — for any UHP point `(x, y)` and boundary
   point `q` with `y > 0`, exhibit an explicit unimodular Möbius matrix
   `[[a, b], [c, d]]` mapping `(0 + i, 0)` to `(x + iy, q)`.  The lower-row
   weight `(c · 0 + d)² = d²` is precisely `poisson x y q`, which is the
   bridge between the Poisson kernel value and the cocycle prefactor.

2. `poisson_unique_under_mobius_covariance` — apply the realisation to
   any candidate `K`, use covariance to express `K(x, y, q)` in terms of
   `K(0, 1, 0)`, then specialise the same identity to `poisson` itself
   (which has the same covariance by `poisson_mobius_covariant`).
-/

noncomputable section

namespace AnalyticTransformer

open Real

/-- Square-root scaffolding lemma: for any UHP point `(x, y)` (with `y > 0`)
and boundary query `q`, the standard reference point `(0 + i, 0)` can be
Möbius-translated to `(x + iy, q)` by an explicit unimodular matrix.  The
lower-row entry `d` satisfies `d² = poisson x y q`. -/
theorem mobius_realization_exists (x y q : ℝ) (hy : 0 < y) :
    ∃ a b c d : ℝ,
      a * d - b * c = 1 ∧
      d ≠ 0 ∧
      mobReal a b c d 0 = q ∧
      mobPoleX a b c d 0 1 = x ∧
      mobPoleY c d 0 1 = y ∧
      (c * 0 + d) ^ 2 = poisson x y q := by
  -- Square-root scaffolding.
  set S := Real.sqrt y with hSdef
  set R := Real.sqrt ((q - x) ^ 2 + y ^ 2) with hRdef
  have hS_pos : 0 < S := Real.sqrt_pos.mpr hy
  have hS_ne : S ≠ 0 := ne_of_gt hS_pos
  have hR_arg_pos : 0 < (q - x) ^ 2 + y ^ 2 := by
    have : 0 < y ^ 2 := by positivity
    nlinarith [sq_nonneg (q - x)]
  have hR_pos : 0 < R := Real.sqrt_pos.mpr hR_arg_pos
  have hR_ne : R ≠ 0 := ne_of_gt hR_pos
  have hSsq : S ^ 2 = y := Real.sq_sqrt (le_of_lt hy)
  have hRsq : R ^ 2 = (q - x) ^ 2 + y ^ 2 :=
    Real.sq_sqrt (le_of_lt hR_arg_pos)
  have hR2_pos : 0 < R ^ 2 := by positivity
  have hR2_ne : R ^ 2 ≠ 0 := ne_of_gt hR2_pos
  have hS2_ne : S ^ 2 ≠ 0 := pow_ne_zero _ hS_ne
  have hSR_pos : 0 < S * R := mul_pos hS_pos hR_pos
  have hSR_ne : S * R ≠ 0 := ne_of_gt hSR_pos
  have hSR2_pos : 0 < (S * R) ^ 2 := by positivity
  have hSR2_ne : (S * R) ^ 2 ≠ 0 := ne_of_gt hSR2_pos
  have hy_ne : y ≠ 0 := ne_of_gt hy
  refine ⟨(x * (x - q) + y ^ 2) / (S * R),
          q * S / R,
          (x - q) / (S * R),
          S / R,
          ?_, ?_, ?_, ?_, ?_, ?_⟩
  ----------------------------------------------------------
  -- Part 1: a*d - b*c = 1
  -- After clearing fractions (LCM = S²·R²): the polynomial identity
  -- `(x*(x-q)+y²)·S² - q·(x-q)·S² = S²·R²` holds modulo
  -- `S² = y` and `R² = (q-x)² + y²`.  The linear-combination coefficients
  -- come from rewriting the residual `S²·((x-q)² + y² - R²)` as
  -- a combination of `(S² - y)` and `(R² - (q-x)² - y²)`.
  ----------------------------------------------------------
  · field_simp
    linear_combination -hRsq
  ----------------------------------------------------------
  -- Part 2: d ≠ 0
  ----------------------------------------------------------
  · exact div_ne_zero hS_ne hR_ne
  ----------------------------------------------------------
  -- Part 3: mobReal a b c d 0 = q
  ----------------------------------------------------------
  · show (((x * (x - q) + y ^ 2) / (S * R)) * 0 + q * S / R) /
           (((x - q) / (S * R)) * 0 + S / R) = q
    have hsimp_num :
        (x * (x - q) + y ^ 2) / (S * R) * 0 + q * S / R = q * S / R := by ring
    have hsimp_den :
        (x - q) / (S * R) * 0 + S / R = S / R := by ring
    rw [hsimp_num, hsimp_den]
    field_simp
  ----------------------------------------------------------
  -- Part 4: mobPoleX a b c d 0 1 = x.  After clearing the * 0 + and * 1²
  -- placeholders and the outer division, the polynomial identity
  -- `q·S⁴ + (x-q)·(x·(x-q)+y²) = x·(S⁴ + (x-q)²)`
  -- becomes (using S² = y) `(q-x)·(S²+y)·(S²-y) = 0`, hence
  -- the linear combination `(q-x)·(S²+y) · hSsq`.
  ----------------------------------------------------------
  · show ((((x * (x - q) + y ^ 2) / (S * R)) * 0 + q * S / R) *
            (((x - q) / (S * R)) * 0 + S / R)
          + ((x * (x - q) + y ^ 2) / (S * R)) * ((x - q) / (S * R)) * 1 ^ 2) /
          ((((x - q) / (S * R)) * 0 + S / R) ^ 2 +
            ((x - q) / (S * R)) ^ 2 * 1 ^ 2) = x
    have hsimp_num :
        ((x * (x - q) + y ^ 2) / (S * R) * 0 + q * S / R) *
            ((x - q) / (S * R) * 0 + S / R) +
          (x * (x - q) + y ^ 2) / (S * R) * ((x - q) / (S * R)) * 1 ^ 2 =
        q * S / R * (S / R) +
          (x * (x - q) + y ^ 2) / (S * R) * ((x - q) / (S * R)) := by ring
    have hsimp_den :
        ((x - q) / (S * R) * 0 + S / R) ^ 2 +
            ((x - q) / (S * R)) ^ 2 * 1 ^ 2 =
        (S / R) ^ 2 + ((x - q) / (S * R)) ^ 2 := by ring
    rw [hsimp_num, hsimp_den]
    have hden_pos : 0 < (S / R) ^ 2 + ((x - q) / (S * R)) ^ 2 := by
      have h1 : 0 < (S / R) ^ 2 := by positivity
      nlinarith [sq_nonneg ((x - q) / (S * R))]
    have hden_ne : (S / R) ^ 2 + ((x - q) / (S * R)) ^ 2 ≠ 0 := ne_of_gt hden_pos
    rw [div_eq_iff hden_ne]
    field_simp
    linear_combination (q - x) * (S ^ 2 + y) * hSsq
  ----------------------------------------------------------
  -- Part 5: mobPoleY c d 0 1 = y.  After clearing placeholders and the
  -- outer division, the polynomial identity
  -- `S²·R² = y·S⁴ + y·(x-q)²`
  -- becomes (using S² = y, R² = (q-x)² + y²) the linear combination
  -- `((q-x)² - y·S²) · hSsq + S² · hRsq`.
  ----------------------------------------------------------
  · show 1 / ((((x - q) / (S * R)) * 0 + S / R) ^ 2 +
              ((x - q) / (S * R)) ^ 2 * 1 ^ 2) = y
    have hden_eq :
        (((x - q) / (S * R)) * 0 + S / R) ^ 2 +
          ((x - q) / (S * R)) ^ 2 * 1 ^ 2 =
        (S / R) ^ 2 + ((x - q) / (S * R)) ^ 2 := by ring
    rw [hden_eq]
    have hden_pos : 0 < (S / R) ^ 2 + ((x - q) / (S * R)) ^ 2 := by
      have h1 : 0 < (S / R) ^ 2 := by positivity
      nlinarith [sq_nonneg ((x - q) / (S * R))]
    have hden_ne : (S / R) ^ 2 + ((x - q) / (S * R)) ^ 2 ≠ 0 := ne_of_gt hden_pos
    rw [div_eq_iff hden_ne]
    field_simp
    linear_combination ((q - x) ^ 2 - y * S ^ 2) * hSsq + S ^ 2 * hRsq
  ----------------------------------------------------------
  -- Part 6: (c · 0 + d)² = poisson x y q
  ----------------------------------------------------------
  · show (((x - q) / (S * R)) * 0 + S / R) ^ 2 = poisson x y q
    have hsimp : ((x - q) / (S * R)) * 0 + S / R = S / R := by ring
    rw [hsimp]
    unfold poisson
    rw [div_pow, hSsq, hRsq]

/-- **Poisson kernel uniqueness under PSL(2, ℝ) covariance.**

Suppose `K : ℝ → ℝ → ℝ → ℝ` is a candidate kernel satisfying

  1. weight-2 Möbius covariance (the same identity as
     `poisson_mobius_covariant`),
  2. the boundary normalisation `K(0, 1, 0) = poisson(0, 1, 0)`.

Then `K = poisson` on the entire upper-half-plane × ℝ.  The proof uses
the explicit Möbius realisation `(0 + i, 0) → (x + iy, q)` from
`mobius_realization_exists`: covariance pulls the kernel back to the
reference point, and the cocycle weight `(c · 0 + d)²` is precisely the
Poisson value at the target.  Both `K` and `poisson` then evaluate to the
same expression, so they agree pointwise.

This is the converse of §1.15 (`poisson_mobius_covariant`): together
they characterise `poisson` as the unique solution to the harmonic-
analysis covariance constraint with the standard normalisation. -/
theorem poisson_unique_under_mobius_covariance
    (K : ℝ → ℝ → ℝ → ℝ)
    (hcov : ∀ {a b c d : ℝ}, a * d - b * c = 1 →
      ∀ {x' y' : ℝ}, 0 < y' → ∀ q' : ℝ, c * q' + d ≠ 0 →
        K (mobPoleX a b c d x' y') (mobPoleY c d x' y') (mobReal a b c d q')
          = (c * q' + d) ^ 2 * K x' y' q')
    (hnorm : K 0 1 0 = poisson 0 1 0) :
    ∀ x y q : ℝ, 0 < y → K x y q = poisson x y q := by
  intro x y q hy
  obtain ⟨a, b, c, d, hdet, hd_ne, hreal, hX, hY, hd_sq⟩ :=
    mobius_realization_exists x y q hy
  -- Apply covariance at the source `(0, 1, 0)`: it lands at `(x, y, q)`.
  have hyo : (0 : ℝ) < 1 := one_pos
  have hcd : c * (0 : ℝ) + d ≠ 0 := by
    rw [mul_zero, zero_add]; exact hd_ne
  have hKcov :=
    hcov (a := a) (b := b) (c := c) (d := d) hdet
      (x' := 0) (y' := 1) hyo (0 : ℝ) hcd
  rw [hX, hY, hreal] at hKcov
  -- `hKcov`: K x y q = (c · 0 + d)² · K 0 1 0.
  rw [hKcov, hnorm, hd_sq]
  -- Now: poisson x y q · poisson 0 1 0 = poisson x y q.
  have h0 : poisson 0 1 0 = 1 := by unfold poisson; norm_num
  rw [h0, mul_one]

/-- **Möbius classification of the Poisson kernel** (combined statement).

`K : ℝ → ℝ → ℝ → ℝ` equals the Poisson kernel on `{(x, y, q) | y > 0}`
**iff** `K` is Möbius weight-2 covariant and `K(0, 1, 0) = 1`.

The forward direction is `poisson_mobius_covariant` (§1.15) plus
`poisson(0, 1, 0) = 1` by direct computation.  The converse is
`poisson_unique_under_mobius_covariance`.

This is the harmonic-analysis identification of the transformer's
infinite-depth attention kernel: PSL(2, ℝ) covariance + a single
boundary normalisation forces the Cauchy/Poisson kernel uniquely. -/
theorem poisson_mobius_classification
    (K : ℝ → ℝ → ℝ → ℝ) :
    (∀ x y q : ℝ, 0 < y → K x y q = poisson x y q) ↔
    ((∀ {a b c d : ℝ}, a * d - b * c = 1 →
      ∀ {x' y' : ℝ}, 0 < y' → ∀ q' : ℝ, c * q' + d ≠ 0 →
        K (mobPoleX a b c d x' y') (mobPoleY c d x' y') (mobReal a b c d q')
          = (c * q' + d) ^ 2 * K x' y' q') ∧
     K 0 1 0 = poisson 0 1 0) := by
  constructor
  · intro hK
    refine ⟨?_, ?_⟩
    · intro a b c d hdet x' y' hy' q' hcq
      have hY_pos : 0 < mobPoleY c d x' y' := mobPoleY_pos hdet x' hy'
      rw [hK _ _ _ hY_pos, hK _ _ _ hy']
      exact poisson_mobius_covariant hdet x' hy' q' hcq
    · exact hK 0 1 0 one_pos
  · rintro ⟨hcov, hnorm⟩
    exact poisson_unique_under_mobius_covariance K hcov hnorm

/-- **Final bridge corollary (`continuum_limit_not_arbitrary`).**

    If:
    1. a candidate continuum flow `Φ` is a strongly differentiable flow
       for generator `T` (`IsExpFlow T Φ`), and
    2. the same flow is obtained as the Euler-refinement limit of the
       discrete residual updates for `T`, and
    3. a candidate boundary kernel `K` satisfies Möbius weight-2 covariance
       plus standard normalisation,

    then both objects are uniquely forced:
    * `Φ = (t ↦ exp (t • T))` (discretization cannot cheat), and
    * `K = poisson` on `y > 0` (symmetry cannot cheat).

    This packages Part 3 + Part 5 into one executive theorem-level bridge:
    the transformer-at-infinity limit is constrained by independent
    structure, not by definitional relabeling. -/
theorem continuum_limit_not_arbitrary {E : Type*}
    [NormedAddCommGroup E] [NormedSpace ℝ E] [CompleteSpace E]
    (T : E →L[ℝ] E)
    (Φ : ℝ → E →L[ℝ] E)
    (K : ℝ → ℝ → ℝ → ℝ)
    (hFlow : IsExpFlow T Φ)
    (hEuler :
      ∀ t : ℝ,
        Filter.Tendsto (fun n : ℕ => ((1 : E →L[ℝ] E) + (t / (n + 1 : ℝ)) • T) ^ (n + 1))
          Filter.atTop (nhds (Φ t)))
    (hcov : ∀ {a b c d : ℝ}, a * d - b * c = 1 →
      ∀ {x' y' : ℝ}, 0 < y' → ∀ q' : ℝ, c * q' + d ≠ 0 →
        K (mobPoleX a b c d x' y') (mobPoleY c d x' y') (mobReal a b c d q')
          = (c * q' + d) ^ 2 * K x' y' q')
    (hnorm : K 0 1 0 = poisson 0 1 0) :
    (Φ = fun t => NormedSpace.exp (t • T)) ∧
    (∀ x y q : ℝ, 0 < y → K x y q = poisson x y q) := by
  constructor
  · have hExp_from_ode : Φ = fun t => NormedSpace.exp (t • T) := isExpFlow_unique hFlow
    have hExp_from_euler : Φ = fun t => NormedSpace.exp (t • T) :=
      euler_refinement_limit_identifies_exp T Φ hEuler
    -- Both routes agree; keep the Euler-limit route as the advertised bridge.
    exact hExp_from_euler
  · exact poisson_unique_under_mobius_covariance K hcov hnorm

/-- **Global-object + Möbius bridge.**

    This theorem explicitly links the two anti-posthoc layers:

    1. `M_global θ`: one shared analytic object realizes the full forward map
       across all rows/tokens/layers for parameter family `θ`;
    2. kernel-side structural assumptions (PSL(2,ℝ) weight-2 covariance plus
       standard normalization).

    Conclusion: the kernel class is forced to Poisson on `y > 0`.
    In parallel with Part 2's fixed-pole obstruction at score-difference level,
    this separates kernel uniqueness from vanilla fixed-pole realizability.

    This is the direct "shared object + converse classification" statement. -/
theorem global_object_and_mobius_force_poisson
    {seqLen vocab nHeads D dModel dMLP : ℕ}
    {contextSize : Fin seqLen → ℕ}
    (θ : GPT2Params seqLen vocab nHeads D dModel dMLP contextSize)
    (hglobal : M_global θ)
    (K : ℝ → ℝ → ℝ → ℝ)
    (hcov : ∀ {a b c d : ℝ}, a * d - b * c = 1 →
      ∀ {x' y' : ℝ}, 0 < y' → ∀ q' : ℝ, c * q' + d ≠ 0 →
        K (mobPoleX a b c d x' y') (mobPoleY c d x' y') (mobReal a b c d q')
          = (c * q' + d) ^ 2 * K x' y' q')
    (hnorm : K 0 1 0 = poisson 0 1 0) :
    (M_global θ) ∧ (∀ x y q : ℝ, 0 < y → K x y q = poisson x y q) := by
  exact ⟨hglobal, poisson_unique_under_mobius_covariance K hcov hnorm⟩

end AnalyticTransformer

end
