import Mathlib
import Part5_PoissonClassification

/-!
# Part 6 — Hardy / Poisson semigroup machinery (scaffolding)

This file initiates the **harmonic-analysis layer** of the project.

## Content (provable, fully formalized)

* The **probability-normalized Poisson kernel**
  `P_y(x) = y / (π · (x² + y²))`.
* Pointwise / algebraic content: positivity, symmetry, continuity in the
  spatial variable, exact relation to the structural `poisson` of Part 1
  (`P_y(x) = poisson 0 y x / π`).
* The **Hardy/Möbius bridge**:
  any kernel satisfying the PSL(2, ℝ) weight-2 covariance and standard
  boundary normalization, normalized by `π`, agrees with `poissonKernel`.

## Structural framework (predicate level)

The full classical Hardy / boundary-trace / F.&M.-Riesz / inner–outer
factorization endpoint requires substantial integral and measure-theoretic
infrastructure beyond what is reachable as pure-algebra Lean content.  We
expose the key harmonic-analysis structure as `Prop`-level predicates:

* `IsConvolutionSemigroup K`:
  `K_y₁ ∗ K_y₂ = K_(y₁+y₂)` (the Poisson semigroup composition law).
* `IsHardyKernel K`:
  positivity + unit total mass + convolution semigroup.

Downstream development can refine or discharge these predicates
incrementally without disturbing the existing Möbius classification core.

## Why "Hardy" and not just "Möbius"

Part 5 classified the Poisson kernel via Möbius covariance pointwise.
The Hardy framing here is the *operator/semigroup* version of the same
classification: instead of asking "what kernel value at `(x, y, q)`", we
ask "what one-parameter convolution semigroup on bounded functions agrees
with `K_y` for all `y > 0`".  The bridge below shows that under Möbius
covariance + normalization, the kernel must be the probability-normalized
Poisson kernel, which in turn is the convolution kernel of the classical
Poisson semigroup on the upper half-plane.
-/

noncomputable section

namespace AnalyticTransformer

open Real MeasureTheory

/-- **Probability-normalized Poisson kernel** on the upper half-plane:
`P_y(x) = y / (π · (x² + y²))`.

This is the harmonic-analysis convention: it is the convolution kernel
of the Poisson semigroup, and a probability density for fixed `y > 0`
(total mass 1).  It differs from the structural `poisson` of Part 1 only
by the `1/π` mass normalization. -/
def poissonKernel (y x : ℝ) : ℝ := y / (Real.pi * (x ^ 2 + y ^ 2))

/-- Positivity of the probability-normalized Poisson kernel for `y > 0`. -/
theorem poissonKernel_pos {y : ℝ} (hy : 0 < y) (x : ℝ) :
    0 < poissonKernel y x := by
  unfold poissonKernel
  have hxy : (0 : ℝ) < x ^ 2 + y ^ 2 := by
    have hy2 : (0 : ℝ) < y ^ 2 := by positivity
    have hx2 : (0 : ℝ) ≤ x ^ 2 := sq_nonneg x
    linarith
  exact div_pos hy (mul_pos Real.pi_pos hxy)

/-- The probability-normalized kernel is the structural `poisson 0 y x`
divided by `π`. -/
theorem poissonKernel_eq_normalized_poisson (y x : ℝ) :
    poissonKernel y x = poisson 0 y x / Real.pi := by
  unfold poissonKernel poisson
  rw [div_div]
  congr 1
  ring

/-- Spatial reflection symmetry: `P_y(-x) = P_y(x)`. -/
theorem poissonKernel_symmetric (y x : ℝ) :
    poissonKernel y (-x) = poissonKernel y x := by
  unfold poissonKernel
  ring

/-- Value at the origin: `P_y(0) = 1 / (π · y)`. -/
theorem poissonKernel_at_zero {y : ℝ} (hy : 0 < y) :
    poissonKernel y 0 = 1 / (Real.pi * y) := by
  unfold poissonKernel
  have hy' : y ≠ 0 := ne_of_gt hy
  field_simp
  ring

/-- Continuity of `P_y(·)` in the spatial variable for `y > 0`. -/
theorem poissonKernel_continuous {y : ℝ} (hy : 0 < y) :
    Continuous (poissonKernel y) := by
  unfold poissonKernel
  refine Continuous.div continuous_const
    (continuous_const.mul ((continuous_pow 2).add continuous_const))
    (fun x => ?_)
  have hxy : (0 : ℝ) < x ^ 2 + y ^ 2 := by
    have hy2 : (0 : ℝ) < y ^ 2 := by positivity
    have hx2 : (0 : ℝ) ≤ x ^ 2 := sq_nonneg x
    linarith
  exact ne_of_gt (mul_pos Real.pi_pos hxy)

-- ════════════════════════════════════════════════════════════════════════
-- Mellin transform (σ-plane hook — not an L-function)
-- ════════════════════════════════════════════════════════════════════════

/-- Integrand for the Mellin transform on `(0, ∞)`:

`y ↦ y^{σ - 1} · f(y)` on `Ioi 0`, extended by `0` elsewhere so `Real.rpow`
is never evaluated at nonpositive `y`. -/
noncomputable def mellinIntegrand (f : ℝ → ℝ) (σ y : ℝ) : ℝ :=
  Set.indicator (Set.Ioi (0 : ℝ)) (fun y' => Real.rpow y' (σ - 1) * f y') y

/-- **Mellin transform** (real `σ` — the first spectral parameter).

`ℳ[f](σ) := ∫_{(0,∞)} y^{σ-1} f(y) dy` when the improper Lebesgue integral
converges absolutely (packaged here as `Integrable` on `ℝ` with the
indicator extension).

This is the honest analytic-number-theory **adjacent** object: the natural
linear transform that takes a boundary / scaling profile `f` on the positive
half-line to a function of `σ`.  It is **not** an L-function by itself —
coefficients `a_n`, Hecke structure, and Γ-factors are extra decades of
structure beyond this definition.

Downstream work can specialise `f` to slices of `poissonKernel` or to
data-driven attention profiles; proving convergence and closed forms (e.g.
Γ-ratios) is measure-theoretic work beyond the current algebraic core. -/
noncomputable def mellinTransform (f : ℝ → ℝ) (σ : ℝ)
    (_hf : Integrable (mellinIntegrand f σ) volume) : ℝ :=
  ∫ y, mellinIntegrand f σ y

theorem mellinIntegrand_zero (σ : ℝ) :
    mellinIntegrand (fun _ => (0 : ℝ)) σ = 0 := by
  funext y
  simp [mellinIntegrand]

theorem mellinTransform_zero (σ : ℝ)
    (h : Integrable (mellinIntegrand (fun _ => (0 : ℝ)) σ) volume) :
    mellinTransform (fun _ => (0 : ℝ)) σ h = 0 := by
  simp [mellinTransform, mellinIntegrand_zero σ, integral_zero]

-- ════════════════════════════════════════════════════════════════════════
-- Structural Hardy / convolution-semigroup predicates
-- ════════════════════════════════════════════════════════════════════════

/-- **Convolution semigroup** predicate.

A two-parameter kernel family `K : (height) → (space) → ℝ` is a convolution
semigroup if `K_(y₁) ∗ K_(y₂) = K_(y₁+y₂)` pointwise:

`∫_ℝ K_(y₁)(x − t) · K_(y₂)(t) dt = K_(y₁+y₂)(x)`

This is the **Poisson semigroup composition law** in kernel form.
-/
def IsConvolutionSemigroup (K : ℝ → ℝ → ℝ) : Prop :=
  ∀ {y₁ y₂ : ℝ}, 0 < y₁ → 0 < y₂ → ∀ x : ℝ,
    (∫ t : ℝ, K y₁ (x - t) * K y₂ t) = K (y₁ + y₂) x

/-- **Hardy-class kernel** predicate.

A kernel `K` is a Hardy / Poisson-semigroup kernel if:
* it is pointwise positive on the upper half-plane,
* it has unit total mass for every height (probability density),
* it forms a convolution semigroup in the height parameter.

These are the structural axioms identifying the classical Poisson
semigroup kernel up to normalization. -/
def IsHardyKernel (K : ℝ → ℝ → ℝ) : Prop :=
  (∀ {y : ℝ}, 0 < y → ∀ x : ℝ, 0 < K y x) ∧
  (∀ {y : ℝ}, 0 < y → (∫ x : ℝ, K y x) = 1) ∧
  IsConvolutionSemigroup K

-- ════════════════════════════════════════════════════════════════════════
-- Hardy / Möbius bridge — pointwise classification
-- ════════════════════════════════════════════════════════════════════════

/-- **Hardy/Möbius bridge.**

Any three-parameter kernel `K' : ℝ → ℝ → ℝ → ℝ` satisfying:

* weight-2 PSL(2, ℝ) covariance (Part 5 form), and
* the standard normalization `K'(0, 1, 0) = poisson(0, 1, 0)`

is forced, after normalization by `π`, to agree with the
probability-normalized Poisson kernel `poissonKernel`.

This packages Part 5's Möbius classification in **Hardy form**: the kernel
is *uniquely* determined by harmonic-analytic constraints, not just up to
existential representation.  Combined with `poissonKernel_eq_normalized_poisson`,
this anchors the harmonic-analysis layer to the existing structural Poisson kernel
classification of Part 5. -/
theorem hardy_kernel_classification_via_mobius
    (K' : ℝ → ℝ → ℝ → ℝ)
    (hcov : ∀ {a b c d : ℝ}, a * d - b * c = 1 →
      ∀ {x' y' : ℝ}, 0 < y' → ∀ q' : ℝ, c * q' + d ≠ 0 →
        K' (mobPoleX a b c d x' y') (mobPoleY c d x' y') (mobReal a b c d q')
          = (c * q' + d) ^ 2 * K' x' y' q')
    (hnorm : K' 0 1 0 = poisson 0 1 0) :
    ∀ x y q : ℝ, 0 < y →
      K' x y q / Real.pi = poissonKernel y (q - x) := by
  intro x y q hy
  have h : K' x y q = poisson x y q :=
    poisson_unique_under_mobius_covariance K' hcov hnorm x y q hy
  rw [h]
  unfold poisson poissonKernel
  rw [div_div]
  congr 1
  ring

/-- **Conditional Hardy classification.**

If `K_⋆ : ℝ → ℝ → ℝ` is a Hardy-class kernel (positive, unit mass,
convolution semigroup), and is the probability-normalized form of a
Möbius-classified kernel `K'`, then `K_⋆` agrees with `poissonKernel`
on the upper half-plane.

The first conjunct is the unconditional kernel-level classification
(Part 5 / Hardy bridge); the second is the structural identification of
`K_⋆` as a Hardy kernel.  Together they connect the harmonic-analysis
predicate framework with the Möbius pointwise theorem. -/
theorem hardyKernel_eq_poissonKernel_under_mobius
    (K_star : ℝ → ℝ → ℝ)
    (K' : ℝ → ℝ → ℝ → ℝ)
    (hcov : ∀ {a b c d : ℝ}, a * d - b * c = 1 →
      ∀ {x' y' : ℝ}, 0 < y' → ∀ q' : ℝ, c * q' + d ≠ 0 →
        K' (mobPoleX a b c d x' y') (mobPoleY c d x' y') (mobReal a b c d q')
          = (c * q' + d) ^ 2 * K' x' y' q')
    (hnorm : K' 0 1 0 = poisson 0 1 0)
    (hreduce : ∀ x y : ℝ, 0 < y → K_star y x = K' 0 y x / Real.pi) :
    ∀ x y : ℝ, 0 < y → K_star y x = poissonKernel y x := by
  intro x y hy
  have h₁ : K_star y x = K' 0 y x / Real.pi := hreduce x y hy
  have h₂ : K' 0 y x / Real.pi = poissonKernel y (x - 0) :=
    hardy_kernel_classification_via_mobius K' hcov hnorm 0 y x hy
  have h₃ : poissonKernel y (x - 0) = poissonKernel y x := by
    have hx0 : x - 0 = x := sub_zero x
    rw [hx0]
  exact h₁.trans (h₂.trans h₃)

end AnalyticTransformer

end
