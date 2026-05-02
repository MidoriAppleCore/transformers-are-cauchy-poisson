import Mathlib
import Part1_TheIdentity

/-!
# Part 3 — The Continuum Limit (replaces the tautology)

This file replaces the tautology
"Cauchy/Poisson attention is just attention with a particular kernel"
with a real classification at the level of the residual-flow generator.

**Setting.**  Part 1 §1.19 builds the continuous Poisson semigroup
`cauchyResidualFlow F = exp(t • cauchyLinearOp F)` on the embedding space
`(Fin D → ℝ)` from a linear Cauchy-Poisson vector field `F`, and proves
the operator-level ODE `dΦ/dt = T · Φ` (where `T = cauchyLinearOp F`).
At that stage the flow exists and is differentiable, but a priori
nothing rules out *another* flow with the same generator that disagrees
with the Mathlib power-series construction off the diagonal.

**This file.**  We prove that there is exactly one such flow:

1. `IsExpFlow T Φ` — a strongly differentiable one-parameter
   contraction-style certificate: `Φ 0 = 1` and `dΦ/dt = T · Φ`.
2. `exp_smul_isExpFlow` — the Mathlib power-series flow satisfies the
   certificate (existence).
3. `isExpFlow_unique` — **classification**: any flow satisfying the
   certificate equals `t ↦ exp(t • T)` (uniqueness via Grönwall /
   `ODE_solution_unique_univ`).
4. `cauchyResidualFlow_classification` — specialised to the embedding
   space: any flow with generator `cauchyLinearOp F` is
   `cauchyResidualFlow F`.

**Why this drops the tautology.**  Saying "the residual stream is a
Cauchy/Poisson exponential" is no longer a renaming of `exp`, because
*any* strongly differentiable operator-valued flow with the right
generator is forced to be the §1.19 flow.  The architectural choice
"Cauchy-Poisson attention" classifies the entire continuous-time
trajectory of the residual stream.

**What this is *not*.**  The full doubly-continuous PSL(2,ℝ)-covariant
Hardy classification — "any strongly continuous contraction semigroup
on `H²(ℍ)` that is PSL(2,ℝ)-covariant in a precise sense is the
harmonic Poisson semigroup" — is not formalised here.  Mathlib does
not currently expose Hardy spaces of the upper half-plane with the
needed PSL(2,ℝ) representation theory.  The statement of that
classification is recorded in the docstring of
`isExpFlow_unique` as the natural strengthening once the prerequisites
land in mathlib.

**Theorems.**
* `IsExpFlow` (definition).
* `exp_smul_isExpFlow` — existence.
* `isExpFlow_unique` — uniqueness (the headline result).
* `cauchyResidualFlow_classification` — uniqueness specialised to the
  §1.19 setting.
-/

noncomputable section

open Filter Set Topology
open scoped NNReal Topology

namespace AnalyticTransformer

-- ═══════════════════════════════════════════════════════════════════════
-- § 3.1  Operator-flow certificate
-- ═══════════════════════════════════════════════════════════════════════

/-- A *strongly differentiable* one-parameter operator flow certificate.

    `IsExpFlow T Φ` says: `Φ 0 = 1` and `Φ` satisfies the operator-valued
    ODE `dΦ/dt = T · Φ` at every `t : ℝ`.  These are *exactly* the two
    properties Part 1 §1.19 proves of `cauchyResidualFlow`
    (`cauchyResidualFlow_zero` and `cauchyResidualFlow_hasDerivAt`).

    We do *not* assume `Φ` is built from a power series.  The point of
    Part 3 is: those two properties already pin `Φ` uniquely. -/
structure IsExpFlow {E : Type*}
    [NormedAddCommGroup E] [NormedSpace ℝ E] [CompleteSpace E]
    (T : E →L[ℝ] E) (Φ : ℝ → E →L[ℝ] E) : Prop where
  init : Φ 0 = 1
  ode  : ∀ t, HasDerivAt Φ (T * Φ t) t

-- ═══════════════════════════════════════════════════════════════════════
-- § 3.2  The Mathlib power-series flow is one such certificate
-- ═══════════════════════════════════════════════════════════════════════

/-- **Existence.**  The Mathlib power-series flow `t ↦ exp(t • T)`
    satisfies the certificate.  Both clauses are immediate from
    standard Mathlib lemmas (`exp_zero`, `hasDerivAt_exp_smul_const'`). -/
theorem exp_smul_isExpFlow {E : Type*}
    [NormedAddCommGroup E] [NormedSpace ℝ E] [CompleteSpace E]
    (T : E →L[ℝ] E) :
    IsExpFlow T (fun t => NormedSpace.exp (t • T)) where
  init := by
    show NormedSpace.exp ((0 : ℝ) • T) = 1
    rw [zero_smul, NormedSpace.exp_zero]
  ode  := fun t => hasDerivAt_exp_smul_const' T t

-- ═══════════════════════════════════════════════════════════════════════
-- § 3.3  Classification: generator pins the flow (replaces tautology)
-- ═══════════════════════════════════════════════════════════════════════

/-- **The classification theorem.**  Any `IsExpFlow T Φ` satisfies
    `Φ = fun t => exp(t • T)`.  In particular: among all strongly
    differentiable operator flows on a complete normed `ℝ`-space, the
    generator pins the entire trajectory.

    *Proof.*  Both `Φ` and `exp(t • T)` solve the operator-valued ODE
    `dY/dt = T · Y` on the Banach algebra `E →L[ℝ] E`, with the same
    initial value `Y(0) = 1`.  The right-hand side is Lipschitz in `Y`
    with constant `‖T‖` (this is the operator norm inequality
    `‖T · X‖ ≤ ‖T‖ · ‖X‖`), so Mathlib's
    `ODE_solution_unique_univ` (a Grönwall-style uniqueness statement)
    forces `Φ = exp(· • T)`.

    *Why this replaces the tautology.*  At finite size the equality
    "this kernel happens to be Poisson" is a labelling.  After taking
    the depth → ∞ limit (Part 1 §1.19), the residual flow becomes a
    one-parameter operator flow on the embedding space, and **the
    architectural choice (Cauchy-Poisson generator) classifies the
    entire continuous-time trajectory**.  No alternative flow with the
    same generator exists.

    *What is not yet formalised.*  The full Hardy classification —
    "the doubly-continuous PSL(2,ℝ)-covariant contraction semigroup on
    `H²(ℍ)` is unique" — would additionally pin the *generator* via
    symmetry, not just pin the flow given the generator.  That requires
    PSL(2,ℝ)-representation infrastructure on Hardy spaces of the
    upper half-plane that Mathlib does not currently provide. -/
theorem isExpFlow_unique {E : Type*}
    [NormedAddCommGroup E] [NormedSpace ℝ E] [CompleteSpace E]
    {T : E →L[ℝ] E} {Φ : ℝ → E →L[ℝ] E}
    (h : IsExpFlow T Φ) :
    Φ = fun t => NormedSpace.exp (t • T) := by
  -- Vector field on the operator Banach algebra: v t Y = T · Y.
  set v : ℝ → (E →L[ℝ] E) → (E →L[ℝ] E) := fun _ Y => T * Y with hv_def
  -- Lipschitz: ‖T·Y - T·Z‖ = ‖T·(Y - Z)‖ ≤ ‖T‖ · ‖Y - Z‖.
  have hLip : ∀ t, LipschitzOnWith ‖T‖₊ (v t) Set.univ := by
    intro _
    refine LipschitzWith.lipschitzOnWith ?_
    refine LipschitzWith.of_dist_le_mul ?_
    intro Y Z
    have hsub : T * Y - T * Z = T * (Y - Z) := (mul_sub T Y Z).symm
    have hbound : ‖T * Y - T * Z‖ ≤ ‖T‖ * ‖Y - Z‖ := by
      rw [hsub]
      exact norm_mul_le T (Y - Z)
    have hcoe : ((‖T‖₊ : ℝ≥0) : ℝ) = ‖T‖ := by
      simp [coe_nnnorm]
    simpa [dist_eq_norm, hcoe] using hbound
  -- Both Φ and exp(t • T) solve the ODE with the same initial value.
  have hC := exp_smul_isExpFlow T
  have hΦsol : ∀ t, HasDerivAt Φ (v t (Φ t)) t ∧ Φ t ∈ (Set.univ : Set _) :=
    fun t => ⟨h.ode t, trivial⟩
  have hCsol : ∀ t, HasDerivAt (fun s => NormedSpace.exp (s • T))
        (v t ((fun s => NormedSpace.exp (s • T)) t)) t ∧
      (fun s => NormedSpace.exp (s • T)) t ∈ (Set.univ : Set _) :=
    fun t => ⟨hC.ode t, trivial⟩
  have hinit : Φ 0 = (fun s : ℝ => NormedSpace.exp (s • T)) 0 := by
    rw [h.init]
    show (1 : E →L[ℝ] E) = NormedSpace.exp ((0 : ℝ) • T)
    rw [zero_smul, NormedSpace.exp_zero]
  exact ODE_solution_unique_univ hLip hΦsol hCsol hinit

/-- **Generator uniqueness from the same flow.**

    If one and the same flow `Φ` satisfies the `IsExpFlow` certificate for
    two generators `T` and `S`, then `T = S`.

    This is the key non-tautological strengthening beyond mere existence:
    once the flow is fixed, the infinitesimal generator is *forced*. -/
theorem isExpFlow_generator_unique {E : Type*}
    [NormedAddCommGroup E] [NormedSpace ℝ E] [CompleteSpace E]
    {T S : E →L[ℝ] E} {Φ : ℝ → E →L[ℝ] E}
    (hT : IsExpFlow T Φ) (hS : IsExpFlow S Φ) :
    T = S := by
  have hderiv_eq : T * Φ 0 = S * Φ 0 := by
    exact (hT.ode 0).unique (hS.ode 0)
  have hT1 : T * Φ 0 = T := by
    rw [hT.init, mul_one]
  have hS1 : S * Φ 0 = S := by
    rw [hS.init, mul_one]
  rw [hT1, hS1] at hderiv_eq
  exact hderiv_eq

/-- **Generator recovered from the flow.**

    For an `IsExpFlow T Φ`, the derivative at time `0` is exactly `T`.
    So the generator is read off from the trajectory itself, not chosen
    externally. -/
theorem isExpFlow_generator_eq_deriv0 {E : Type*}
    [NormedAddCommGroup E] [NormedSpace ℝ E] [CompleteSpace E]
    {T : E →L[ℝ] E} {Φ : ℝ → E →L[ℝ] E}
    (h : IsExpFlow T Φ) :
    deriv Φ 0 = T := by
  have h0 : HasDerivAt Φ (T * Φ 0) 0 := h.ode 0
  have hmul : T * Φ 0 = T := by rw [h.init, mul_one]
  rw [← hmul]
  exact h0.deriv

-- ═══════════════════════════════════════════════════════════════════════
-- § 3.3b  Discretization pins the continuum limit
-- ═══════════════════════════════════════════════════════════════════════

/-- **No-cheating discretization theorem.**

    Suppose `Φ` is proposed as a continuum limit for the Euler refinements
    of generator `T`, in the exact sense that for each time `t`, the
    sequence `((1 + (t/(n+1))•T)^(n+1))` tends to `Φ t`.

    Then `Φ` is forced to be the exponential flow `t ↦ exp(t • T)`.
    The proof is one line of limit uniqueness: Part 1 already proves that
    the same Euler sequence tends to `exp(t • T)` (`operatorEulerLimit`),
    so any other claimed limit must coincide with it.

    This is the direct formal witness that "the continuum object is not
    hand-picked": once you demand it be the Euler-refinement limit, there
    is exactly one possible answer. -/
theorem euler_refinement_limit_identifies_exp {E : Type*}
    [NormedAddCommGroup E] [NormedSpace ℝ E] [CompleteSpace E]
    (T : E →L[ℝ] E)
    (Φ : ℝ → E →L[ℝ] E)
    (hEuler :
      ∀ t : ℝ,
        Tendsto (fun n : ℕ => ((1 : E →L[ℝ] E) + (t / (n + 1 : ℝ)) • T) ^ (n + 1))
          atTop (nhds (Φ t))) :
    Φ = fun t => NormedSpace.exp (t • T) := by
  funext t
  exact tendsto_nhds_unique (hEuler t) (operatorEulerLimit T t)

-- ═══════════════════════════════════════════════════════════════════════
-- § 3.4  Specialisation to the §1.19 embedding-space flow
-- ═══════════════════════════════════════════════════════════════════════

/-- **Specialisation to the embedding space.**  Any `IsExpFlow` on
    `(Fin D → ℝ)` with generator `cauchyLinearOp F` is exactly the
    Part 1 §1.19 flow `cauchyResidualFlow F`.

    This is the form that directly addresses the original "Transformer
    = Cauchy-Poisson" claim: at the continuum-depth limit, *every*
    strongly differentiable operator-flow with the Cauchy-Poisson
    generator is the §1.19 flow.  No alternative continuum dynamics
    can match the generator and disagree with `cauchyResidualFlow`. -/
theorem cauchyResidualFlow_classification {D : ℕ}
    (F : (Fin D → ℝ) → Fin D → ℝ) (hF : IsRLinear F)
    {Φ : ℝ → (Fin D → ℝ) →L[ℝ] (Fin D → ℝ)}
    (h : IsExpFlow (cauchyLinearOp F hF) Φ) :
    Φ = cauchyResidualFlow F hF := by
  have hΦ_eq : Φ = fun t => NormedSpace.exp (t • cauchyLinearOp F hF) :=
    isExpFlow_unique h
  funext t
  rw [hΦ_eq]
  rfl

/-- **Pointwise corollary.**  For every starting embedding `v` and
    every time `t`, the trajectory of any `IsExpFlow` matches the
    §1.19 trajectory.  This is the "transformer-friendly" form of the
    classification: pick a starting embedding, run any candidate
    continuum flow with the right generator, and the orbit is exactly
    the Cauchy-Poisson orbit. -/
theorem cauchyResidualFlow_classification_apply {D : ℕ}
    (F : (Fin D → ℝ) → Fin D → ℝ) (hF : IsRLinear F)
    {Φ : ℝ → (Fin D → ℝ) →L[ℝ] (Fin D → ℝ)}
    (h : IsExpFlow (cauchyLinearOp F hF) Φ) (t : ℝ) (v : Fin D → ℝ) :
    Φ t v = cauchyResidualFlow F hF t v := by
  have := cauchyResidualFlow_classification F hF h
  rw [this]

/-- **Existence side:** the §1.19 flow itself satisfies the
    certificate.  Together with `cauchyResidualFlow_classification`
    this gives an `↔`-style statement: a flow on the embedding space
    has generator `cauchyLinearOp F` (in the strong-derivative sense
    of `IsExpFlow`) iff it equals `cauchyResidualFlow F`. -/
theorem cauchyResidualFlow_isExpFlow {D : ℕ}
    (F : (Fin D → ℝ) → Fin D → ℝ) (hF : IsRLinear F) :
    IsExpFlow (cauchyLinearOp F hF) (cauchyResidualFlow F hF) where
  init := cauchyResidualFlow_zero F hF
  ode  := cauchyResidualFlow_hasDerivAt F hF

/-- **Cauchy generator is recovered from the Cauchy flow at `t=0`.**

    This is the embedding-space specialization of
    `isExpFlow_generator_eq_deriv0`: the infinitesimal generator of the
    §1.19 flow is not an extra assumption; it is exactly the derivative
    of `t ↦ cauchyResidualFlow F hF t` at zero. -/
theorem cauchyLinearOp_eq_deriv_cauchyResidualFlow_zero {D : ℕ}
    (F : (Fin D → ℝ) → Fin D → ℝ) (hF : IsRLinear F) :
    cauchyLinearOp F hF = deriv (cauchyResidualFlow F hF) 0 := by
  symm
  exact isExpFlow_generator_eq_deriv0 (cauchyResidualFlow_isExpFlow F hF)

end AnalyticTransformer
