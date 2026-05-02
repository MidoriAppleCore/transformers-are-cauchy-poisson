import Mathlib
import Part2_TheFullModel
import Part3_ContinuumLimit

/-!
# Part 4 — Replacement Axioms Block

This file packages the "non-tautology replacement block" into one reusable
structure and one constructor theorem.  The intent is practical:
downstream theorems can depend on this single block instead of depending on
separate tautology-prone links.

Given `θ : GPT2Params ...`, the block contains:

1. function-level identity (`analyticForward (scoreDerived...) = gpt2Forward`);
2. existential realization in the independent analytic parameter type;
3. canonicality of score-derived poles on the realised vertical slice;
4. continuum-flow uniqueness from the Cauchy generator;
5. generator recoverability at `t=0` from the Cauchy flow.

No new axioms are introduced; this is a packaging layer over Parts 2 and 3.
-/

noncomputable section

namespace AnalyticTransformer

/-- A single reusable theorem-block replacing the old tautology-style link.

    This is intentionally a `Prop`-valued structure so it can be passed as a
    compact assumption to later developments. -/
structure ReplacementBlock
    {seqLen vocab nHeads D dModel dMLP : ℕ}
    {contextSize : Fin seqLen → ℕ}
    (θ : GPT2Params seqLen vocab nHeads D dModel dMLP contextSize) : Prop where
  forward_eq :
    analyticForward (scoreDerivedAnalyticParams θ) = gpt2Forward θ
  exists_analytic :
    ∃ (φ : AnalyticGPT2Params seqLen vocab nHeads D dModel dMLP contextSize),
      analyticForward φ = gpt2Forward θ
  row_canonical :
    ∀ {N : ℕ} [NeZero N] (scores : Fin N → ℝ) (q : ℝ) (p : Poles N),
      (∀ k, p.x k = q) →
      (∀ k, poisson (p.x k) (p.y k) q = Real.exp (scores k) / ∑ j : Fin N, Real.exp (scores j)) →
      p = scorePoles scores q
  flow_unique :
    ∀ {D0 : ℕ} (F : (Fin D0 → ℝ) → Fin D0 → ℝ) (hF : IsRLinear F)
      {Φ : ℝ → (Fin D0 → ℝ) →L[ℝ] (Fin D0 → ℝ)},
      IsExpFlow (cauchyLinearOp F hF) Φ →
      Φ = cauchyResidualFlow F hF
  generator_recoverable :
    ∀ {D0 : ℕ} (F : (Fin D0 → ℝ) → Fin D0 → ℝ) (hF : IsRLinear F),
      cauchyLinearOp F hF = deriv (cauchyResidualFlow F hF) 0

/-- Explicit assumptions interface for deriving `ReplacementBlock`.

    This is intentionally separated from `ReplacementBlock` so downstream
    work can prove domain-specific hypotheses once, then obtain the full
    replacement block by a single implication theorem. -/
structure ReplacementAxioms
    {seqLen vocab nHeads D dModel dMLP : ℕ}
    {contextSize : Fin seqLen → ℕ}
    (θ : GPT2Params seqLen vocab nHeads D dModel dMLP contextSize) : Prop where
  forward_eq :
    analyticForward (scoreDerivedAnalyticParams θ) = gpt2Forward θ
  exists_analytic :
    ∃ (φ : AnalyticGPT2Params seqLen vocab nHeads D dModel dMLP contextSize),
      analyticForward φ = gpt2Forward θ
  row_canonical :
    ∀ {N : ℕ} [NeZero N] (scores : Fin N → ℝ) (q : ℝ) (p : Poles N),
      (∀ k, p.x k = q) →
      (∀ k, poisson (p.x k) (p.y k) q = Real.exp (scores k) / ∑ j : Fin N, Real.exp (scores j)) →
      p = scorePoles scores q
  flow_unique :
    ∀ {D0 : ℕ} (F : (Fin D0 → ℝ) → Fin D0 → ℝ) (hF : IsRLinear F)
      {Φ : ℝ → (Fin D0 → ℝ) →L[ℝ] (Fin D0 → ℝ)},
      IsExpFlow (cauchyLinearOp F hF) Φ →
      Φ = cauchyResidualFlow F hF
  generator_recoverable :
    ∀ {D0 : ℕ} (F : (Fin D0 → ℝ) → Fin D0 → ℝ) (hF : IsRLinear F),
      cauchyLinearOp F hF = deriv (cauchyResidualFlow F hF) 0

/-- Main implication: explicit assumptions imply the replacement block. -/
theorem replacementAxioms_imply_replacementBlock
    {seqLen vocab nHeads D dModel dMLP : ℕ}
    {contextSize : Fin seqLen → ℕ}
    {θ : GPT2Params seqLen vocab nHeads D dModel dMLP contextSize} :
    ReplacementAxioms θ → ReplacementBlock θ := by
  intro h
  exact
    { forward_eq := h.forward_eq
      exists_analytic := h.exists_analytic
      row_canonical := h.row_canonical
      flow_unique := h.flow_unique
      generator_recoverable := h.generator_recoverable }

/-- Constructor theorem: every GPT2 parameter family admits the full
    replacement block.  This is the theorem downstream files should import
    instead of leaning on older link-style equalities individually. -/
theorem transformer_tautology_replacement_block
    {seqLen vocab nHeads D dModel dMLP : ℕ}
    {contextSize : Fin seqLen → ℕ}
    (θ : GPT2Params seqLen vocab nHeads D dModel dMLP contextSize) :
    ReplacementBlock θ := by
  apply replacementAxioms_imply_replacementBlock
  refine
    { forward_eq := (transformers_are_boundary_value_solvers θ).symm
      exists_analytic := gpt2_has_pole_realization_functional θ
      row_canonical := ?_
      flow_unique := ?_
      generator_recoverable := ?_ }
  · intro N _ scores q p hx hw
    exact scorePoles_unique_on_vertical_slice scores q p hx hw
  · intro D0 F hF Φ hΦ
    exact cauchyResidualFlow_classification F hF hΦ
  · intro D0 F hF
    exact cauchyLinearOp_eq_deriv_cauchyResidualFlow_zero F hF

end AnalyticTransformer

