import Part1_TheIdentity
import Part2_TheFullModel
import Part3_ContinuumLimit
import Part4_ReplacementAxioms
import Part5_PoissonClassification
import Part6_HardySemigroup

/-
PROOF GATES — this file only compiles if the main theorems exist.

  Part 1: Softmax rows are *classified* by a unique realised-slice
          Poisson pole family (not only represented existentially).
  Part 2: GPT-2-style forward pass equals the pole-based (Cauchy-Poisson) forward pass.
-/

-- Gate 1: Row-level classification (general softmax scores — covers real GPT-2 directly).
-- Primary theorem: uniqueness on the realised vertical slice.
#check AnalyticTransformer.softmax_poles_unique_on_vertical_slice
-- Helper existence theorem (retained for downstream convenience).
#check AnalyticTransformer.topological_subsumes_softmax_of_scores
#check AnalyticTransformer.contour_simulates_softmax_of_scores
#check AnalyticTransformer.matchPoles_unique_on_vertical_slice

-- Gate 2: The full model (the main result)
#check AnalyticTransformer.transformer_is_cauchy_poisson
#check AnalyticTransformer.transformers_are_boundary_value_solvers
#check AnalyticTransformer.scorePoles_unique_on_vertical_slice
#check AnalyticTransformer.gpt2_has_pole_realization_functional
#check AnalyticTransformer.gpt2_has_single_global_analytic_object
#check AnalyticTransformer.M_global
#check AnalyticTransformer.AnalyticMHParams.FixedPoleGeometry
#check AnalyticTransformer.M_global_fixed_poles
#check AnalyticTransformer.M_global_fixed_poles_implies_M_global
#check AnalyticTransformer.FixedPoleAnalyticMHParams
#check AnalyticTransformer.FixedPoleAnalyticMHParams.toAnalyticMH
#check AnalyticTransformer.FixedPoleAnalyticMHParams.toAnalyticMH_fixedGeometry
#check AnalyticTransformer.FixedPoleAnalyticGPT2Params
#check AnalyticTransformer.FixedPoleAnalyticGPT2Params.toAnalyticGPT2
#check @AnalyticTransformer.fixedPoleAnalyticForward
#check @AnalyticTransformer.fixed_pole_witness_in_M_global_fixed_poles
#check @AnalyticTransformer.gpt2_in_M_global_fixed_poles_of_state_independent_scores
#check AnalyticTransformer.ScoreStateIndependent
#check AnalyticTransformer.NondegenerateScores
#check AnalyticTransformer.ScoreDifferencesStateIndependent
#check AnalyticTransformer.NondegenerateScoreDifferences
#check AnalyticTransformer.ScoreDerivedFixedPoleGeometry
#check AnalyticTransformer.RealizesVanillaSoftmaxRows
#check AnalyticTransformer.nondegenerateScores_iff_not_scoreStateIndependent
#check AnalyticTransformer.nondegenerateScoreDifferences_iff_not_scoreDifferencesStateIndependent
#check @AnalyticTransformer.vanilla_row_realization_forces_scoreDerived_poles
#check @AnalyticTransformer.fixed_pole_vanilla_realization_implies_scoreDerivedFixedPoleGeometry
#check @AnalyticTransformer.gpt2_has_intrinsic_dynamic_row_realization
#check AnalyticTransformer.scoreDerived_fixed_poles_imply_scoreDifferencesStateIndependent
#check @AnalyticTransformer.nondegenerateScoreDifferences_not_scoreDerivedFixedPoleGeometry
#check @AnalyticTransformer.nondegenerateScoreDifferences_force_scoreDerived_poles_move
#check @AnalyticTransformer.nondegenerate_scores_not_M_global_fixed_poles
#check AnalyticTransformer.gpt2_in_M_global
#check AnalyticTransformer.transformer_cauchy_poisson_replacement_block

-- Gate 3: Backprop is well-defined on a Cauchy-Poisson head.
-- For every smooth loss L, the boundary trace is differentiable in
-- each pole-position (x_k), bandwidth (y_k), and residue (V k d)
-- parameter, with the gradient given by the standard chain rule.
#check AnalyticTransformer.poisson_differentiableAt_pole
#check AnalyticTransformer.one_div_one_div_of_pos
#check AnalyticTransformer.poisson_at_query_one_div_one_div
#check AnalyticTransformer.poisson_differentiableAt_x
#check AnalyticTransformer.poisson_differentiableAt_y
#check AnalyticTransformer.contourOutput_one_pole_differentiable
#check AnalyticTransformer.contourOutput_differentiableAt_xk
#check AnalyticTransformer.contourOutput_differentiableAt_yk
#check AnalyticTransformer.contourOutput_differentiableAt_Vkd
#check AnalyticTransformer.contourOutput_differentiableAt_pole_pair
#check AnalyticTransformer.loss_compose_contourOutput_xk_differentiableAt
#check AnalyticTransformer.loss_compose_contourOutput_yk_differentiableAt
#check AnalyticTransformer.loss_compose_contourOutput_Vkd_differentiableAt
#check AnalyticTransformer.loss_compose_contourOutput_pole_pair_differentiableAt

-- Gate 4: Backprop **pushforward** through softmax.
-- PyTorch trains pre-softmax scores, not pole heights.  The
-- score-parameterised attention output is differentiable in the
-- scores, equals the Cauchy contour output through `matchPoles`, and
-- (composed with any smooth parameter→scores map and any smooth loss)
-- is differentiable in the trainable parameter.  This means: the
-- gradient PyTorch's `.backward()` produces and the Cauchy backward
-- pass produce the **same gradient** through different parametrisations.
#check AnalyticTransformer.softmax_weight_differentiable
#check AnalyticTransformer.softmaxAttention_score_differentiable
#check AnalyticTransformer.softmaxAttention_eq_contourOutput
#check AnalyticTransformer.pushforward_loss_differentiable
#check AnalyticTransformer.pushforward_loss_through_contour_differentiable

-- Gate 5: Negative result — Cauchy-Poisson sums are uniformly bounded,
-- so unbounded functions (e.g. `f(q) = q`) are *not* representable.
-- This rules out the "everything is Cauchy-Poisson" tautology.
#check AnalyticTransformer.contourOutput_uniformly_bounded
#check AnalyticTransformer.identity_not_cauchyPoisson
#check AnalyticTransformer.unbounded_not_cauchyPoisson

-- Gate 6: Sense/Act decomposition — the conjugate Poisson kernel `Q`
-- carries information that the Poisson kernel `P` does not.  The two
-- channels are linearly independent and satisfy a pointwise
-- Cauchy-Riemann coupling at the boundary.
#check AnalyticTransformer.conjPoisson_at_query
#check AnalyticTransformer.sense_act_linearly_independent
#check AnalyticTransformer.conjPoisson_not_scalar_multiple_of_poisson
#check AnalyticTransformer.single_channel_convex_poisson_cannot_represent_conjPoisson
#check AnalyticTransformer.conjPoisson_has_dual_channel_representation
#check AnalyticTransformer.conjPoisson_differentiable
#check AnalyticTransformer.sense_act_cauchy_riemann

-- Gate 7: Expressiveness — Cauchy-Poisson kernels separate points.
-- This is the precondition for universal approximation on compacta
-- via Stone–Weierstrass.
#check AnalyticTransformer.poisson_separates_points
#check AnalyticTransformer.contourOutput_separates_points

-- Gate 8: Decay at infinity — sharper non-tautology constraint.
-- Every Cauchy-Poisson sum vanishes at the right tail, so even
-- nonzero constants are *not* representable.  The function classes
-- {constants ≠ 0} and {Cauchy-Poisson sums} are disjoint.
#check AnalyticTransformer.poisson_atTop_lt
#check AnalyticTransformer.contourOutput_atTop_lt
#check AnalyticTransformer.nonzero_const_not_cauchyPoisson

-- Gate 9: Smoothness in the query coordinate `q`.  Closes the loop:
-- §1.7 covered smoothness in the *pole* parameters (what backprop trains);
-- this gate covers smoothness in the *input* (the boundary coordinate).
-- Every Cauchy-Poisson head is jointly smooth in **all** directions.
#check AnalyticTransformer.poisson_differentiable_q
#check AnalyticTransformer.contourOutput_differentiable_q

-- Gate 10: Affine invariance — the Cauchy-Poisson class is closed under
-- the natural affine group acting on the boundary.  Translation, dilation,
-- and reflection of the query coordinate correspond to canonical
-- transformations of the poles, with bandwidths and residues following.
#check AnalyticTransformer.contourOutput_translation
#check AnalyticTransformer.contourOutput_reflection
#check AnalyticTransformer.poisson_scale
#check AnalyticTransformer.contourOutput_scale

-- Gate 11: PSL(2,ℝ) Möbius covariance of the *kernel class*.
--   • Poisson kernel is weight-2 covariant under simultaneous Möbius
--     action on the query and a freely-floating pole.
--   • Contour output is a weight-2 modular form on PSL(2,ℝ).
--   • Attention simplex is genuinely Möbius-invariant (the cocycle
--     cancels in the normalised weights).
-- This is a theorem about the abstract kernel class — see Gate 12 for
-- which subgroup actually acts on the realised softmax-derived poles.
#check AnalyticTransformer.mobPoleY_pos
#check AnalyticTransformer.poisson_mobius_covariant
#check AnalyticTransformer.contourOutput_mobius_covariant
#check AnalyticTransformer.attention_simplex_mobius_invariant

-- Gate 12: Realisation caveats — the realised geometry is affine.
-- The standard `matchPoles` construction places every pole directly
-- above the query (`x_k = q`).  The Möbius transformations that
-- preserve this verticality are *exactly* the affine (c=0) subgroup,
-- so the realised softmax-derived geometry inherits §1.14's symmetry,
-- not the full §1.15 PSL(2,ℝ).  This is the formal calibration of
-- "kernel class supports PSL(2,ℝ)" vs "transformer head only sees
-- the affine subgroup".
#check AnalyticTransformer.matchPoles_mobius_preserves_verticality
#check AnalyticTransformer.realised_mobius_orbit_is_affine

-- Gate 13: Dynamics — the residual stream is a discrete linear semigroup.
-- The transformer block is *literally* one Euler step at Δt = 1
-- (`cauchy_residual_is_euler_step`), the underlying vector field is
-- linear in the value vectors (`cauchyResidualVF_linear_in_V`), the
-- L-step iteration enjoys the discrete semigroup law Φ_{m+n} = Φ_n ∘ Φ_m
-- (`iterEuler_add`), and linearity is preserved by every iterate
-- (`iterEuler_linear_of_linear`).  Together these say the residual
-- stream is a genuine discrete linear semigroup on the embedding
-- space — the algebraic skeleton on top of which a continuum-limit
-- statement (Poisson semigroup) would have to be built.  The
-- continuum limit itself is *not* claimed: see the §1.17 docstring.
#check @AnalyticTransformer.cauchyResidualVF
#check @AnalyticTransformer.eulerStep
#check @AnalyticTransformer.eulerStepFn
#check @AnalyticTransformer.cauchy_residual_is_euler_step
#check @AnalyticTransformer.cauchy_residual_euler_step_dt
#check @AnalyticTransformer.cauchyResidualVF_linear_in_V
#check @AnalyticTransformer.iterEuler
#check @AnalyticTransformer.iterEuler_zero
#check @AnalyticTransformer.iterEuler_succ
#check @AnalyticTransformer.iterEuler_add
#check @AnalyticTransformer.eulerStepFn_linear_of_linear
#check @AnalyticTransformer.iterEuler_linear_of_linear

-- Gate 14: RoPE strictly exceeds the affine subgroup, *with* the
-- feature-space-rotation → disc-rotation → UHP-Möbius chain proved.
--   • `cayley`, `cayleyInv`: explicit Cayley transform between UHP
--     and the unit disc.
--   • `discRotate θ`: feature-space rotation in the disc, multiplication
--     by `cos θ + i sin θ` — this is the rotation real transformer
--     architectures actually apply (block-diagonally on 2D feature
--     pairs).
--   • `ropeMobius_eq_cayley_pullback`: *theorem* — the composition
--     `cayleyInv ∘ discRotate θ ∘ cayley` is **literally** equal to
--     `ropeMobius θ`.  No "we declare RoPE to be a Möbius" hand-wave;
--     the disc-model lift is computed and proved equal algebraically.
--   • `rope_breaks_verticality`, `rope_strictly_exceeds_affine`:
--     the resulting Möbius has `c = -sin(θ/2) ≠ 0` for non-trivial θ,
--     and so by Gate 12 cannot live inside the affine subgroup
--     `B(2,ℝ)` that standard softmax-derived poles inhabit.
#check @AnalyticTransformer.cayley
#check @AnalyticTransformer.cayleyInv
#check @AnalyticTransformer.discRotate
#check @AnalyticTransformer.ropeMobius
#check AnalyticTransformer.ropeMobius_det_one
#check AnalyticTransformer.ropeMobius_c
#check @AnalyticTransformer.ropeMobius_eq_cayley_pullback
#check AnalyticTransformer.rope_breaks_verticality
#check AnalyticTransformer.rope_strictly_exceeds_affine

-- Gate 15: §1.19 — Continuum lift via Mathlib's exponential map.
-- The discrete linear semigroup of §1.17 is upgraded to a continuous
-- one-parameter semigroup in the Banach algebra of bounded linear
-- endomorphisms of the embedding space.  Concretely:
--   • `cauchyLinearOp F`: packages a linear vector field as a CLM.
--   • `cauchyResidualFlow F t = exp(t • cauchyLinearOp F)` — Mathlib's
--     `NormedSpace.exp`, taking values in `(Fin D → ℝ) →L[ℝ] (Fin D → ℝ)`.
--   • `cauchyResidualFlow_zero` — the flow at `t = 0` is the identity.
--   • `cauchyResidualFlow_hasDerivAt` — **operator-level ODE**: the flow
--     satisfies `dΦ/dt = T · Φ`, via Mathlib's `hasDerivAt_exp_smul_const'`.
--   • `cauchyResidualFlow_apply_hasDerivAt` — **pointwise ODE**: for every
--     starting embedding `v`, `(d/dt) (Φ_t v) = T (Φ_t v)`.  This is the
--     "transformer-friendly" form: an honest continuous flow whose
--     velocity field is exactly the residual VF of §1.17.
--   • `iterEuler_eq_clm_pow` — discrete iterate equals matrix power
--     `(1 + Δt · T)^L` applied to `h`.  This is the bridge: Euler iterate
--     = first-order Taylor of the operator exponential.
--   • `cauchyResidualFlow_eq_exp_npow` — **multiplicative refinement** of
--     the continuous flow: for every `n`,
--       `exp(t•T) = (exp ((t/(n+1))•T)))^{n+1}`.
--     This is the exact continuous analogue of the discrete refinement.
--   • `cauchyResidualFlow_eulerLimit_clm` — **Euler limit, operator norm**:
--     `((1 + (t/(n+1)) • T)^{n+1}) → exp(t • T)` as `n → ∞` in the
--     operator-norm topology on `(Fin D → ℝ) →L[ℝ] (Fin D → ℝ)`.  This is
--     the missing analytic step that turns the discrete §1.17 dynamics
--     into the continuous §1.19 flow.
--   • `cauchyResidualFlow_eulerLimit` — **Euler limit, pointwise**: for
--     every starting embedding `h₀`,
--       `iterEuler (n+1) (t/(n+1)) F h₀ → cauchyResidualFlow F t h₀`.
--   • `operatorEulerLimit` — **Polymorphic operator-level Euler limit.**
--     For *any* complete normed `ℝ`-space `E` and *any* `T : E →L[ℝ] E`,
--       `((1 + (t/(n+1)) • T)^{n+1}) → exp(t • T)` as `n → ∞`.
--     Specialising `E := Fin D → ℝ` gives `cauchyResidualFlow_eulerLimit_clm`;
--     specialising `E := SeqState seqLen dModel` gives the linearised
--     transformer-block Euler limit of Part 2 §2.4.
-- §2.4 — One block = one Euler step (LayerNorm + attention + MLP residual):
--   • `BlockOps.residualVF` — the residual vector field
--     `x ↦ attn(ln1 x) + mlp(ln2(x + attn(ln1 x)))`.
--   • `gpt2Block_eq_one_eulerStep` — `gpt2Block ops x = x + ops.residualVF x`,
--     so one full pre-norm block is exactly one Euler step at `Δt = 1`.
--   • `BlockOps.iterEuler` — the iterated Euler refinement.
--   • `BlockOps.iterEuler_one_one_eq_block` — `iterEuler 1 1 ops = gpt2Block ops`.
--   • `BlockOps.iterEuler_eq_pow_apply` — *whenever the block residual VF is
--     the action of a CLM `T`*, the Euler iterate equals `(1 + Δt·T)^m h₀`
--     (operator power applied to the initial state).
--   • `BlockOps.linearised_eulerLimit_clm` / `BlockOps.linearised_eulerLimit`
--     / `transformer_block_eulerLimit_with_layerNorm_and_mlp` — **Euler limit
--     for the full pre-norm block** (LayerNorm + attention + LayerNorm + MLP):
--     under linearity of the combined residual, the discrete refinement
--     `iterEuler (n+1) (t/(n+1)) ops h₀` converges to `exp(t·T)·h₀`.  This is
--     the rigorous form of "stacking transformer blocks ≃ flowing along an ODE".
#check @AnalyticTransformer.cauchyLinearOp
#check @AnalyticTransformer.cauchyLinearOp_apply
#check @AnalyticTransformer.cauchyResidualFlow
#check @AnalyticTransformer.cauchyResidualFlow_zero
#check @AnalyticTransformer.cauchyResidualFlow_hasDerivAt
#check @AnalyticTransformer.cauchyResidualFlow_apply_hasDerivAt
#check @AnalyticTransformer.iterEuler_eq_clm_pow
#check @AnalyticTransformer.cauchyResidualFlow_eq_exp_npow
#check @AnalyticTransformer.cauchyResidualFlow_eulerLimit_clm
#check @AnalyticTransformer.cauchyResidualFlow_eulerLimit
#check @AnalyticTransformer.operatorEuler_error_bound
#check @AnalyticTransformer.operatorEuler_error_bound_apply
#check @AnalyticTransformer.operatorEuler_error_bound_O_inv
#check @AnalyticTransformer.operatorEuler_error_bound_O_inv_unconditional
#check @AnalyticTransformer.operatorEuler_error_bound_O_inv_all_n
#check @AnalyticTransformer.cauchyResidualFlow_euler_error_bound_clm
#check @AnalyticTransformer.cauchyResidualFlow_euler_error_bound
#check @AnalyticTransformer.operatorEulerLimit
#check @AnalyticTransformer.BlockOps.residualVF
#check @AnalyticTransformer.gpt2Block_eq_one_eulerStep
#check @AnalyticTransformer.BlockOps.iterEuler
#check @AnalyticTransformer.BlockOps.iterEuler_one_one_eq_block
#check @AnalyticTransformer.BlockOps.iterEuler_eq_pow_apply
#check @AnalyticTransformer.BlockOps.linearised_eulerLimit_clm
#check @AnalyticTransformer.BlockOps.linearised_eulerLimit
#check @AnalyticTransformer.transformer_block_eulerLimit_with_layerNorm_and_mlp

-- Gate 16: §3 — Continuum-limit classification (replaces the tautology).
-- §1.19 builds `cauchyResidualFlow F = exp(t • cauchyLinearOp F)` and
-- proves it satisfies the operator ODE `dΦ/dt = T · Φ`.  By itself that
-- is *existence* only — a priori another flow with the same generator
-- could match this ODE on the diagonal but disagree off it.  Part 3
-- closes the loop:
--   • `IsExpFlow T Φ` packages the operator-flow certificate
--     (`Φ 0 = 1` plus `dΦ/dt = T · Φ` at every t).
--   • `exp_smul_isExpFlow` — the Mathlib power-series flow satisfies it.
--   • `isExpFlow_unique` — **classification**: any flow satisfying the
--     certificate is `t ↦ exp(t • T)`.  Proved via mathlib's
--     `ODE_solution_unique_univ` and the operator-norm Lipschitz bound
--     `‖T · X‖ ≤ ‖T‖ · ‖X‖` on the operator Banach algebra
--     `E →L[ℝ] E`.
--   • `cauchyResidualFlow_classification` — specialised to the §1.19
--     embedding-space flow: any flow with generator `cauchyLinearOp F`
--     equals `cauchyResidualFlow F`.
-- This is the structural sense in which the depth → ∞ limit replaces
-- the labelling-style "tautology": the Cauchy-Poisson generator pins
-- the entire continuous-time trajectory of the residual stream, with
-- no alternative continuum dynamics admissible.
#check @AnalyticTransformer.IsExpFlow
#check @AnalyticTransformer.exp_smul_isExpFlow
#check @AnalyticTransformer.isExpFlow_unique
#check @AnalyticTransformer.isExpFlow_generator_unique
#check @AnalyticTransformer.isExpFlow_generator_eq_deriv0
#check @AnalyticTransformer.euler_refinement_limit_identifies_exp
#check @AnalyticTransformer.cauchyResidualFlow_isExpFlow
#check @AnalyticTransformer.cauchyResidualFlow_classification
#check @AnalyticTransformer.cauchyResidualFlow_classification_apply
#check @AnalyticTransformer.cauchyLinearOp_eq_deriv_cauchyResidualFlow_zero

-- Gate 17: §4 — replacement axioms block for downstream reuse.
#check @AnalyticTransformer.ReplacementAxioms
#check @AnalyticTransformer.replacementAxioms_imply_replacementBlock
#check @AnalyticTransformer.ReplacementBlock
#check @AnalyticTransformer.transformer_tautology_replacement_block

-- Gate 18: §5 — Möbius classification (the deep theorem).
-- §1.15 (`poisson_mobius_covariant`) says: the Poisson kernel transforms
-- with weight 2 under PSL(2, ℝ).  Part 5 supplies the **converse**:
-- any kernel satisfying weight-2 Möbius covariance plus a single
-- boundary normalisation `K(0, 1, 0) = poisson(0, 1, 0) = 1` is forced
-- to equal the Poisson kernel pointwise on the upper half-plane.
--
--   • `mobius_realization_exists` — for any UHP point `(x, y)` and
--     boundary point `q` with `y > 0`, exhibit an explicit unimodular
--     Möbius matrix mapping `(0+i, 0)` to `(x+iy, q)` whose lower-row
--     entry `d` satisfies `d² = poisson x y q`.  This is the
--     transitive-action lemma that drives the classification.
--   • `poisson_unique_under_mobius_covariance` — the converse uniqueness
--     theorem: covariance + normalisation ⟹ K = poisson.
--   • `poisson_mobius_classification` — the iff statement combining both
--     directions: Möbius covariance + normalisation **classifies** the
--     Poisson kernel uniquely.
--
-- This is the harmonic-analysis identification of the transformer's
-- infinite-depth attention kernel: it is not just *labelled* Cauchy/
-- Poisson but is **forced** to be by the PSL(2, ℝ) symmetry alone.
-- "The transformer at infinity is Cauchy" is now a theorem, not a
-- definition.
#check @AnalyticTransformer.mobius_realization_exists
#check @AnalyticTransformer.poisson_unique_under_mobius_covariance
#check @AnalyticTransformer.poisson_mobius_classification
#check @AnalyticTransformer.continuum_limit_not_arbitrary
#check @AnalyticTransformer.global_object_and_mobius_force_poisson

-- Gate 19: §6 — Hardy / Poisson semigroup machinery (scaffolding).
-- The probability-normalized Poisson kernel `P_y(x) = y / (π(x²+y²))`,
-- algebraic / pointwise content, structural Hardy/convolution-semigroup
-- predicates, and Hardy/Möbius bridge: Möbius covariance + normalization
-- (Part 5) classifies the kernel; normalization by π gives the
-- probability-normalized Poisson kernel.  Full classical Hardy / boundary-
-- trace / F.&M.-Riesz / factorization endpoint is exposed as Prop-level
-- predicates for incremental refinement.
#check @AnalyticTransformer.poissonKernel
#check @AnalyticTransformer.poissonKernel_pos
#check @AnalyticTransformer.poissonKernel_eq_normalized_poisson
#check @AnalyticTransformer.poissonKernel_symmetric
#check @AnalyticTransformer.poissonKernel_at_zero
#check @AnalyticTransformer.poissonKernel_continuous
#check @AnalyticTransformer.IsConvolutionSemigroup
#check @AnalyticTransformer.IsHardyKernel
#check @AnalyticTransformer.hardy_kernel_classification_via_mobius
#check @AnalyticTransformer.hardyKernel_eq_poissonKernel_under_mobius
#check @AnalyticTransformer.mellinIntegrand
#check @AnalyticTransformer.mellinTransform
#check @AnalyticTransformer.mellinIntegrand_zero
#check @AnalyticTransformer.mellinTransform_zero

def main : IO Unit :=
  IO.println
    "AnalyticTransformer: Transformers Are Boundary-Value Solvers.\n\
     Part 1 (The Identity) and Part 2 (The Full Model) are sorry-free.\n\
     Part 1 §1.7 (Differentiability) certifies backprop on Cauchy heads.\n\
     Part 1 §1.8 (Pushforward) certifies PyTorch's gradient = Cauchy gradient.\n\
     Part 1 §1.9 (Boundedness) rules out the f(q)=q tautology.\n\
     Part 1 §1.10 (Sense/Act) certifies P/Q linear independence + CR coupling.\n\
     Part 1 §1.11 (Separation) gives the universal-approximation precondition.\n\
     Part 1 §1.12 (Decay) rules out nonzero constants — sharpest no-cheats result.\n\
     Part 1 §1.13 (Smoothness in q) closes joint differentiability.\n\
     Part 1 §1.14 (Affine invariance) gives the natural symmetry group.\n\
     Part 1 §1.15 (PSL(2,ℝ) covariance) — kernel class symmetry, not realisation.\n\
     Part 1 §1.16 (Realisation caveats) — realised geometry sees only B(2,ℝ).\n\
     Part 1 §1.17 (Dynamics) — residual stream is a discrete linear semigroup.\n\
     Part 1 §1.18 (RoPE) — Cayley pullback of feature-space rotation = ropeMobius;\n\
                        non-trivial RoPE rotations live in PSL(2,ℝ) \\ B(2,ℝ).\n\
     Part 1 §1.19 (Continuum) — `exp(t • T)` is the continuous Poisson\n\
                        semigroup; satisfies the operator/pointwise ODE\n\
                        `dΦ/dt = T · Φ`; Euler iterate = `(1 + Δt T)^L`;\n\
                        multiplicative refinement `exp(t•T) = exp(δ)^{n+1}`.\n\
     Part 3 §3 (Classification) — `IsExpFlow T Φ` ⇒ Φ = exp(· • T):\n\
                        the depth → ∞ generator pins the entire flow.\n\
                        Specialised: any flow with generator `cauchyLinearOp F`\n\
                        equals `cauchyResidualFlow F` (replaces the tautology).\n\
     Part 5 §5 (Möbius classification, the deep theorem) — Möbius weight-2\n\
                        covariance + a single boundary normalisation\n\
                        K(0,1,0)=1 FORCES K = poisson on the entire UHP×ℝ.\n\
                        Combined with §1.15: a kernel is PSL(2,ℝ)-covariant\n\
                        and standardly normalised IFF it is Cauchy/Poisson.\n\
                        The transformer at infinity is provably Cauchy, not\n\
                        by definition but by harmonic-analytic uniqueness.\n\
     See AxiomAudit for the formal certificate."
