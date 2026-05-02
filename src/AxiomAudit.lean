import Part1_TheIdentity
import Part2_TheFullModel
import Part3_ContinuumLimit
import Part4_ReplacementAxioms
import Part5_PoissonClassification
import Part6_HardySemigroup

/-!
# Axiom Audit — Formal Verification Certificate

Every `#print axioms T` below lists the complete axiom dependencies of the
proof of `T`.  A clean `sorry`-free theorem in Lean 4 / Mathlib depends on
at most:

  * `Classical.choice`
  * `Quot.sound`
  * `propext`

These are the three standard axioms of classical higher-order logic that
every Mathlib theorem depends on.  Any `sorry`, `sorryAx`, or user-introduced
`axiom` would appear here.

To regenerate this audit:

  lake env lean src/AxiomAudit.lean

The output should mention only the three axioms above (and sometimes a
strict subset, when `propext` alone suffices).
-/

open AnalyticTransformer

/-! ## Part 1 — The Identity -/

section Part1

/- Core row-level theorems (classification-first, then helper existence) -/
#print axioms topological_subsumes_simplex
#print axioms softmax_poles_unique_on_vertical_slice
#print axioms topological_subsumes_softmax_of_scores
#print axioms one_div_one_div_of_pos
#print axioms poisson_at_query_one_div_one_div
#print axioms topological_subsumes_positive_weights
#print axioms matchPoles_unique_on_vertical_slice

/- The genuine softmax-specific identities (Gibbs structure) -/
#print axioms softmax_pole_log_ratio
#print axioms softmax_iff_gibbs_poles

/- Off-query bandwidth limit -/
#print axioms poisson_offquery_eq

/- § 1.7 — Cauchy transform: genuine complex-analytic content -/
#print axioms cauchyTransform_differentiableAt
#print axioms im_cauchyTransform_ofReal_eq_poisson_weighted_sum
#print axioms contourOutput_eq_im_cauchyTransform_vertical

/- Operator upgrade: scalar weights → full vector head output -/
#print axioms contour_output_eq_transformer_of_weight_match
#print axioms contour_simulates_softmax_of_scores

/- § 1.7 — Differentiability of the boundary trace in pole parameters.
   Backprop is well-defined on a Cauchy-Poisson head. -/
#print axioms poisson_differentiableAt_pole
#print axioms poisson_differentiableAt_x
#print axioms poisson_differentiableAt_y
#print axioms contourOutput_differentiableAt_xk
#print axioms contourOutput_differentiableAt_yk
#print axioms contourOutput_differentiableAt_Vkd
#print axioms contourOutput_differentiableAt_pole_pair
#print axioms loss_compose_contourOutput_xk_differentiableAt
#print axioms loss_compose_contourOutput_yk_differentiableAt
#print axioms loss_compose_contourOutput_Vkd_differentiableAt
#print axioms loss_compose_contourOutput_pole_pair_differentiableAt

/- § 1.8 — Backprop pushforward through softmax.
   PyTorch's gradient = Cauchy gradient through the chain rule. -/
#print axioms softmax_weight_differentiable
#print axioms softmaxAttention_score_differentiable
#print axioms softmaxAttention_eq_contourOutput
#print axioms pushforward_loss_differentiable
#print axioms pushforward_loss_through_contour_differentiable

/- § 1.9 — Negative result: Cauchy-Poisson sums are uniformly bounded,
   so unbounded functions are not representable.  This rules out the
   "everything is Cauchy-Poisson" tautology. -/
#print axioms poisson_le_inv_y
#print axioms poisson_nonneg
#print axioms contourOutput_uniformly_bounded
#print axioms identity_not_cauchyPoisson
#print axioms unbounded_not_cauchyPoisson

/- § 1.10 — Conjugate Poisson `Q` and Sense/Act coupling.
   `P` (sense) and `Q` (act) are linearly independent functions of the
   query, and satisfy a pointwise Cauchy-Riemann identity at the
   boundary.  The act channel carries information the sense channel
   cannot. -/
#print axioms conjPoisson_at_query
#print axioms poisson_pos
#print axioms sense_act_linearly_independent
#print axioms conjPoisson_not_scalar_multiple_of_poisson
#print axioms single_channel_convex_poisson_cannot_represent_conjPoisson
#print axioms conjPoisson_has_dual_channel_representation
#print axioms conjPoisson_differentiable
#print axioms sense_act_cauchy_riemann

/- § 1.11 — Expressiveness: Cauchy-Poisson kernels separate points.
   Precondition for universal approximation via Stone–Weierstrass. -/
#print axioms poisson_separates_points
#print axioms contourOutput_separates_points

/- § 1.12 — Decay at infinity: every Cauchy-Poisson sum vanishes at
   the right tail.  Sharper than §1.9 (boundedness) — rules out nonzero
   constants.  Strongest "no cheats" result so far. -/
#print axioms poisson_atTop_lt
#print axioms contourOutput_atTop_lt
#print axioms nonzero_const_not_cauchyPoisson

/- § 1.13 — Smoothness in the query coordinate.  Closes the loop on
   joint differentiability begun in §1.7 (which covered the pole
   parameters PyTorch trains). -/
#print axioms poisson_differentiable_q
#print axioms contourOutput_differentiable_q

/- § 1.14 — Affine invariance: translation, dilation, reflection of
   the boundary coordinate correspond to canonical transformations
   of the poles.  The CP class is closed under the affine group. -/
#print axioms contourOutput_translation
#print axioms contourOutput_reflection
#print axioms poisson_scale
#print axioms contourOutput_scale

/- § 1.15 — PSL(2, ℝ) Möbius covariance of the abstract kernel class.
   On freely-floating poles in the upper half plane, the Poisson kernel
   is weight-2 covariant, the contour output is a weight-2 modular
   form, and the attention simplex is Möbius-invariant.  See §1.16 for
   what this does (and does *not*) say about realised softmax heads. -/
#print axioms mobPoleY_pos
#print axioms poisson_mobius_covariant
#print axioms contourOutput_mobius_covariant
#print axioms attention_simplex_mobius_invariant

/- § 1.16 — Realisation caveats: the slice `x_k = q` produced by
   `matchPoles` is preserved exactly by the affine subgroup
   `B(2, ℝ) = { c = 0 } ⊂ PSL(2, ℝ)`.  So the realised softmax-derived
   geometry inherits §1.14's symmetry, not the full §1.15.  This is
   the formal answer to the "are you sure you have a global isomorphism?"
   pushback: no, only the kernel class does; the realised geometry has
   exactly the affine symmetry. -/
#print axioms matchPoles_mobius_preserves_verticality
#print axioms realised_mobius_orbit_is_affine

/- § 1.17 — Dynamics: the residual stream is a discrete linear
   semigroup.  Each layer is exactly one Euler step of an explicit
   ℝ-linear vector field on the embedding space; the discrete
   semigroup property Φ_{m+n} = Φ_n ∘ Φ_m holds for the full L-step
   iteration; and linearity is preserved by every iterate.  These
   three facts (Euler-step shape + semigroup law + linearity
   preservation) constitute the genuine discrete-time analogue of the
   continuous Poisson semigroup `e^{-t√(-Δ)}` — which is *itself*
   left as an open continuum-limit target. -/
#print axioms cauchy_residual_is_euler_step
#print axioms cauchy_residual_euler_step_dt
#print axioms cauchyResidualVF_linear_in_V
#print axioms iterEuler_zero
#print axioms iterEuler_one
#print axioms iterEuler_succ
#print axioms iterEuler_add
#print axioms eulerStepFn_linear_of_linear
#print axioms iterEuler_linear_of_linear

/- § 1.18 — RoPE strictly exceeds the affine subgroup, with the
   feature-space-rotation → disc-rotation → UHP-Möbius chain proved.

   `ropeMobius_eq_cayley_pullback` is the algebraic content: the
   composition `cayleyInv ∘ discRotate θ ∘ cayley` is *literally*
   equal to `ropeMobius θ` (the half-angle PSL(2,ℝ) representative).
   Combined with `ropeMobius_c` (whose value is `−sin(θ/2)`), this
   forces `c ≠ 0` for every non-trivial rotation, and §1.16 then
   rules out preservation of the realised verticality slice. -/
#print axioms ropeMobius_det_one
#print axioms ropeMobius_c
#print axioms ropeMobius_eq_cayley_pullback
#print axioms rope_breaks_verticality
#print axioms rope_strictly_exceeds_affine

/- § 1.19 — Continuum lift via Mathlib's `NormedSpace.exp`.  The §1.17
   *discrete* linear semigroup is upgraded to a genuine *continuous*
   one-parameter semigroup `Φ_t = exp(t • T)` valued in the Banach
   algebra of bounded linear endomorphisms of the embedding space.

   `cauchyResidualFlow_hasDerivAt` and `cauchyResidualFlow_apply_hasDerivAt`
   are the headline content: the operator-valued and pointwise ODEs
   `dΦ/dt = T · Φ` and `(d/dt)(Φ_t v) = T (Φ_t v)` are *theorems*, not
   metaphors.  `cauchyResidualFlow_zero` provides the initial condition
   `Φ_0 = 1`.  `iterEuler_eq_clm_pow` shows the §1.17 discrete iterate
   is *exactly* `(1 + Δt · T)^L` applied to `h₀` — the explicit
   first-order Taylor truncation of the operator exponential.

   `cauchyResidualFlow_eq_exp_npow` is the **multiplicative refinement**
   identity `exp(t•T) = (exp ((t/(n+1))•T)))^{n+1}` for every `n : ℕ`,
   matching the `(n+1)`-fold Euler refinement at timestep `t/(n+1)`.

   `cauchyResidualFlow_eulerLimit_clm` and `cauchyResidualFlow_eulerLimit`
   close the discrete-to-continuous gap: the Euler refinement
   `(1 + (t/(n+1))•T)^{n+1}` converges (in operator norm, and pointwise) to
   `exp(t•T)` as `n → ∞`.  `operatorEulerLimit` is the polymorphic
   Banach-algebra version (any complete normed `ℝ`-space `E`, any
   `T : E →L[ℝ] E`); the two `cauchy*` versions specialise it to
   `E = Fin D → ℝ`. -/
#print axioms cauchyLinearOp_apply
#print axioms cauchyResidualFlow_zero
#print axioms cauchyResidualFlow_hasDerivAt
#print axioms cauchyResidualFlow_apply_hasDerivAt
#print axioms iterEuler_eq_clm_pow
#print axioms cauchyResidualFlow_eq_exp_npow
#print axioms cauchyResidualFlow_eulerLimit_clm
#print axioms cauchyResidualFlow_eulerLimit
#print axioms operatorEuler_error_bound
#print axioms operatorEuler_error_bound_apply
#print axioms operatorEuler_error_bound_O_inv
#print axioms operatorEuler_error_bound_O_inv_unconditional
#print axioms operatorEuler_error_bound_O_inv_all_n
#print axioms cauchyResidualFlow_euler_error_bound_clm
#print axioms cauchyResidualFlow_euler_error_bound
#print axioms operatorEulerLimit

end Part1

/-! ## Part 2 — The Full Model -/

section Part2

/- Constructive score-derived identity -/
#print axioms softmax_is_poisson_at_score_poles

/- Head-level equality -/
#print axioms head_output_equiv
#print axioms multiHeadAttn_equiv

/- Pipeline machinery -/
#print axioms gpt2_end_to_end_equiv
#print axioms gpt2Block_equiv_of_components

/- The main theorems -/
#print axioms transformer_is_cauchy_poisson
#print axioms transformers_are_boundary_value_solvers
#print axioms gpt2_has_pole_realization
#print axioms gpt2_has_pole_realization_functional
#print axioms gpt2_has_single_global_analytic_object
#print axioms M_global
#print axioms AnalyticMHParams.FixedPoleGeometry
#print axioms M_global_fixed_poles
#print axioms M_global_fixed_poles_implies_M_global
#print axioms FixedPoleAnalyticMHParams
#print axioms FixedPoleAnalyticMHParams.toAnalyticMH
#print axioms FixedPoleAnalyticMHParams.toAnalyticMH_fixedGeometry
#print axioms FixedPoleAnalyticGPT2Params
#print axioms FixedPoleAnalyticGPT2Params.toAnalyticGPT2
#print axioms fixedPoleAnalyticForward
#print axioms fixed_pole_witness_in_M_global_fixed_poles
#print axioms gpt2_in_M_global_fixed_poles_of_state_independent_scores
#print axioms ScoreStateIndependent
#print axioms NondegenerateScores
#print axioms ScoreDifferencesStateIndependent
#print axioms NondegenerateScoreDifferences
#print axioms ScoreDerivedFixedPoleGeometry
#print axioms RealizesVanillaSoftmaxRows
#print axioms nondegenerateScores_iff_not_scoreStateIndependent
#print axioms nondegenerateScoreDifferences_iff_not_scoreDifferencesStateIndependent
#print axioms vanilla_row_realization_forces_scoreDerived_poles
#print axioms fixed_pole_vanilla_realization_implies_scoreDerivedFixedPoleGeometry
#print axioms gpt2_has_intrinsic_dynamic_row_realization
#print axioms scoreDerived_fixed_poles_imply_scoreDifferencesStateIndependent
#print axioms nondegenerateScoreDifferences_not_scoreDerivedFixedPoleGeometry
#print axioms nondegenerateScoreDifferences_force_scoreDerived_poles_move
#print axioms nondegenerate_scores_not_M_global_fixed_poles
#print axioms gpt2_in_M_global
#print axioms scorePoles_unique_on_vertical_slice
#print axioms transformer_cauchy_poisson_replacement_block
#print axioms pole_witness_is_score_derived

/- § 2.4 — Continuum lift of one pre-norm block (LayerNorm + attention + MLP).
   `gpt2Block_eq_one_eulerStep` says one pre-norm block IS one Euler step at
   `Δt = 1` of the residual vector field
   `x ↦ attn(ln1 x) + mlp(ln2(x + attn(ln1 x)))`.
   `BlockOps.iterEuler_eq_pow_apply` turns the iterated refinement into
   the `m`-th power of `1 + Δt·T` (under linearity of the combined residual).
   `BlockOps.linearised_eulerLimit_clm` and the pointwise / "user-level"
   versions deliver the **Euler limit for the full pre-norm block (with
   LayerNorm and MLP)**: under the linearity hypothesis on the combined
   residual VF, the iterated refinement converges to the continuous
   semigroup `exp(t·T)`. -/
#print axioms gpt2Block_eq_one_eulerStep
#print axioms BlockOps.iterEuler_one_one_eq_block
#print axioms BlockOps.iterEuler_eq_pow_apply
#print axioms BlockOps.linearised_eulerLimit_clm
#print axioms BlockOps.linearised_eulerLimit
#print axioms transformer_block_eulerLimit_with_layerNorm_and_mlp

end Part2

/-! ## Part 3 — The Continuum Limit (replaces the tautology)

The §1.19 flow `cauchyResidualFlow F = exp(t • cauchyLinearOp F)`
*exists* and satisfies the operator ODE `dΦ/dt = T · Φ`.  Part 3
adds the matching *uniqueness* statement: any operator flow
satisfying `Φ 0 = 1` and the same ODE is forced to equal
`t ↦ exp(t • T)`.  Specialising to the embedding space, any such
flow with generator `cauchyLinearOp F` equals `cauchyResidualFlow F`.

This replaces the tautology
"Cauchy/Poisson attention is just attention with a particular kernel"
with a real classification at the level of the continuous-time flow:
the Cauchy/Poisson generator pins the entire residual-stream
trajectory in the depth → ∞ limit. -/

section Part3

#print axioms IsExpFlow
#print axioms exp_smul_isExpFlow
#print axioms isExpFlow_unique
#print axioms isExpFlow_generator_unique
#print axioms isExpFlow_generator_eq_deriv0
#print axioms euler_refinement_limit_identifies_exp
#print axioms cauchyResidualFlow_isExpFlow
#print axioms cauchyResidualFlow_classification
#print axioms cauchyResidualFlow_classification_apply
#print axioms cauchyLinearOp_eq_deriv_cauchyResidualFlow_zero

end Part3

/-! ## Part 5 — The Möbius classification (the deep theorem)

The §1.15 covariance `poisson_mobius_covariant` says: the Poisson kernel
*satisfies* PSL(2, ℝ) weight-2 covariance.  Part 5 supplies the converse:
any kernel `K(x, y, q)` satisfying the same covariance plus the
single-point boundary normalisation `K(0, 1, 0) = 1` **must** be the
Poisson kernel.

Combined with §1.15, this gives a genuine harmonic-analysis
classification: an attention-style boundary kernel is Möbius covariant
of weight 2 with the standard normalisation **iff** it is the Cauchy/
Poisson kernel.  This is the rigorous sense in which the transformer at
infinity is *forced* to be Cauchy: no other kernel can satisfy the same
group-theoretic constraints. -/

section Part5

#print axioms mobius_realization_exists
#print axioms poisson_unique_under_mobius_covariance
#print axioms poisson_mobius_classification
#print axioms continuum_limit_not_arbitrary
#print axioms global_object_and_mobius_force_poisson

end Part5

/-! ## Part 6 — Hardy / Poisson semigroup machinery (scaffolding)

The probability-normalized Poisson kernel `P_y(x) = y / (π·(x²+y²))`,
its pointwise / algebraic theory, structural Hardy/convolution-semigroup
predicates, and the bridge from Part 5's Möbius classification to the
Hardy form via algebra alone.  The integral-side Hardy classical
endpoint is exposed as `Prop`-level predicates so downstream development
can refine or discharge them incrementally without disturbing the
existing classification core. -/

section Part6

#print axioms poissonKernel
#print axioms poissonKernel_pos
#print axioms poissonKernel_eq_normalized_poisson
#print axioms poissonKernel_symmetric
#print axioms poissonKernel_at_zero
#print axioms poissonKernel_continuous
#print axioms IsConvolutionSemigroup
#print axioms IsHardyKernel
#print axioms hardy_kernel_classification_via_mobius
#print axioms hardyKernel_eq_poissonKernel_under_mobius
#print axioms mellinIntegrand
#print axioms mellinTransform
#print axioms mellinIntegrand_zero
#print axioms mellinTransform_zero

end Part6
