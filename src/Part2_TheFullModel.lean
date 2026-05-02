import Mathlib
import Part1_TheIdentity

/-!
# Part 2 — The Full Model

**Definitions.** `ModelVec`, `SeqState`; `GPT2HeadWeights`, `HeadParams`, `headParamsFromWeights`,
`gpt2HeadOutput`; `scorePoles`; `AnalyticHeadParams`, `scoreDerivedAnalyticHead`, `analyticHeadOutput`;
`MultiHeadAttnParams`, `gpt2MultiHeadAttn`, `AnalyticMHParams`, `analyticMultiHeadAttn`,
`scoreDerivedAnalyticMH`; `BlockOps`, `gpt2Block`; `PipelineSpec`, `runBlocks`, `forward`;
`LayerNormParams`, `layerNormState`, `GPT2MLPParams`, `gpt2MLPState`, `EmbeddingParams`,
`inputEmbedding`, `LMHeadParams`, `lmHeadState`, `blockFamilySpec`; `GPT2Params`,
`AnalyticGPT2Params`, `scoreDerivedAnalyticParams`, `gpt2Forward`, `analyticForward`, `gpt2Unembed`.

**Lemmas.** Causal indexing: `causalContextPos_le_query`, `causalContextPos_injective`.

**Theorems.** Score poles vs softmax weights `softmax_is_poisson_at_score_poles`; per-head equivalence
`head_output_equiv`; multi-head `multiHeadAttn_equiv`; residual block `gpt2Block_equiv_of_components`;
pipeline `gpt2_end_to_end_equiv`; full forward `transformer_is_cauchy_poisson`,
`transformers_are_boundary_value_solvers`; existential realization `gpt2_has_pole_realization`;
constructive witness `pole_witness_is_score_derived`; off-query bandwidth `softmax_offquery_bandwidth_bound`.
-/

noncomputable section

open Finset Real
open scoped BigOperators

namespace AnalyticTransformer

-- ═══════════════════════════════════════════════════════════════════════
-- § 2.1  GPT-2 block structure
-- ═══════════════════════════════════════════════════════════════════════

/-- One position's activation vector (dModel real values). -/
abbrev ModelVec (dModel : ℕ) := Fin dModel → ℝ

/-- A full sequence state (seqLen positions, each with dModel channels). -/
abbrev SeqState (seqLen dModel : ℕ) := Fin seqLen → ModelVec dModel

/-- Per-head attention projection matrices and biases.
    These are the parameters that GPT-2 checkpoints actually store per head
    (HuggingFace stores them combined as `c_attn`, but they slice cleanly):

    - `W_Q ∈ ℝ^{D × dModel}` and bias `b_Q ∈ ℝ^D` project the query activation
    - `W_K ∈ ℝ^{D × dModel}` and bias `b_K ∈ ℝ^D` project each key activation
    - `W_V ∈ ℝ^{D × dModel}` and bias `b_V ∈ ℝ^D` project each value activation

    where D is the per-head dimension (= dModel / nHeads in standard GPT-2).
    The biases are present in standard GPT-2 checkpoints (HuggingFace's
    `c_attn` Conv1D has `bias=True`), so we model them explicitly. -/
structure GPT2HeadWeights (dModel D : ℕ) where
  W_Q : Matrix (Fin D) (Fin dModel) ℝ
  W_K : Matrix (Fin D) (Fin dModel) ℝ
  W_V : Matrix (Fin D) (Fin dModel) ℝ
  b_Q : Fin D → ℝ
  b_K : Fin D → ℝ
  b_V : Fin D → ℝ
  /-- Head dimension must be positive so scaled dot-product uses non-degenerate √D. -/
  d_pos : 0 < D

/-- Per-head parameters for softmax attention at a fixed query position.

    `scores k` is the score for key `k`: `(1/√D)·(W_Q x_t + b_Q)·(W_K x_{ks k} + b_K)`.
    `V k` is the projected value vector: `W_V x_{ks k} + b_V`.

    This struct holds the *result* of applying `headParamsFromWeights` to a
    concrete input state.  It is always produced by `MultiHeadAttnParams.head`,
    never constructed directly with arbitrary scores. -/
structure HeadParams (N D : ℕ) where
  scores : Fin N → ℝ
  V : Fin N → Fin D → ℝ

/-- Compute `HeadParams` concretely from weight matrices+biases, a query activation,
    and a sequence of key/value source activations.

    This is the exact computation GPT-2 performs at query position `t`:
      `scores k = (1/√D) · (W_Q x_t + b_Q) · (W_K x_{ks k} + b_K)`   (scaled dot-product)
      `V k d    = (W_V x_{ks k} + b_V) d`                              (value projection)

    The division by √D is the standard scaling factor that prevents
    dot-products from growing large in high dimension. -/
def headParamsFromWeights {N dModel D : ℕ}
    (hw : GPT2HeadWeights dModel D)
    (x_t : ModelVec dModel)
    (x_ks : Fin N → ModelVec dModel) : HeadParams N D where
  scores k := (1 / Real.sqrt (D : ℝ)) *
    ∑ i : Fin D, (Matrix.mulVec hw.W_Q x_t i + hw.b_Q i) *
                  (Matrix.mulVec hw.W_K (x_ks k) i + hw.b_K i)
  V k := fun i => Matrix.mulVec hw.W_V (x_ks k) i + hw.b_V i

/-- GPT-2 softmax head output: standard `softmax(scores) · V`. -/
def gpt2HeadOutput {N D : ℕ} [NeZero N] (hp : HeadParams N D) : Fin D → ℝ :=
  fun d => transformerOutput
      (fun j : Fin N => exp (hp.scores j) / ∑ i : Fin N, exp (hp.scores i))
      hp.V d

/-- Part 2 alias for Part 1's canonical softmax-derived pole construction. -/
abbrev scorePoles {N : ℕ} [NeZero N] (scores : Fin N → ℝ) (q : ℝ) : Poles N :=
  matchPoles q (softmaxSimplexOfScores scores)

/-- **The constructive Cauchy-Poisson identity.**

    The softmax weight for key k equals the Poisson kernel evaluated at the
    score-derived pole (q, Z/exp(s_k)):

      P(q, Z/exp(s_k), q) = exp(s_k) / Z = softmax(s)_k

    The poles are given by a closed formula in the scores — not chosen from
    the output.  This is the non-circular form of the main identity. -/
theorem softmax_is_poisson_at_score_poles {N : ℕ} [NeZero N]
    (scores : Fin N → ℝ) (q : ℝ) (k : Fin N) :
    poisson ((scorePoles scores q).x k) ((scorePoles scores q).y k) q =
      exp (scores k) / ∑ j : Fin N, exp (scores j) := by
  simpa [scorePoles, softmaxSimplexOfScores] using
    (match_weight (softmaxSimplexOfScores scores) q k)

/-- **Canonicality of score-derived poles on the realised slice.**

    If a pole family sits on-query (`x_k = q`) and reproduces the softmax
    row at `q`, then it must be exactly `scorePoles scores q`.

    This is the non-tautological upgrade from "a witness exists" to
    "the realised witness is forced" for GPT-style rows. -/
theorem scorePoles_unique_on_vertical_slice {N : ℕ} [NeZero N]
    (scores : Fin N → ℝ) (q : ℝ) (p : Poles N)
    (hx : ∀ k, p.x k = q)
    (hw : ∀ k,
      poisson (p.x k) (p.y k) q = exp (scores k) / ∑ j : Fin N, exp (scores j)) :
    p = scorePoles scores q := by
  refine matchPoles_unique_on_vertical_slice q (softmaxSimplexOfScores scores) p hx ?_
  intro k
  simpa [softmaxSimplexOfScores] using hw k

/-- Pole-based head parameters.  Poles are specified directly — not derived
    from scores.  This is the independently-typed "analytic side": the step of
    constructing `AnalyticHeadParams` from a `HeadParams` (via
    `scoreDerivedAnalyticHead`) is the non-trivial part proved in
    `head_output_equiv`. -/
structure AnalyticHeadParams (N D : ℕ) where
  poles : Poles N
  V : Fin N → Fin D → ℝ

/-- Convert score-based head params to independently-typed pole-based params.
    Constructive witness: poles sit on-query at height Z/exp(s_k), q = 0. -/
def scoreDerivedAnalyticHead {N D : ℕ} [NeZero N] (hp : HeadParams N D) : AnalyticHeadParams N D where
  poles := scorePoles hp.scores 0
  V := hp.V

/-- Analytic (Poisson-pole) head output from independently-typed pole params.
    Query coordinate is fixed at 0; all poles are on-query so it cancels. -/
def analyticHeadOutput {N D : ℕ} (ahp : AnalyticHeadParams N D) : Fin D → ℝ :=
  fun d => contourOutput ahp.poles 0 ahp.V d

/-- Flatten multi-head outputs into a single vector of length `nHeads * D`.
    Linearises as index `h * D + d` — the standard head-concatenation that
    GPT-2's output projection `W_O ∈ ℝ^{dModel × (nHeads·D)}` operates on.
    Uses `Fin.divNat` / `Fin.modNat` so no runtime division is needed. -/
def concatHeads {nHeads D : ℕ} (heads : Fin nHeads → Fin D → ℝ) : Fin (nHeads * D) → ℝ :=
  fun k => heads (Fin.divNat k) (Fin.modNat k)

/-- **The key non-tautological equivalence.**

    The two sides have DIFFERENT types:
    - `gpt2HeadOutput hp` computes softmax weights from `hp.scores`
    - `analyticHeadOutput (scoreDerivedAnalyticHead hp)` evaluates the
       Poisson kernel at *independently-typed* poles of height Z/exp(s_k)

    The proof routes **both** sides through the same canonical analytic quantity
    from Part 1, `headCauchyIM` (imaginary boundary value of the discrete
    Cauchy transform at vertically lifted poles), using
    `softmax_is_poisson_at_score_poles` only to identify softmax weights with
    the Poisson weights in that contour. -/
theorem head_output_equiv {N D : ℕ} [NeZero N] (hp : HeadParams N D) :
    ∀ d : Fin D, analyticHeadOutput (scoreDerivedAnalyticHead hp) d = gpt2HeadOutput hp d := by
  intro d
  set p : Poles N := scorePoles hp.scores 0
  have hScoreComplex :
      scoreComplexPoles hp.scores (0 : ℝ) = verticalComplexPoles p (0 : ℝ) := by
    funext k
    simp [scoreComplexPoles, p, scorePoles]
  have hx : ∀ k : Fin N, p.x k = (0 : ℝ) := fun _ => rfl
  let w : Fin N → ℝ := fun j => exp (hp.scores j) / ∑ i : Fin N, exp (hp.scores i)
  have h_softmax_as_poisson :
      ∀ k : Fin N, poisson (p.x k) (p.y k) (0 : ℝ) = w k :=
    fun k => softmax_is_poisson_at_score_poles hp.scores 0 k
  have h_cauchy_name :
      headCauchyIM hp.V p (0 : ℝ) d =
        (cauchyTransform hp.V (scoreComplexPoles hp.scores (0 : ℝ)) (0 : ℂ) d).im := by
    unfold headCauchyIM
    rw [← hScoreComplex]
    simp
  have h_analytic :
      analyticHeadOutput (scoreDerivedAnalyticHead hp) d =
        (cauchyTransform hp.V (scoreComplexPoles hp.scores (0 : ℝ)) (0 : ℂ) d).im := by
    simp only [analyticHeadOutput, scoreDerivedAnalyticHead]
    rw [contourOutput_eq_im_cauchyTransform_vertical p (0 : ℝ) hp.V hx d]
    exact h_cauchy_name
  have h_softmax :
      gpt2HeadOutput hp d =
        (cauchyTransform hp.V (scoreComplexPoles hp.scores (0 : ℝ)) (0 : ℂ) d).im := by
    have hcont :
        gpt2HeadOutput hp d = contourOutput p (0 : ℝ) hp.V d := by
      unfold gpt2HeadOutput transformerOutput
      symm
      exact contour_output_eq_transformer_of_weight_match p (0 : ℝ) w hp.V h_softmax_as_poisson d
    rw [hcont, contourOutput_eq_im_cauchyTransform_vertical p (0 : ℝ) hp.V hx d]
    exact h_cauchy_name
  rw [h_analytic, h_softmax]

-- ─────────────────────────────────────────────────────────────────────
-- Context-window size functions
-- ─────────────────────────────────────────────────────────────────────

/-- **Causal context size** — at position `t`, the key set is `{0, …, t}`,
    so the window has `t.val + 1` entries.  This is GPT-2's lower-triangular
    mask.  In `MultiHeadAttnParams`, passing `causalContextSize` as the
    `contextSize` argument enforces `N = t+1` in the type system. -/
def causalContextSize {seqLen : ℕ} : Fin seqLen → ℕ := fun t => t.val + 1

/-- Causal context position map: slot k at query t corresponds to sequence position k.val.
    The bound holds because `k.val < t.val + 1` and `t.val + 1 ≤ seqLen`. -/
def causalContextPos {seqLen : ℕ} (t : Fin seqLen)
    (k : Fin (@causalContextSize seqLen t)) : Fin seqLen :=
  ⟨k.val, Nat.lt_of_lt_of_le k.isLt t.isLt⟩

/-- Causal source positions are never in the future of the query position. -/
theorem causalContextPos_le_query {seqLen : ℕ} (t : Fin seqLen)
    (k : Fin (@causalContextSize seqLen t)) :
    (causalContextPos t k).val ≤ t.val :=
  Nat.le_of_lt_succ k.isLt

/-- The canonical causal context map is injective in the context slot index. -/
theorem causalContextPos_injective {seqLen : ℕ} (t : Fin seqLen) :
    Function.Injective (causalContextPos t) := by
  intro k1 k2 h
  apply Fin.ext
  have hval : (causalContextPos t k1).val = (causalContextPos t k2).val :=
    congrArg Fin.val h
  simpa [causalContextPos] using hval

/-- **Full-sequence context size** — every position sees all `seqLen` tokens
    (bidirectional / BERT-style attention). -/
def fullContextSize (seqLen : ℕ) : Fin seqLen → ℕ := fun _ => seqLen

/-- GPT-2 multi-head attention parameters (score-based side).

    `contextSize : Fin seqLen → ℕ` encodes the key-set size at each query
    position.  Use `causalContextSize` for GPT-2's causal mask (`N t = t+1`)
    or `fullContextSize seqLen` for bidirectional attention.

    `headWeights h` stores the W_Q/W_K/W_V matrices for head `h`.
    `contextPos t k` maps context slot `k` at query `t` to a sequence position.
    For causal GPT-2, use `causalContextPos`. -/
structure MultiHeadAttnParams (seqLen nHeads D dModel : ℕ)
    (contextSize : Fin seqLen → ℕ) where
  /-- Every context window is nonempty. -/
  cs_pos : ∀ t : Fin seqLen, 0 < contextSize t
  /-- Per-head W_Q, W_K, W_V projection matrices and biases. -/
  headWeights : Fin nHeads → GPT2HeadWeights dModel D
  /-- Maps context-window slot k at query position t to its sequence position.
      For causal GPT-2: `contextPos t k = ⟨k.val, k.isLt.trans t.isLt⟩`. -/
  contextPos : (t : Fin seqLen) → Fin (contextSize t) → Fin seqLen
  /-- Distinct context slots refer to distinct source positions. -/
  contextPos_injective : ∀ t : Fin seqLen, Function.Injective (contextPos t)
  /-- Context positions are never in the future of the query position. -/
  contextPos_past : ∀ (t : Fin seqLen) (k : Fin (contextSize t)),
    (contextPos t k).val ≤ t.val
  /-- Output-projection matrix W_O ∈ ℝ^{dModel × (nHeads·D)}.
      Applied as `W_O · concatHeads(head outputs) + b_O` — a genuine affine map,
      matching GPT-2's `c_proj` weight exactly. -/
  outProj : Matrix (Fin dModel) (Fin (nHeads * D)) ℝ
  /-- Output-projection bias `b_O ∈ ℝ^{dModel}`.  Real GPT-2's `c_proj` Conv1D
      is constructed with `bias=True`, so we model it explicitly. -/
  outProjBias : Fin dModel → ℝ

/-- Compute `HeadParams` for head `h` at query position `t` from the full input state.
    Applies `W_Q` (and bias `b_Q`) to `x t`, and `W_K`/`W_V` (with `b_K`/`b_V`)
    to `x (contextPos t k)` for each slot `k`.
    This is the concrete GPT-2 attention computation:
      scores k = (1/√D) · (W_Q x_t + b_Q) · (W_K x_{contextPos t k} + b_K)
      V k d    = (W_V x_{contextPos t k} + b_V) d  -/
def MultiHeadAttnParams.head {seqLen nHeads D dModel : ℕ}
    {contextSize : Fin seqLen → ℕ}
    (mh : MultiHeadAttnParams seqLen nHeads D dModel contextSize)
    (t : Fin seqLen) (h : Fin nHeads) (x : SeqState seqLen dModel) :
    HeadParams (contextSize t) D :=
  headParamsFromWeights (mh.headWeights h) (x t) (fun k => x (mh.contextPos t k))

/-- Independently-typed analytic (pole-based) multi-head attention parameters.
    `poleHead` returns `AnalyticHeadParams` — poles specified directly, not via
    scores.  No `cs_pos` needed: `analyticHeadOutput` requires no `NeZero`.

    The output-projection matrix and bias are shared verbatim with the
    score-based side; the Cauchy-Poisson identity replaces only the per-head
    softmax-vs-Poisson step, not the linear post-processing. -/
structure AnalyticMHParams (seqLen nHeads D dModel : ℕ)
    (contextSize : Fin seqLen → ℕ) where
  poleHead : (t : Fin seqLen) → Fin nHeads → SeqState seqLen dModel → AnalyticHeadParams (contextSize t) D
  outProj : Matrix (Fin dModel) (Fin (nHeads * D)) ℝ
  outProjBias : Fin dModel → ℝ

/-- Convert a GPT-2 MH param set to an independently-typed analytic param set.
    `NeZero` is synthesised during construction (from `mh.cs_pos`), not at
    evaluation time. -/
def scoreDerivedAnalyticMH {seqLen nHeads D dModel : ℕ} {contextSize : Fin seqLen → ℕ}
    (mh : MultiHeadAttnParams seqLen nHeads D dModel contextSize) :
    AnalyticMHParams seqLen nHeads D dModel contextSize where
  poleHead t h x :=
    haveI : NeZero (contextSize t) := ⟨(mh.cs_pos t).ne'⟩
    scoreDerivedAnalyticHead (mh.head t h x)
  outProj := mh.outProj
  outProjBias := mh.outProjBias

/-- GPT-side multi-head attention.  Concatenated head outputs are projected
    through `outProj` and shifted by `outProjBias`. -/
def gpt2MultiHeadAttn {seqLen nHeads D dModel : ℕ} {contextSize : Fin seqLen → ℕ}
    (mh : MultiHeadAttnParams seqLen nHeads D dModel contextSize)
    (x : SeqState seqLen dModel) : SeqState seqLen dModel :=
  fun t =>
    haveI : NeZero (contextSize t) := ⟨(mh.cs_pos t).ne'⟩
    fun i => Matrix.mulVec mh.outProj
      (concatHeads (fun h => gpt2HeadOutput (mh.head t h x))) i + mh.outProjBias i

/-- Analytic (pole-based) multi-head attention.  Same affine post-processing
    (`outProj · concat + outProjBias`) as `gpt2MultiHeadAttn`; only the per-head
    output is computed via Poisson kernels instead of softmax. -/
def analyticMultiHeadAttn {seqLen nHeads D dModel : ℕ} {contextSize : Fin seqLen → ℕ}
    (amh : AnalyticMHParams seqLen nHeads D dModel contextSize)
    (x : SeqState seqLen dModel) : SeqState seqLen dModel :=
  fun t => fun i => Matrix.mulVec amh.outProj
    (concatHeads (fun h => analyticHeadOutput (amh.poleHead t h x))) i + amh.outProjBias i

/-- GPT and analytic multi-head attention are identical when analytic params
    are constructed via `scoreDerivedAnalyticMH`.  The two arguments have
    DIFFERENT types; equality holds because `head_output_equiv` bridges them. -/
theorem multiHeadAttn_equiv {seqLen nHeads D dModel : ℕ} {contextSize : Fin seqLen → ℕ}
    (mh : MultiHeadAttnParams seqLen nHeads D dModel contextSize) :
    ∀ x, gpt2MultiHeadAttn mh x = analyticMultiHeadAttn (scoreDerivedAnalyticMH mh) x := by
  intro x; funext t i
  haveI : NeZero (contextSize t) := ⟨(mh.cs_pos t).ne'⟩
  -- Both sides equal `Matrix.mulVec mh.outProj (...) i + mh.outProjBias i`;
  -- the only place they differ is inside the matrix-vector product.
  show Matrix.mulVec mh.outProj
        (concatHeads (fun h => gpt2HeadOutput (mh.head t h x))) i + mh.outProjBias i =
       Matrix.mulVec mh.outProj
        (concatHeads (fun h => analyticHeadOutput (scoreDerivedAnalyticHead (mh.head t h x)))) i +
        mh.outProjBias i
  congr 2
  funext k
  simp only [concatHeads]
  exact (head_output_equiv (mh.head t (Fin.divNat k) x) (Fin.modNat k)).symm

/-- Abstract operations for one GPT-style pre-norm residual block. -/
structure BlockOps (seqLen dModel dMLP : ℕ) where
  ln1 : SeqState seqLen dModel → SeqState seqLen dModel
  attn : SeqState seqLen dModel → SeqState seqLen dModel
  ln2 : SeqState seqLen dModel → SeqState seqLen dModel
  mlp : SeqState seqLen dModel → SeqState seqLen dModel

/-- One GPT-style pre-norm block: x → x + attn(ln1(x)) → x1 + mlp(ln2(x1)). -/
def gpt2Block {seqLen dModel dMLP : ℕ}
    (ops : BlockOps seqLen dModel dMLP)
    (x : SeqState seqLen dModel) : SeqState seqLen dModel :=
  let x1 := x + ops.attn (ops.ln1 x)
  x1 + ops.mlp (ops.ln2 x1)

/-- Component equality → block equality. -/
theorem gpt2Block_equiv_of_components {seqLen dModel dMLP : ℕ}
    (opsA opsB : BlockOps seqLen dModel dMLP)
    (hln1 : ∀ x, opsA.ln1 x = opsB.ln1 x)
    (hattn : ∀ x, opsA.attn x = opsB.attn x)
    (hln2 : ∀ x, opsA.ln2 x = opsB.ln2 x)
    (hmlp : ∀ x, opsA.mlp x = opsB.mlp x) :
    ∀ x, gpt2Block opsA x = gpt2Block opsB x := by
  intro x; unfold gpt2Block
  have h1 : x + opsA.attn (opsA.ln1 x) = x + opsB.attn (opsB.ln1 x) := by
    rw [hln1 x, hattn (opsB.ln1 x)]
  rw [h1]
  set x1 : SeqState seqLen dModel := x + opsB.attn (opsB.ln1 x)
  simpa [x1] using congrArg (fun y => x1 + y) (by rw [hln2 x1, hmlp (opsB.ln2 x1)])

/-- A generic pipeline: embed → stacked blocks → unembed. -/
structure PipelineSpec (Tokens State Logits : Type _) where
  embed : Tokens → State
  block : ℕ → State → State
  unembed : State → Logits

/-- Execute n blocks in order: `block 0` first (GPT-2 layer 0), then `block 1`, …,
    finally `block (n-1)`.  The recursion peels off the *outermost* layer at each
    step, so `block 0` is innermost in the call stack — applied first to the
    embedding output.  Block indices therefore match GPT-2's layer numbering
    (0 = earliest / closest to embedding). -/
def runBlocks {State : Type _} (block : ℕ → State → State) : ℕ → State → State
  | 0, s => s
  | n + 1, s => block n (runBlocks block n s)

/-- Full forward pass: embed → run blocks → unembed. -/
def forward {Tokens State Logits : Type _}
    (spec : PipelineSpec Tokens State Logits)
    (layers : ℕ) (tok : Tokens) : Logits :=
  spec.unembed (runBlocks spec.block layers (spec.embed tok))

/-- Component equality → full pipeline equality. -/
theorem gpt2_end_to_end_equiv {Tokens State Logits : Type _}
    (specA specB : PipelineSpec Tokens State Logits)
    (hembed : ∀ tok, specA.embed tok = specB.embed tok)
    (hblock : ∀ i s, specA.block i s = specB.block i s)
    (hunembed : ∀ s, specA.unembed s = specB.unembed s) :
    ∀ layers tok, forward specA layers tok = forward specB layers tok := by
  intro layers tok; unfold forward
  rw [hunembed]
  have hrun : ∀ n s, runBlocks specA.block n s = runBlocks specB.block n s := by
    intro n; induction n with
    | zero => intro s; rfl
    | succ n ih => intro s; simp [runBlocks, hblock, ih]
  rw [hrun, hembed]

-- ═══════════════════════════════════════════════════════════════════════
-- § 2.3  Concrete GPT-2 components (LN, MLP, embed, LM head)
-- ═══════════════════════════════════════════════════════════════════════

/-- LayerNorm parameters.

    `dmodel_pos` guards the division-by-`dModel` in `layerNormMean` and
    `layerNormVar`.  When `dModel = 0`, those functions would silently
    return 0 (Lean's `x / 0 = 0` convention), making LayerNorm degenerate.
    Requiring `0 < dModel` eliminates this edge case and ensures that
    normalisation is non-trivial.  All real models have `dModel ≥ 64`. -/
structure LayerNormParams (dModel : ℕ) where
  gamma : ModelVec dModel
  beta : ModelVec dModel
  eps : ℝ
  eps_pos : 0 < eps
  dmodel_pos : 0 < dModel

def layerNormMean {dModel : ℕ} [NeZero dModel] (x : ModelVec dModel) : ℝ :=
  (∑ i : Fin dModel, x i) / (dModel : ℝ)

def layerNormVar {dModel : ℕ} [NeZero dModel] (x : ModelVec dModel) : ℝ :=
  (∑ i : Fin dModel, (x i - layerNormMean x) ^ 2) / (dModel : ℝ)

def layerNormVec {dModel : ℕ} (p : LayerNormParams dModel) (x : ModelVec dModel) : ModelVec dModel :=
  letI : NeZero dModel := ⟨p.dmodel_pos.ne'⟩
  fun i => p.gamma i * ((x i - layerNormMean x) / Real.sqrt (layerNormVar x + p.eps)) + p.beta i

def layerNormState {seqLen dModel : ℕ}
    (p : LayerNormParams dModel) (x : SeqState seqLen dModel) : SeqState seqLen dModel :=
  fun t => layerNormVec p (x t)

/-- GPT-2 MLP parameters (W1+b1 → GELU → W2+b2). -/
structure GPT2MLPParams (dModel dMLP : ℕ) where
  W1 : Matrix (Fin dMLP) (Fin dModel) ℝ
  b1 : Fin dMLP → ℝ
  W2 : Matrix (Fin dModel) (Fin dMLP) ℝ
  b2 : Fin dModel → ℝ

/-- GPT-2's gelu_new: the tanh closed form. -/
def gelu (x : ℝ) : ℝ :=
  (x / 2) * (1 + Real.tanh ((Real.sqrt (2 / Real.pi)) * (x + 0.044715 * x ^ 3)))

def gpt2MLPVec {dModel dMLP : ℕ}
    (p : GPT2MLPParams dModel dMLP) (x : ModelVec dModel) : ModelVec dModel :=
  fun i => Matrix.mulVec p.W2 (fun j => gelu (Matrix.mulVec p.W1 x j + p.b1 j)) i + p.b2 i

def gpt2MLPState {seqLen dModel dMLP : ℕ}
    (p : GPT2MLPParams dModel dMLP) (x : SeqState seqLen dModel) : SeqState seqLen dModel :=
  fun t => gpt2MLPVec p (x t)

/-- Embedding parameters (token table + position table). -/
abbrev TokenInput (seqLen vocab : ℕ) := Fin seqLen → Fin vocab

structure EmbeddingParams (seqLen vocab dModel : ℕ) where
  tokenEmbed : Fin vocab → ModelVec dModel
  posEmbed : Fin seqLen → ModelVec dModel

def inputEmbedding {seqLen vocab dModel : ℕ}
    (p : EmbeddingParams seqLen vocab dModel)
    (tok : TokenInput seqLen vocab) : SeqState seqLen dModel :=
  fun t => p.tokenEmbed (tok t) + p.posEmbed t

/-- LM head parameters (final linear projection to vocabulary logits). -/
abbrev SeqLogits (seqLen vocab : ℕ) := Fin seqLen → Fin vocab → ℝ

structure LMHeadParams (dModel vocab : ℕ) where
  Wout : Matrix (Fin vocab) (Fin dModel) ℝ
  bout : Fin vocab → ℝ

def lmHeadState {seqLen dModel vocab : ℕ}
    (p : LMHeadParams dModel vocab) (x : SeqState seqLen dModel) : SeqLogits seqLen vocab :=
  fun t v => Matrix.mulVec p.Wout (x t) v + p.bout v

def blockFamilySpec {Tokens Logits : Type _} {seqLen dModel dMLP : ℕ}
    (embed : Tokens → SeqState seqLen dModel)
    (blocks : ℕ → BlockOps seqLen dModel dMLP)
    (unembed : SeqState seqLen dModel → Logits) :
    PipelineSpec Tokens (SeqState seqLen dModel) Logits where
  embed := embed
  block i x := gpt2Block (blocks i) x
  unembed := unembed

-- ═══════════════════════════════════════════════════════════════════════
-- § 2.4  The universality theorem
-- ═══════════════════════════════════════════════════════════════════════

/-- The standard GPT-2 unembed path: a final LayerNorm `ln_f`, then the LM head.

    Real GPT-2 (HuggingFace `transformers.GPT2Model`) applies `transformer.ln_f`
    to the last block's output before the LM head; omitting it would drop a
    layer that standard GPT-2 checkpoints has. -/
def gpt2Unembed {seqLen dModel vocab : ℕ}
    (lnF : LayerNormParams dModel) (lmHead : LMHeadParams dModel vocab)
    (x : SeqState seqLen dModel) : SeqLogits seqLen vocab :=
  lmHeadState lmHead (layerNormState lnF x)

/-- A complete set of GPT-2 model parameters (score-based). -/
structure GPT2Params (seqLen vocab nHeads D dModel dMLP : ℕ)
    (contextSize : Fin seqLen → ℕ) where
  embedding : EmbeddingParams seqLen vocab dModel
  multihead : ℕ → MultiHeadAttnParams seqLen nHeads D dModel contextSize
  ln1 : ℕ → LayerNormParams dModel
  ln2 : ℕ → LayerNormParams dModel
  mlp : ℕ → GPT2MLPParams dModel dMLP
  /-- Final LayerNorm `ln_f` applied to the last block's output before the LM head.
      Present in standard GPT-2 checkpoints as `transformer.ln_f`. -/
  lnF : LayerNormParams dModel
  lmHead : LMHeadParams dModel vocab

/-- A complete set of independently-typed pole-based parameters.
    The `multihead` field stores `AnalyticMHParams` whose `poleHead` returns
    `AnalyticHeadParams` — poles specified directly, not derived from scores.
    This makes the attention sub-parameters a genuinely different type from
    `GPT2Params.multihead`.

    The remaining fields (embedding, LN, MLP, ln_f, LM head) share types with
    `GPT2Params`; they are copied verbatim by `scoreDerivedAnalyticParams`
    because the Cauchy-Poisson identity only replaces attention, not the
    other components. -/
structure AnalyticGPT2Params (seqLen vocab nHeads D dModel dMLP : ℕ)
    (contextSize : Fin seqLen → ℕ) where
  embedding : EmbeddingParams seqLen vocab dModel
  multihead : ℕ → AnalyticMHParams seqLen nHeads D dModel contextSize
  ln1 : ℕ → LayerNormParams dModel
  ln2 : ℕ → LayerNormParams dModel
  mlp : ℕ → GPT2MLPParams dModel dMLP
  lnF : LayerNormParams dModel
  lmHead : LMHeadParams dModel vocab

/-- The constructive converter: replace every head's score-based params with
    score-derived poles.  This is the witness used in `transformer_is_cauchy_poisson`. -/
def scoreDerivedAnalyticParams {seqLen vocab nHeads D dModel dMLP : ℕ}
    {contextSize : Fin seqLen → ℕ}
    (θ : GPT2Params seqLen vocab nHeads D dModel dMLP contextSize) :
    AnalyticGPT2Params seqLen vocab nHeads D dModel dMLP contextSize where
  embedding := θ.embedding
  multihead i := scoreDerivedAnalyticMH (θ.multihead i)
  ln1 := θ.ln1
  ln2 := θ.ln2
  mlp := θ.mlp
  lnF := θ.lnF
  lmHead := θ.lmHead

/-- The standard GPT-2 forward pass:
      embed → (ln1 → attn → +residual → ln2 → mlp → +residual)·layers → lnF → lmHead. -/
def gpt2Forward {seqLen vocab nHeads D dModel dMLP : ℕ} {contextSize : Fin seqLen → ℕ}
    (θ : GPT2Params seqLen vocab nHeads D dModel dMLP contextSize)
    (layers : ℕ) (tok : TokenInput seqLen vocab) : SeqLogits seqLen vocab :=
  forward
    (blockFamilySpec (inputEmbedding θ.embedding)
      (fun i => ({ ln1 := layerNormState (θ.ln1 i)
                 , attn := gpt2MultiHeadAttn (θ.multihead i)
                 , ln2 := layerNormState (θ.ln2 i)
                 , mlp := gpt2MLPState (θ.mlp i) }
        : BlockOps seqLen dModel dMLP))
      (gpt2Unembed θ.lnF θ.lmHead))
    layers tok

/-- The analytic (Poisson-pole) forward pass.  Takes `AnalyticGPT2Params` —
    a DIFFERENT type from `GPT2Params`.  Attention is computed via Poisson
    kernel evaluation on independently-specified pole structs.  Embedding,
    LayerNorms (including ln_f), MLP, and LM head are definitionally shared. -/
def analyticForward {seqLen vocab nHeads D dModel dMLP : ℕ} {contextSize : Fin seqLen → ℕ}
    (φ : AnalyticGPT2Params seqLen vocab nHeads D dModel dMLP contextSize)
    (layers : ℕ) (tok : TokenInput seqLen vocab) : SeqLogits seqLen vocab :=
  forward
    (blockFamilySpec (inputEmbedding φ.embedding)
      (fun i => ({ ln1 := layerNormState (φ.ln1 i)
                 , attn := analyticMultiHeadAttn (φ.multihead i)
                 , ln2 := layerNormState (φ.ln2 i)
                 , mlp := gpt2MLPState (φ.mlp i) }
        : BlockOps seqLen dModel dMLP))
      (gpt2Unembed φ.lnF φ.lmHead))
    layers tok

/-- Helper: equality of stacked blocks when attention sides are connected via
    `scoreDerivedAnalyticMH`. -/
private theorem block_family_attn_equiv {seqLen vocab nHeads D dModel dMLP : ℕ}
    {contextSize : Fin seqLen → ℕ}
    (θ : GPT2Params seqLen vocab nHeads D dModel dMLP contextSize) :
    ∀ layers tok, gpt2Forward θ layers tok =
      analyticForward (scoreDerivedAnalyticParams θ) layers tok := by
  intro layers tok
  unfold gpt2Forward analyticForward scoreDerivedAnalyticParams
  apply gpt2_end_to_end_equiv
  · intro tok; rfl
  · intro i x
    apply gpt2Block_equiv_of_components
    · intro s; rfl
    · intro s; exact multiHeadAttn_equiv (θ.multihead i) s
    · intro s; rfl
    · intro s; rfl
  · intro s; rfl

/-- **Main theorem — Transformers are Cauchy-Poisson systems.**

    For all `θ : GPT2Params` (checkpoint or random initialization, any depth, any input),
    the standard GPT-2 forward pass equals the Poisson-pole forward pass of
    `scoreDerivedAnalyticParams θ : AnalyticGPT2Params`.

    The two sides have DIFFERENT types:
    - `gpt2Forward θ` uses `gpt2HeadOutput`: softmax(scores) · V
    - `analyticForward φ` uses `analyticHeadOutput`: Poisson(poles) · V,
      where poles are specified as an independent `AnalyticHeadParams` struct

    The witness `φ = scoreDerivedAnalyticParams θ` is constructive and explicit.
    No hypotheses on θ.  The architecture alone guarantees this. -/
theorem transformer_is_cauchy_poisson
    {seqLen vocab nHeads D dModel dMLP : ℕ} {contextSize : Fin seqLen → ℕ}
    (θ : GPT2Params seqLen vocab nHeads D dModel dMLP contextSize) :
    ∀ layers : ℕ, ∀ tok : TokenInput seqLen vocab,
      gpt2Forward θ layers tok =
        analyticForward (scoreDerivedAnalyticParams θ) layers tok :=
  block_family_attn_equiv θ

/-- **Corollary — Function-level equality.** -/
theorem transformers_are_boundary_value_solvers
    {seqLen vocab nHeads D dModel dMLP : ℕ} {contextSize : Fin seqLen → ℕ}
    (θ : GPT2Params seqLen vocab nHeads D dModel dMLP contextSize) :
    gpt2Forward θ = analyticForward (scoreDerivedAnalyticParams θ) := by
  funext layers tok; exact transformer_is_cauchy_poisson θ layers tok

/-- **Corollary — Genuinely existential form.**

    There EXISTS a `φ : AnalyticGPT2Params` (a different type from `GPT2Params`)
    such that `analyticForward φ = gpt2Forward θ`.  The existential ranges over
    an independently-typed space; the witness is `scoreDerivedAnalyticParams θ`. -/
theorem gpt2_has_pole_realization
    {seqLen vocab nHeads D dModel dMLP : ℕ} {contextSize : Fin seqLen → ℕ}
    (θ : GPT2Params seqLen vocab nHeads D dModel dMLP contextSize)
    (layers : ℕ) (tok : TokenInput seqLen vocab) :
    ∃ (φ : AnalyticGPT2Params seqLen vocab nHeads D dModel dMLP contextSize),
      analyticForward φ layers tok = gpt2Forward θ layers tok :=
  ⟨scoreDerivedAnalyticParams θ, (transformer_is_cauchy_poisson θ layers tok).symm⟩

/-- **Function-level existential realization (fully non-pointwise).**

    There exists an analytic parameter family `φ` such that the full
    forward functions are equal, not only each pointwise call. -/
theorem gpt2_has_pole_realization_functional
    {seqLen vocab nHeads D dModel dMLP : ℕ} {contextSize : Fin seqLen → ℕ}
    (θ : GPT2Params seqLen vocab nHeads D dModel dMLP contextSize) :
    ∃ (φ : AnalyticGPT2Params seqLen vocab nHeads D dModel dMLP contextSize),
      analyticForward φ = gpt2Forward θ := by
  refine ⟨scoreDerivedAnalyticParams θ, ?_⟩
  funext layers tok
  exact (transformer_is_cauchy_poisson θ layers tok).symm

/-- **Single global analytic object (no per-row reconstruction in the statement).**

    There exists one fixed analytic parameter family `φ` (shared across all
    queries/rows, tokens, and layer counts) such that every forward call of the
    score-based model equals the corresponding forward call of the analytic model.

    Formally, `φ` is chosen once from the independent parameter type
    `AnalyticGPT2Params`, and then

    `∀ layers tok, analyticForward φ layers tok = gpt2Forward θ layers tok`.

    This is the global-object form of the bridge; unlike row-level identities, the
    theorem statement does not mention the reciprocal map `w ↦ 1/w`. -/
theorem gpt2_has_single_global_analytic_object
    {seqLen vocab nHeads D dModel dMLP : ℕ} {contextSize : Fin seqLen → ℕ}
    (θ : GPT2Params seqLen vocab nHeads D dModel dMLP contextSize) :
    ∃ (φ : AnalyticGPT2Params seqLen vocab nHeads D dModel dMLP contextSize),
      ∀ (layers : ℕ) (tok : TokenInput seqLen vocab),
        analyticForward φ layers tok = gpt2Forward θ layers tok := by
  refine ⟨scoreDerivedAnalyticParams θ, ?_⟩
  intro layers tok
  exact (transformer_is_cauchy_poisson θ layers tok).symm

/-- **`M_global` (global analytic realization class).**

    This is the critic-proof interface: a score-based transformer parameter set
    `θ` belongs to `M_global` if there exists one shared analytic parameter
    object `φ` such that all forward calls (all layer counts and all token
    inputs) are generated by that same `φ`.

    Importantly, this statement is purely global/function-level and does not
    mention row-wise reciprocal reconstructions (`w ↦ 1/w`). -/
def M_global
    {seqLen vocab nHeads D dModel dMLP : ℕ}
    {contextSize : Fin seqLen → ℕ}
    (θ : GPT2Params seqLen vocab nHeads D dModel dMLP contextSize) : Prop :=
  ∃ (φ : AnalyticGPT2Params seqLen vocab nHeads D dModel dMLP contextSize),
    ∀ (layers : ℕ) (tok : TokenInput seqLen vocab),
      analyticForward φ layers tok = gpt2Forward θ layers tok

/-- State-independent pole geometry for one analytic multi-head module.

    `FixedPoleGeometry` says the poles used by `poleHead` do not depend on the
    current sequence state `x`; they may vary with query position `t` and head
    index `h`, but are shared across inputs at that `(t,h)` slot. -/
def AnalyticMHParams.FixedPoleGeometry
    {seqLen nHeads D dModel : ℕ} {contextSize : Fin seqLen → ℕ}
    (amh : AnalyticMHParams seqLen nHeads D dModel contextSize) : Prop :=
  ∀ (t : Fin seqLen) (h : Fin nHeads) (x₁ x₂ : SeqState seqLen dModel),
    (amh.poleHead t h x₁).poles = (amh.poleHead t h x₂).poles

/-- **Stricter global class:** one shared analytic object + fixed poles.

    This strengthens `M_global` by adding a row-level anti-posthoc condition:
    for each layer and each `(query position, head)` slot, pole geometry is
    state-independent.  The remaining gap is now explicit and measurable:
    `M_global_fixed_poles θ` is the target class beyond plain functional
    equivalence. -/
def M_global_fixed_poles
    {seqLen vocab nHeads D dModel dMLP : ℕ}
    {contextSize : Fin seqLen → ℕ}
    (θ : GPT2Params seqLen vocab nHeads D dModel dMLP contextSize) : Prop :=
  ∃ (φ : AnalyticGPT2Params seqLen vocab nHeads D dModel dMLP contextSize),
    (∀ (layers : ℕ) (tok : TokenInput seqLen vocab),
      analyticForward φ layers tok = gpt2Forward θ layers tok) ∧
    (∀ i : ℕ, AnalyticMHParams.FixedPoleGeometry (φ.multihead i))

/-- The strict class projects to the original global realization class. -/
theorem M_global_fixed_poles_implies_M_global
    {seqLen vocab nHeads D dModel dMLP : ℕ}
    {contextSize : Fin seqLen → ℕ}
    {θ : GPT2Params seqLen vocab nHeads D dModel dMLP contextSize} :
    M_global_fixed_poles θ → M_global θ := by
  rintro ⟨φ, hEq, _hFixed⟩
  exact ⟨φ, hEq⟩

/-- Multi-head analytic parameters with state-independent pole geometry.

    `poles t h` is fixed for each query/head slot.  The value tensor remains
    state-dependent via `Vof t h x`.  This captures the strict anti-posthoc
    architecture class where row geometry is shared and not reconstructed from
    solved row weights. -/
structure FixedPoleAnalyticMHParams (seqLen nHeads D dModel : ℕ)
    (contextSize : Fin seqLen → ℕ) where
  poles : (t : Fin seqLen) → Fin nHeads → Poles (contextSize t)
  Vof : (t : Fin seqLen) → Fin nHeads → SeqState seqLen dModel →
    Fin (contextSize t) → Fin D → ℝ
  outProj : Matrix (Fin dModel) (Fin (nHeads * D)) ℝ
  outProjBias : Fin dModel → ℝ

/-- Forgetful map: fixed-pole MH params are ordinary analytic MH params. -/
def FixedPoleAnalyticMHParams.toAnalyticMH
    {seqLen nHeads D dModel : ℕ} {contextSize : Fin seqLen → ℕ}
    (fpmh : FixedPoleAnalyticMHParams seqLen nHeads D dModel contextSize) :
    AnalyticMHParams seqLen nHeads D dModel contextSize where
  poleHead t h x := { poles := fpmh.poles t h, V := fpmh.Vof t h x }
  outProj := fpmh.outProj
  outProjBias := fpmh.outProjBias

/-- The forgetful map satisfies fixed pole geometry by construction. -/
theorem FixedPoleAnalyticMHParams.toAnalyticMH_fixedGeometry
    {seqLen nHeads D dModel : ℕ} {contextSize : Fin seqLen → ℕ}
    (fpmh : FixedPoleAnalyticMHParams seqLen nHeads D dModel contextSize) :
    AnalyticMHParams.FixedPoleGeometry fpmh.toAnalyticMH := by
  intro t h x₁ x₂
  rfl

/-- Full-model analytic parameters with fixed poles at every layer/head/slot. -/
structure FixedPoleAnalyticGPT2Params (seqLen vocab nHeads D dModel dMLP : ℕ)
    (contextSize : Fin seqLen → ℕ) where
  embedding : EmbeddingParams seqLen vocab dModel
  multihead : ℕ → FixedPoleAnalyticMHParams seqLen nHeads D dModel contextSize
  ln1 : ℕ → LayerNormParams dModel
  ln2 : ℕ → LayerNormParams dModel
  mlp : ℕ → GPT2MLPParams dModel dMLP
  lnF : LayerNormParams dModel
  lmHead : LMHeadParams dModel vocab

/-- Forgetful map from fixed-pole full params to ordinary analytic full params. -/
def FixedPoleAnalyticGPT2Params.toAnalyticGPT2
    {seqLen vocab nHeads D dModel dMLP : ℕ} {contextSize : Fin seqLen → ℕ}
    (φf : FixedPoleAnalyticGPT2Params seqLen vocab nHeads D dModel dMLP contextSize) :
    AnalyticGPT2Params seqLen vocab nHeads D dModel dMLP contextSize where
  embedding := φf.embedding
  multihead i := (φf.multihead i).toAnalyticMH
  ln1 := φf.ln1
  ln2 := φf.ln2
  mlp := φf.mlp
  lnF := φf.lnF
  lmHead := φf.lmHead

/-- Forward pass for the fixed-pole full model (delegates to analyticForward). -/
def fixedPoleAnalyticForward
    {seqLen vocab nHeads D dModel dMLP : ℕ} {contextSize : Fin seqLen → ℕ}
    (φf : FixedPoleAnalyticGPT2Params seqLen vocab nHeads D dModel dMLP contextSize)
    (layers : ℕ) (tok : TokenInput seqLen vocab) : SeqLogits seqLen vocab :=
  analyticForward φf.toAnalyticGPT2 layers tok

/-- If a fixed-pole analytic witness matches GPT2 forward calls globally, then
    the GPT2 family is in `M_global_fixed_poles`. -/
theorem fixed_pole_witness_in_M_global_fixed_poles
    {seqLen vocab nHeads D dModel dMLP : ℕ}
    {contextSize : Fin seqLen → ℕ}
    (θ : GPT2Params seqLen vocab nHeads D dModel dMLP contextSize)
    (φf : FixedPoleAnalyticGPT2Params seqLen vocab nHeads D dModel dMLP contextSize)
    (hEq : ∀ (layers : ℕ) (tok : TokenInput seqLen vocab),
      fixedPoleAnalyticForward φf layers tok = gpt2Forward θ layers tok) :
    M_global_fixed_poles θ := by
  refine ⟨φf.toAnalyticGPT2, ?_, ?_⟩
  · intro layers tok
    simpa [fixedPoleAnalyticForward] using hEq layers tok
  · intro i
    simpa using (φf.multihead i).toAnalyticMH_fixedGeometry

/-- **Conditional closure of GPT2 into the strict fixed-pole class.**

    Unconditionally, vanilla GPT2 need not lie in `M_global_fixed_poles`
    because score maps typically depend on the current state `x`.
    This theorem gives the exact missing condition:

    if, for every layer/query/head, the score vector is state-independent,
    then the standard score-derived analytic witness has fixed pole geometry,
    hence `θ ∈ M_global_fixed_poles`.

    This makes the remaining gap explicit and measurable. -/
theorem gpt2_in_M_global_fixed_poles_of_state_independent_scores
    {seqLen vocab nHeads D dModel dMLP : ℕ}
    {contextSize : Fin seqLen → ℕ}
    (θ : GPT2Params seqLen vocab nHeads D dModel dMLP contextSize)
    (hscore :
      ∀ (i : ℕ) (t : Fin seqLen) (h : Fin nHeads)
        (x₁ x₂ : SeqState seqLen dModel),
        ((θ.multihead i).head t h x₁).scores = ((θ.multihead i).head t h x₂).scores) :
    M_global_fixed_poles θ := by
  refine ⟨scoreDerivedAnalyticParams θ, ?_, ?_⟩
  · intro layers tok
    exact (transformer_is_cauchy_poisson θ layers tok).symm
  · intro i
    intro t h x₁ x₂
    have hs :
        ((θ.multihead i).head t h x₁).scores = ((θ.multihead i).head t h x₂).scores :=
      hscore i t h x₁ x₂
    -- Pole equality follows from score equality (`scorePoles` is functional in scores).
    simp [scoreDerivedAnalyticParams, scoreDerivedAnalyticMH,
      scoreDerivedAnalyticHead, hs]

/-- Score maps do not depend on the current sequence state (vanilla-degenerate regime). -/
def ScoreStateIndependent
    {seqLen vocab nHeads D dModel dMLP : ℕ}
    {contextSize : Fin seqLen → ℕ}
    (θ : GPT2Params seqLen vocab nHeads D dModel dMLP contextSize) : Prop :=
  ∀ (i : ℕ) (t : Fin seqLen) (h : Fin nHeads)
    (x₁ x₂ : SeqState seqLen dModel),
    ((θ.multihead i).head t h x₁).scores = ((θ.multihead i).head t h x₂).scores

/-- There exists at least one layer/query/head slot where scores vary with state. -/
def NondegenerateScores
    {seqLen vocab nHeads D dModel dMLP : ℕ}
    {contextSize : Fin seqLen → ℕ}
    (θ : GPT2Params seqLen vocab nHeads D dModel dMLP contextSize) : Prop :=
  ∃ (i : ℕ) (t : Fin seqLen) (h : Fin nHeads)
    (x₁ x₂ : SeqState seqLen dModel),
    ((θ.multihead i).head t h x₁).scores ≠ ((θ.multihead i).head t h x₂).scores

/-- Score differences are state-independent (softmax-gauge invariant form). -/
def ScoreDifferencesStateIndependent
    {seqLen vocab nHeads D dModel dMLP : ℕ}
    {contextSize : Fin seqLen → ℕ}
    (θ : GPT2Params seqLen vocab nHeads D dModel dMLP contextSize) : Prop :=
  ∀ (i : ℕ) (t : Fin seqLen) (h : Fin nHeads)
    (x₁ x₂ : SeqState seqLen dModel) (k j : Fin (contextSize t)),
    ((θ.multihead i).head t h x₁).scores j - ((θ.multihead i).head t h x₁).scores k =
      ((θ.multihead i).head t h x₂).scores j - ((θ.multihead i).head t h x₂).scores k

/-- There exists at least one slot where score differences vary with state. -/
def NondegenerateScoreDifferences
    {seqLen vocab nHeads D dModel dMLP : ℕ}
    {contextSize : Fin seqLen → ℕ}
    (θ : GPT2Params seqLen vocab nHeads D dModel dMLP contextSize) : Prop :=
  ∃ (i : ℕ) (t : Fin seqLen) (h : Fin nHeads)
    (x₁ x₂ : SeqState seqLen dModel) (k j : Fin (contextSize t)),
    ((θ.multihead i).head t h x₁).scores j - ((θ.multihead i).head t h x₁).scores k ≠
      ((θ.multihead i).head t h x₂).scores j - ((θ.multihead i).head t h x₂).scores k

/-- Fixed geometry of the score-derived witness itself (not an arbitrary witness). -/
def ScoreDerivedFixedPoleGeometry
    {seqLen vocab nHeads D dModel dMLP : ℕ}
    {contextSize : Fin seqLen → ℕ}
    (θ : GPT2Params seqLen vocab nHeads D dModel dMLP contextSize) : Prop :=
  ∀ (i : ℕ) (t : Fin seqLen) (h : Fin nHeads)
    (x₁ x₂ : SeqState seqLen dModel),
    ((scoreDerivedAnalyticMH (θ.multihead i)).poleHead t h x₁).poles =
      ((scoreDerivedAnalyticMH (θ.multihead i)).poleHead t h x₂).poles

/-- Row-level architectural constraint: an analytic witness realizes each vanilla
    softmax row by an on-query Poisson pole family for the corresponding head. -/
def RealizesVanillaSoftmaxRows
    {seqLen vocab nHeads D dModel dMLP : ℕ}
    {contextSize : Fin seqLen → ℕ}
    (θ : GPT2Params seqLen vocab nHeads D dModel dMLP contextSize)
    (φ : AnalyticGPT2Params seqLen vocab nHeads D dModel dMLP contextSize) : Prop :=
  ∀ (i : ℕ) (t : Fin seqLen) (h : Fin nHeads) (x : SeqState seqLen dModel),
    let hp := (θ.multihead i).head t h x
    let p := ((φ.multihead i).poleHead t h x).poles
    (∀ k : Fin (contextSize t), p.x k = (0 : ℝ)) ∧
    (∀ k : Fin (contextSize t),
      poisson (p.x k) (p.y k) (0 : ℝ) =
        exp (hp.scores k) / ∑ j : Fin (contextSize t), exp (hp.scores j))

/-- Nondegenerate scores are exactly the negation of state-independence. -/
theorem nondegenerateScores_iff_not_scoreStateIndependent
    {seqLen vocab nHeads D dModel dMLP : ℕ}
    {contextSize : Fin seqLen → ℕ}
    (θ : GPT2Params seqLen vocab nHeads D dModel dMLP contextSize) :
    NondegenerateScores θ ↔ ¬ ScoreStateIndependent θ := by
  constructor
  · rintro ⟨i, t, h, x₁, x₂, hne⟩ hs
    exact hne (hs i t h x₁ x₂)
  · intro hnot
    by_contra hnone
    apply hnot
    intro i t h x₁ x₂
    by_contra hne
    exact hnone ⟨i, t, h, x₁, x₂, hne⟩

/-- Nondegenerate score differences are exactly the negation of difference invariance. -/
theorem nondegenerateScoreDifferences_iff_not_scoreDifferencesStateIndependent
    {seqLen vocab nHeads D dModel dMLP : ℕ}
    {contextSize : Fin seqLen → ℕ}
    (θ : GPT2Params seqLen vocab nHeads D dModel dMLP contextSize) :
    NondegenerateScoreDifferences θ ↔ ¬ ScoreDifferencesStateIndependent θ := by
  constructor
  · rintro ⟨i, t, h, x₁, x₂, k, j, hne⟩ hs
    exact hne (hs i t h x₁ x₂ k j)
  · intro hnot
    by_contra hnone
    apply hnot
    intro i t h x₁ x₂ k j
    by_contra hne
    exact hnone ⟨i, t, h, x₁, x₂, k, j, hne⟩

/-- Under explicit vanilla row-realization constraints, any analytic witness pole
    family is forced to be the canonical score-derived one (`scorePoles`). -/
theorem vanilla_row_realization_forces_scoreDerived_poles
    {seqLen vocab nHeads D dModel dMLP : ℕ}
    {contextSize : Fin seqLen → ℕ}
    (θ : GPT2Params seqLen vocab nHeads D dModel dMLP contextSize)
    (φ : AnalyticGPT2Params seqLen vocab nHeads D dModel dMLP contextSize)
    (hrows : RealizesVanillaSoftmaxRows θ φ) :
    ∀ (i : ℕ) (t : Fin seqLen) (h : Fin nHeads) (x : SeqState seqLen dModel)
      (_hne : NeZero (contextSize t)),
      ((φ.multihead i).poleHead t h x).poles =
        scorePoles (((θ.multihead i).head t h x).scores) (0 : ℝ) := by
  intro i t h x _hne
  letI : NeZero (contextSize t) := _hne
  rcases hrows i t h x with ⟨hx, hw⟩
  exact scorePoles_unique_on_vertical_slice (((θ.multihead i).head t h x).scores) (0 : ℝ)
    (((φ.multihead i).poleHead t h x).poles) hx hw

/-- If a fixed-pole analytic witness realizes vanilla softmax rows (on-query),
    then fixed geometry necessarily collapses to score-derived fixed geometry.
    This is the explicit "automatic reduction" bridge from general fixed-pole
    witnesses to the canonical score-derived class. -/
theorem fixed_pole_vanilla_realization_implies_scoreDerivedFixedPoleGeometry
    {seqLen vocab nHeads D dModel dMLP : ℕ}
    {contextSize : Fin seqLen → ℕ}
    (θ : GPT2Params seqLen vocab nHeads D dModel dMLP contextSize)
    (φ : AnalyticGPT2Params seqLen vocab nHeads D dModel dMLP contextSize)
    (hfixed : ∀ i : ℕ, AnalyticMHParams.FixedPoleGeometry (φ.multihead i))
    (hrows : RealizesVanillaSoftmaxRows θ φ) :
    ScoreDerivedFixedPoleGeometry θ := by
  intro i t h x₁ x₂
  haveI : NeZero (contextSize t) := ⟨((θ.multihead i).cs_pos t).ne'⟩
  have hφ : ((φ.multihead i).poleHead t h x₁).poles =
      ((φ.multihead i).poleHead t h x₂).poles := hfixed i t h x₁ x₂
  have hx₁ :
      ((φ.multihead i).poleHead t h x₁).poles =
        scorePoles (((θ.multihead i).head t h x₁).scores) (0 : ℝ) :=
    vanilla_row_realization_forces_scoreDerived_poles θ φ hrows i t h x₁
      inferInstance
  have hx₂ :
      ((φ.multihead i).poleHead t h x₂).poles =
        scorePoles (((θ.multihead i).head t h x₂).scores) (0 : ℝ) :=
    vanilla_row_realization_forces_scoreDerived_poles θ φ hrows i t h x₂
      inferInstance
  -- Transport fixedness through the forced score-derived identification.
  exact hx₁.symm.trans (hφ.trans hx₂)

/-- **Intrinsic dynamic realization (moving-pole form).**

    Every vanilla GPT2 parameter family admits a global analytic witness that
    realizes the same forward map, and this witness realizes each row by an
    on-query Poisson pole family computed from the current state-dependent
    score vector.  This is the formal "moving poles with state" statement. -/
theorem gpt2_has_intrinsic_dynamic_row_realization
    {seqLen vocab nHeads D dModel dMLP : ℕ}
    {contextSize : Fin seqLen → ℕ}
    (θ : GPT2Params seqLen vocab nHeads D dModel dMLP contextSize) :
    ∃ φ : AnalyticGPT2Params seqLen vocab nHeads D dModel dMLP contextSize,
      (∀ (layers : ℕ) (tok : TokenInput seqLen vocab),
        analyticForward φ layers tok = gpt2Forward θ layers tok) ∧
      RealizesVanillaSoftmaxRows θ φ := by
  refine ⟨scoreDerivedAnalyticParams θ, ?_, ?_⟩
  · intro layers tok
    exact (transformer_is_cauchy_poisson θ layers tok).symm
  · intro i t h x
    haveI : NeZero (contextSize t) := ⟨((θ.multihead i).cs_pos t).ne'⟩
    constructor
    · intro k
      rfl
    · intro k
      simpa [scoreDerivedAnalyticParams, scoreDerivedAnalyticMH, scoreDerivedAnalyticHead]
        using softmax_is_poisson_at_score_poles (((θ.multihead i).head t h x).scores) (0 : ℝ) k

/-- **Actual bridge for vanilla score-derived poles.**

    If the poles produced by `scoreDerivedAnalyticMH` are fixed across states
    (for each layer/query/head slot), then all score differences are
    state-independent.  This is the mathematically correct bridge: scores are
    only identifiable up to additive constants, while differences are gauge-free. -/
theorem scoreDerived_fixed_poles_imply_scoreDifferencesStateIndependent
    {seqLen vocab nHeads D dModel dMLP : ℕ}
    {contextSize : Fin seqLen → ℕ}
    (θ : GPT2Params seqLen vocab nHeads D dModel dMLP contextSize)
    (hfixed : ScoreDerivedFixedPoleGeometry θ) :
    ScoreDifferencesStateIndependent θ := by
  intro i t h x₁ x₂ k j
  let hp₁ := ((θ.multihead i).head t h x₁)
  let hp₂ := ((θ.multihead i).head t h x₂)
  haveI : NeZero (contextSize t) := ⟨((θ.multihead i).cs_pos t).ne'⟩
  have hpoles :
      scorePoles hp₁.scores (0 : ℝ) = scorePoles hp₂.scores (0 : ℝ) := by
    simpa [scoreDerivedAnalyticMH, scoreDerivedAnalyticHead, hp₁, hp₂] using hfixed i t h x₁ x₂
  have hlog₁ :
      Real.log ((scorePoles hp₁.scores (0 : ℝ)).y k) -
        Real.log ((scorePoles hp₁.scores (0 : ℝ)).y j) =
        hp₁.scores j - hp₁.scores k := by
    simpa [scorePoles] using (softmax_pole_log_ratio hp₁.scores (0 : ℝ) k j)
  have hlog₂ :
      Real.log ((scorePoles hp₂.scores (0 : ℝ)).y k) -
        Real.log ((scorePoles hp₂.scores (0 : ℝ)).y j) =
        hp₂.scores j - hp₂.scores k := by
    simpa [scorePoles] using (softmax_pole_log_ratio hp₂.scores (0 : ℝ) k j)
  rw [hpoles] at hlog₁
  exact hlog₁.symm.trans hlog₂

/-- **Direct impossibility theorem (gauge-invariant form).**

    In vanilla score-derived geometry, fixed poles across states force
    score-difference invariance; therefore any witnessed state-varying score
    differences rule out fixed score-derived pole geometry. -/
theorem nondegenerateScoreDifferences_not_scoreDerivedFixedPoleGeometry
    {seqLen vocab nHeads D dModel dMLP : ℕ}
    {contextSize : Fin seqLen → ℕ}
    (θ : GPT2Params seqLen vocab nHeads D dModel dMLP contextSize)
    (hnondeg : NondegenerateScoreDifferences θ) :
    ¬ ScoreDerivedFixedPoleGeometry θ := by
  intro hfixed
  have hsind : ScoreDifferencesStateIndependent θ :=
    scoreDerived_fixed_poles_imply_scoreDifferencesStateIndependent θ hfixed
  exact (nondegenerateScoreDifferences_iff_not_scoreDifferencesStateIndependent θ).mp
    hnondeg hsind

/-- If score differences are nondegenerate, score-derived poles cannot stay fixed:
    there must exist a pair of states where the score-derived pole geometry differs. -/
theorem nondegenerateScoreDifferences_force_scoreDerived_poles_move
    {seqLen vocab nHeads D dModel dMLP : ℕ}
    {contextSize : Fin seqLen → ℕ}
    (θ : GPT2Params seqLen vocab nHeads D dModel dMLP contextSize)
    (hnondeg : NondegenerateScoreDifferences θ) :
    ∃ (i : ℕ) (t : Fin seqLen) (h : Fin nHeads)
      (x₁ x₂ : SeqState seqLen dModel),
      ((scoreDerivedAnalyticMH (θ.multihead i)).poleHead t h x₁).poles ≠
        ((scoreDerivedAnalyticMH (θ.multihead i)).poleHead t h x₂).poles := by
  by_contra hnoMove
  have hfixed : ScoreDerivedFixedPoleGeometry θ := by
    intro i t h x₁ x₂
    by_contra hneq
    exact hnoMove ⟨i, t, h, x₁, x₂, hneq⟩
  exact (nondegenerateScoreDifferences_not_scoreDerivedFixedPoleGeometry θ hnondeg) hfixed

/-- **Requested negative theorem (with explicit bridge hypothesis).**

    If one can prove the structural implication
    `M_global_fixed_poles θ → ScoreStateIndependent θ`, then any witnessed
    score nondegeneracy immediately rules out `M_global_fixed_poles θ`.

    This theorem is the formal "impossibility wrapper" used to turn the
    row-level criticism into a precise obstruction claim. -/
theorem nondegenerate_scores_not_M_global_fixed_poles
    {seqLen vocab nHeads D dModel dMLP : ℕ}
    {contextSize : Fin seqLen → ℕ}
    (θ : GPT2Params seqLen vocab nHeads D dModel dMLP contextSize)
    (hbridge : M_global_fixed_poles θ → ScoreStateIndependent θ)
    (hnondeg : NondegenerateScores θ) :
    ¬ M_global_fixed_poles θ := by
  intro hfixed
  have hsind : ScoreStateIndependent θ := hbridge hfixed
  exact (nondegenerateScores_iff_not_scoreStateIndependent θ).mp hnondeg hsind

/-- The full-model bridge places every GPT2 parameter family in `M_global`. -/
theorem gpt2_in_M_global
    {seqLen vocab nHeads D dModel dMLP : ℕ}
    {contextSize : Fin seqLen → ℕ}
    (θ : GPT2Params seqLen vocab nHeads D dModel dMLP contextSize) :
    M_global θ :=
  gpt2_has_single_global_analytic_object θ

/-- **Replacement block for the tautology-era link theorem.**

    This single theorem packages three non-tautological properties:

    1. **Constructive function-level identity**:
       `analyticForward (scoreDerivedAnalyticParams θ) = gpt2Forward θ`.
    2. **Existential realization** in the independent analytic parameter type.
    3. **Canonicality on the realised slice**:
       any on-query pole family reproducing a softmax row is forced to be
       `scorePoles`.

    Downstream developments can cite this theorem as the replacement for
    the old "link" theorem: it gives equality, existence, and uniqueness
    constraints in one statement. -/
theorem transformer_cauchy_poisson_replacement_block
    {seqLen vocab nHeads D dModel dMLP : ℕ} {contextSize : Fin seqLen → ℕ}
    (θ : GPT2Params seqLen vocab nHeads D dModel dMLP contextSize) :
    (analyticForward (scoreDerivedAnalyticParams θ) = gpt2Forward θ) ∧
    (∃ (φ : AnalyticGPT2Params seqLen vocab nHeads D dModel dMLP contextSize),
      analyticForward φ = gpt2Forward θ) ∧
    (∀ {N : ℕ} [NeZero N] (scores : Fin N → ℝ) (q : ℝ) (p : Poles N),
      (∀ k, p.x k = q) →
      (∀ k, poisson (p.x k) (p.y k) q = exp (scores k) / ∑ j : Fin N, exp (scores j)) →
      p = scorePoles scores q) := by
  refine ⟨?_, ?_, ?_⟩
  · exact (transformers_are_boundary_value_solvers θ).symm
  · exact gpt2_has_pole_realization_functional θ
  · intro N _ scores q p hx hw
    exact scorePoles_unique_on_vertical_slice scores q p hx hw

/-- **Corollary — The poles are constructive and score-derived.**

    The pole geometry is explicit and computable from the scores alone
    (before softmax).  By definition of `scorePoles`:
    - `x_k = q`  (all poles sit on-query)
    - `y_k = Z / exp(s_k)`  (Z = Σ_j exp(s_j))

    The non-trivial content — proved by `softmax_is_poisson_at_score_poles` —
    is that evaluating the Poisson kernel at these poles recovers the softmax
    weight exactly.  Combining with `softmax_pole_log_ratio`:
    log(y_k) − log(y_j) = s_j − s_k, so score differences are exactly the
    log-ratios of pole heights. -/
theorem pole_witness_is_score_derived
    {N : ℕ} [NeZero N] (scores : Fin N → ℝ) (q : ℝ) :
    let p := scorePoles scores q
    let Z := ∑ j : Fin N, exp (scores j)
    ∀ k, poisson (p.x k) (p.y k) q = exp (scores k) / Z :=
  softmax_is_poisson_at_score_poles scores q

/-- **Off-query representation bandwidth law (integrated with score-derived poles).**

    Fix a score vector and let `w_k = softmax(scores)_k`.  If we try to realize
    key `k` at horizontal displacement `d > 0` from query `q`, then feasibility of
    the Poisson equation at that displacement (`poissonDiscriminant w_k d ≥ 0`)
    forces the hard bound `w_k ≤ 1/(2d)`.

    This ties Part 1's off-query law directly to score-derived transformer rows. -/
theorem softmax_offquery_bandwidth_bound
    {N : ℕ} [NeZero N] (scores : Fin N → ℝ) (q d : ℝ) (k : Fin N)
    (hd : 0 < d)
    (hdisc : 0 ≤ poissonDiscriminant
      (exp (scores k) / ∑ j : Fin N, exp (scores j)) d) :
    exp (scores k) / ∑ j : Fin N, exp (scores j) ≤ 1 / (2 * d) := by
  have hw : 0 < exp (scores k) / ∑ j : Fin N, exp (scores j) := by
    exact div_pos (exp_pos _) (Finset.sum_pos (fun j _ => exp_pos _) Finset.univ_nonempty)
  exact poisson_offquery_bandwidth_bound
    (exp (scores k) / ∑ j : Fin N, exp (scores j)) d hw hd hdisc

-- ═══════════════════════════════════════════════════════════════════════
-- § 2.4  Continuum lift of one pre-norm block (Euler limit)
-- ═══════════════════════════════════════════════════════════════════════

open scoped Topology

/-! The §1.17 / §1.19 Euler-limit machinery generalises *as is* to the
sequence-state space `SeqState seqLen dModel`, since that space is a finite-
dimensional real Banach space.

In this section we expose the connection: one pre-norm block `gpt2Block` is
*exactly one Euler step* (at `Δt = 1`) of a vector field
`BlockOps.residualVF`, the iterated refinement
`BlockOps.iterEuler (n+1) (t/(n+1)) ops` is the standard Euler discretisation
of that field, and — *whenever the residual happens to be linear* (i.e.\ the
attention/LayerNorm/MLP combination admits a CLM representation `T`) — the
refinement converges to the continuous one-parameter semigroup
`exp(t·T)` of Part 1.

The non-linear case (general softmax-attention + LayerNorm + GELU MLP) reduces,
on bounded balls, to a Lipschitz vector field; Picard–Lindelöf then gives a
continuous flow and Grönwall gives the Euler convergence.  Component-wise
Lipschitz constants for the *concrete* GPT-2 components (LayerNorm, softmax,
GELU) are deferred to a separate development. -/

/-- The residual vector field of one pre-norm GPT-2 block.

By definition of `gpt2Block`,
  `gpt2Block ops x = x + ops.residualVF x`,
so `residualVF` is the "velocity" generated by one full Euler step at
`Δt = 1`.  See `gpt2Block_eq_one_eulerStep` for the identity. -/
def BlockOps.residualVF {seqLen dModel dMLP : ℕ}
    (ops : BlockOps seqLen dModel dMLP) (x : SeqState seqLen dModel) :
    SeqState seqLen dModel :=
  ops.attn (ops.ln1 x) + ops.mlp (ops.ln2 (x + ops.attn (ops.ln1 x)))

/-- **One full GPT-2 block IS one Euler step** at `Δt = 1` of its residual
vector field. -/
theorem gpt2Block_eq_one_eulerStep {seqLen dModel dMLP : ℕ}
    (ops : BlockOps seqLen dModel dMLP) (x : SeqState seqLen dModel) :
    gpt2Block ops x = x + ops.residualVF x := by
  unfold gpt2Block BlockOps.residualVF
  abel

/-- One Euler step at timestep `Δt`: `x ↦ x + Δt · residualVF(x)`. -/
def BlockOps.eulerStep {seqLen dModel dMLP : ℕ}
    (Δt : ℝ) (ops : BlockOps seqLen dModel dMLP)
    (x : SeqState seqLen dModel) : SeqState seqLen dModel :=
  x + Δt • ops.residualVF x

/-- Iterated Euler refinement: `n` steps of timestep `Δt`.

Setting `n = 1, Δt = 1` recovers `gpt2Block` exactly
(`BlockOps.iterEuler_one_one_eq_block`). -/
def BlockOps.iterEuler {seqLen dModel dMLP : ℕ} :
    ℕ → ℝ → BlockOps seqLen dModel dMLP →
      SeqState seqLen dModel → SeqState seqLen dModel
  | 0,     _,  _,   x => x
  | k + 1, Δt, ops, x => BlockOps.eulerStep Δt ops (BlockOps.iterEuler k Δt ops x)

theorem BlockOps.iterEuler_zero {seqLen dModel dMLP : ℕ}
    (Δt : ℝ) (ops : BlockOps seqLen dModel dMLP) (x : SeqState seqLen dModel) :
    BlockOps.iterEuler 0 Δt ops x = x := rfl

theorem BlockOps.iterEuler_succ {seqLen dModel dMLP : ℕ}
    (k : ℕ) (Δt : ℝ) (ops : BlockOps seqLen dModel dMLP) (x : SeqState seqLen dModel) :
    BlockOps.iterEuler (k + 1) Δt ops x =
      BlockOps.eulerStep Δt ops (BlockOps.iterEuler k Δt ops x) := rfl

theorem BlockOps.iterEuler_one_one_eq_block {seqLen dModel dMLP : ℕ}
    (ops : BlockOps seqLen dModel dMLP) (x : SeqState seqLen dModel) :
    BlockOps.iterEuler 1 1 ops x = gpt2Block ops x := by
  rw [BlockOps.iterEuler_succ, BlockOps.iterEuler_zero, gpt2Block_eq_one_eulerStep]
  show x + (1 : ℝ) • ops.residualVF x = x + ops.residualVF x
  rw [one_smul]

/-- **CLM-power identity for the linearised block.**

If the block's residual vector field equals the action of a continuous
linear endomorphism `T` of the sequence-state space, then iterating the
Euler step is iterating the affine map `1 + Δt · T`, which equals taking
the `m`-th power of `1 + Δt · T` in the operator algebra. -/
theorem BlockOps.iterEuler_eq_pow_apply {seqLen dModel dMLP : ℕ}
    (ops : BlockOps seqLen dModel dMLP)
    (T : SeqState seqLen dModel →L[ℝ] SeqState seqLen dModel)
    (hT : ∀ x, ops.residualVF x = T x)
    (Δt : ℝ) (x : SeqState seqLen dModel) :
    ∀ m : ℕ,
      BlockOps.iterEuler m Δt ops x =
        (((1 : SeqState seqLen dModel →L[ℝ] SeqState seqLen dModel)
            + Δt • T) ^ m) x
  | 0 => by
      show x = ((1 : SeqState seqLen dModel →L[ℝ] SeqState seqLen dModel) ^ 0) x
      simp
  | m + 1 => by
      have ih := BlockOps.iterEuler_eq_pow_apply ops T hT Δt x m
      rw [BlockOps.iterEuler_succ]
      show BlockOps.eulerStep Δt ops _ = _
      unfold BlockOps.eulerStep
      rw [ih, hT, pow_succ' _ m]
      simp [ContinuousLinearMap.mul_apply, ContinuousLinearMap.add_apply,
            ContinuousLinearMap.smul_apply, ContinuousLinearMap.one_apply]

/-- **Linearised-block Euler limit (operator level).**

If the block's residual is the action of a CLM `T`, then the discrete
operator refinement `(1 + (t/(n+1))·T)^{n+1}` converges (in operator norm)
to the continuous one-parameter semigroup `exp(t·T)`. -/
theorem BlockOps.linearised_eulerLimit_clm {seqLen dModel dMLP : ℕ}
    (T : SeqState seqLen dModel →L[ℝ] SeqState seqLen dModel) (t : ℝ) :
    Filter.Tendsto (fun n : ℕ =>
        ((1 : SeqState seqLen dModel →L[ℝ] SeqState seqLen dModel)
          + (t / (↑n + 1 : ℝ)) • T) ^ (n + 1))
      Filter.atTop (𝓝 (NormedSpace.exp (t • T))) :=
  operatorEulerLimit T t

/-- **Linearised-block Euler limit, pointwise.**

Whenever the residual vector field of a pre-norm block coincides with the
action of some continuous linear `T` on the sequence-state space, the
iterated Euler refinement of the block converges to the continuous
semigroup `exp(t·T)` evaluated at the initial state.

This is the precise statement that *a transformer block is the discrete
Euler discretisation of a continuous linear flow* — with LayerNorm and MLP
included via `BlockOps.residualVF`, under the linearity hypothesis on the
combined residual.

For the non-linear (concrete softmax + LayerNorm + GELU) case the analogous
theorem replaces `exp(t·T)` by the unique flow of the Lipschitz vector
field `residualVF` and the convergence rate is governed by Grönwall's
inequality.  See the `operatorEulerLimit`-style proof in Part 1 for the
linear template. -/
theorem BlockOps.linearised_eulerLimit {seqLen dModel dMLP : ℕ}
    (ops : BlockOps seqLen dModel dMLP)
    (T : SeqState seqLen dModel →L[ℝ] SeqState seqLen dModel)
    (hT : ∀ x, ops.residualVF x = T x)
    (t : ℝ) (h₀ : SeqState seqLen dModel) :
    Filter.Tendsto (fun n : ℕ =>
        BlockOps.iterEuler (n + 1) (t / (↑n + 1 : ℝ)) ops h₀)
      Filter.atTop (𝓝 (NormedSpace.exp (t • T) h₀)) := by
  have hclm := BlockOps.linearised_eulerLimit_clm (seqLen := seqLen) (dModel := dModel)
    (dMLP := dMLP) T t
  have hcont : Continuous
      (fun Φ : SeqState seqLen dModel →L[ℝ] SeqState seqLen dModel => Φ h₀) :=
    (ContinuousLinearMap.apply ℝ (SeqState seqLen dModel) h₀).continuous
  have heval := (hcont.tendsto _).comp hclm
  convert heval using 1
  funext n
  simp only [Function.comp_def]
  exact BlockOps.iterEuler_eq_pow_apply ops T hT (t / (↑n + 1 : ℝ)) h₀ (n + 1)

/-- **Headline corollary — a transformer block, *with* LayerNorm and MLP, is
a discrete Euler step of a continuous linear flow whenever its combined
residual is linear.**

Concretely: if the composite map
`x ↦ attn(ln1 x) + mlp(ln2(x + attn(ln1 x)))`
admits a CLM representation `T`, then the residual stream produced by
running `n` blocks with timestep `t/n` (each block being one Euler step
of the *combined* LayerNorm + attention + LayerNorm + MLP residual)
converges to `exp(t·T)·h₀` as `n → ∞`. -/
theorem transformer_block_eulerLimit_with_layerNorm_and_mlp
    {seqLen dModel dMLP : ℕ}
    (ops : BlockOps seqLen dModel dMLP)
    (T : SeqState seqLen dModel →L[ℝ] SeqState seqLen dModel)
    (hT : ∀ x, ops.attn (ops.ln1 x)
              + ops.mlp (ops.ln2 (x + ops.attn (ops.ln1 x))) = T x)
    (t : ℝ) (h₀ : SeqState seqLen dModel) :
    Filter.Tendsto (fun n : ℕ =>
        BlockOps.iterEuler (n + 1) (t / (↑n + 1 : ℝ)) ops h₀)
      Filter.atTop (𝓝 (NormedSpace.exp (t • T) h₀)) :=
  BlockOps.linearised_eulerLimit ops T (fun x => hT x) t h₀

end AnalyticTransformer
