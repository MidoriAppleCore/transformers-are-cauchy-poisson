import Mathlib

/-!
# Cauchy theory: softmax as a forced kernel.

## Scope: single-orbit geometry, not multi-token products

This file is **rigorous on one affine homogeneous story**; it is **not** a Lean
model of “`N` independent keys/poles in general position” inside a single
`KernelClassification`.  Concretely:

* **`KernelClassification.transitive`** means: one base pair `(d₀, q₀)` and one
  group `G` reach **every** `(d, q)` in the formalised domain.  That is a **single
  orbit** (e.g. one pole height and one boundary query swept by `AffinePos`), not
  a product of `N` unrelated `(dᵢ, qᵢ)` patches.
* **`eisenstein_sum_factors`** sums over a `Finset G` — group elements acting on
  **one** fixed `(d, q)`.  It is **not** a substitute for summing attention over
  `Fin N` arbitrary tokens unless you add extra structure identifying keys with
  such a group family.
* **Section 1** (`softmax_is_poisson_at_scoreHeight`) uses `Fin N` only for the **softmax
  algebra** (scores and the partition sum).  That theorem does **not** extend the
  classification in later sections to an `N`-body independent-pole space.

Extending this spine to `N` unrelated geometric “engines” (e.g. per-token poles)
requires a separate formalisation: a product structure, a family of patches, or
partial orbits — not proved here.

---

The story this file proves, in plain English.

We start from a tautology: a softmax weight `w_k = exp(s_k) / Σ exp(s_j)` is its
own *inverse pole height*.  If you place a pole at `y_k = 1 / w_k` directly above
the query, the **Poisson kernel** evaluated at that pole gives back the softmax
weight.  Geometrically: every softmax weight pins exactly one point on a
vertical line in the upper half-plane.

That observation is rigid.  Once you say "the kernel transforms covariantly
under a group `G` acting on a domain `D` and a boundary `B`, with cocycle `j`,
and equals 1 at one chosen basepoint," the kernel is **forced**.  This is the
abstract content of `KernelClassification.unique`: covariance + reachability +
one normalisation pin the kernel pointwise.  On the same affine group action we
discharge **two** rows: the covariant pole-height kernel `K(y,q) = y`
(`affineSlice`, cocycle `g ↦ g.a`) and the inverse-height / softmax-scale kernel
`W(y,q) = 1/y` (`affineInverseSlice`, cocycle `g ↦ 1/g.a`).  The Section 1 Poisson
identity `poisson q y q = 1/y` is exactly `W` evaluated on the vertical through
`q`, not `K`.

The schema is small enough to compose.  A morphism between two rows pulls the
kernel back along the morphism (`kernel_pullback`); the concrete witness here
is `trivial → affine`, where the singleton row maps to the affine basepoint.
Pointwise product of two covariant kernels is covariant for the product
cocycle (`covariance_product`, the Hecke-style operation).  Finite sums along
a group orbit factor as `(Σ cocycle) · K` (`eisenstein_sum_factors`,
specialised to the affine row in `affineKernel_eisenstein`).

Finally we record the small atlas of discharged rows in `CPAtlas` (both affine
rows).

What is NOT in this file: the full `PSL(2, ℝ)` Möbius classification, the Hardy
bridge, GPT-2 forward identification, the RoPE obstruction, and the continuum
semigroup — those live in the multi-file project.  This is the rigidity spine
the catalogue is about.
-/

noncomputable section

namespace CauchyTheory

open Finset Real

-- ════════════════════════════════════════════════════════════════════════════
-- Section 1.  Poisson kernel and the softmax pinning tautology
-- ════════════════════════════════════════════════════════════════════════════

/-- The classical Poisson kernel `P(x, y, q) = y / ((q - x)² + y²)`. -/
def poisson (x y q : ℝ) : ℝ := y / ((q - x) ^ 2 + y ^ 2)

theorem poisson_at_query (q : ℝ) {y : ℝ} (hy : 0 < y) :
    poisson q y q = 1 / y := by
  have hyne : y ≠ 0 := ne_of_gt hy
  unfold poisson
  field_simp
  ring

/-- The pinning identity: the softmax weight of token `k` equals the Poisson
kernel evaluated at the score-derived pole height `y_k = (Σ exp s_j) / exp s_k`,
with the pole placed directly above the query (`x_k = q`).

**Scope:** `Fin N` indexes scores only (partition-sum algebra).  This is **not** the
multi-pole `KernelClassification` of later sections; see module doc “Scope: single-orbit
geometry.” -/
theorem softmax_is_poisson_at_scoreHeight
    {N : ℕ} (scores : Fin N → ℝ) (q : ℝ) (k : Fin N) :
    poisson q ((∑ j, exp (scores j)) / exp (scores k)) q
      = exp (scores k) / ∑ j, exp (scores j) := by
  have hZ : (0 : ℝ) < ∑ j, exp (scores j) :=
    Finset.sum_pos (fun _ _ => Real.exp_pos _) ⟨k, Finset.mem_univ _⟩
  have hy : (0 : ℝ) < (∑ j, exp (scores j)) / exp (scores k) :=
    div_pos hZ (Real.exp_pos _)
  rw [poisson_at_query _ hy, one_div, inv_div]

-- ════════════════════════════════════════════════════════════════════════════
-- Section 2.  KernelClassification schema and its uniqueness theorem
-- ════════════════════════════════════════════════════════════════════════════

/-- Symmetry data + three hypotheses that force the kernel pointwise.

**Remark:** the schema is one `(D, B, G)` setting with a single reachability
hypothesis `transitive`; it does not by itself model `N` independent geometric factors. -/
structure KernelClassification (D B G : Type*) where
  action     : G → D → D
  bdy_action : G → B → B
  cocycle    : G → D → B → ℝ
  kernel     : D → B → ℝ
  basepoint  : D
  base_query : B
  covariant  : ∀ g d q, kernel (action g d) (bdy_action g q) = cocycle g d q * kernel d q
  /-- Every `(d, q)` lies in the **same** `G`-orbit as `(basepoint, base_query)`.
  Uniqueness is rigidity on this homogeneous patch — **not** a theorem about `N`
  unrelated token poles. -/
  transitive : ∀ d q, ∃ g, action g basepoint = d ∧ bdy_action g base_query = q
  normalized : kernel basepoint base_query = 1

/-- Any covariant normalised candidate `K'` agrees with `K.kernel` pointwise. -/
theorem KernelClassification.unique
    {D B G : Type*} (K : KernelClassification D B G) (K' : D → B → ℝ)
    (covar' : ∀ g d q,
      K' (K.action g d) (K.bdy_action g q) = K.cocycle g d q * K' d q)
    (norm' : K' K.basepoint K.base_query = 1) :
    ∀ d q, K' d q = K.kernel d q := by
  intro d q
  obtain ⟨g, hd, hq⟩ := K.transitive d q
  have h1 := covar' g K.basepoint K.base_query
  have h2 := K.covariant g K.basepoint K.base_query
  rw [hd, hq] at h1 h2
  rw [h1, h2, norm', K.normalized]

-- ════════════════════════════════════════════════════════════════════════════
-- Section 3.  The trivial (terminal) classification
-- ════════════════════════════════════════════════════════════════════════════

/-- Trivial row on a singleton; serves as the morphism source in Section 5. -/
def trivialClassification : KernelClassification Unit Unit Unit where
  action     := fun _ _ => ()
  bdy_action := fun _ _ => ()
  cocycle    := fun _ _ _ => 1
  kernel     := fun _ _ => 1
  basepoint  := ()
  base_query := ()
  covariant  := by intros; ring
  transitive := by intros; exact ⟨(), rfl, rfl⟩
  normalized := rfl

-- ════════════════════════════════════════════════════════════════════════════
-- Section 4.  The affine softmax row (discharged KernelClassification)
-- ════════════════════════════════════════════════════════════════════════════

/-- `AffinePos = ℝ_{>0} ⋉ ℝ`, the dilation–translation semidirect product. -/
@[ext]
structure AffinePos where
  a : ℝ
  b : ℝ
  ha : 0 < a

namespace AffinePos

def mul (g h : AffinePos) : AffinePos :=
  ⟨g.a * h.a, g.a * h.b + g.b, mul_pos g.ha h.ha⟩

def one : AffinePos := ⟨1, 0, zero_lt_one⟩

@[simp] lemma mul_a (g h : AffinePos) : (mul g h).a = g.a * h.a := rfl
@[simp] lemma mul_b (g h : AffinePos) : (mul g h).b = g.a * h.b + g.b := rfl
@[simp] lemma one_a : one.a = 1 := rfl
@[simp] lemma one_b : one.b = 0 := rfl

instance : Mul AffinePos := ⟨mul⟩
instance : One AffinePos := ⟨one⟩

@[simp] lemma mul_def (g h : AffinePos) : g * h = mul g h := rfl
@[simp] lemma one_def : (1 : AffinePos) = one := rfl

end AffinePos

/-- Positive pole heights `{ y : ℝ | 0 < y }`. -/
abbrev AffineDom := { y : ℝ // 0 < y }

def affineDomAction (g : AffinePos) (d : AffineDom) : AffineDom :=
  ⟨g.a * d.val, mul_pos g.ha d.prop⟩

def affineBdyAction (g : AffinePos) (q : ℝ) : ℝ := g.a * q + g.b

def affineCocycle (g : AffinePos) (_d : AffineDom) (_q : ℝ) : ℝ := g.a

def affineKernel (d : AffineDom) (_q : ℝ) : ℝ := d.val

/-- Every `(d, q)` is reached from the affine basepoint `(1, 0)` by some
`g ∈ AffinePos` (same action as in `affineSlice`). -/
theorem affine_transitive (d : AffineDom) (q : ℝ) :
    ∃ g : AffinePos,
      affineDomAction g ⟨1, zero_lt_one⟩ = d ∧ affineBdyAction g (0 : ℝ) = q := by
  refine ⟨{ a := d.val, b := q, ha := d.prop }, ?_, ?_⟩
  · apply Subtype.ext; simp [affineDomAction]
  · simp [affineBdyAction]

/-- The affine row discharged: covariant kernel `K(y, q) = y` (cocycle `g ↦ g.a`). -/
def affineSlice : KernelClassification AffineDom ℝ AffinePos where
  action     := affineDomAction
  bdy_action := affineBdyAction
  cocycle    := affineCocycle
  kernel     := affineKernel
  basepoint  := ⟨1, zero_lt_one⟩
  base_query := 0
  covariant  := by
    intros
    simp [affineDomAction, affineCocycle, affineKernel]
  transitive := by intro d q; exact affine_transitive d q
  normalized := rfl

/-- Dual cocycle `g ↦ (g.a)⁻¹` for the inverse-height / softmax-scale row. -/
def affineCocycleInv (g : AffinePos) (_d : AffineDom) (_q : ℝ) : ℝ := (g.a)⁻¹

/-- Kernel `W(y,q) = y⁻¹`: transforms with the **inverse** affine cocycle. -/
def inverseAffineKernel (d : AffineDom) (_q : ℝ) : ℝ := (d.val)⁻¹

/-- Affine inverse row: same `AffinePos` action, kernel `1/y`, cocycle `1/g.a`. -/
def affineInverseSlice : KernelClassification AffineDom ℝ AffinePos where
  action     := affineDomAction
  bdy_action := affineBdyAction
  cocycle    := affineCocycleInv
  kernel     := inverseAffineKernel
  basepoint  := ⟨1, zero_lt_one⟩
  base_query := 0
  covariant  := by
    intro g d q
    simp only [affineDomAction, inverseAffineKernel, affineCocycleInv, Subtype.coe_mk]
    have hg : g.a ≠ 0 := ne_of_gt g.ha
    have hd : d.val ≠ 0 := ne_of_gt d.prop
    field_simp [mul_ne_zero hg hd]
  transitive := by intro d q; exact affine_transitive d q
  normalized := by simp [inverseAffineKernel]

/-- Affine covariance + normalisation force `K' d q = d.val`. -/
theorem affine_slice_unique (K' : AffineDom → ℝ → ℝ)
    (cov : ∀ g d q,
      K' (affineDomAction g d) (affineBdyAction g q) = affineCocycle g d q * K' d q)
    (norm : K' ⟨1, zero_lt_one⟩ 0 = 1) :
    ∀ d q, K' d q = d.val := by
  intro d q
  have h := affineSlice.unique K' cov norm d q
  simpa [affineSlice, affineKernel] using h

/-- Covariance + normalisation force the inverse-height kernel `W' d q = (d.val)⁻¹`. -/
theorem affine_inverse_slice_unique (W' : AffineDom → ℝ → ℝ)
    (cov : ∀ g d q,
      W' (affineDomAction g d) (affineBdyAction g q) =
        affineCocycleInv g d q * W' d q)
    (norm : W' ⟨1, zero_lt_one⟩ 0 = 1) :
    ∀ d q, W' d q = (d.val)⁻¹ := by
  intro d q
  have h := affineInverseSlice.unique W' cov norm d q
  simpa [affineInverseSlice, inverseAffineKernel] using h

/-- On the vertical through `q`, `W` agrees with the Poisson kernel at height `y`. -/
theorem inverseAffineKernel_eq_poisson_at_query (q : ℝ) (d : AffineDom) :
    inverseAffineKernel d q = poisson q d.val q := by
  simp [inverseAffineKernel, poisson_at_query q d.prop]

-- ════════════════════════════════════════════════════════════════════════════
-- Section 5.  Functoriality: morphisms of classifications and a concrete witness
-- ════════════════════════════════════════════════════════════════════════════

/-- Morphism of rows: maps on `(D, B, G)` intertwining all schema fields. -/
structure KernelClassificationHom
    {D₁ B₁ G₁ D₂ B₂ G₂ : Type*}
    (K₁ : KernelClassification D₁ B₁ G₁)
    (K₂ : KernelClassification D₂ B₂ G₂) where
  fD : D₁ → D₂
  fB : B₁ → B₂
  fG : G₁ → G₂
  preserves_action     : ∀ g d, fD (K₁.action g d) = K₂.action (fG g) (fD d)
  preserves_bdy_action : ∀ g q, fB (K₁.bdy_action g q) = K₂.bdy_action (fG g) (fB q)
  preserves_cocycle    : ∀ g d q, K₁.cocycle g d q = K₂.cocycle (fG g) (fD d) (fB q)
  preserves_basepoint  : fD K₁.basepoint = K₂.basepoint
  preserves_base_query : fB K₁.base_query = K₂.base_query

/-- A morphism `φ : K₁ ⟶ K₂` pulls back: `K₂ ∘ (φ.fD, φ.fB) = K₁.kernel`. -/
theorem KernelClassificationHom.kernel_pullback
    {D₁ B₁ G₁ D₂ B₂ G₂ : Type*}
    {K₁ : KernelClassification D₁ B₁ G₁}
    {K₂ : KernelClassification D₂ B₂ G₂}
    (φ : KernelClassificationHom K₁ K₂) :
    ∀ d q, K₂.kernel (φ.fD d) (φ.fB q) = K₁.kernel d q := by
  refine K₁.unique (fun d q => K₂.kernel (φ.fD d) (φ.fB q)) ?_ ?_
  · intro g d q
    show K₂.kernel (φ.fD (K₁.action g d)) (φ.fB (K₁.bdy_action g q))
          = K₁.cocycle g d q * K₂.kernel (φ.fD d) (φ.fB q)
    rw [φ.preserves_action g d, φ.preserves_bdy_action g q, K₂.covariant,
        φ.preserves_cocycle g d q]
  · show K₂.kernel (φ.fD K₁.basepoint) (φ.fB K₁.base_query) = 1
    rw [φ.preserves_basepoint, φ.preserves_base_query]
    exact K₂.normalized

/-- Concrete morphism `trivial → affine`: send every singleton to the affine
basepoint and to the identity element. -/
def trivialToAffine :
    KernelClassificationHom trivialClassification affineSlice where
  fD _ := ⟨1, zero_lt_one⟩
  fB _ := (0 : ℝ)
  fG _ := (1 : AffinePos)
  preserves_action := by
    intro _ _
    apply Subtype.ext
    show (1 : ℝ) = (affineSlice.action (1 : AffinePos) ⟨1, zero_lt_one⟩).val
    simp [affineSlice, affineDomAction]
  preserves_bdy_action := by
    intro _ _
    show (0 : ℝ) = affineSlice.bdy_action (1 : AffinePos) 0
    simp [affineSlice, affineBdyAction]
  preserves_cocycle := by
    intro _ _ _
    show (1 : ℝ) = affineSlice.cocycle (1 : AffinePos) ⟨1, zero_lt_one⟩ 0
    simp [affineSlice, affineCocycle]
  preserves_basepoint := rfl
  preserves_base_query := rfl

-- ════════════════════════════════════════════════════════════════════════════
-- Section 6.  Automorphy factors and Hecke products
-- ════════════════════════════════════════════════════════════════════════════

/-- Automorphy factor for an action: `j(1, x) = 1` and the cocycle identity. -/
structure IsAutomorphyFactor
    {G X : Type*} [One G] [Mul G]
    (action : G → X → X) (j : G → X → ℝ) : Prop where
  one : ∀ x, j 1 x = 1
  mul : ∀ g h x, j (g * h) x = j g (action h x) * j h x

/-- Pointwise product of automorphy factors is an automorphy factor. -/
theorem IsAutomorphyFactor.mul_factors
    {G X : Type*} [One G] [Mul G]
    {action : G → X → X} {j₁ j₂ : G → X → ℝ}
    (h₁ : IsAutomorphyFactor action j₁) (h₂ : IsAutomorphyFactor action j₂) :
    IsAutomorphyFactor action (fun g x => j₁ g x * j₂ g x) where
  one := by intro x; simp [h₁.one x, h₂.one x]
  mul := by
    intro g h x
    have e₁ := h₁.mul g h x
    have e₂ := h₂.mul g h x
    calc j₁ (g * h) x * j₂ (g * h) x
        = (j₁ g (action h x) * j₁ h x) * (j₂ g (action h x) * j₂ h x) := by rw [e₁, e₂]
      _ = (j₁ g (action h x) * j₂ g (action h x)) * (j₁ h x * j₂ h x) := by ring

/-- The affine cocycle `g ↦ g.a` is an automorphy factor. -/
theorem affineCocycle_isAutomorphyFactor :
    IsAutomorphyFactor (fun g d => affineDomAction g d)
        (fun (g : AffinePos) (_d : AffineDom) => g.a) where
  one := by intro; simp
  mul := by intro g h d; simp

/-- Pointwise product of `G`-covariant kernels is covariant for the product
cocycle (the Hecke-style operation on attention kernels). -/
theorem covariance_product
    {D B G : Type*}
    (action : G → D → D) (bdy_action : G → B → B)
    (jK jL : G → D → B → ℝ)
    (K L : D → B → ℝ)
    (covK : ∀ g d q, K (action g d) (bdy_action g q) = jK g d q * K d q)
    (covL : ∀ g d q, L (action g d) (bdy_action g q) = jL g d q * L d q) :
    ∀ g d q,
      K (action g d) (bdy_action g q) * L (action g d) (bdy_action g q)
        = (jK g d q * jL g d q) * (K d q * L d q) := by
  intro g d q
  rw [covK, covL]; ring

-- ════════════════════════════════════════════════════════════════════════════
-- Section 7.  Eisenstein finite-orbit factorisation
-- ════════════════════════════════════════════════════════════════════════════

/-- **Finite Eisenstein-style sum.**  For any finset `s : Finset G`,
`Σ γ ∈ s, K(γ·d, γ·q) = (Σ γ ∈ s, jK(γ, d, q)) · K(d, q)`.

**Remark:** the sum is indexed by **group elements** acting on one base `(d, q)`;
it is not, without further structure, a sum over `Fin N` unrelated attention keys. -/
theorem eisenstein_sum_factors
    {D B G : Type*}
    (action : G → D → D) (bdy_action : G → B → B)
    (jK : G → D → B → ℝ) (K : D → B → ℝ)
    (covK : ∀ g d q, K (action g d) (bdy_action g q) = jK g d q * K d q)
    (s : Finset G) (d : D) (q : B) :
    (∑ γ ∈ s, K (action γ d) (bdy_action γ q))
      = (∑ γ ∈ s, jK γ d q) * K d q := by
  rw [Finset.sum_mul]
  refine Finset.sum_congr rfl ?_
  intro γ _; exact covK γ d q

/-- **Concrete Eisenstein sum on the affine row.** -/
theorem affineKernel_eisenstein
    (s : Finset AffinePos) (d : AffineDom) (q : ℝ) :
    (∑ γ ∈ s, affineKernel (affineDomAction γ d) (affineBdyAction γ q))
      = (∑ γ ∈ s, γ.a) * affineKernel d q := by
  have hcov := affineSlice.covariant
  have h := eisenstein_sum_factors
    affineDomAction affineBdyAction affineCocycle affineKernel
    (by intro g d' q'; simpa [affineSlice] using hcov g d' q') s d q
  simpa [affineCocycle] using h

/-- Eisenstein factorisation on the inverse affine row (`Σ 1/a` as cocycle sum). -/
theorem inverseAffineKernel_eisenstein
    (s : Finset AffinePos) (d : AffineDom) (q : ℝ) :
    (∑ γ ∈ s, inverseAffineKernel (affineDomAction γ d) (affineBdyAction γ q))
      = (∑ γ ∈ s, (γ.a)⁻¹) * inverseAffineKernel d q := by
  have hcov := affineInverseSlice.covariant
  have h := eisenstein_sum_factors
    affineDomAction affineBdyAction affineCocycleInv inverseAffineKernel
    (by intro g d' q'; simpa [affineInverseSlice] using hcov g d' q') s d q
  simpa [affineCocycleInv] using h

-- ════════════════════════════════════════════════════════════════════════════
-- Section 8.  The (small) atlas
-- ════════════════════════════════════════════════════════════════════════════

/-- The atlas of discharged Cauchy–Poisson rows in this file.

**Remark:** both rows use the **same** single-orbit affine geometry; `CPAtlas` does
not formalise `N` simultaneous unrelated pole configurations. -/
structure CPAtlas where
  affine : KernelClassification AffineDom ℝ AffinePos
  affineInverse : KernelClassification AffineDom ℝ AffinePos

/-- Canonical atlas: covariant pole-height row and inverse softmax-scale row. -/
def canonicalAtlas : CPAtlas := ⟨affineSlice, affineInverseSlice⟩

@[simp] theorem canonicalAtlas_affine : canonicalAtlas.affine = affineSlice := rfl

@[simp] theorem canonicalAtlas_affineInverse :
    canonicalAtlas.affineInverse = affineInverseSlice := rfl

end CauchyTheory

end
