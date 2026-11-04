import Paper.TwoVector
import Paper.Rayleigh

set_option linter.style.header false
set_option linter.style.longLine false

/-!
# Theorem 5: every subspace interchange lies in one range

Pairs are indexed by a finite type `ι` with weights `p` (the mean over test pairs is `p i = 1/N`). For pair `i`, `w i`
is the gradient of the log odds `z` in the state at the location, taken at the midpoint of the full swap, and
`Δ i = h_s - h_b` the difference of the states. To first order, a patch `v` changes the output by `⟪w i, v⟫`. The
rank-one interchange along a unit `d` patches `⟪d, Δ i⟫ • d`; the network's own interchange patches `Δ i`. The
first-order effect matrix is `M = sym (∑ p_i w_i Δ_iᵀ)`, equation (1) of the paper. Parts (i), (ii) and (iv) are below;
part (iii) is in `Paper.KyFan`. Eigenvalues follow Mathlib's order: `λ₀` is the paper's `λ₁`, and `λ_{n-1}` its
`λ_D`.
-/

namespace Paper

open scoped RealInnerProductSpace

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]
variable {ι : Type*} [Fintype ι]

/-- The first-order effect matrix `M = sym (∑ᵢ pᵢ wᵢ Δᵢᵀ)`, as a linear map. -/
noncomputable def effM (p : ι → ℝ) (w Δ : ι → E) : E →ₗ[ℝ] E :=
  (1 / 2 : ℝ) • ∑ i, p i • ((innerₛₗ ℝ (Δ i)).smulRight (w i) + (innerₛₗ ℝ (w i)).smulRight (Δ i))

theorem effM_apply (p : ι → ℝ) (w Δ : ι → E) (x : E) :
    effM p w Δ x = (1 / 2 : ℝ) • ∑ i, p i • (⟪Δ i, x⟫ • w i + ⟪w i, x⟫ • Δ i) := by
  simp [effM, LinearMap.sum_apply, LinearMap.smul_apply, LinearMap.add_apply, LinearMap.smulRight_apply,
    innerₛₗ_apply_apply]

/-- `⟪x, M y⟫ = ½ ∑ pᵢ (⟪Δᵢ, y⟫⟪x, wᵢ⟫ + ⟪wᵢ, y⟫⟪x, Δᵢ⟫)`. -/
theorem inner_effM (p : ι → ℝ) (w Δ : ι → E) (x y : E) :
    ⟪x, effM p w Δ y⟫ = (1 / 2 : ℝ) * ∑ i, p i * (⟪Δ i, y⟫ * ⟪x, w i⟫ + ⟪w i, y⟫ * ⟪x, Δ i⟫) := by
  rw [effM_apply, real_inner_smul_right, inner_sum]
  congr 1
  apply Finset.sum_congr rfl
  intro i _
  rw [real_inner_smul_right, inner_add_right, real_inner_smul_right, real_inner_smul_right]

/-- `M` is symmetric. -/
theorem effM_isSymmetric (p : ι → ℝ) (w Δ : ι → E) : (effM p w Δ).IsSymmetric := by
  intro x y
  rw [real_inner_comm, inner_effM, inner_effM]
  congr 1
  apply Finset.sum_congr rfl
  intro i _
  rw [real_inner_comm x (w i), real_inner_comm x (Δ i), real_inner_comm y (w i), real_inner_comm y (Δ i)]
  ring

/-- **Theorem 5(i), the identity.** The mean first-order effect of the rank-one interchange along `d`,
`∑ pᵢ ⟪wᵢ, ⟪d, Δᵢ⟫ d⟫ = ∑ pᵢ ⟪wᵢ, d⟫⟪d, Δᵢ⟫`, is the Rayleigh form `⟪d, M d⟫`. -/
theorem rank_one_effect_eq (p : ι → ℝ) (w Δ : ι → E) (d : E) :
    ∑ i, p i * ⟪w i, ⟪d, Δ i⟫ • d⟫ = ⟪d, effM p w Δ d⟫ := by
  rw [inner_effM, Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro i _
  rw [real_inner_smul_right, real_inner_comm d (w i), real_inner_comm (Δ i) d]
  ring

/-- **Theorem 5(ii), first order.** The mean first-order effect of the network's own interchange (the full swap,
patch `Δᵢ`) is the trace of `M`. -/
theorem trace_effM [FiniteDimensional ℝ E] (p : ι → ℝ) (w Δ : ι → E) :
    LinearMap.trace ℝ E (effM p w Δ) = ∑ i, p i * ⟪w i, Δ i⟫ := by
  set b := stdOrthonormalBasis ℝ E
  rw [LinearMap.trace_eq_sum_inner _ b]
  simp_rw [inner_effM]
  rw [← Finset.mul_sum, Finset.sum_comm]
  have h : ∀ i, ∑ k, p i * (⟪Δ i, b k⟫ * ⟪b k, w i⟫ + ⟪w i, b k⟫ * ⟪b k, Δ i⟫) = 2 * (p i * ⟪w i, Δ i⟫) := by
    intro i
    have h1 := b.sum_inner_mul_inner (Δ i) (w i)
    have h2 := b.sum_inner_mul_inner (w i) (Δ i)
    rw [← Finset.mul_sum, Finset.sum_add_distrib, h1, h2, real_inner_comm (Δ i) (w i)]
    ring
  simp_rw [h]
  rw [← Finset.mul_sum]
  ring

/-- **Theorem 5(ii), the eigenvalue form.** The trace is the sum of the eigenvalues. -/
theorem trace_effM_eq_sum_eigenvalues [FiniteDimensional ℝ E] {n : ℕ} (hn : Module.finrank ℝ E = n)
    (p : ι → ℝ) (w Δ : ι → E) :
    ∑ i, p i * ⟪w i, Δ i⟫ = ∑ j, (effM_isSymmetric p w Δ).eigenvalues hn j := by
  rw [← trace_effM, (effM_isSymmetric p w Δ).trace_eq_sum_eigenvalues hn]
  simp [RCLike.ofReal_real_eq_id]

/-- **Theorem 5(i), the range.** In dimension `n ≥ 2`, the mean first-order effect of a rank-one interchange takes
exactly the values in `[λ_{n-1}, λ₀]` as `d` ranges over unit vectors. -/
theorem rank_one_effect_range [FiniteDimensional ℝ E] {n : ℕ} (hn : Module.finrank ℝ E = n) (h2 : 2 ≤ n)
    (p : ι → ℝ) (w Δ : ι → E) :
    {r | ∃ d : E, ‖d‖ = 1 ∧ ∑ i, p i * ⟪w i, ⟪d, Δ i⟫ • d⟫ = r} =
      Set.Icc ((effM_isSymmetric p w Δ).eigenvalues hn ⟨n - 1, by omega⟩)
        ((effM_isSymmetric p w Δ).eigenvalues hn ⟨0, by omega⟩) := by
  rw [← rayleigh_range hn (effM_isSymmetric p w Δ) h2]
  simp_rw [rank_one_effect_eq]

/-- For a single pair (`ι = Unit`, weight 1) the Rayleigh form is `⟪w, d⟫⟪d, Δ⟫`. -/
theorem rayleigh_single (w Δ d : E) :
    ⟪d, effM (fun _ : Unit => (1 : ℝ)) (fun _ => w) (fun _ => Δ) d⟫ = ⟪w, d⟫ * ⟪d, Δ⟫ := by
  rw [← rank_one_effect_eq]
  simp [real_inner_smul_right]
  ring

/-- **Theorem 5(iv).** For a single pair, the top eigenvalue of `sym(w Δᵀ)` is `(⟪w, Δ⟫ + ‖w‖‖Δ‖)/2` (dimension at
least 2). -/
theorem top_eigenvalue_single [FiniteDimensional ℝ E] {n : ℕ} (hn : Module.finrank ℝ E = n) (h2 : 2 ≤ n) (w Δ : E) :
    (effM_isSymmetric (fun _ : Unit => (1 : ℝ)) (fun _ => w) (fun _ => Δ)).eigenvalues hn ⟨0, by omega⟩ =
      (⟪w, Δ⟫ + ‖w‖ * ‖Δ‖) / 2 := by
  set hT := effM_isSymmetric (fun _ : Unit => (1 : ℝ)) (fun _ => w) (fun _ => Δ)
  have h0 : 0 < n := by omega
  -- the top eigenvalue is the greatest value of the Rayleigh form on the unit sphere
  have hgreat : IsGreatest {r | ∃ d : E, ‖d‖ = 1 ∧ ⟪w, d⟫ * ⟪d, Δ⟫ = r} (hT.eigenvalues hn ⟨0, h0⟩) := by
    refine ⟨⟨hT.eigenvectorBasis hn ⟨0, h0⟩, (hT.eigenvectorBasis hn).orthonormal.1 _, ?_⟩, ?_⟩
    · rw [← rayleigh_single]; exact rayleigh_eigenvector hn hT _
    · rintro r ⟨d, hd, rfl⟩
      rw [← rayleigh_single]
      exact rayleigh_le_top hn hT h0 d hd
  exact hgreat.unique (two_vector_isGreatest (by omega) w Δ)

/-- **Theorem 5(iv), the comparison.** `(⟪w, Δ⟫ + ‖w‖‖Δ‖)/2 ≥ ⟪w, Δ⟫`, with equality if and only if
`‖Δ‖ w = ‖w‖ Δ` (`w` and `Δ` parallel with `⟪w, Δ⟫ ≥ 0`). -/
theorem top_ge_trace_single (w Δ : E) :
    ⟪w, Δ⟫ ≤ (⟪w, Δ⟫ + ‖w‖ * ‖Δ‖) / 2 ∧
      (⟪w, Δ⟫ = (⟪w, Δ⟫ + ‖w‖ * ‖Δ‖) / 2 ↔ ‖Δ‖ • w = ‖w‖ • Δ) := by
  have hcs : ⟪w, Δ⟫ ≤ ‖w‖ * ‖Δ‖ := real_inner_le_norm w Δ
  refine ⟨by linarith, ?_⟩
  rw [← inner_eq_norm_mul_iff_real]
  constructor <;> intro h <;> linarith

end Paper
