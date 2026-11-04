import Mathlib

set_option linter.style.header false
set_option linter.style.longLine false

/-!
# The Rayleigh form of a symmetric map: its range on the unit sphere

For a symmetric linear map `T` on a finite-dimensional real inner product space, with eigenvalues
`λ₀ ≥ λ₁ ≥ … ≥ λ_{n-1}` (Mathlib numbers them from 0 in decreasing order, so `λ₀` and `λ_{n-1}` are the paper's `λ₁`
and `λ_D`), the Rayleigh form `⟪d, T d⟫` over unit vectors `d` takes exactly the values in `[λ_{n-1}, λ₀]` (dimension
at least 2 for the "every value" part).
This is Theorem 5(i) of the paper once `T` is the first-order effect matrix `M` (see `Paper.Range`).
-/

namespace Paper

open scoped RealInnerProductSpace

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] [FiniteDimensional ℝ E]
variable {n : ℕ} (hn : Module.finrank ℝ E = n) {T : E →ₗ[ℝ] E} (hT : T.IsSymmetric)

/-- `⟪e_i, T d⟫ = λ_i ⟪e_i, d⟫` for the eigenvector basis. -/
theorem inner_eigenvector_apply (d : E) (i : Fin n) :
    ⟪hT.eigenvectorBasis hn i, T d⟫ = hT.eigenvalues hn i * ⟪hT.eigenvectorBasis hn i, d⟫ := by
  rw [← hT (hT.eigenvectorBasis hn i) d, hT.apply_eigenvectorBasis hn i, real_inner_smul_left]
  simp

/-- Eigen-expansion of the Rayleigh form. -/
theorem rayleigh_expansion (d : E) :
    ⟪d, T d⟫ = ∑ i, hT.eigenvalues hn i * ⟪hT.eigenvectorBasis hn i, d⟫ ^ 2 := by
  rw [← (hT.eigenvectorBasis hn).sum_inner_mul_inner d (T d)]
  apply Finset.sum_congr rfl
  intro i _
  rw [inner_eigenvector_apply hn hT d i, real_inner_comm d]
  ring

/-- Parseval in the eigenvector basis. -/
theorem sum_sq_inner_eigenvector (d : E) :
    ∑ i, ⟪hT.eigenvectorBasis hn i, d⟫ ^ 2 = ‖d‖ ^ 2 := by
  rw [← real_inner_self_eq_norm_sq, ← (hT.eigenvectorBasis hn).sum_inner_mul_inner d d]
  apply Finset.sum_congr rfl
  intro i _
  rw [real_inner_comm d]
  ring

/-- The top eigenvalue bounds the Rayleigh form on the unit sphere. -/
theorem rayleigh_le_top (h0 : 0 < n) (d : E) (hd : ‖d‖ = 1) :
    ⟪d, T d⟫ ≤ hT.eigenvalues hn ⟨0, h0⟩ := by
  rw [rayleigh_expansion hn hT d]
  have hsum := sum_sq_inner_eigenvector hn hT d
  rw [hd, one_pow] at hsum
  calc ∑ i, hT.eigenvalues hn i * ⟪hT.eigenvectorBasis hn i, d⟫ ^ 2
      ≤ ∑ i, hT.eigenvalues hn ⟨0, h0⟩ * ⟪hT.eigenvectorBasis hn i, d⟫ ^ 2 := by
        apply Finset.sum_le_sum
        intro i _
        apply mul_le_mul_of_nonneg_right _ (sq_nonneg _)
        exact hT.eigenvalues_antitone hn (Fin.mk_le_of_le_val (Nat.zero_le _))
    _ = hT.eigenvalues hn ⟨0, h0⟩ := by rw [← Finset.mul_sum, hsum, mul_one]

/-- The bottom eigenvalue bounds the Rayleigh form from below on the unit sphere. -/
theorem bot_le_rayleigh (h0 : 0 < n) (d : E) (hd : ‖d‖ = 1) :
    hT.eigenvalues hn ⟨n - 1, Nat.sub_lt h0 one_pos⟩ ≤ ⟪d, T d⟫ := by
  rw [rayleigh_expansion hn hT d]
  have hsum := sum_sq_inner_eigenvector hn hT d
  rw [hd, one_pow] at hsum
  calc hT.eigenvalues hn ⟨n - 1, Nat.sub_lt h0 one_pos⟩
      = ∑ i, hT.eigenvalues hn ⟨n - 1, Nat.sub_lt h0 one_pos⟩ * ⟪hT.eigenvectorBasis hn i, d⟫ ^ 2 := by
        rw [← Finset.mul_sum, hsum, mul_one]
    _ ≤ ∑ i, hT.eigenvalues hn i * ⟪hT.eigenvectorBasis hn i, d⟫ ^ 2 := by
        apply Finset.sum_le_sum
        intro i _
        apply mul_le_mul_of_nonneg_right _ (sq_nonneg _)
        have hi := i.isLt
        exact hT.eigenvalues_antitone hn (Fin.le_iff_val_le_val.mpr (by simp only; omega))

/-- An eigenvector of the basis attains its eigenvalue. -/
theorem rayleigh_eigenvector (i : Fin n) :
    ⟪hT.eigenvectorBasis hn i, T (hT.eigenvectorBasis hn i)⟫ = hT.eigenvalues hn i := by
  rw [inner_eigenvector_apply hn hT, real_inner_self_eq_norm_sq, (hT.eigenvectorBasis hn).orthonormal.1 i]
  ring

/-- On the arc from one eigenvector to another, the Rayleigh form is `cos² t λ_j + sin² t λ_i`. -/
theorem rayleigh_arc (i j : Fin n) (hij : i ≠ j) (t : ℝ) :
    let γ := Real.cos t • hT.eigenvectorBasis hn j + Real.sin t • hT.eigenvectorBasis hn i
    ‖γ‖ = 1 ∧ ⟪γ, T γ⟫ = Real.cos t ^ 2 * hT.eigenvalues hn j + Real.sin t ^ 2 * hT.eigenvalues hn i := by
  intro γ
  have ho := (hT.eigenvectorBasis hn).orthonormal
  have hii : ⟪hT.eigenvectorBasis hn i, hT.eigenvectorBasis hn i⟫ = 1 := by
    rw [real_inner_self_eq_norm_sq, ho.1 i]; ring
  have hjj : ⟪hT.eigenvectorBasis hn j, hT.eigenvectorBasis hn j⟫ = 1 := by
    rw [real_inner_self_eq_norm_sq, ho.1 j]; ring
  have hij0 : ⟪hT.eigenvectorBasis hn i, hT.eigenvectorBasis hn j⟫ = 0 := ho.2 hij
  have hji0 : ⟪hT.eigenvectorBasis hn j, hT.eigenvectorBasis hn i⟫ = 0 := ho.2 (Ne.symm hij)
  have hγγ : ⟪γ, γ⟫ = 1 := by
    simp only [γ, inner_add_left, inner_add_right, real_inner_smul_left, real_inner_smul_right, hii, hjj, hij0,
      hji0]
    nlinarith [Real.cos_sq_add_sin_sq t]
  constructor
  · have h := real_inner_self_eq_norm_sq γ
    rw [hγγ] at h
    have hnn : 0 ≤ ‖γ‖ := norm_nonneg _
    nlinarith
  · have hTγ : T γ = (Real.cos t * hT.eigenvalues hn j) • hT.eigenvectorBasis hn j +
        (Real.sin t * hT.eigenvalues hn i) • hT.eigenvectorBasis hn i := by
      simp only [γ, map_add, map_smul, hT.apply_eigenvectorBasis hn, smul_smul]
      simp
    rw [hTγ]
    simp only [γ, inner_add_left, inner_add_right, real_inner_smul_left, real_inner_smul_right, hii, hjj, hij0,
      hji0]
    ring

/-- **Theorem 5(i), range part.** In dimension at least 2 the Rayleigh form on the unit sphere takes exactly the
values between the bottom and the top eigenvalue. -/
theorem rayleigh_range (h2 : 2 ≤ n) :
    {r | ∃ d : E, ‖d‖ = 1 ∧ ⟪d, T d⟫ = r} =
      Set.Icc (hT.eigenvalues hn ⟨n - 1, by omega⟩) (hT.eigenvalues hn ⟨0, by omega⟩) := by
  have h0 : 0 < n := by omega
  ext r
  constructor
  · rintro ⟨d, hd, rfl⟩
    exact ⟨bot_le_rayleigh hn hT h0 d hd, rayleigh_le_top hn hT h0 d hd⟩
  · intro hr
    set i : Fin n := ⟨0, h0⟩
    set j : Fin n := ⟨n - 1, by omega⟩
    have hij : i ≠ j := by
      intro h
      have := congrArg Fin.val h
      simp [i, j] at this
      omega
    set f : ℝ → ℝ := fun t =>
      ⟪Real.cos t • hT.eigenvectorBasis hn j + Real.sin t • hT.eigenvectorBasis hn i,
        T (Real.cos t • hT.eigenvectorBasis hn j + Real.sin t • hT.eigenvectorBasis hn i)⟫
    have hf : ∀ t, f t = Real.cos t ^ 2 * hT.eigenvalues hn j + Real.sin t ^ 2 * hT.eigenvalues hn i :=
      fun t => (rayleigh_arc hn hT i j hij t).2
    have hcont : ContinuousOn f (Set.Icc 0 (Real.pi / 2)) := by
      have : f = fun t => Real.cos t ^ 2 * hT.eigenvalues hn j + Real.sin t ^ 2 * hT.eigenvalues hn i :=
        funext hf
      rw [this]
      fun_prop
    have h0' : f 0 = hT.eigenvalues hn j := by rw [hf]; simp
    have hpi : f (Real.pi / 2) = hT.eigenvalues hn i := by rw [hf]; simp
    have hivt := intermediate_value_Icc (by positivity : (0 : ℝ) ≤ Real.pi / 2) hcont
    rw [h0', hpi] at hivt
    obtain ⟨t, -, ht⟩ := hivt hr
    exact ⟨_, (rayleigh_arc hn hT i j hij t).1, ht⟩

end Paper
