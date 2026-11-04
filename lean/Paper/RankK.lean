import Paper.Natural

set_option linter.style.header false
set_option linter.style.longLine false

/-!
# Appendix A.3 under one shared direction: a larger rank does not raise the top

When `𝔼[w Δᵀ] = w̄ mᵀ` (Proposition 7), a rank-`k` interchange along an orthonormal family `q₁, …, q_k` has mean
first-order effect `∑ₗ ⟪qₗ, w̄⟫ ⟪qₗ, m⟫`. This sum never exceeds the rank-one top `(⟪w̄, m⟫ + ‖w̄‖ ‖m‖)/2`, whatever `k`:
with `v` the projection of `w̄` on the family's span, the reflection `2v - w̄` has the norm of `w̄`, and Cauchy–Schwarz for
it and `m` is the bound. The top is reached by the bisector completed with directions orthogonal to `w̄`, which exist
whenever `k` is below the dimension.
-/

namespace Paper

open scoped RealInnerProductSpace

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

/-- **One shared direction, rank `k`: the bound.** For every orthonormal family, the rank-`k` first-order effect is at
most the rank-one top `(⟪w̄, m⟫ + ‖w̄‖‖m‖)/2`. -/
theorem natural_rank_k_le {k : ℕ} (wbar m : E) (q : Fin k → E) (hq : Orthonormal ℝ q) :
    ∑ l, ⟪q l, effM1 wbar m (q l)⟫ ≤ (⟪wbar, m⟫ + ‖wbar‖ * ‖m‖) / 2 := by
  simp_rw [natural_effect]
  set v : E := ∑ l, ⟪q l, wbar⟫ • q l with hv
  have hvm : ⟪v, m⟫ = ∑ l, ⟪q l, wbar⟫ * ⟪q l, m⟫ := by
    simp only [hv, sum_inner, real_inner_smul_left]
  have hvw : ⟪v, wbar⟫ = ∑ l, ⟪q l, wbar⟫ * ⟪q l, wbar⟫ := by
    simp only [hv, sum_inner, real_inner_smul_left]
  have hvv : ⟪v, v⟫ = ∑ l, ⟪q l, wbar⟫ * ⟪q l, wbar⟫ := by
    rw [hv, hq.inner_sum]
    simp
  -- the reflection `2v - w̄` keeps the norm of `w̄`
  have hsq : ‖(2 : ℝ) • v - wbar‖ ^ 2 = ‖wbar‖ ^ 2 := by
    rw [← real_inner_self_eq_norm_sq, ← real_inner_self_eq_norm_sq]
    simp only [inner_sub_left, inner_sub_right, real_inner_smul_left, real_inner_smul_right]
    rw [real_inner_comm v wbar, hvv, hvw]
    ring
  have hnorm : ‖(2 : ℝ) • v - wbar‖ = ‖wbar‖ := by
    have h := congrArg Real.sqrt hsq
    rwa [Real.sqrt_sq (norm_nonneg _), Real.sqrt_sq (norm_nonneg _)] at h
  have hcs : ⟪(2 : ℝ) • v - wbar, m⟫ ≤ ‖(2 : ℝ) • v - wbar‖ * ‖m‖ := real_inner_le_norm _ _
  have hlin : ⟪(2 : ℝ) • v - wbar, m⟫ = 2 * ⟪v, m⟫ - ⟪wbar, m⟫ := by
    simp only [inner_sub_left, real_inner_smul_left]
  rw [hnorm, hlin, hvm] at hcs
  linarith

/-- **One shared direction, rank `k`: the top is reached.** A family whose first vector reaches the rank-one top and whose
other vectors are orthogonal to `w̄` has the same first-order effect. -/
theorem natural_rank_k_attained {k : ℕ} (wbar m : E) (q : Fin (k + 1) → E)
    (htop : ⟪q 0, effM1 wbar m (q 0)⟫ = (⟪wbar, m⟫ + ‖wbar‖ * ‖m‖) / 2)
    (horth : ∀ l : Fin k, ⟪q l.succ, wbar⟫ = 0) :
    ∑ l, ⟪q l, effM1 wbar m (q l)⟫ = (⟪wbar, m⟫ + ‖wbar‖ * ‖m‖) / 2 := by
  rw [Fin.sum_univ_succ, htop]
  have hz : ∑ l : Fin k, ⟪q l.succ, effM1 wbar m (q l.succ)⟫ = 0 := by
    apply Finset.sum_eq_zero
    intro l _
    rw [natural_effect, horth l, zero_mul]
  rw [hz, add_zero]

/-- **One shared direction, rank `k`: the top for every rank below the dimension.** In dimension `n ≥ 2`, for every
`k + 1 ≤ n - 1` some orthonormal family of size `k + 1` reaches the rank-one top `(⟪w̄, m⟫ + ‖w̄‖‖m‖)/2`: a unit vector
that reaches it, completed by an orthonormal family orthogonal to it and to `w̄`. With `natural_rank_k_le`, the top of the
rank-`(k + 1)` range is the rank-one top. -/
theorem natural_rank_k_exists [FiniteDimensional ℝ E] {n k : ℕ} (hn : Module.finrank ℝ E = n) (h2 : 2 ≤ n)
    (hk : k + 1 ≤ n - 1) (wbar m : E) :
    ∃ q : Fin (k + 1) → E, Orthonormal ℝ q ∧
      ∑ l, ⟪q l, effM1 wbar m (q l)⟫ = (⟪wbar, m⟫ + ‖wbar‖ * ‖m‖) / 2 := by
  classical
  obtain ⟨d, hd1, hdtop⟩ := (two_vector_isGreatest (hn ▸ h2) wbar m).1
  -- the directions orthogonal to `d` and `w̄`
  set K : Submodule ℝ E := Submodule.span ℝ (({d, wbar} : Finset E) : Set E) with hKdef
  have hK : Module.finrank ℝ K ≤ 2 :=
    (finrank_span_finset_le_card _).trans (Finset.card_le_two)
  have hKo : n - 2 ≤ Module.finrank ℝ Kᗮ := by
    have := Submodule.finrank_add_finrank_orthogonal K
    omega
  have hkK : k ≤ Module.finrank ℝ Kᗮ := by omega
  set b := stdOrthonormalBasis ℝ Kᗮ
  let c : Fin k → E := fun i => (b (Fin.castLE hkK i) : E)
  have hdK : d ∈ K := Submodule.subset_span (by simp)
  have hwK : wbar ∈ K := Submodule.subset_span (by simp)
  have hdc : ∀ i, ⟪d, c i⟫ = 0 := fun i => Submodule.inner_right_of_mem_orthogonal hdK (b _).2
  have hwc : ∀ i, ⟪c i, wbar⟫ = 0 := fun i => by
    rw [real_inner_comm]; exact Submodule.inner_right_of_mem_orthogonal hwK (b _).2
  have hcc : ∀ i j, ⟪c i, c j⟫ = if i = j then 1 else 0 := by
    intro i j
    have h := orthonormal_iff_ite.1 b.orthonormal (Fin.castLE hkK i) (Fin.castLE hkK j)
    rw [Submodule.coe_inner] at h
    simpa [c, Fin.castLE_inj] using h
  refine ⟨Fin.cons d c, ?_, ?_⟩
  · rw [orthonormal_iff_ite]
    intro i j
    refine Fin.cases ?_ (fun i' => ?_) i <;> refine Fin.cases ?_ (fun j' => ?_) j
    · simp [hd1]
    · simp [hdc, (Fin.succ_ne_zero j').symm]
    · simp [real_inner_comm, hdc, Fin.succ_ne_zero i']
    · simp [hcc, Fin.succ_inj]
  · apply natural_rank_k_attained wbar m (Fin.cons d c)
    · simp only [Fin.cons_zero]
      rw [natural_effect, real_inner_comm wbar d]
      exact hdtop
    · intro l
      simp only [Fin.cons_succ]
      exact hwc l

end Paper
