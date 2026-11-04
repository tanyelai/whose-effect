import Paper.Range

set_option linter.style.header false
set_option linter.style.longLine false

/-!
# Theorem 5(iii) and Appendix A.3: rank-`k` interchanges (Ky Fan's maximum principle)

A rank-`k` interchange along an orthonormal family `q₁, …, q_k` patches `∑ₗ ⟪qₗ, Δᵢ⟫ qₗ` (that is, `P Pᵀ Δᵢ`). Its mean
first-order effect is `∑ₗ ⟪qₗ, M qₗ⟫ = tr(Pᵀ M P)`, which lies between the sum of the `k` smallest and the sum of the `k`
largest eigenvalues of `M`. Mathlib does not have Ky Fan's principle; it is proved here from the eigen-expansion,
Bessel's inequality and a rearrangement step.
-/

namespace Paper

open scoped RealInnerProductSpace

/-- The rearrangement step: for sorted (decreasing) `λ` and weights `0 ≤ cⱼ ≤ 1` summing to `k ≤ n`,
`∑ λⱼ cⱼ ≤ ∑_{j < k} λⱼ`. -/
theorem sum_le_top_k {n k : ℕ} (hk : k ≤ n) (lam : Fin n → ℝ) (hlam : Antitone lam) (c : Fin n → ℝ)
    (hc0 : ∀ j, 0 ≤ c j) (hc1 : ∀ j, c j ≤ 1) (hsum : ∑ j, c j = k) :
    ∑ j, lam j * c j ≤ ∑ j ∈ Finset.univ.filter (fun j : Fin n => (j : ℕ) < k), lam j := by
  rcases Nat.eq_zero_or_pos k with rfl | hkpos
  · have hz : ∀ j, c j = 0 := by
      intro j
      have h0 : ∑ j, c j = 0 := by simpa using hsum
      exact (Finset.sum_eq_zero_iff_of_nonneg (fun j _ => hc0 j)).mp h0 j (Finset.mem_univ j)
    simp [hz]
  · set μ := lam ⟨k - 1, by omega⟩
    have key : ∀ j : Fin n, lam j * c j - (if (j : ℕ) < k then lam j else 0) ≤
        μ * (c j - if (j : ℕ) < k then 1 else 0) := by
      intro j
      by_cases hj : (j : ℕ) < k
      · simp only [hj, if_true]
        have hlj : μ ≤ lam j := hlam (Fin.le_iff_val_le_val.mpr (by simp only; omega))
        nlinarith [hc1 j]
      · simp only [hj, if_false, sub_zero]
        have hlj : lam j ≤ μ := hlam (Fin.le_iff_val_le_val.mpr (by simp only; omega))
        nlinarith [hc0 j]
    have hcard : ((Finset.univ.filter (fun j : Fin n => (j : ℕ) < k)).card : ℝ) = k := by
      rw [Fin.card_filter_val_lt, min_eq_right hk]
    have hR : ∑ j : Fin n, μ * (c j - if (j : ℕ) < k then 1 else 0) = 0 := by
      rw [← Finset.mul_sum, Finset.sum_sub_distrib, hsum, Finset.sum_boole, hcard]
      ring
    have hL : ∑ j : Fin n, (if (j : ℕ) < k then lam j else 0) =
        ∑ j ∈ Finset.univ.filter (fun j : Fin n => (j : ℕ) < k), lam j := by
      rw [Finset.sum_filter]
    have := Finset.sum_le_sum (fun j (_ : j ∈ Finset.univ) => key j)
    rw [Finset.sum_sub_distrib, hR, hL] at this
    linarith

/-- The rearrangement step from below: `∑ λⱼ cⱼ ≥ ∑_{j ≥ n - k} λⱼ`. -/
theorem sum_ge_bot_k {n k : ℕ} (hk : k ≤ n) (lam : Fin n → ℝ) (hlam : Antitone lam) (c : Fin n → ℝ)
    (hc0 : ∀ j, 0 ≤ c j) (hc1 : ∀ j, c j ≤ 1) (hsum : ∑ j, c j = k) :
    ∑ j ∈ Finset.univ.filter (fun j : Fin n => n - k ≤ (j : ℕ)), lam j ≤ ∑ j, lam j * c j := by
  -- apply the upper bound to the complementary weights `1 - cⱼ`, which sum to `n - k`
  have h := sum_le_top_k (Nat.sub_le n k) lam hlam (fun j => 1 - c j) (fun j => by linarith [hc1 j])
    (fun j => by linarith [hc0 j]) (by
      rw [Finset.sum_sub_distrib, hsum]; simp [Nat.cast_sub hk])
  have hsplit := Finset.sum_filter_add_sum_filter_not Finset.univ (fun j : Fin n => (j : ℕ) < n - k) lam
  have hcompl : Finset.univ.filter (fun j : Fin n => ¬ (j : ℕ) < n - k) =
      Finset.univ.filter (fun j : Fin n => n - k ≤ (j : ℕ)) := by
    ext j; simp
  rw [hcompl] at hsplit
  have hexp : ∑ j, lam j * (1 - c j) = ∑ j, lam j - ∑ j, lam j * c j := by
    rw [← Finset.sum_sub_distrib]; apply Finset.sum_congr rfl; intro j _; ring
  rw [hexp] at h
  linarith

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] [FiniteDimensional ℝ E]
variable {n : ℕ} (hn : Module.finrank ℝ E = n) {T : E →ₗ[ℝ] E} (hT : T.IsSymmetric)

/-- The weights `cⱼ = ∑ₗ ⟪eⱼ, qₗ⟫²` of an orthonormal family in the eigenvector basis. -/
theorem kyfan_weights {k : ℕ} (q : Fin k → E) (hq : Orthonormal ℝ q) :
    (∀ j, 0 ≤ ∑ l, ⟪hT.eigenvectorBasis hn j, q l⟫ ^ 2) ∧
      (∀ j, ∑ l, ⟪hT.eigenvectorBasis hn j, q l⟫ ^ 2 ≤ 1) ∧
      ∑ j, ∑ l, ⟪hT.eigenvectorBasis hn j, q l⟫ ^ 2 = k := by
  refine ⟨fun j => Finset.sum_nonneg (fun l _ => sq_nonneg _), fun j => ?_, ?_⟩
  · have hb := hq.sum_inner_products_le (x := hT.eigenvectorBasis hn j) (s := Finset.univ)
    rw [(hT.eigenvectorBasis hn).orthonormal.1 j] at hb
    have : ∀ l, ‖⟪q l, hT.eigenvectorBasis hn j⟫‖ ^ 2 = ⟪hT.eigenvectorBasis hn j, q l⟫ ^ 2 := by
      intro l; rw [Real.norm_eq_abs, sq_abs, real_inner_comm]
    simp only [this, one_pow] at hb
    exact hb
  · rw [Finset.sum_comm]
    have : ∀ l, ∑ j, ⟪hT.eigenvectorBasis hn j, q l⟫ ^ 2 = 1 := by
      intro l; rw [sum_sq_inner_eigenvector hn hT, hq.1 l]; ring
    simp [this]

/-- `∑ₗ ⟪qₗ, T qₗ⟫ = ∑ⱼ λⱼ cⱼ`. -/
theorem trace_compression_expansion {k : ℕ} (q : Fin k → E) :
    ∑ l, ⟪q l, T (q l)⟫ = ∑ j, hT.eigenvalues hn j * ∑ l, ⟪hT.eigenvectorBasis hn j, q l⟫ ^ 2 := by
  simp_rw [rayleigh_expansion hn hT, Finset.mul_sum]
  rw [Finset.sum_comm]

/-- **Ky Fan's maximum principle, upper half.** -/
theorem kyfan_upper {k : ℕ} (hk : k ≤ n) (q : Fin k → E) (hq : Orthonormal ℝ q) :
    ∑ l, ⟪q l, T (q l)⟫ ≤ ∑ j ∈ Finset.univ.filter (fun j : Fin n => (j : ℕ) < k), hT.eigenvalues hn j := by
  obtain ⟨h0, h1, hs⟩ := kyfan_weights hn hT q hq
  rw [trace_compression_expansion hn hT q]
  exact sum_le_top_k hk _ (hT.eigenvalues_antitone hn) _ h0 h1 hs

/-- **Ky Fan's maximum principle, lower half.** -/
theorem kyfan_lower {k : ℕ} (hk : k ≤ n) (q : Fin k → E) (hq : Orthonormal ℝ q) :
    ∑ j ∈ Finset.univ.filter (fun j : Fin n => n - k ≤ (j : ℕ)), hT.eigenvalues hn j ≤ ∑ l, ⟪q l, T (q l)⟫ := by
  obtain ⟨h0, h1, hs⟩ := kyfan_weights hn hT q hq
  rw [trace_compression_expansion hn hT q]
  exact sum_ge_bot_k hk _ (hT.eigenvalues_antitone hn) _ h0 h1 hs

variable {ι : Type*} [Fintype ι]

/-- **Theorem 5(iii).** The mean first-order effect of the rank-`k` interchange along an orthonormal family is
`tr(Pᵀ M P) = ∑ₗ ⟪qₗ, M qₗ⟫`, and it lies between the sum of the `k` smallest and the sum of the `k` largest
eigenvalues of `M`. -/
theorem rank_k_effect {k : ℕ} (hk : k ≤ n) (p : ι → ℝ) (w Δ : ι → E) (q : Fin k → E) (hq : Orthonormal ℝ q) :
    ∑ i, p i * ⟪w i, ∑ l, ⟪q l, Δ i⟫ • q l⟫ = ∑ l, ⟪q l, effM p w Δ (q l)⟫ ∧
      ∑ j ∈ Finset.univ.filter (fun j : Fin n => n - k ≤ (j : ℕ)), (effM_isSymmetric p w Δ).eigenvalues hn j ≤
        ∑ l, ⟪q l, effM p w Δ (q l)⟫ ∧
      ∑ l, ⟪q l, effM p w Δ (q l)⟫ ≤
        ∑ j ∈ Finset.univ.filter (fun j : Fin n => (j : ℕ) < k), (effM_isSymmetric p w Δ).eigenvalues hn j := by
  refine ⟨?_, kyfan_lower hn _ hk q hq, kyfan_upper hn _ hk q hq⟩
  simp_rw [← rank_one_effect_eq, inner_sum, Finset.mul_sum]
  rw [Finset.sum_comm]

end Paper
