import Paper.KyFan
import Paper.Natural

set_option linter.style.header false
set_option linter.style.longLine false

/-!
# Appendix A.4: confining the direction to a subspace that contains the differences

* (i) one code makes the rank-one interchange the network's own; (ii) the compression keeps the trace of `M`; (iii) the
  confined top, which also gives "size alone"; and the paper's example, `1.207 tr M`.
-/

namespace Paper

open scoped RealInnerProductSpace
open Real Set

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

/-! ### B.9 Confinement -/

/-- **A.4(i).** With a single code `Δᵢ = sᵢ Δ̄`, the rank-one interchange along `Δ̄/‖Δ̄‖` is the network's own
interchange, pair by pair. -/
theorem single_code_rank_one (s : ℝ) (Δbar : E) (h : Δbar ≠ 0) :
    ⟪‖Δbar‖⁻¹ • Δbar, s • Δbar⟫ • (‖Δbar‖⁻¹ • Δbar) = s • Δbar := by
  have hn : ‖Δbar‖ ≠ 0 := norm_ne_zero_iff.mpr h
  rw [real_inner_smul_left, real_inner_smul_right, real_inner_self_eq_norm_sq, smul_smul]
  congr 1
  field_simp

variable {ι : Type*} [Fintype ι]

/-- **A.4(ii).** If the orthonormal family `q` spans a subspace containing every difference (`hspan`, which says that the
rank-`k` interchange along it is the network's own interchange pair by pair, B.9(i)), the trace of the compression
`PᵀMP` is the trace of `M`. -/
theorem confined_trace {k : ℕ} (p : ι → ℝ) (w Δ : ι → E) (q : Fin k → E)
    (hspan : ∀ i, ∑ l, ⟪q l, Δ i⟫ • q l = Δ i) :
    ∑ l, ⟪q l, effM p w Δ (q l)⟫ = ∑ i, p i * ⟪w i, Δ i⟫ := by
  calc ∑ l, ⟪q l, effM p w Δ (q l)⟫ = ∑ l, ∑ i, p i * ⟪w i, ⟪q l, Δ i⟫ • q l⟫ := by
        simp_rw [rank_one_effect_eq]
    _ = ∑ i, p i * ⟪w i, ∑ l, ⟪q l, Δ i⟫ • q l⟫ := by
        rw [Finset.sum_comm]; simp_rw [inner_sum, Finset.mul_sum]
    _ = ∑ i, p i * ⟪w i, Δ i⟫ := by simp_rw [hspan]

/-- **A.4(iii)** and "size alone". For `d` and `Δ` in a subspace `K`, `⟪a, d⟫ = ⟪Π a, d⟫`, so over unit `d ∈ K` the
score change `⟪a, d⟫⟪d, Δ⟫` is at most `(⟪a, Δ⟫ + ‖Π a‖‖Δ‖)/2`, attained at the bisector of `Π a` and `Δ` (which lies in
`K`). With `a = w̄` and `Δ = m ∈ K` this is the confined top of Proposition 7. -/
theorem confined_two_vector (K : Submodule ℝ E) [K.HasOrthogonalProjection] (a Δ : E) (hΔ : Δ ∈ K) :
    (∀ d ∈ K, ‖d‖ = 1 → ⟪a, d⟫ * ⟪d, Δ⟫ ≤ (⟪a, Δ⟫ + ‖K.starProjection a‖ * ‖Δ‖) / 2) ∧
      (‖Δ‖ • K.starProjection a + ‖K.starProjection a‖ • Δ ≠ 0 →
        ∃ d ∈ K, ‖d‖ = 1 ∧ ⟪a, d⟫ * ⟪d, Δ⟫ = (⟪a, Δ⟫ + ‖K.starProjection a‖ * ‖Δ‖) / 2) := by
  have hproj : ∀ d ∈ K, ⟪a, d⟫ = ⟪K.starProjection a, d⟫ := by
    intro d hd
    rw [Submodule.inner_starProjection_left_eq_right, Submodule.starProjection_eq_self_iff.mpr hd]
  have hPaΔ : ⟪K.starProjection a, Δ⟫ = ⟪a, Δ⟫ := (hproj Δ hΔ).symm
  refine ⟨fun d hd hd1 => ?_, fun hc => ?_⟩
  · rw [hproj d hd, ← hPaΔ]
    exact two_vector_le _ _ d hd1
  · obtain ⟨hd1, hv⟩ := two_vector_attained (K.starProjection a) Δ _ rfl hc
    refine ⟨_, ?_, hd1, ?_⟩
    · apply K.smul_mem
      exact K.add_mem (K.smul_mem _ (Submodule.starProjection_apply_mem K a)) (K.smul_mem _ hΔ)
    · rw [hproj _ (K.smul_mem _ (K.add_mem (K.smul_mem _ (Submodule.starProjection_apply_mem K a))
        (K.smul_mem _ hΔ))), hv, hPaΔ]

/-- **A.4, the example.** With orthonormal `e₁, e₂`, the pair `w = e₁ + e₂`, `Δ = e₁` has top eigenvalue
`(1 + √2)/2` and trace `1` (dimension at least 2): a top of `1.207 tr M`. This is Example 1 of
Section 3.2. The two pairs of Appendix A.4 (`Δ = e₁, e₂` with `w = e₁ + e₂, 0`, weights `1/2`) give half this `M`
(the second pair adds nothing) and so the same ratio; the top eigenvector lies in `span(e₁, e₂)`, so confining the
direction to that span keeps it. -/
theorem confined_example [FiniteDimensional ℝ E] {n : ℕ} (hn : Module.finrank ℝ E = n) (h2 : 2 ≤ n)
    (e1 e2 : E) (h1 : ‖e1‖ = 1) (h2' : ‖e2‖ = 1) (h12 : ⟪e1, e2⟫ = 0) :
    (effM_isSymmetric (fun _ : Unit => (1 : ℝ)) (fun _ => e1 + e2) (fun _ => e1)).eigenvalues hn ⟨0, by omega⟩ =
        (1 + √2) / 2 ∧
      LinearMap.trace ℝ E (effM1 (e1 + e2) e1) = 1 := by
  have h11 : ⟪e1, e1⟫ = 1 := by rw [real_inner_self_eq_norm_sq, h1]; ring
  have h22 : ⟪e2, e2⟫ = 1 := by rw [real_inner_self_eq_norm_sq, h2']; ring
  have h21 : ⟪e2, e1⟫ = 0 := by rw [real_inner_comm]; exact h12
  have hin : ⟪e1 + e2, e1⟫ = 1 := by rw [inner_add_left, h11, h21]; ring
  have hnorm : ‖e1 + e2‖ = √2 := by
    rw [← sqrt_sq (norm_nonneg _), ← real_inner_self_eq_norm_sq]
    congr 1
    rw [inner_add_left, inner_add_right, inner_add_right, h11, h22, h12, h21]; ring
  constructor
  · rw [top_eigenvalue_single hn h2, hin, hnorm, h1]; ring
  · rw [natural_trace, hin]

end Paper
