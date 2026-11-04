import Mathlib

set_option linter.style.header false
set_option linter.style.longLine false

/-!
# The two-vector lemma

For vectors `a`, `b` of a real inner product space and a unit vector `d`,
`⟪a, d⟫ ⟪d, b⟫ ≤ (⟪a, b⟫ + ‖a‖ ‖b‖) / 2`, with equality at the bisector of `a/‖a‖` and `b/‖b‖`.
This is the core of Theorem 5(iv), Proposition 7 and Proposition 8(i) of the paper.
-/

namespace Paper

open scoped RealInnerProductSpace

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

/-- Two-vector lemma, the bound: for every unit `d`, `⟪a, d⟫ ⟪d, b⟫ ≤ (⟪a, b⟫ + ‖a‖‖b‖)/2`. -/
theorem two_vector_le (a b d : E) (hd : ‖d‖ = 1) :
    ⟪a, d⟫ * ⟪d, b⟫ ≤ (⟪a, b⟫ + ‖a‖ * ‖b‖) / 2 := by
  set s := ‖a‖
  set t := ‖b‖
  set x := ⟪a, d⟫
  set y := ⟪d, b⟫
  have hs : 0 ≤ s := norm_nonneg _
  have ht : 0 ≤ t := norm_nonneg _
  -- `c = t a + s b` has `‖c‖² = 2 s t (s t + ⟪a, b⟫)`
  set c := t • a + s • b with hc
  have hcd : ⟪c, d⟫ = t * x + s * y := by
    simp only [hc, inner_add_left, inner_smul_left, x, y, real_inner_comm d b]
    simp
  have hcc : ⟪c, c⟫ = 2 * (s * t) * (s * t + ⟪a, b⟫) := by
    have ha : ⟪a, a⟫ = s ^ 2 := real_inner_self_eq_norm_sq a
    have hb : ⟪b, b⟫ = t ^ 2 := real_inner_self_eq_norm_sq b
    simp only [hc, inner_add_left, inner_add_right, inner_smul_left, inner_smul_right, ha, hb,
      real_inner_comm b a]
    simp
    ring
  -- Cauchy–Schwarz for the unit vector `d`
  have hCS : ⟪c, d⟫ ^ 2 ≤ ⟪c, c⟫ := by
    have h1 : |⟪c, d⟫| ≤ ‖c‖ * ‖d‖ := abs_real_inner_le_norm c d
    rw [hd, mul_one] at h1
    have h2 : ⟪c, d⟫ ^ 2 ≤ ‖c‖ ^ 2 := by
      have := sq_le_sq' (neg_le_of_abs_le h1) (le_of_abs_le h1)
      simpa using this
    rwa [real_inner_self_eq_norm_sq]
  rcases (mul_nonneg hs ht).lt_or_eq with hst | hst
  · -- `s t > 0`: `4 s t x y = (t x + s y)² - (t x - s y)² ≤ ‖c‖²`
    have key : 4 * (s * t) * (x * y) ≤ 2 * (s * t) * (s * t + ⟪a, b⟫) := by
      have h3 : 4 * (s * t) * (x * y) = (t * x + s * y) ^ 2 - (t * x - s * y) ^ 2 := by ring
      have h4 : 0 ≤ (t * x - s * y) ^ 2 := sq_nonneg _
      rw [h3, ← hcd, ← hcc]
      linarith
    have : x * y ≤ (s * t + ⟪a, b⟫) / 2 := by
      have h5 : 0 < 2 * (s * t) := by positivity
      nlinarith
    linarith
  · -- `s t = 0`: one of `a`, `b` is zero
    have h0 : s = 0 ∨ t = 0 := by
      rcases mul_eq_zero.mp hst.symm with h | h
      · exact Or.inl h
      · exact Or.inr h
    rcases h0 with h | h
    · have ha0 : a = 0 := norm_eq_zero.mp h
      simp [x, y, ha0, s, t]
    · have hb0 : b = 0 := norm_eq_zero.mp h
      simp [x, y, hb0, s, t]

end Paper

namespace Paper

open scoped RealInnerProductSpace

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

/-- Two-vector lemma, attainment: when `c = ‖b‖ a + ‖a‖ b ≠ 0`, the unit vector `c/‖c‖` (the bisector of the two
directions) attains the bound. -/
theorem two_vector_attained (a b c : E) (hcdef : c = ‖b‖ • a + ‖a‖ • b) (hc : c ≠ 0) :
    ‖‖c‖⁻¹ • c‖ = 1 ∧ ⟪a, ‖c‖⁻¹ • c⟫ * ⟪‖c‖⁻¹ • c, b⟫ = (⟪a, b⟫ + ‖a‖ * ‖b‖) / 2 := by
  have hcn : ‖c‖ ≠ 0 := norm_ne_zero_iff.mpr hc
  refine ⟨by rw [norm_smul, norm_inv, norm_norm]; exact inv_mul_cancel₀ hcn, ?_⟩
  have ha : ⟪a, a⟫ = ‖a‖ ^ 2 := real_inner_self_eq_norm_sq a
  have hb : ⟪b, b⟫ = ‖b‖ ^ 2 := real_inner_self_eq_norm_sq b
  have hac : ⟪a, c⟫ = ‖b‖ * ‖a‖ ^ 2 + ‖a‖ * ⟪a, b⟫ := by
    rw [hcdef, inner_add_right, real_inner_smul_right, real_inner_smul_right, ha]
  have hcb : ⟪c, b⟫ = ‖b‖ * ⟪a, b⟫ + ‖a‖ * ‖b‖ ^ 2 := by
    rw [hcdef, inner_add_left, real_inner_smul_left, real_inner_smul_left, hb]
  have hcc : ‖c‖ ^ 2 = 2 * (‖a‖ * ‖b‖) * (‖a‖ * ‖b‖ + ⟪a, b⟫) := by
    rw [← real_inner_self_eq_norm_sq, hcdef]
    simp only [inner_add_left, inner_add_right, real_inner_smul_left, real_inner_smul_right, ha, hb,
      real_inner_comm b a]
    ring
  have hc2 : ‖c‖ ^ 2 ≠ 0 := pow_ne_zero 2 hcn
  rw [real_inner_smul_right, real_inner_smul_left, hac, hcb]
  have key : (‖b‖ * ‖a‖ ^ 2 + ‖a‖ * ⟪a, b⟫) * (‖b‖ * ⟪a, b⟫ + ‖a‖ * ‖b‖ ^ 2) =
      (‖a‖ * ‖b‖) * (‖a‖ * ‖b‖ + ⟪a, b⟫) ^ 2 := by ring
  calc ‖c‖⁻¹ * (‖b‖ * ‖a‖ ^ 2 + ‖a‖ * ⟪a, b⟫) * (‖c‖⁻¹ * (‖b‖ * ⟪a, b⟫ + ‖a‖ * ‖b‖ ^ 2))
      = (‖a‖ * ‖b‖) * (‖a‖ * ‖b‖ + ⟪a, b⟫) ^ 2 / ‖c‖ ^ 2 := by
        rw [← key]; field_simp
    _ = (⟪a, b⟫ + ‖a‖ * ‖b‖) / 2 := by
        rw [hcc]
        have h1 : 2 * (‖a‖ * ‖b‖) * (‖a‖ * ‖b‖ + ⟪a, b⟫) ≠ 0 := by rw [← hcc]; exact hc2
        have h2 : ‖a‖ * ‖b‖ + ⟪a, b⟫ ≠ 0 := by
          intro h; apply h1; rw [h]; ring
        have h3 : ‖a‖ * ‖b‖ ≠ 0 := by
          intro h; apply h1; rw [h]; ring
        have ha0 : ‖a‖ ≠ 0 := left_ne_zero_of_mul h3
        have hb0 : ‖b‖ ≠ 0 := right_ne_zero_of_mul h3
        field_simp
        ring

/-- In dimension at least 2 every vector has a unit vector orthogonal to it. -/
theorem exists_unit_orthogonal [FiniteDimensional ℝ E] (h2 : 2 ≤ Module.finrank ℝ E) (b : E) :
    ∃ d : E, ‖d‖ = 1 ∧ ⟪d, b⟫ = 0 := by
  set e := stdOrthonormalBasis ℝ E
  have hcard : 2 ≤ Fintype.card (Fin (Module.finrank ℝ E)) := by simpa using h2
  set i0 : Fin (Module.finrank ℝ E) := ⟨0, by omega⟩
  set i1 : Fin (Module.finrank ℝ E) := ⟨1, by omega⟩
  have h01 : i0 ≠ i1 := by
    intro h; have := congrArg Fin.val h; simp [i0, i1] at this
  have ho := e.orthonormal
  by_cases hb0 : ⟪e i0, b⟫ = 0
  · exact ⟨e i0, ho.1 i0, hb0⟩
  · set u := ⟪e i1, b⟫ • e i0 - ⟪e i0, b⟫ • e i1 with hu
    have hub : ⟪u, b⟫ = 0 := by
      simp only [hu, inner_sub_left, real_inner_smul_left]; ring
    have hu1 : ⟪u, e i1⟫ = -⟪e i0, b⟫ := by
      simp only [hu, inner_sub_left, real_inner_smul_left, ho.2 h01, real_inner_self_eq_norm_sq, ho.1 i1]
      ring
    have hune : u ≠ 0 := by
      intro h
      rw [h, inner_zero_left] at hu1
      exact hb0 (by linarith)
    refine ⟨‖u‖⁻¹ • u, ?_, ?_⟩
    · rw [norm_smul, norm_inv, norm_norm]; exact inv_mul_cancel₀ (norm_ne_zero_iff.mpr hune)
    · rw [real_inner_smul_left, hub, mul_zero]

/-- **The two-vector maximum** (dimension at least 2): over unit vectors, the greatest value of `⟪a, d⟫ ⟪d, b⟫` is
`(⟪a, b⟫ + ‖a‖‖b‖)/2`. -/
theorem two_vector_isGreatest [FiniteDimensional ℝ E] (h2 : 2 ≤ Module.finrank ℝ E) (a b : E) :
    IsGreatest {r | ∃ d : E, ‖d‖ = 1 ∧ ⟪a, d⟫ * ⟪d, b⟫ = r} ((⟪a, b⟫ + ‖a‖ * ‖b‖) / 2) := by
  refine ⟨?_, ?_⟩
  · by_cases hc : ‖b‖ • a + ‖a‖ • b = 0
    · -- antiparallel or a zero vector: the value is 0, attained orthogonally to `b`
      obtain ⟨d, hd, hdb⟩ := exists_unit_orthogonal h2 b
      refine ⟨d, hd, ?_⟩
      have hab : ⟪a, b⟫ + ‖a‖ * ‖b‖ = 0 := by
        have h1 : ⟪‖b‖ • a + ‖a‖ • b, b⟫ = 0 := by rw [hc, inner_zero_left]
        simp only [inner_add_left, real_inner_smul_left, real_inner_self_eq_norm_sq] at h1
        by_cases hb : ‖b‖ = 0
        · have : b = 0 := norm_eq_zero.mp hb
          simp [this]
        · have : ‖b‖ * (⟪a, b⟫ + ‖a‖ * ‖b‖) = 0 := by linarith
          rcases mul_eq_zero.mp this with h | h
          · exact absurd h hb
          · exact h
      rw [hdb, mul_zero, hab, zero_div]
    · obtain ⟨hd, hv⟩ := two_vector_attained a b _ rfl hc
      exact ⟨_, hd, hv⟩
  · rintro r ⟨d, hd, rfl⟩
    exact two_vector_le a b d hd

end Paper

namespace Paper

open scoped RealInnerProductSpace

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

/-- **Uniqueness up to sign.** For nonzero `a`, `b` with `c = ‖b‖ a + ‖a‖ b ≠ 0`, a unit `d` attains the two-vector
bound if and only if `d = ± c/‖c‖`, the bisector of the two directions. -/
theorem two_vector_eq_iff (a b c d : E) (hcdef : c = ‖b‖ • a + ‖a‖ • b) (hc : c ≠ 0) (hd : ‖d‖ = 1) :
    ⟪a, d⟫ * ⟪d, b⟫ = (⟪a, b⟫ + ‖a‖ * ‖b‖) / 2 ↔ d = ‖c‖⁻¹ • c ∨ d = -(‖c‖⁻¹ • c) := by
  constructor
  · intro heq
    set s := ‖a‖
    set t := ‖b‖
    have ha : ⟪a, a⟫ = s ^ 2 := real_inner_self_eq_norm_sq a
    have hb : ⟪b, b⟫ = t ^ 2 := real_inner_self_eq_norm_sq b
    have hcd : ⟪c, d⟫ = t * ⟪a, d⟫ + s * ⟪d, b⟫ := by
      rw [hcdef, inner_add_left, real_inner_smul_left, real_inner_smul_left, real_inner_comm b d]
    have hcc : ‖c‖ ^ 2 = 2 * (s * t) * (s * t + ⟪a, b⟫) := by
      rw [← real_inner_self_eq_norm_sq, hcdef]
      simp only [inner_add_left, inner_add_right, real_inner_smul_left, real_inner_smul_right, ha, hb,
        real_inner_comm b a]
      ring
    -- `⟪c, d⟫² = ‖c‖²`
    have hsq : ⟪c, d⟫ ^ 2 = ‖c‖ ^ 2 := by
      have h1 : ⟪c, d⟫ ^ 2 - (t * ⟪a, d⟫ - s * ⟪d, b⟫) ^ 2 = 4 * (s * t) * (⟪a, d⟫ * ⟪d, b⟫) := by rw [hcd]; ring
      rw [heq] at h1
      have h2 : 4 * (s * t) * ((⟪a, b⟫ + s * t) / 2) = ‖c‖ ^ 2 := by rw [hcc]; ring
      have hCS : ⟪c, d⟫ ^ 2 ≤ ‖c‖ ^ 2 := by
        have := abs_real_inner_le_norm c d
        rw [hd, mul_one] at this
        have h3 := sq_le_sq' (neg_le_of_abs_le this) (le_of_abs_le this)
        simpa using h3
      nlinarith [sq_nonneg (t * ⟪a, d⟫ - s * ⟪d, b⟫)]
    -- equality in Cauchy–Schwarz
    have hcn : ‖c‖ ≠ 0 := norm_ne_zero_iff.mpr hc
    have habs : |⟪c, d⟫ / (‖c‖ * ‖d‖)| = 1 := by
      rw [hd, mul_one, abs_div, abs_norm]
      have : |⟪c, d⟫| = ‖c‖ := by
        have h4 : |⟪c, d⟫| ^ 2 = ‖c‖ ^ 2 := by rw [sq_abs]; exact hsq
        exact (sq_eq_sq₀ (abs_nonneg _) (norm_nonneg _)).mp h4
      rw [this, div_self hcn]
    obtain ⟨-, r, -, hr⟩ := (abs_real_inner_div_norm_mul_norm_eq_one_iff c d).mp habs
    have hr1 : |r| * ‖c‖ = 1 := by rw [← hd, hr, norm_smul, Real.norm_eq_abs]
    have hrabs : |r| = ‖c‖⁻¹ := by field_simp; linarith
    rcases abs_eq (inv_nonneg.mpr (norm_nonneg c)) |>.mp hrabs with h | h
    · left; rw [hr, h]
    · right; rw [hr, h, neg_smul]
  · rintro (h | h)
    · rw [h]; exact (two_vector_attained a b c hcdef hc).2
    · rw [h, inner_neg_right, inner_neg_left, neg_mul_neg]; exact (two_vector_attained a b c hcdef hc).2

end Paper
