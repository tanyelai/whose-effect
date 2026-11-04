import Paper.KyFan
import Paper.Natural

set_option linter.style.header false
set_option linter.style.longLine false

/-!
# Appendix A.5: the counterexamples

* An indefinite `M` whose top lies below the trace; the first-order undershoot; a per-pair squared error not minimised
  where the mean effect is `T` (its minimum: `Paper.SqErrorMinimiser`).

The basis vectors `e₀, e₁, e₂` are the paper's `e₁, e₂, e₃`, and eigenvalues follow Mathlib's order (`λ₀` is the paper's
`λ₁`, `λ_{n-1}` its `λ_D`).
-/

namespace Paper

open scoped RealInnerProductSpace
open Real Set

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

variable {ι : Type*} [Fintype ι]

/-! ### B.10 and Corollary 6: counterexamples -/

/-- **An indefinite `M` does not make the top exceed the trace.** For orthonormal `e₀, e₁, e₂` and the two pairs
`w₁ = e₀, Δ₁ = e₀ + e₂/2`, `w₂ = e₁, Δ₂ = e₁ - e₂/2` with weights `1/2`: `tr M = 1`, every unit `d` has
`⟪d, M d⟫ ≤ 0.6 < tr M`, and some unit `d` has `⟪d, M d⟫ < 0`. -/
theorem counterexample_indefinite [FiniteDimensional ℝ E] (e : Fin 3 → E) (he : Orthonormal ℝ e) :
    let w : Fin 2 → E := ![e 0, e 1]
    let Δ : Fin 2 → E := ![e 0 + (1 / 2 : ℝ) • e 2, e 1 - (1 / 2 : ℝ) • e 2]
    let p : Fin 2 → ℝ := fun _ => 1 / 2
    LinearMap.trace ℝ E (effM p w Δ) = 1 ∧
      (∀ d : E, ‖d‖ = 1 → ⟪d, effM p w Δ d⟫ ≤ 3 / 5) ∧
      ∃ d : E, ‖d‖ = 1 ∧ ⟪d, effM p w Δ d⟫ < 0 := by
  intro w Δ p
  have hii : ∀ i, ⟪e i, e i⟫ = 1 := fun i => by rw [real_inner_self_eq_norm_sq, he.1 i]; ring
  have hij : ∀ i j, i ≠ j → ⟪e i, e j⟫ = 0 := fun i j h => he.2 h
  have h01 := hij 0 1 (by decide); have h02 := hij 0 2 (by decide); have h12 := hij 1 2 (by decide)
  have h10 := hij 1 0 (by decide); have h20 := hij 2 0 (by decide); have h21 := hij 2 1 (by decide)
  -- the Rayleigh form in the coordinates x = ⟪e₀,d⟫, y = ⟪e₁,d⟫, z = ⟪e₂,d⟫
  have hform : ∀ d : E, ⟪d, effM p w Δ d⟫ =
      (1 / 2) * (⟪e 0, d⟫ * (⟪e 0, d⟫ + (1 / 2) * ⟪e 2, d⟫)) + (1 / 2) * (⟪e 1, d⟫ * (⟪e 1, d⟫ - (1 / 2) * ⟪e 2, d⟫)) := by
    intro d
    rw [← rank_one_effect_eq]
    simp only [Fin.sum_univ_two, p, w, Δ, Matrix.cons_val_zero, Matrix.cons_val_one, real_inner_smul_right,
      inner_add_right, inner_sub_right, real_inner_comm d]
    ring
  refine ⟨?_, ?_, ?_⟩
  · rw [trace_effM]
    simp only [Fin.sum_univ_two, p, w, Δ, Matrix.cons_val_zero, Matrix.cons_val_one, inner_add_right, inner_sub_right,
      real_inner_smul_right, hii, h01, h02, h12]
    norm_num
  · intro d hd
    rw [hform]
    set x := ⟪e 0, d⟫; set y := ⟪e 1, d⟫; set z := ⟪e 2, d⟫
    have hbessel : x ^ 2 + y ^ 2 + z ^ 2 ≤ 1 := by
      have := he.sum_inner_products_le (x := d) (s := Finset.univ)
      simp only [Fin.sum_univ_three, Real.norm_eq_abs, sq_abs, hd, one_pow] at this
      linarith
    nlinarith [sq_nonneg (x - 5 / 4 * z), sq_nonneg (y + 5 / 4 * z), sq_nonneg z]
  · set v : E := (1 / 4 : ℝ) • e 0 - (1 / 4 : ℝ) • e 1 - e 2
    have hvv : ⟪v, v⟫ = 9 / 8 := by
      simp only [v, inner_sub_left, inner_sub_right, real_inner_smul_left, real_inner_smul_right, hii, h01, h02, h12,
        h10, h20, h21]
      norm_num
    have hvn : ‖v‖ ≠ 0 := by
      intro h; rw [real_inner_self_eq_norm_sq, h] at hvv; norm_num at hvv
    refine ⟨‖v‖⁻¹ • v, by rw [norm_smul, norm_inv, norm_norm]; exact inv_mul_cancel₀ hvn, ?_⟩
    rw [hform]
    set r := ‖v‖⁻¹ with hr
    have hpos : 0 < r := inv_pos.mpr (lt_of_le_of_ne (norm_nonneg _) (Ne.symm hvn))
    have hx : ⟪e 0, r • v⟫ = r * (1 / 4) := by
      simp only [v, real_inner_smul_right, inner_sub_right, hii, h01, h02]; ring
    have hy : ⟪e 1, r • v⟫ = r * (-(1 / 4)) := by
      simp only [v, real_inner_smul_right, inner_sub_right, hii, h10, h12]; ring
    have hz : ⟪e 2, r • v⟫ = r * (-1) := by
      simp only [v, real_inner_smul_right, inner_sub_right, hii, h20, h21]; ring
    rw [hx, hy, hz]
    nlinarith [mul_pos hpos hpos]

/-- **Undershoot (first order).** For orthonormal `e₀, e₁` and two pairs whose readout directions are orthogonal,
`w₁ = Δ₁ = e₀` and `w₂ = Δ₂ = e₁` (weights `1/2`), `tr M = 1` while every unit `d` has `⟪d, M d⟫ ≤ 1/2`: the best
rank-one interchange reaches half the trace, and a target-aimed objective with `τ = tr M` has an empty level set
(Corollary 6(iii) needs `tr M ∈ [λ_{n-1}, λ₀]`). -/
theorem counterexample_undershoot [FiniteDimensional ℝ E] (e : Fin 2 → E) (he : Orthonormal ℝ e) :
    let w : Fin 2 → E := ![e 0, e 1]
    let p : Fin 2 → ℝ := fun _ => 1 / 2
    LinearMap.trace ℝ E (effM p w w) = 1 ∧ (∀ d : E, ‖d‖ = 1 → ⟪d, effM p w w d⟫ ≤ 1 / 2) := by
  intro w p
  have hii : ∀ i, ⟪e i, e i⟫ = 1 := fun i => by rw [real_inner_self_eq_norm_sq, he.1 i]; ring
  refine ⟨?_, ?_⟩
  · rw [trace_effM]
    simp only [Fin.sum_univ_two, p, w, Matrix.cons_val_zero, Matrix.cons_val_one, hii]
    norm_num
  · intro d hd
    have hform : ⟪d, effM p w w d⟫ = (1 / 2) * ⟪e 0, d⟫ ^ 2 + (1 / 2) * ⟪e 1, d⟫ ^ 2 := by
      rw [← rank_one_effect_eq]
      simp only [Fin.sum_univ_two, p, w, Matrix.cons_val_zero, Matrix.cons_val_one, real_inner_smul_right]
      have c0 : ⟪d, e 0⟫ = ⟪e 0, d⟫ := real_inner_comm _ _
      have c1 : ⟪d, e 1⟫ = ⟪e 1, d⟫ := real_inner_comm _ _
      rw [c0, c1]; ring
    rw [hform]
    have hbessel : ⟪e 0, d⟫ ^ 2 + ⟪e 1, d⟫ ^ 2 ≤ 1 := by
      have := he.sum_inner_products_le (x := d) (s := Finset.univ)
      simp only [Fin.sum_univ_two, Real.norm_eq_abs, sq_abs, hd, one_pow] at this
      linarith
    nlinarith

/-- **A per-pair squared error does not aim at `T` (Appendix A.5).** For orthonormal `e₀, e₁, e₂` and the
two pairs `w₁ = e₀, Δ₁ = e₀ + 3e₁`, `w₂ = Δ₂ = e₂` (weights `1/2`, per-pair effects `Tᵢ = 1`): the target `T = tr M = 1`
is reached by the unit `(e₀ + e₁)/√2`, yet the per-pair squared error `∑ pᵢ (eᵢ(d) - Tᵢ)²` is at least `100/121` at every
unit `d` whose mean effect is `T`, and only `1/4` at the unit `(e₀ + e₂)/√2`; so it is not minimised where the mean
effect equals `T`. -/
theorem counterexample_sq_error [FiniteDimensional ℝ E] (e : Fin 3 → E) (he : Orthonormal ℝ e) :
    let w : Fin 2 → E := ![e 0, e 2]
    let Δ : Fin 2 → E := ![e 0 + (3 : ℝ) • e 1, e 2]
    let p : Fin 2 → ℝ := fun _ => 1 / 2
    let O3 : E → ℝ := fun d => ∑ i, p i * (⟪w i, d⟫ * ⟪d, Δ i⟫ - ⟪w i, Δ i⟫) ^ 2
    (∑ i, p i * ⟪w i, Δ i⟫ = 1) ∧
      (∃ d : E, ‖d‖ = 1 ∧ ⟪d, effM p w Δ d⟫ = 1) ∧
      (∀ d : E, ‖d‖ = 1 → ⟪d, effM p w Δ d⟫ = 1 → 100 / 121 ≤ O3 d) ∧
      ∃ d0 : E, ‖d0‖ = 1 ∧ O3 d0 = 1 / 4 := by
  intro w Δ p O3
  have hii : ∀ i, ⟪e i, e i⟫ = 1 := fun i => by rw [real_inner_self_eq_norm_sq, he.1 i]; ring
  have hij : ∀ i j, i ≠ j → ⟪e i, e j⟫ = 0 := fun i j h => he.2 h
  have h01 := hij 0 1 (by decide); have h02 := hij 0 2 (by decide); have h12 := hij 1 2 (by decide)
  have h10 := hij 1 0 (by decide); have h20 := hij 2 0 (by decide); have h21 := hij 2 1 (by decide)
  have hT1 : ⟪w 0, Δ 0⟫ = 1 := by
    simp only [w, Δ, Matrix.cons_val_zero, inner_add_right, real_inner_smul_right, hii, h01]; ring
  have hT2 : ⟪w 1, Δ 1⟫ = 1 := by simp only [w, Δ, Matrix.cons_val_one, Matrix.cons_val_zero, hii]
  -- the Rayleigh form in coordinates x = ⟪e₀,d⟫, y = ⟪e₁,d⟫, z = ⟪e₂,d⟫
  have hform : ∀ d : E, ⟪d, effM p w Δ d⟫ =
      (1 / 2) * (⟪e 0, d⟫ * (⟪e 0, d⟫ + 3 * ⟪e 1, d⟫)) + (1 / 2) * (⟪e 2, d⟫ * ⟪e 2, d⟫) := by
    intro d
    rw [← rank_one_effect_eq]
    simp only [Fin.sum_univ_two, p, w, Δ, Matrix.cons_val_zero, Matrix.cons_val_one, real_inner_smul_right,
      inner_add_right]
    have c0 : ⟪d, e 0⟫ = ⟪e 0, d⟫ := real_inner_comm _ _
    have c1 : ⟪d, e 1⟫ = ⟪e 1, d⟫ := real_inner_comm _ _
    have c2 : ⟪d, e 2⟫ = ⟪e 2, d⟫ := real_inner_comm _ _
    rw [c0, c1, c2]; ring
  have hO3 : ∀ d : E, O3 d = (1 / 2) * (⟪e 0, d⟫ * (⟪e 0, d⟫ + 3 * ⟪e 1, d⟫) - 1) ^ 2 +
      (1 / 2) * (⟪e 2, d⟫ * ⟪e 2, d⟫ - 1) ^ 2 := by
    intro d
    simp only [O3, Fin.sum_univ_two, p, hT1, hT2]
    simp only [w, Δ, Matrix.cons_val_zero, Matrix.cons_val_one, inner_add_right, real_inner_smul_right]
    have c0 : ⟪d, e 0⟫ = ⟪e 0, d⟫ := real_inner_comm _ _
    have c1 : ⟪d, e 1⟫ = ⟪e 1, d⟫ := real_inner_comm _ _
    have c2 : ⟪d, e 2⟫ = ⟪e 2, d⟫ := real_inner_comm _ _
    rw [c0, c1, c2]
  have hbessel : ∀ d : E, ‖d‖ = 1 → ⟪e 0, d⟫ ^ 2 + ⟪e 1, d⟫ ^ 2 + ⟪e 2, d⟫ ^ 2 ≤ 1 := by
    intro d hd
    have := he.sum_inner_products_le (x := d) (s := Finset.univ)
    simp only [Fin.sum_univ_three, Real.norm_eq_abs, sq_abs, hd, one_pow] at this
    linarith
  have hs2 : (√2) ^ 2 = 2 := Real.sq_sqrt (by norm_num)
  have hs2pos : 0 < √2 := by positivity
  refine ⟨?_, ?_, ?_, ?_⟩
  · simp only [Fin.sum_univ_two, p, hT1, hT2]; norm_num
  · -- the unit (e₀ + e₁)/√2 has mean effect 1
    refine ⟨(√2)⁻¹ • (e 0 + e 1), ?_, ?_⟩
    · rw [norm_smul, norm_inv, Real.norm_eq_abs, abs_of_pos hs2pos]
      have : ‖e 0 + e 1‖ = √2 := by
        rw [← Real.sqrt_sq (norm_nonneg _), ← real_inner_self_eq_norm_sq]
        congr 1
        rw [inner_add_left, inner_add_right, inner_add_right, hii, hii, h01, h10]; ring
      rw [this, inv_mul_cancel₀ hs2pos.ne']
    · rw [hform]
      simp only [real_inner_smul_right, inner_add_right, hii, h01, h10, h20, h21]
      field_simp
      nlinarith [hs2]
  · intro d hd hlev
    rw [hform] at hlev
    rw [hO3]
    have hb := hbessel d hd
    set x := ⟪e 0, d⟫; set y := ⟪e 1, d⟫; set z := ⟪e 2, d⟫
    -- on the level set: z² = 2 - x² - 3xy, so both pairs are off by x² + 3xy - 1 ≥ x² + y² ≥ 10/11
    have hz : z * z = 2 - x * (x + 3 * y) := by linarith
    have hge : 3 * x * y - y ^ 2 ≥ 1 := by nlinarith
    have hpsd : 0 ≤ 11 * x ^ 2 - 30 * x * y + 21 * y ^ 2 := by nlinarith [sq_nonneg (11 * x - 15 * y), sq_nonneg y]
    have hr : 10 / 11 ≤ x ^ 2 + y ^ 2 := by nlinarith
    have hoff : x ^ 2 + y ^ 2 ≤ x * (x + 3 * y) - 1 := by nlinarith
    rw [hz]
    nlinarith [hr, hoff]
  · -- the unit (e₀ + e₂)/√2 has per-pair effects 1/2 and 1/2
    refine ⟨(√2)⁻¹ • (e 0 + e 2), ?_, ?_⟩
    · rw [norm_smul, norm_inv, Real.norm_eq_abs, abs_of_pos hs2pos]
      have : ‖e 0 + e 2‖ = √2 := by
        rw [← Real.sqrt_sq (norm_nonneg _), ← real_inner_self_eq_norm_sq]
        congr 1
        rw [inner_add_left, inner_add_right, inner_add_right, hii, hii, h02, h20]; ring
      rw [this, inv_mul_cancel₀ hs2pos.ne']
    · rw [hO3]
      simp only [real_inner_smul_right, inner_add_right, hii, h01, h02, h10, h12, h20, h21]
      field_simp
      nlinarith [hs2]

end Paper
