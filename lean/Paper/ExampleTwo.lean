import Paper.Natural

set_option linter.style.header false
set_option linter.style.longLine false

/-!
# Example 1: two dimensions

The change of the state is `Δ = e₀`, along the variable, and the gradient is `w = e₀ + e₁`, at `45°` to it, for orthonormal
`e₀, e₁` (the paper's coordinates `(1, 0)` and `(0, 1)`). The single pair's matrix is `M = sym(w Δᵀ)` (`Paper.effM1`). The
paper's decimals are exact values: `1.207 = (1 + √2)/2`, `-0.207 = (1 - √2)/2` and `0.354 = √2/4`. The bisector of `w` and
`Δ` is `d = cos(π/8) e₀ + sin(π/8) e₁`, the orthogonal direction `d⊥ = -sin(π/8) e₀ + cos(π/8) e₁`.
-/

namespace Paper

open scoped RealInnerProductSpace
open Real

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

/-- **Example 1.** With `Δ = e₀`, `w = e₀ + e₁`, `c = cos(π/8)`, `s = sin(π/8)`, `d = c e₀ + s e₁` and `d⊥ = -s e₀ + c e₁`:
(a) the direction of the variable gives exactly `1`, the trace `⟪w, Δ⟫`; (b) `d` and `d⊥` are orthonormal, with effects
`(1 + √2)/2` and `(1 - √2)/2`, which sum to the trace; (c) they are eigenvectors of `M` with these eigenvalues; (d) the
patch along `d` moves the state by `√2/4` along `e₁`, which the input does not change, and the patch along `d⊥` by `-√2/4`;
(e) with the gradient turned to `w = e₁` the trace is `0`, while the unit bisector `(e₀ + e₁)/√2` gives `1/2`. -/
theorem two_dimensions_example (e : Fin 2 → E) (he : Orthonormal ℝ e) :
    let Δ := e 0
    let w := e 0 + e 1
    let c := cos (π / 8)
    let s := sin (π / 8)
    let d := c • e 0 + s • e 1
    let d' := (-s) • e 0 + c • e 1
    (⟪e 0, effM1 w Δ (e 0)⟫ = 1 ∧ ⟪w, Δ⟫ = 1) ∧
      (‖d‖ = 1 ∧ ‖d'‖ = 1 ∧ ⟪d, d'⟫ = 0 ∧ ⟪d, effM1 w Δ d⟫ = (1 + √2) / 2 ∧
        ⟪d', effM1 w Δ d'⟫ = (1 - √2) / 2 ∧ ⟪d, effM1 w Δ d⟫ + ⟪d', effM1 w Δ d'⟫ = ⟪w, Δ⟫) ∧
      (effM1 w Δ d = ((1 + √2) / 2) • d ∧ effM1 w Δ d' = ((1 - √2) / 2) • d') ∧
      (⟪e 1, ⟪d, Δ⟫ • d⟫ = √2 / 4 ∧ ⟪e 1, ⟪d', Δ⟫ • d'⟫ = -(√2 / 4)) ∧
      (⟪e 1, Δ⟫ = 0 ∧ ‖(√2 / 2) • (e 0 + e 1)‖ = 1 ∧
        ⟪(√2 / 2) • (e 0 + e 1), effM1 (e 1) Δ ((√2 / 2) • (e 0 + e 1))⟫ = 1 / 2) := by
  intro Δ w c s d d'
  have hii : ∀ i, ⟪e i, e i⟫ = 1 := fun i => by rw [real_inner_self_eq_norm_sq, he.1 i]; ring
  have h01 : ⟪e 0, e 1⟫ = 0 := he.2 (by decide)
  have h10 : ⟪e 1, e 0⟫ = 0 := he.2 (by decide)
  have hunit : ∀ x : E, ⟪x, x⟫ = 1 → ‖x‖ = 1 := fun x h => by
    rw [real_inner_self_eq_norm_sq] at h; nlinarith [norm_nonneg x]
  -- `c` and `s` from the double-angle formulas at `π/4 = 2 (π/8)`
  have hr : √2 ^ 2 = 2 := Real.sq_sqrt (by norm_num)
  have hs : 0 < s := sin_pos_of_pos_of_lt_pi (by positivity) (by linarith [pi_pos])
  have hcs : c * s = √2 / 4 := by
    have h := sin_two_mul (π / 8)
    rw [show 2 * (π / 8) = π / 4 by ring, sin_pi_div_four] at h
    linarith
  have hcc : c ^ 2 = (2 + √2) / 4 := by
    have h := cos_sq (π / 8)
    rw [show 2 * (π / 8) = π / 4 by ring, cos_pi_div_four] at h
    linarith
  have hss : s ^ 2 = (2 - √2) / 4 := by nlinarith [sin_sq_add_cos_sq (π / 8)]
  have hc : c = (1 + √2) * s := by
    apply mul_right_cancel₀ hs.ne'
    linear_combination hcs - (1 + √2) * hss + (1 / 4) * hr
  refine ⟨⟨?_, ?_⟩, ⟨?_, ?_, ?_, ?_, ?_, ?_⟩, ⟨?_, ?_⟩, ⟨?_, ?_⟩, ⟨?_, ?_, ?_⟩⟩
  -- (a) the direction of the variable, and the trace
  · rw [natural_effect]; simp only [w, Δ, inner_add_right, hii, h01]; ring
  · simp only [w, Δ, inner_add_left, hii, h10]; ring
  -- (b) the orthonormal pair `d, d'` and the effects along it
  · apply hunit
    simp only [d, inner_add_left, inner_add_right, real_inner_smul_left, real_inner_smul_right, hii, h01, h10]
    linear_combination hcc + hss
  · apply hunit
    simp only [d', inner_add_left, inner_add_right, real_inner_smul_left, real_inner_smul_right, hii, h01, h10]
    linear_combination hcc + hss
  · simp only [d, d', inner_add_left, inner_add_right, real_inner_smul_left, real_inner_smul_right, hii, h01, h10]
    ring
  · rw [natural_effect]
    simp only [d, w, Δ, inner_add_left, inner_add_right, real_inner_smul_left, hii, h01, h10]
    linear_combination hcc + hcs
  · rw [natural_effect]
    simp only [d', w, Δ, inner_add_left, inner_add_right, real_inner_smul_left, hii, h01, h10]
    linear_combination hss - hcs
  · rw [natural_effect, natural_effect]
    simp only [d, d', w, Δ, inner_add_left, inner_add_right, real_inner_smul_left, hii, h01, h10]
    linear_combination hcc + hss
  -- (c) `d` and `d'` are eigenvectors
  · rw [effM_apply]
    simp only [Finset.univ_unique, Finset.sum_singleton, one_smul, d, w, Δ, inner_add_left, inner_add_right,
      real_inner_smul_right, hii, h01, h10]
    linear_combination (norm := module) (1 / 2 : ℝ) • ((1 - √2) * hc - s * hr) • e 0 + (1 / 2 : ℝ) • hc • e 1
  · rw [effM_apply]
    simp only [Finset.univ_unique, Finset.sum_singleton, one_smul, d', w, Δ, inner_add_left, inner_add_right,
      real_inner_smul_right, hii, h01, h10]
    linear_combination (norm := module) (1 / 2 : ℝ) • hc • e 0 + (1 / 2 : ℝ) • (-(1 - √2) * hc + s * hr) • e 1
  -- (d) the patches' components along `e 1`
  · simp only [d, Δ, inner_add_left, inner_add_right, real_inner_smul_left, real_inner_smul_right, hii, h10]
    linear_combination hcs
  · simp only [d', Δ, inner_add_left, inner_add_right, real_inner_smul_left, real_inner_smul_right, hii, h10]
    linear_combination -hcs
  -- (e) the gradient turned to `e 1`
  · simp only [Δ, h10]
  · apply hunit
    simp only [inner_add_left, inner_add_right, real_inner_smul_left, real_inner_smul_right, hii, h01, h10]
    linear_combination (1 / 2) * hr
  · rw [natural_effect]
    simp only [Δ, inner_add_left, real_inner_smul_left, hii, h01, h10]
    linear_combination (1 / 4) * hr

end Paper
