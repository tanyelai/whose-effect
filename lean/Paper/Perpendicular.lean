import Mathlib

set_option linter.style.header false
set_option linter.style.longLine false

/-!
# Section 3.2, after Example 1: a part exceeds the whole only through its perpendicular component

For one pair with readout gradient `w` and difference `Δ`, the rank-one interchange along a unit `d` moves the state by
`p = ⟪d, Δ⟫ d`. Split `p` into its component along `Δ`, `(⟪p, Δ⟫ / ‖Δ‖²) Δ`, and the rest, perpendicular to `Δ`. The part
along `Δ` is at most `Δ` itself (`⟪p, Δ⟫ = ⟪d, Δ⟫² ≤ ‖Δ‖²`), so where the whole effect `⟪w, Δ⟫` is positive, the
perpendicular component alone moves the output by at least the excess `⟪w, p⟫ - ⟪w, Δ⟫`, and a part exceeds the whole only
if that component moves the output upward.
-/

namespace Paper

open scoped RealInnerProductSpace

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

/-- **The perpendicular component carries the excess.** With `‖d‖ = 1`, `Δ ≠ 0` and `⟪w, Δ⟫ > 0`, the component of
`p = ⟪d, Δ⟫ d` perpendicular to `Δ` moves the output by at least `⟪w, p⟫ - ⟪w, Δ⟫`. -/
theorem part_exceeds_through_perpendicular (w Δ d : E) (hd : ‖d‖ = 1) (hΔ : Δ ≠ 0) (hpos : 0 < ⟪w, Δ⟫) :
    ⟪w, ⟪d, Δ⟫ • d⟫ - ⟪w, Δ⟫ ≤ ⟪w, ⟪d, Δ⟫ • d - (⟪⟪d, Δ⟫ • d, Δ⟫ / ‖Δ‖ ^ 2) • Δ⟫ := by
  have hn : 0 < ‖Δ‖ ^ 2 := by
    have := norm_pos_iff.mpr hΔ
    positivity
  have hp : ⟪⟪d, Δ⟫ • d, Δ⟫ = ⟪d, Δ⟫ ^ 2 := by
    rw [real_inner_smul_left]
    ring
  have hcs : ⟪d, Δ⟫ ^ 2 ≤ ‖Δ‖ ^ 2 := by
    have h := abs_real_inner_le_norm d Δ
    rw [hd, one_mul] at h
    calc ⟪d, Δ⟫ ^ 2 = |⟪d, Δ⟫| ^ 2 := (sq_abs _).symm
      _ ≤ ‖Δ‖ ^ 2 := by gcongr
  have hc1 : ⟪⟪d, Δ⟫ • d, Δ⟫ / ‖Δ‖ ^ 2 ≤ 1 := by
    rw [hp, div_le_one hn]
    exact hcs
  have hmul := mul_le_mul_of_nonneg_right hc1 hpos.le
  simp only [inner_sub_right, real_inner_smul_right] at hmul ⊢
  linarith

/-- **A part exceeds the whole only through its perpendicular component.** Where `0 < ⟪w, Δ⟫ < ⟪w, ⟪d, Δ⟫ d⟫` for a unit
`d`, the component of `⟪d, Δ⟫ d` perpendicular to `Δ` moves the output upward. -/
theorem part_exceeds_only_perpendicular (w Δ d : E) (hd : ‖d‖ = 1) (hΔ : Δ ≠ 0) (hpos : 0 < ⟪w, Δ⟫)
    (hex : ⟪w, Δ⟫ < ⟪w, ⟪d, Δ⟫ • d⟫) :
    0 < ⟪w, ⟪d, Δ⟫ • d - (⟪⟪d, Δ⟫ • d, Δ⟫ / ‖Δ‖ ^ 2) • Δ⟫ := by
  have := part_exceeds_through_perpendicular w Δ d hd hΔ hpos
  linarith

end Paper
