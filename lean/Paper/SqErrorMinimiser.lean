import Paper.Range

set_option linter.style.header false
set_option linter.style.longLine false

/-!
# Appendix A.5: where the per-pair squared error is minimised

The example of `counterexample_sq_error` (Appendix A.5, "A per-pair squared error can miss `T`"): orthonormal
`e₀, e₁, e₂` (the paper's `e₁, e₂, e₃`), two pairs `w₁ = e₀, Δ₁ = e₀ + 3e₁` and `w₂ = Δ₂ = e₂`, weights `1/2`, per-pair
effects `Tᵢ = 1`, target `T = 1`. That theorem shows the per-pair squared error `½ ∑ (eᵢ(d) - 1)²` is not minimised
where the mean effect is `T`. The theorems below find the minimum exactly.

In the coordinates `x = ⟪e₀, d⟫`, `y = ⟪e₁, d⟫`, `z = ⟪e₂, d⟫` the per-pair effects are `e₁(d) = x(x + 3y)` and
`e₂(d) = z²`, and `x² + y² + z² ≤ 1` for a unit `d`. The form `x(x + 3y)` has top eigenvalue `c = (1 + √10)/2` (so
`c² = c + 9/4`), hence `e₁ ≤ c(x² + y²) ≤ c(1 - z²)`: every unit `d` puts `(e₁, e₂)` in the half-plane
`e₁ + c e₂ ≤ c`. The point `(1, 1)` lies outside it, and the loss is half the squared distance from `(e₁, e₂)` to
`(1, 1)`. So

* the loss is at least `1/(2(1 + c²)) ≈ 0.0938` (`sq_error_lower_bound`);
* a unit `d` reaches that value (`sq_error_attained`);
* at every unit `d` that reaches it, the mean effect is `m* = (2c² - c + 1)/(2(1 + c²)) ≈ 0.711`
  (`sq_error_minimiser_mean`), which lies in `(0.71, 0.712)` and below `T = 1` (`sq_error_minimiser_mean_bounds`).

In closed form, `m* = (160 - 9√10)/185` and the minimum is `(30 - 4√10)/185` (`sq_error_closed_forms`).

The contrast (Appendix A.5, "Objectives aimed at a target"): when every pair reads the output along one direction, the same loss
reaches `0`, and every minimiser has mean effect exactly `T` (`sq_error_shared_readout`).
-/

namespace Paper

open scoped RealInnerProductSpace
open Real

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

/-- `c = (1 + √10)/2`, the top eigenvalue of the quadratic form `x(x + 3y)` (the matrix `[[1, 3/2], [3/2, 0]]`). -/
noncomputable def sqErrC : ℝ := (1 + √10) / 2

/-- The minimum of the per-pair squared error in the example, `1/(2(1 + c²))`. -/
noncomputable def sqErrMin : ℝ := 1 / (2 * (1 + sqErrC ^ 2))

/-- The mean effect at every minimiser of the per-pair squared error, `m* = (2c² - c + 1)/(2(1 + c²))`. -/
noncomputable def sqErrMean : ℝ := (2 * sqErrC ^ 2 - sqErrC + 1) / (2 * (1 + sqErrC ^ 2))

/-- `c` is a root of `c² = c + 9/4`, the characteristic equation of `[[1, 3/2], [3/2, 0]]`. -/
theorem sqErrC_sq : sqErrC ^ 2 = sqErrC + 9 / 4 := by
  have h : (√10) ^ 2 = 10 := Real.sq_sqrt (by norm_num)
  unfold sqErrC
  linear_combination (1 / 4 : ℝ) * h

/-- `c > 0`. -/
theorem sqErrC_pos : 0 < sqErrC := by
  unfold sqErrC
  positivity

/-- **The bound in coordinates.** If `x² + y² + z² ≤ 1`, the loss `½(x(x + 3y) - 1)² + ½(z² - 1)²` is at least
`1/(2(1 + c²))`. The reason: `x(x + 3y) ≤ c(x² + y²)`, so the two per-pair effects satisfy `e₁ + c e₂ ≤ c`, and the loss
is half the squared distance from `(e₁, e₂)` to `(1, 1)`, which is at least half the squared distance from `(1, 1)` to the
line `e₁ + c e₂ = c`. -/
theorem sq_error_coord_lower_bound (x y z : ℝ) (h : x ^ 2 + y ^ 2 + z ^ 2 ≤ 1) :
    sqErrMin ≤ (1 / 2) * (x * (x + 3 * y) - 1) ^ 2 + (1 / 2) * (z * z - 1) ^ 2 := by
  have hc2 := sqErrC_sq
  have hc := sqErrC_pos
  unfold sqErrMin
  set c := sqErrC
  -- the form `x(x + 3y)` is at most `c (x² + y²)`: `c (c(x² + y²) - x(x + 3y)) = (3x/2 - c y)²`
  have key : c * (c * (x ^ 2 + y ^ 2) - x * (x + 3 * y)) = (3 / 2 * x - c * y) ^ 2 := by
    linear_combination x ^ 2 * hc2
  have hform : x * (x + 3 * y) ≤ c * (x ^ 2 + y ^ 2) := by
    have h0 : 0 ≤ c * (c * (x ^ 2 + y ^ 2) - x * (x + 3 * y)) := by rw [key]; positivity
    have := (mul_nonneg_iff_of_pos_left hc).mp h0
    linarith
  -- so the per-pair effects lie in the half-plane `e₁ + c e₂ ≤ c`
  have hplane : x * (x + 3 * y) + c * (z * z) ≤ c := by nlinarith
  set u := 1 - x * (x + 3 * y)
  set v := 1 - z * z
  have huv : 1 ≤ u + c * v := by simp only [u, v]; nlinarith
  have hsq : 1 ≤ (u + c * v) ^ 2 := by nlinarith
  have hid : (1 + c ^ 2) * (u ^ 2 + v ^ 2) = (u + c * v) ^ 2 + (c * u - v) ^ 2 := by ring
  have hu : x * (x + 3 * y) - 1 = -u := by simp only [u]; ring
  have hv : z * z - 1 = -v := by simp only [v]; ring
  rw [hu, hv, div_le_iff₀ (by positivity)]
  nlinarith [sq_nonneg (c * u - v)]

/-- **The minimisers in coordinates.** If `x² + y² + z² ≤ 1` and the loss equals `1/(2(1 + c²))`, then the mean effect
`½ x(x + 3y) + ½ z²` is `m*`: equality forces `(1 - e₁, 1 - e₂)` to be the multiple `(1, c)/(1 + c²)` of the line's
normal, that is `e₁ = c²/(1 + c²)` and `e₂ = 1 - c/(1 + c²)`. -/
theorem sq_error_coord_minimiser_mean (x y z : ℝ) (h : x ^ 2 + y ^ 2 + z ^ 2 ≤ 1)
    (heq : (1 / 2) * (x * (x + 3 * y) - 1) ^ 2 + (1 / 2) * (z * z - 1) ^ 2 = sqErrMin) :
    (1 / 2) * (x * (x + 3 * y)) + (1 / 2) * (z * z) = sqErrMean := by
  have hc2 := sqErrC_sq
  have hc := sqErrC_pos
  unfold sqErrMin at heq
  unfold sqErrMean
  set c := sqErrC
  have key : c * (c * (x ^ 2 + y ^ 2) - x * (x + 3 * y)) = (3 / 2 * x - c * y) ^ 2 := by
    linear_combination x ^ 2 * hc2
  have hform : x * (x + 3 * y) ≤ c * (x ^ 2 + y ^ 2) := by
    have h0 : 0 ≤ c * (c * (x ^ 2 + y ^ 2) - x * (x + 3 * y)) := by rw [key]; positivity
    have := (mul_nonneg_iff_of_pos_left hc).mp h0
    linarith
  have hplane : x * (x + 3 * y) + c * (z * z) ≤ c := by nlinarith
  set u := 1 - x * (x + 3 * y)
  set v := 1 - z * z
  have huv : 1 ≤ u + c * v := by simp only [u, v]; nlinarith
  have hid : (1 + c ^ 2) * (u ^ 2 + v ^ 2) = (u + c * v) ^ 2 + (c * u - v) ^ 2 := by ring
  have hu : x * (x + 3 * y) - 1 = -u := by simp only [u]; ring
  have hv : z * z - 1 = -v := by simp only [v]; ring
  rw [hu, hv] at heq
  have hpos : 0 < 2 * (1 + c ^ 2) := by positivity
  -- the loss at its floor: `(u + c v)² + (c u - v)² = 1`
  have hone : (u + c * v) ^ 2 + (c * u - v) ^ 2 = 1 := by
    rw [← hid]
    field_simp at heq
    linear_combination heq
  -- so `c u = v` and `u + c v = 1`
  have hperp : c * u - v = 0 := by
    have : (c * u - v) ^ 2 ≤ 0 := by nlinarith
    exact pow_eq_zero_iff (n := 2) (by norm_num) |>.mp (le_antisymm this (sq_nonneg _))
  have hline : u + c * v = 1 := by
    have h1 : (u + c * v) ^ 2 = 1 := by rw [hperp] at hone; linarith
    nlinarith
  have hx : x * (x + 3 * y) = 1 - u := by simp only [u]; ring
  have hz : z * z = 1 - v := by simp only [v]; ring
  rw [hx, hz, eq_div_iff hpos.ne']
  have hv' : v = c * u := by linarith
  rw [hv'] at hline ⊢
  linear_combination (-(1 + c)) * hline

/-- **The per-pair squared error is at least `1/(2(1 + c²))`.** In the example of `counterexample_sq_error` (orthonormal
`e₀, e₁, e₂`; pairs `w₁ = e₀, Δ₁ = e₀ + 3e₁` and `w₂ = Δ₂ = e₂`; weights `1/2`; per-pair effects `Tᵢ = 1`), every unit `d`
has per-pair squared error `∑ pᵢ (eᵢ(d) - Tᵢ)²` at least `1/(2(1 + c²))`, with `c = (1 + √10)/2`. -/
theorem sq_error_lower_bound (e : Fin 3 → E) (he : Orthonormal ℝ e) :
    let w : Fin 2 → E := ![e 0, e 2]
    let Δ : Fin 2 → E := ![e 0 + (3 : ℝ) • e 1, e 2]
    let p : Fin 2 → ℝ := fun _ => 1 / 2
    let O3 : E → ℝ := fun d => ∑ i, p i * (⟪w i, d⟫ * ⟪d, Δ i⟫ - ⟪w i, Δ i⟫) ^ 2
    ∀ d : E, ‖d‖ = 1 → sqErrMin ≤ O3 d := by
  intro w Δ p O3 d hd
  have hii : ∀ i, ⟪e i, e i⟫ = 1 := fun i => by rw [real_inner_self_eq_norm_sq, he.1 i]; ring
  have h01 : ⟪e 0, e 1⟫ = 0 := he.2 (by decide)
  have hT1 : ⟪w 0, Δ 0⟫ = 1 := by
    simp only [w, Δ, Matrix.cons_val_zero, inner_add_right, real_inner_smul_right, hii, h01]; ring
  have hT2 : ⟪w 1, Δ 1⟫ = 1 := by simp only [w, Δ, Matrix.cons_val_one, Matrix.cons_val_zero, hii]
  have hO3 : O3 d = (1 / 2) * (⟪e 0, d⟫ * (⟪e 0, d⟫ + 3 * ⟪e 1, d⟫) - 1) ^ 2 +
      (1 / 2) * (⟪e 2, d⟫ * ⟪e 2, d⟫ - 1) ^ 2 := by
    simp only [O3, Fin.sum_univ_two, p, hT1, hT2]
    simp only [w, Δ, Matrix.cons_val_zero, Matrix.cons_val_one, inner_add_right, real_inner_smul_right]
    rw [real_inner_comm d (e 0), real_inner_comm d (e 1), real_inner_comm d (e 2)]
  have hbessel : ⟪e 0, d⟫ ^ 2 + ⟪e 1, d⟫ ^ 2 + ⟪e 2, d⟫ ^ 2 ≤ 1 := by
    have := he.sum_inner_products_le (x := d) (s := Finset.univ)
    simp only [Fin.sum_univ_three, Real.norm_eq_abs, sq_abs, hd, one_pow] at this
    linarith
  rw [hO3]
  exact sq_error_coord_lower_bound _ _ _ hbessel

/-- **The bound `1/(2(1 + c²))` is attained.** Some unit `d` has per-pair squared error exactly `1/(2(1 + c²))`: the
unit `d = t(2c e₀ + 3e₁) + √(1 - c/(1 + c²)) e₂`, where `(2c, 3)` is the top eigenvector of `[[1, 3/2], [3/2, 0]]` and `t`
gives its part in the span of `e₀, e₁` the squared norm `c/(1 + c²)`. There the per-pair effects are `e₁ = c²/(1 + c²)`
and `e₂ = 1 - c/(1 + c²)`. -/
theorem sq_error_attained (e : Fin 3 → E) (he : Orthonormal ℝ e) :
    let w : Fin 2 → E := ![e 0, e 2]
    let Δ : Fin 2 → E := ![e 0 + (3 : ℝ) • e 1, e 2]
    let p : Fin 2 → ℝ := fun _ => 1 / 2
    let O3 : E → ℝ := fun d => ∑ i, p i * (⟪w i, d⟫ * ⟪d, Δ i⟫ - ⟪w i, Δ i⟫) ^ 2
    ∃ d : E, ‖d‖ = 1 ∧ O3 d = sqErrMin := by
  intro w Δ p O3
  have hii : ∀ i, ⟪e i, e i⟫ = 1 := fun i => by rw [real_inner_self_eq_norm_sq, he.1 i]; ring
  have hij : ∀ i j, i ≠ j → ⟪e i, e j⟫ = 0 := fun i j h => he.2 h
  have h01 := hij 0 1 (by decide); have h02 := hij 0 2 (by decide); have h12 := hij 1 2 (by decide)
  have h10 := hij 1 0 (by decide); have h20 := hij 2 0 (by decide); have h21 := hij 2 1 (by decide)
  have hT1 : ⟪w 0, Δ 0⟫ = 1 := by
    simp only [w, Δ, Matrix.cons_val_zero, inner_add_right, real_inner_smul_right, hii, h01]; ring
  have hT2 : ⟪w 1, Δ 1⟫ = 1 := by simp only [w, Δ, Matrix.cons_val_one, Matrix.cons_val_zero, hii]
  have hc2 := sqErrC_sq
  have hc := sqErrC_pos
  set c := sqErrC
  have hq : 0 < 1 + c ^ 2 := by positivity
  set t := √(c / ((1 + c ^ 2) * (4 * c ^ 2 + 9)))
  set β := √(1 - c / (1 + c ^ 2))
  have ht : t ^ 2 = c / ((1 + c ^ 2) * (4 * c ^ 2 + 9)) := Real.sq_sqrt (by positivity)
  have hβ : β ^ 2 = 1 - c / (1 + c ^ 2) := by
    refine Real.sq_sqrt ?_
    rw [sub_nonneg, div_le_one hq]
    nlinarith
  set d : E := (2 * c * t) • e 0 + (3 * t) • e 1 + β • e 2
  have hx : ⟪e 0, d⟫ = 2 * c * t := by
    simp only [d, inner_add_right, real_inner_smul_right, hii, h01, h02]; ring
  have hy : ⟪e 1, d⟫ = 3 * t := by
    simp only [d, inner_add_right, real_inner_smul_right, hii, h10, h12]; ring
  have hz : ⟪e 2, d⟫ = β := by
    simp only [d, inner_add_right, real_inner_smul_right, hii, h20, h21]; ring
  -- the squared norm of the part in the span of `e₀, e₁` is `c/(1 + c²)`, and the form there is `c` times that
  have hxy : (2 * c * t) ^ 2 + (3 * t) ^ 2 = c / (1 + c ^ 2) := by
    have : (2 * c * t) ^ 2 + (3 * t) ^ 2 = (4 * c ^ 2 + 9) * t ^ 2 := by ring
    rw [this, ht]
    field_simp
  have he1 : (2 * c * t) * (2 * c * t + 3 * (3 * t)) = c ^ 2 / (1 + c ^ 2) := by
    have : (2 * c * t) * (2 * c * t + 3 * (3 * t)) = c * ((2 * c * t) ^ 2 + (3 * t) ^ 2) := by
      linear_combination (-(4 : ℝ) * c * t ^ 2) * hc2
    rw [this, hxy]
    ring
  refine ⟨d, ?_, ?_⟩
  · have hdd : ⟪d, d⟫ = 1 := by
      have : ⟪d, d⟫ = (2 * c * t) * ⟪e 0, d⟫ + (3 * t) * ⟪e 1, d⟫ + β * ⟪e 2, d⟫ := by
        change ⟪(2 * c * t) • e 0 + (3 * t) • e 1 + β • e 2, d⟫ = _
        simp only [inner_add_left, real_inner_smul_left]
      rw [this, hx, hy, hz]
      nlinarith [hxy, hβ]
    rw [real_inner_self_eq_norm_sq] at hdd
    rw [← Real.sqrt_sq (norm_nonneg d), hdd, Real.sqrt_one]
  · have hO3 : O3 d = (1 / 2) * (⟪e 0, d⟫ * (⟪e 0, d⟫ + 3 * ⟪e 1, d⟫) - 1) ^ 2 +
        (1 / 2) * (⟪e 2, d⟫ * ⟪e 2, d⟫ - 1) ^ 2 := by
      simp only [O3, Fin.sum_univ_two, p, hT1, hT2]
      simp only [w, Δ, Matrix.cons_val_zero, Matrix.cons_val_one, inner_add_right, real_inner_smul_right]
      rw [real_inner_comm d (e 0), real_inner_comm d (e 1), real_inner_comm d (e 2)]
    rw [hO3, hx, hy, hz, he1, ← sq, hβ]
    unfold sqErrMin
    field_simp
    ring

/-- **Every minimiser has mean effect `m* ≈ 0.711`, not `T`.** In the same example, every unit `d` whose per-pair
squared error equals its minimum `1/(2(1 + c²))` has mean effect `⟪d, M d⟫ = m* = (2c² - c + 1)/(2(1 + c²))`. -/
theorem sq_error_minimiser_mean (e : Fin 3 → E) (he : Orthonormal ℝ e) :
    let w : Fin 2 → E := ![e 0, e 2]
    let Δ : Fin 2 → E := ![e 0 + (3 : ℝ) • e 1, e 2]
    let p : Fin 2 → ℝ := fun _ => 1 / 2
    let O3 : E → ℝ := fun d => ∑ i, p i * (⟪w i, d⟫ * ⟪d, Δ i⟫ - ⟪w i, Δ i⟫) ^ 2
    ∀ d : E, ‖d‖ = 1 → O3 d = sqErrMin → ⟪d, effM p w Δ d⟫ = sqErrMean := by
  intro w Δ p O3 d hd hmin
  have hii : ∀ i, ⟪e i, e i⟫ = 1 := fun i => by rw [real_inner_self_eq_norm_sq, he.1 i]; ring
  have h01 : ⟪e 0, e 1⟫ = 0 := he.2 (by decide)
  have hT1 : ⟪w 0, Δ 0⟫ = 1 := by
    simp only [w, Δ, Matrix.cons_val_zero, inner_add_right, real_inner_smul_right, hii, h01]; ring
  have hT2 : ⟪w 1, Δ 1⟫ = 1 := by simp only [w, Δ, Matrix.cons_val_one, Matrix.cons_val_zero, hii]
  have hO3 : O3 d = (1 / 2) * (⟪e 0, d⟫ * (⟪e 0, d⟫ + 3 * ⟪e 1, d⟫) - 1) ^ 2 +
      (1 / 2) * (⟪e 2, d⟫ * ⟪e 2, d⟫ - 1) ^ 2 := by
    simp only [O3, Fin.sum_univ_two, p, hT1, hT2]
    simp only [w, Δ, Matrix.cons_val_zero, Matrix.cons_val_one, inner_add_right, real_inner_smul_right]
    rw [real_inner_comm d (e 0), real_inner_comm d (e 1), real_inner_comm d (e 2)]
  have hform : ⟪d, effM p w Δ d⟫ =
      (1 / 2) * (⟪e 0, d⟫ * (⟪e 0, d⟫ + 3 * ⟪e 1, d⟫)) + (1 / 2) * (⟪e 2, d⟫ * ⟪e 2, d⟫) := by
    rw [← rank_one_effect_eq]
    simp only [Fin.sum_univ_two, p, w, Δ, Matrix.cons_val_zero, Matrix.cons_val_one, real_inner_smul_right,
      inner_add_right]
    rw [real_inner_comm d (e 0), real_inner_comm d (e 1), real_inner_comm d (e 2)]
    ring
  have hbessel : ⟪e 0, d⟫ ^ 2 + ⟪e 1, d⟫ ^ 2 + ⟪e 2, d⟫ ^ 2 ≤ 1 := by
    have := he.sum_inner_products_le (x := d) (s := Finset.univ)
    simp only [Fin.sum_univ_three, Real.norm_eq_abs, sq_abs, hd, one_pow] at this
    linarith
  rw [hO3] at hmin
  rw [hform]
  exact sq_error_coord_minimiser_mean _ _ _ hbessel hmin

/-- **Closed forms.** `m* = (160 - 9√10)/185` and the minimum `1/(2(1 + c²))` is `(30 - 4√10)/185`. -/
theorem sq_error_closed_forms :
    sqErrMean = (160 - 9 * √10) / 185 ∧ sqErrMin = (30 - 4 * √10) / 185 := by
  have h10 : (√10) ^ 2 = 10 := Real.sq_sqrt (by norm_num)
  have hc2 := sqErrC_sq
  have hc := sqErrC_pos
  have hq : (2 * (1 + sqErrC ^ 2)) ≠ 0 := by positivity
  constructor
  · unfold sqErrMean
    rw [div_eq_div_iff hq (by norm_num)]
    unfold sqErrC
    linear_combination ((9 * √10 + 43) / 2) * h10
  · unfold sqErrMin
    rw [div_eq_div_iff hq (by norm_num)]
    unfold sqErrC
    linear_combination (2 * √10 - 11) * h10

/-- **The numbers.** The mean effect at the minimiser lies in `(0.71, 0.712)`, so below `T = 1`, and the minimum lies in
`(0.0937, 0.0939)`. -/
theorem sq_error_minimiser_mean_bounds :
    0.71 < sqErrMean ∧ sqErrMean < 0.712 ∧ sqErrMean < 1 ∧ 0.0937 < sqErrMin ∧ sqErrMin < 0.0939 := by
  obtain ⟨hm, hmin⟩ := sq_error_closed_forms
  have h10 : (√10) ^ 2 = 10 := Real.sq_sqrt (by norm_num)
  have hlo : (3.162 : ℝ) < √10 := by
    rw [Real.lt_sqrt (by norm_num)]; norm_num
  have hhi : √10 < (3.1623 : ℝ) := by
    rw [Real.sqrt_lt' (by norm_num)]; norm_num
  rw [hm, hmin]
  refine ⟨?_, ?_, ?_, ?_, ?_⟩ <;> rw [← sub_pos] <;> norm_num <;> linarith
section SharedReadout

variable {ι : Type*} [Fintype ι]

/-- **The per-pair squared error hits `T` when one readout direction serves every pair** (Appendix A.5, "Objectives
aimed at a target"). If every pair reads the output along a multiple of one direction, `w i = c i • w̄` with `w̄ ≠ 0` (as the log odds
of the same two labels do under a readout linear in the state), the unit `w̄/‖w̄‖` reproduces every pair's first-order
effect, so the loss reaches `0`; with positive weights, every unit `d` that minimises it reproduces each pair's own
`Tᵢ = ⟪wᵢ, Δᵢ⟫`, so its mean effect `⟪d, M d⟫` is `T = ∑ pᵢ Tᵢ`. -/
theorem sq_error_shared_readout (p c : ι → ℝ) (hp : ∀ i, 0 < p i) (wbar : E) (hw : wbar ≠ 0) (Δ : ι → E) :
    let w : ι → E := fun i => c i • wbar
    let O : E → ℝ := fun d => ∑ i, p i * (⟪w i, d⟫ * ⟪d, Δ i⟫ - ⟪w i, Δ i⟫) ^ 2
    (∃ d0 : E, ‖d0‖ = 1 ∧ O d0 = 0) ∧
      ∀ d : E, ‖d‖ = 1 → (∀ d' : E, ‖d'‖ = 1 → O d ≤ O d') →
        (∀ i, ⟪w i, d⟫ * ⟪d, Δ i⟫ = ⟪w i, Δ i⟫) ∧ ⟪d, effM p w Δ d⟫ = ∑ i, p i * ⟪w i, Δ i⟫ := by
  intro w O
  have hn : ‖wbar‖ ≠ 0 := norm_ne_zero_iff.mpr hw
  have hd0n : ‖(‖wbar‖⁻¹ : ℝ) • wbar‖ = 1 := by
    rw [norm_smul, norm_inv, norm_norm, inv_mul_cancel₀ hn]
  have hrep : ∀ i, ⟪w i, (‖wbar‖⁻¹ : ℝ) • wbar⟫ * ⟪(‖wbar‖⁻¹ : ℝ) • wbar, Δ i⟫ = ⟪w i, Δ i⟫ := by
    intro i
    simp only [w, real_inner_smul_left, real_inner_smul_right, real_inner_self_eq_norm_sq]
    field_simp
  have hO0 : O ((‖wbar‖⁻¹ : ℝ) • wbar) = 0 := by
    simp only [O, hrep, sub_self]; simp
  have hnonneg : ∀ d i, 0 ≤ p i * (⟪w i, d⟫ * ⟪d, Δ i⟫ - ⟪w i, Δ i⟫) ^ 2 :=
    fun d i => mul_nonneg (hp i).le (sq_nonneg _)
  refine ⟨⟨_, hd0n, hO0⟩, ?_⟩
  intro d _ hmin
  have hzero : O d = 0 :=
    le_antisymm ((hmin _ hd0n).trans_eq hO0) (Finset.sum_nonneg fun i _ => hnonneg d i)
  have heach : ∀ i, ⟪w i, d⟫ * ⟪d, Δ i⟫ = ⟪w i, Δ i⟫ := by
    intro i
    have ht := (Finset.sum_eq_zero_iff_of_nonneg (fun j _ => hnonneg d j)).mp hzero i (Finset.mem_univ i)
    rcases mul_eq_zero.mp ht with h | h
    · exact absurd h (hp i).ne'
    · exact sub_eq_zero.mp (pow_eq_zero_iff two_ne_zero |>.mp h)
  refine ⟨heach, ?_⟩
  rw [← rank_one_effect_eq]
  apply Finset.sum_congr rfl
  intro i _
  rw [real_inner_smul_right, ← heach i]
  ring

end SharedReadout

end Paper
