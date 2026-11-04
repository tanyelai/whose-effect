import Mathlib

/-!
# Appendix A.1, one dimension: the symmetric difference to first order, the core of the full swap's expansion

For `φ` three times continuously differentiable with `|φ'''| ≤ L` on `[-1, 1]`,

`φ(1) - φ(-1) = 2 φ'(0) + R`, with `|R| ≤ L/3`, and the constant `1/3` is sharp.

The proof applies Taylor's theorem with the Lagrange remainder to the odd part `g(t) = φ(t) - φ(-t)` on `[0, 1]`: the
constant and quadratic Taylor terms of `g` vanish, the linear term is `2 φ'(0)`, and `g'''(t) = φ'''(t) + φ'''(-t)` is at
most `2L`. `Paper.Expansion` applies it along the segment of a swap (`full_swap_third_order`).
-/

namespace Paper

open Set

/-- The iterated derivatives of the odd part `t ↦ φ t - φ (-t)`. -/
lemma iteratedDeriv_oddPart (φ : ℝ → ℝ) (hφ : ContDiff ℝ 3 φ) (k : ℕ) (hk : k ≤ 3) (t : ℝ) :
    iteratedDeriv k (fun s => φ s - φ (-s)) t =
      iteratedDeriv k φ t - (-1) ^ k * iteratedDeriv k φ (-t) := by
  have hk' : ((k : ℕ) : WithTop ℕ∞) ≤ 3 := by exact_mod_cast hk
  have h1 : ContDiffAt ℝ k φ t := (hφ.of_le hk').contDiffAt
  have h2 : ContDiffAt ℝ k (fun s => φ (-s)) t := ((hφ.of_le hk').comp contDiff_neg).contDiffAt
  have hfun : (fun s => φ s - φ (-s)) = φ - (fun s => φ (-s)) := rfl
  rw [hfun, iteratedDeriv_sub h1 h2, iteratedDeriv_comp_neg, smul_eq_mul]

/-- **The symmetric difference to first order.** If `φ` is `C³` and `|φ'''| ≤ L` on `[-1, 1]`, then
`|φ(1) - φ(-1) - 2 φ'(0)| ≤ L / 3`. -/
theorem symm_diff_first_order (φ : ℝ → ℝ) (hφ : ContDiff ℝ 3 φ) (L : ℝ)
    (hL : ∀ t ∈ Icc (-1 : ℝ) 1, |iteratedDeriv 3 φ t| ≤ L) :
    |φ 1 - φ (-1) - 2 * deriv φ 0| ≤ L / 3 := by
  have hgC : ContDiff ℝ 3 (fun s => φ s - φ (-s)) := hφ.sub (hφ.comp contDiff_neg)
  have h01 : (0 : ℝ) ≠ 1 := by norm_num
  -- Taylor with Lagrange remainder for the odd part, degree 2, from 0 to 1
  obtain ⟨ξ, hξ, hT⟩ := taylor_mean_remainder_lagrange_iteratedDeriv (f := fun s => φ s - φ (-s)) (n := 2) h01
    (hgC.contDiffOn.of_le (by norm_num))
  -- the Taylor polynomial: derivatives within `[0, 1]` at `0` are ordinary derivatives
  have hu : UniqueDiffOn ℝ (uIcc (0 : ℝ) 1) := uniqueDiffOn_uIcc h01
  have h0mem : (0 : ℝ) ∈ uIcc (0 : ℝ) 1 := left_mem_uIcc
  have hwithin : ∀ k : ℕ, k ≤ 3 →
      iteratedDerivWithin k (fun s => φ s - φ (-s)) (uIcc 0 1) 0 = iteratedDeriv k (fun s => φ s - φ (-s)) 0 := by
    intro k hk
    have hk' : ((k : ℕ) : WithTop ℕ∞) ≤ 3 := by exact_mod_cast hk
    exact iteratedDerivWithin_eq_iteratedDeriv hu (hgC.of_le hk').contDiffAt h0mem
  have hpoly : taylorWithinEval (fun s => φ s - φ (-s)) 2 (uIcc 0 1) 0 1 = 2 * deriv φ 0 := by
    rw [taylor_within_apply]
    simp only [Finset.sum_range_succ, Finset.sum_range_zero]
    rw [hwithin 0 (by norm_num), hwithin 1 (by norm_num), hwithin 2 (by norm_num),
      iteratedDeriv_oddPart φ hφ 0 (by norm_num), iteratedDeriv_oddPart φ hφ 1 (by norm_num),
      iteratedDeriv_oddPart φ hφ 2 (by norm_num)]
    simp [iteratedDeriv_one]
    ring
  -- the remainder: `g'''(ξ) = φ'''(ξ) + φ'''(-ξ)`
  rw [hpoly, iteratedDeriv_oddPart φ hφ (2 + 1) (by norm_num)] at hT
  try beta_reduce at hT
  have hξ01 : 0 < ξ ∧ ξ < 1 := by
    rw [uIoo_of_le zero_le_one] at hξ
    exact hξ
  have b1 := hL ξ ⟨by linarith, by linarith⟩
  have b2 := hL (-ξ) ⟨by linarith, by linarith⟩
  rw [hT]
  norm_num
  rw [abs_div, abs_of_pos (by norm_num : (0 : ℝ) < 6)]
  have htri := abs_add_le (iteratedDeriv 3 φ ξ) (iteratedDeriv 3 φ (-ξ))
  rw [div_le_iff₀ (by norm_num : (0 : ℝ) < 6)]
  linarith

/-- The constant `⅓` cannot be improved: the cubic `φ t = L t³ / 6` (third derivative `L` everywhere) attains it. -/
lemma symm_diff_first_order_sharp (L : ℝ) :
    let φ : ℝ → ℝ := fun t => L * t ^ 3 / 6
    iteratedDeriv 3 φ = (fun _ => L) ∧ φ 1 - φ (-1) - 2 * deriv φ 0 = L / 3 := by
  intro φ
  have h1 : deriv φ = fun t => L * t ^ 2 / 2 := by
    funext t
    have := ((hasDerivAt_pow 3 t).const_mul L).div_const 6
    rw [this.deriv]; push_cast; ring
  have h2 : deriv (deriv φ) = fun t => L * t := by
    rw [h1]; funext t
    have := ((hasDerivAt_pow 2 t).const_mul L).div_const 2
    rw [this.deriv]; push_cast; ring
  have h3 : deriv (deriv (deriv φ)) = fun _ => L := by
    rw [h2]; funext t
    have := (hasDerivAt_id' t).const_mul L
    rw [this.deriv]; simp
  refine ⟨?_, ?_⟩
  · rw [iteratedDeriv_succ, iteratedDeriv_succ, iteratedDeriv_one, h3]
  · rw [h1]; simp only [φ]; norm_num; ring

end Paper
