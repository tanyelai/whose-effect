import Paper.Expansion

set_option linter.style.header false
set_option linter.style.longLine false

/-!
# Appendix A.1 for a general patch: the expansion to third order

`full_swap_third_order` bounds the remainder of the first-order expansion for the full swap. For a general patch
`h_b ↦ h_b + v` (a subspace patch, rank one or larger, at any site), the expansion about the midpoint `c = h_b + Δ/2`
is `F(h_b + v) - F(h_b) = ⟪w, v⟫ + ½ D²F(c)(v, v - Δ) + R`. With `a = v - Δ/2` and `b = -Δ/2` the two states are `c + a`
and `c + b`; along each segment `F` has a second-order Taylor expansion with a Lagrange remainder, and the difference of
the two quadratic terms is `½ D²F(c)(v, v - Δ)` by the symmetry of the second derivative (`midpoint_second_order` in
Hessian form). So `|R| ≤ (L_a + L_b)/6`, with `L_a`, `L_b` bounds on the third derivative along the two segments; each
scales as the cube of its segment's length, at most `‖Δ‖³` when `‖v - Δ/2‖ ≤ ‖Δ‖/2`.
-/

namespace Paper

open scoped RealInnerProductSpace
open Real Set

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

/-- One-sided Taylor to second order with the Lagrange remainder: if `φ` is `C³` and `|φ'''| ≤ L` on `[0, 1]`, then
`|φ 1 - φ 0 - φ' 0 - φ'' 0 / 2| ≤ L / 6`. -/
theorem taylor_two_unit (φ : ℝ → ℝ) (hφ : ContDiff ℝ 3 φ) (L : ℝ)
    (hL : ∀ t ∈ Icc (0 : ℝ) 1, |iteratedDeriv 3 φ t| ≤ L) :
    |φ 1 - φ 0 - deriv φ 0 - iteratedDeriv 2 φ 0 / 2| ≤ L / 6 := by
  have h01 : (0 : ℝ) ≠ 1 := by norm_num
  obtain ⟨ξ, hξ, hT⟩ := taylor_mean_remainder_lagrange_iteratedDeriv (f := φ) (n := 2) h01
    (hφ.contDiffOn.of_le (by norm_num))
  have hu : UniqueDiffOn ℝ (uIcc (0 : ℝ) 1) := uniqueDiffOn_uIcc h01
  have h0mem : (0 : ℝ) ∈ uIcc (0 : ℝ) 1 := left_mem_uIcc
  have hwithin : ∀ k : ℕ, k ≤ 3 → iteratedDerivWithin k φ (uIcc 0 1) 0 = iteratedDeriv k φ 0 := by
    intro k hk
    have hk' : ((k : ℕ) : WithTop ℕ∞) ≤ 3 := by exact_mod_cast hk
    exact iteratedDerivWithin_eq_iteratedDeriv hu (hφ.of_le hk').contDiffAt h0mem
  have hpoly : taylorWithinEval φ 2 (uIcc 0 1) 0 1 = φ 0 + deriv φ 0 + iteratedDeriv 2 φ 0 / 2 := by
    rw [taylor_within_apply]
    simp only [Finset.sum_range_succ, Finset.sum_range_zero]
    rw [hwithin 0 (by norm_num), hwithin 1 (by norm_num), hwithin 2 (by norm_num)]
    simp [iteratedDeriv_one]
    ring
  rw [hpoly] at hT
  have hξ01 : 0 < ξ ∧ ξ < 1 := by
    rw [uIoo_of_le zero_le_one] at hξ
    exact hξ
  have hb := hL ξ ⟨hξ01.1.le, hξ01.2.le⟩
  have e : φ 1 - φ 0 - deriv φ 0 - iteratedDeriv 2 φ 0 / 2 =
      φ 1 - (φ 0 + deriv φ 0 + iteratedDeriv 2 φ 0 / 2) := by ring
  rw [e, hT]
  norm_num
  rw [abs_div, abs_of_pos (by norm_num : (0 : ℝ) < 6)]
  linarith

/-- Along the line `t ↦ c + t a`, the second derivative of `F` at `t = 0` is `D²F(c)(a, a)`. -/
theorem second_deriv_along_line (F : E → ℝ) (hF1 : Differentiable ℝ F) (c a : E)
    (hF2 : DifferentiableAt ℝ (fderiv ℝ F) c) :
    iteratedDeriv 2 (fun t : ℝ => F (c + t • a)) 0 = fderiv ℝ (fderiv ℝ F) c a a := by
  have hline : ∀ t : ℝ, HasDerivAt (fun s : ℝ => c + s • a) a t := fun t => by
    simpa using ((hasDerivAt_id t).smul_const a).const_add c
  have hd1 : deriv (fun t : ℝ => F (c + t • a)) = fun t => fderiv ℝ F (c + t • a) a := by
    funext t
    exact ((hF1 (c + t • a)).hasFDerivAt.comp_hasDerivAt t (hline t)).deriv
  rw [iteratedDeriv_succ, iteratedDeriv_one, hd1]
  have hF2' : HasFDerivAt (fderiv ℝ F) (fderiv ℝ (fderiv ℝ F) c) (c + (0 : ℝ) • a) := by
    simpa using hF2.hasFDerivAt
  have hg : HasDerivAt (fun t : ℝ => fderiv ℝ F (c + t • a)) (fderiv ℝ (fderiv ℝ F) c a) 0 :=
    hF2'.comp_hasDerivAt (0 : ℝ) (hline 0)
  have hev := hg.clm_apply (hasDerivAt_const (0 : ℝ) a)
  simpa using hev.deriv

/-- **Appendix A.1, a general patch, to third order.** For a patch `h_b ↦ h_b + v`, with `c = h_b + Δ/2` the midpoint,
`w` the gradient there and `D²F(c)` the second derivative, the effect differs from `⟪w, v⟫ + ½ D²F(c)(v, v - Δ)` by at most
`(L_a + L_b)/6`, where `L_a` and `L_b` bound the third derivative of `F` along the segments from `c` to `h_b + v` and
from `c` to `h_b`. For the full swap (`v = Δ`) the second-order term vanishes (`full_swap_second_order`). -/
theorem general_patch_third_order [CompleteSpace E] (F : E → ℝ) (hb Δ v w : E) (hF1 : Differentiable ℝ F)
    (hF2 : DifferentiableAt ℝ (fderiv ℝ F) (hb + (1 / 2 : ℝ) • Δ))
    (hgrad : HasGradientAt F w (hb + (1 / 2 : ℝ) • Δ))
    (hφa : ContDiff ℝ 3 (fun t : ℝ => F (hb + (1 / 2 : ℝ) • Δ + t • (v - (1 / 2 : ℝ) • Δ))))
    (hφb : ContDiff ℝ 3 (fun t : ℝ => F (hb + (1 / 2 : ℝ) • Δ + t • (-((1 / 2 : ℝ) • Δ))))) (La Lb : ℝ)
    (hLa : ∀ t ∈ Icc (0 : ℝ) 1,
      |iteratedDeriv 3 (fun t : ℝ => F (hb + (1 / 2 : ℝ) • Δ + t • (v - (1 / 2 : ℝ) • Δ))) t| ≤ La)
    (hLb : ∀ t ∈ Icc (0 : ℝ) 1,
      |iteratedDeriv 3 (fun t : ℝ => F (hb + (1 / 2 : ℝ) • Δ + t • (-((1 / 2 : ℝ) • Δ)))) t| ≤ Lb) :
    |F (hb + v) - F hb - ⟪w, v⟫ - (1 / 2 : ℝ) * fderiv ℝ (fderiv ℝ F) (hb + (1 / 2 : ℝ) • Δ) v (v - Δ)|
      ≤ (La + Lb) / 6 := by
  set c := hb + (1 / 2 : ℝ) • Δ with hc
  set a := v - (1 / 2 : ℝ) • Δ with ha
  set b := -((1 / 2 : ℝ) • Δ) with hbdef
  set B := fderiv ℝ (fderiv ℝ F) c with hB
  have Ta := taylor_two_unit (fun t : ℝ => F (c + t • a)) hφa La hLa
  have Tb := taylor_two_unit (fun t : ℝ => F (c + t • b)) hφb Lb hLb
  -- first derivatives along the two lines: the gradient
  have hderiv : ∀ u : E, deriv (fun t : ℝ => F (c + t • u)) 0 = ⟪w, u⟫ := by
    intro u
    have hline : HasDerivAt (fun t : ℝ => c + t • u) u 0 := by
      simpa using ((hasDerivAt_id (0 : ℝ)).smul_const u).const_add c
    have hF : HasFDerivAt F (InnerProductSpace.toDual ℝ E w) (c + (0 : ℝ) • u) := by
      simpa using hgrad.hasFDerivAt
    rw [show (fun t : ℝ => F (c + t • u)) = F ∘ (fun t : ℝ => c + t • u) from rfl,
      (hF.comp_hasDerivAt (0 : ℝ) hline).deriv]
    simp [InnerProductSpace.toDual_apply_apply]
  -- second derivatives along the two lines: the second derivative of `F` at `c`
  have h2a := second_deriv_along_line F hF1 c a hF2
  have h2b := second_deriv_along_line F hF1 c b hF2
  simp only [one_smul, zero_smul, add_zero] at Ta Tb
  rw [hderiv a, h2a] at Ta
  rw [hderiv b, h2b] at Tb
  -- the endpoints
  have ea : c + a = hb + v := by rw [hc, ha]; abel
  have eb : c + b = hb := by rw [hc, hbdef]; abel
  rw [ea] at Ta
  rw [eb] at Tb
  -- the symmetry of the second derivative and the quadratic identity
  have hsymm : ∀ x y : E, B x y = B y x := fun x y =>
    second_derivative_symmetric (fun y => (hF1 y).hasFDerivAt) hF2.hasFDerivAt x y
  have hquad : B a a / 2 - B b b / 2 = (1 / 2 : ℝ) * B v (v - Δ) := by
    have hv : v = a - b := by rw [ha, hbdef]; abel
    have hvd : v - Δ = a + b := by rw [ha, hbdef]; module
    rw [hvd, hv]
    simp only [map_sub, map_add, sub_apply]
    rw [hsymm b a]
    ring
  have hw : ⟪w, a⟫ - ⟪w, b⟫ = ⟪w, v⟫ := by
    rw [← inner_sub_right]; congr 1; rw [ha, hbdef]; abel
  have key : F (hb + v) - F hb - ⟪w, v⟫ - (1 / 2 : ℝ) * B v (v - Δ) =
      (F (hb + v) - F c - ⟪w, a⟫ - B a a / 2) - (F hb - F c - ⟪w, b⟫ - B b b / 2) := by
    rw [← hquad, ← hw]; ring
  rw [key]
  calc |(F (hb + v) - F c - ⟪w, a⟫ - B a a / 2) - (F hb - F c - ⟪w, b⟫ - B b b / 2)|
      ≤ |F (hb + v) - F c - ⟪w, a⟫ - B a a / 2| + |F hb - F c - ⟪w, b⟫ - B b b / 2| := abs_sub _ _
    _ ≤ La / 6 + Lb / 6 := add_le_add Ta Tb
    _ = (La + Lb) / 6 := by ring

end Paper
