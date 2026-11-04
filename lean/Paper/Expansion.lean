import Paper.SymmetricDifference

set_option linter.style.header false
set_option linter.style.longLine false

/-!
# Appendix A.1 and Remark 10: the error of the first-order expansion, and attribution patching

* The midpoint expansion's second-order term `½ vᵀH(v - Δ)`, its vanishing for the full swap, its rank-one form, and its
  cancellation on mirrored pairs; the full swap is first-order exact up to a third-order remainder (Theorem 5(ii)).
* Attribution patching's second-order term cancels on mirrored pairs (Remark 10).
-/

namespace Paper

open scoped RealInnerProductSpace
open Real Set

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

/-! ### B.6 The midpoint expansion -/

/-- The second-order term of the expansion about the midpoint `c = h_b + Δ/2`, for a patch `h_b ↦ h_b + v`, is
`½ vᵀH(v - Δ)` (for symmetric `H`). -/
theorem midpoint_second_order (H : E →ₗ[ℝ] E) (hH : H.IsSymmetric) (v Δ : E) :
    (1 / 2 : ℝ) * ⟪v - (1 / 2 : ℝ) • Δ, H (v - (1 / 2 : ℝ) • Δ)⟫ -
        (1 / 2 : ℝ) * ⟪(1 / 2 : ℝ) • Δ, H ((1 / 2 : ℝ) • Δ)⟫ =
      (1 / 2 : ℝ) * ⟪v, H (v - Δ)⟫ := by
  have hs : ⟪v, H Δ⟫ = ⟪Δ, H v⟫ := by rw [real_inner_comm, hH]
  simp only [map_sub, map_smul, inner_sub_left, inner_sub_right, real_inner_smul_left, real_inner_smul_right]
  linear_combination (1 / 4 : ℝ) * hs

/-- For the full swap (`v = Δ`) the second-order term vanishes. -/
theorem full_swap_second_order (H : E →ₗ[ℝ] E) (Δ : E) : (1 / 2 : ℝ) * ⟪Δ, H (Δ - Δ)⟫ = 0 := by simp

/-- For a rank-one patch `v = ⟪d, Δ⟫ d` the second-order term is `-½ (dᵀΔ) dᵀH(I - ddᵀ)Δ`. -/
theorem rank_one_second_order (H : E →ₗ[ℝ] E) (d Δ : E) :
    (1 / 2 : ℝ) * ⟪⟪d, Δ⟫ • d, H (⟪d, Δ⟫ • d - Δ)⟫ = -(1 / 2 : ℝ) * ⟪d, Δ⟫ * ⟪d, H (Δ - ⟪d, Δ⟫ • d)⟫ := by
  have h : H (⟪d, Δ⟫ • d - Δ) = -(H (Δ - ⟪d, Δ⟫ • d)) := by rw [← map_neg, neg_sub]
  rw [real_inner_smul_left, h, inner_neg_right]
  ring

/-- On a mirrored pair (difference `-Δ`, measure `-z`, so `-H`, and patch `-v`) the second-order term is the negative
of the original's: the mean over a mirrored set has none. -/
theorem mirrored_second_order (H : E →ₗ[ℝ] E) (v Δ : E) :
    (1 / 2 : ℝ) * ⟪-v, (-H) (-v - -Δ)⟫ = -((1 / 2 : ℝ) * ⟪v, H (v - Δ)⟫) := by
  have h : (-H) (-v - -Δ) = H (v - Δ) := by
    simp only [LinearMap.neg_apply, map_sub, map_neg]; abel
  rw [h, inner_neg_left]
  ring

/-- **Theorem 5(ii), to third order.** With `φ t = F (c + t (Δ/2))` three times continuously differentiable and
`|φ'''| ≤ L` on `[-1, 1]`, the full swap's effect `F (c + Δ/2) - F (c - Δ/2)` differs from the first-order value
`⟪w, Δ⟫`, `w` the gradient at the midpoint `c`, by at most `L/3`; `L` scales as `‖Δ‖³`. -/
theorem full_swap_third_order [CompleteSpace E] (F : E → ℝ) (c Δ w : E) (hgrad : HasGradientAt F w c)
    (hφ : ContDiff ℝ 3 (fun t : ℝ => F (c + t • ((1 / 2 : ℝ) • Δ)))) (L : ℝ)
    (hL : ∀ t ∈ Icc (-1 : ℝ) 1, |iteratedDeriv 3 (fun t : ℝ => F (c + t • ((1 / 2 : ℝ) • Δ))) t| ≤ L) :
    |F (c + (1 / 2 : ℝ) • Δ) - F (c - (1 / 2 : ℝ) • Δ) - ⟪w, Δ⟫| ≤ L / 3 := by
  have h := symm_diff_first_order (fun t : ℝ => F (c + t • ((1 / 2 : ℝ) • Δ))) hφ L hL
  -- the derivative of `φ` at 0 is `⟪w, Δ/2⟫`
  have hline : HasDerivAt (fun t : ℝ => c + t • ((1 / 2 : ℝ) • Δ)) ((1 / 2 : ℝ) • Δ) 0 := by
    simpa using ((hasDerivAt_id (0 : ℝ)).smul_const ((1 / 2 : ℝ) • Δ)).const_add c
  have hF : HasFDerivAt F (InnerProductSpace.toDual ℝ E w) (c + (0 : ℝ) • ((1 / 2 : ℝ) • Δ)) := by
    simpa using hgrad.hasFDerivAt
  have hd := hF.comp_hasDerivAt (0 : ℝ) hline
  have hderiv : deriv (fun t : ℝ => F (c + t • ((1 / 2 : ℝ) • Δ))) 0 = ⟪w, (1 / 2 : ℝ) • Δ⟫ := by
    rw [show (fun t : ℝ => F (c + t • ((1 / 2 : ℝ) • Δ))) = F ∘ (fun t : ℝ => c + t • ((1 / 2 : ℝ) • Δ)) from rfl,
      hd.deriv]
    simp [InnerProductSpace.toDual_apply_apply, real_inner_smul_right]
  rw [hderiv, real_inner_smul_right] at h
  have e1 : F (c + (1 : ℝ) • ((1 / 2 : ℝ) • Δ)) = F (c + (1 / 2 : ℝ) • Δ) := by rw [one_smul]
  have e2 : F (c + (-1 : ℝ) • ((1 / 2 : ℝ) • Δ)) = F (c - (1 / 2 : ℝ) • Δ) := by
    rw [neg_one_smul, ← sub_eq_add_neg]
  rw [e1, e2] at h
  convert h using 2
  ring

/-! ### Remark 10 Attribution patching -/

/-- Attribution patching uses the gradient at the base point, `w_b = w - ½ H Δ + r`. On a mirrored pair
(`-Δ`, `-H`, remainder `r'`) the second-order terms cancel, so on a mirrored set it estimates the trace to third order. -/
theorem atp_mirrored (H : E →ₗ[ℝ] E) (w r r' Δ : E) :
    (⟪w - (1 / 2 : ℝ) • H Δ + r, Δ⟫ - ⟪w, Δ⟫) +
        (⟪-w - (1 / 2 : ℝ) • (-H) (-Δ) + r', -Δ⟫ - ⟪-w, -Δ⟫) = ⟪r, Δ⟫ - ⟪r', Δ⟫ := by
  simp only [LinearMap.neg_apply, map_neg, neg_neg, inner_add_left, inner_sub_left, real_inner_smul_left,
    inner_neg_left, inner_neg_right]
  ring

end Paper
