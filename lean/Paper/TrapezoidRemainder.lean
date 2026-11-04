import Mathlib

set_option linter.style.header false
set_option linter.style.longLine false

/-!
# Proposition 9 for a general path: the remainder beyond degree four

`Paper/Trapezoid.lean` proves the relation between the two traces exactly for every path `g` of degree at most four. The
theorems below prove it for every path `g : ℝ → ℝ` that is five times continuously differentiable, with an explicit
remainder.
Write `T = g 1 - g 0` (the whole-input effect), `trapezoid = (g'(0) + g'(1))/2` (the base-point trace) and
`midpoint = g'(1/2)` (the midpoint trace). If `|g⁽⁵⁾| ≤ L` on `[0, 1]`, then

* Simpson's combination misses `T` by little: `|(trapezoid + 2 midpoint)/3 - T| ≤ L/720` (`simpson_remainder`);
* the midpoint's error is `-1/2` of the trapezoid's up to the same order:
  `|(midpoint - T) + (1/2)(trapezoid - T)| ≤ L/480` (`midpoint_neg_half_remainder`; the left side is `3/2` times
  Simpson's error).

If only `|g⁽⁴⁾| ≤ L₄` on `[0, 1]` is known (`g` four times continuously differentiable), the two errors are, to leading
order, `g'''(1/2)/12` and `-g'''(1/2)/24`:

* `|trapezoid - T - g'''(1/2)/12| ≤ 5 L₄/192` (`trapezoid_leading_order`);
* `|midpoint - T + g'''(1/2)/24| ≤ L₄/192` (`midpoint_leading_order`).

Remark 10 compares attribution patching's base-point trace with the midpoint trace directly. On a mirrored pair at a
complete cut, `|trapezoid - midpoint| ≤ L/8` when `|g'''| ≤ L` (`atp_mirrored_third_order`, third order); for a single pair,
`|g'(0) - g'(1/2)| ≤ L₂/2` when `|g''| ≤ L₂` (`atp_single_second_order`, second order).

The proof expands `g` and `g'` by Taylor's theorem with the Lagrange remainder about the midpoint `c = 1/2`, out to
`c ± 1/2`, and checks that every polynomial term cancels. For Simpson's rule the even-order terms (`g`, `g''`, `g''''`
at `c`) cancel by symmetry, the `g'(c)` terms cancel as `1/3 + 2/3 - 1 = 0` and the `g'''(c)` terms as
`1/24 - 1/24 = 0`. What is left is four remainders: two of `g'` at size `L (1/2)⁴/24` each (weight `1/6`) and two of
`g` at size `L (1/2)⁵/120` each (weight `1`), so `2 · L/2304 + 2 · L/3840 = L/720`. The constant is what this Taylor
argument gives, not the best one.
-/

namespace Paper

open Real Set
open scoped Nat

/-- **Taylor's theorem, as a bound.** If `f` is `n + 1` times continuously differentiable on `ℝ` and its `(n + 1)`-st
derivative is at most `M` in absolute value between `x₀` and `x`, the degree-`n` Taylor polynomial about `x₀` misses
`f x` by at most `M |x - x₀|ⁿ⁺¹ / (n + 1)!`. -/
theorem taylor_abs_sub_le (f : ℝ → ℝ) (n : ℕ) (hf : ContDiff ℝ (n + 1) f) (x₀ x M : ℝ)
    (hM : ∀ t ∈ uIcc x₀ x, |iteratedDeriv (n + 1) f t| ≤ M) :
    |f x - ∑ k ∈ Finset.range (n + 1), iteratedDeriv k f x₀ / (k ! : ℝ) * (x - x₀) ^ k| ≤
      M * |x - x₀| ^ (n + 1) / ((n + 1)! : ℝ) := by
  rcases eq_or_ne x₀ x with rfl | hx
  · simp [Finset.sum_range_succ']
  obtain ⟨ξ, hξ, hT⟩ := taylor_mean_remainder_lagrange_iteratedDeriv hx hf.contDiffOn
  have hu : UniqueDiffOn ℝ (uIcc x₀ x) := uniqueDiffOn_uIcc hx
  have hpoly : taylorWithinEval f n (uIcc x₀ x) x₀ x =
      ∑ k ∈ Finset.range (n + 1), iteratedDeriv k f x₀ / (k ! : ℝ) * (x - x₀) ^ k := by
    rw [taylor_within_apply]
    refine Finset.sum_congr rfl fun k hk => ?_
    have hk' : ((k : ℕ) : WithTop ℕ∞) ≤ n + 1 := by
      have := Finset.mem_range.mp hk
      exact_mod_cast this.le
    rw [iteratedDerivWithin_eq_iteratedDeriv hu (hf.of_le hk').contDiffAt left_mem_uIcc, smul_eq_mul]
    ring
  rw [← hpoly, hT, abs_div, abs_mul, abs_pow, Nat.abs_cast]
  gcongr
  exact hM ξ (uIoo_subset_uIcc_self hξ)

/-- **Taylor's theorem for the derivative, as a bound.** The same bound for `f'`, written with the derivatives of `f`:
the degree-`n` Taylor polynomial of `f'` about `x₀` misses `f' x` by at most `M |x - x₀|ⁿ⁺¹ / (n + 1)!` when
`|f⁽ⁿ⁺²⁾| ≤ M` between `x₀` and `x`. -/
theorem taylor_deriv_abs_sub_le (f : ℝ → ℝ) (n : ℕ) (hf : ContDiff ℝ (n + 1 + 1) f) (x₀ x M : ℝ)
    (hM : ∀ t ∈ uIcc x₀ x, |iteratedDeriv (n + 2) f t| ≤ M) :
    |deriv f x - ∑ k ∈ Finset.range (n + 1), iteratedDeriv (k + 1) f x₀ / (k ! : ℝ) * (x - x₀) ^ k| ≤
      M * |x - x₀| ^ (n + 1) / ((n + 1)! : ℝ) := by
  have h := taylor_abs_sub_le (deriv f) n hf.deriv' x₀ x M (fun t ht => by
    rw [← iteratedDeriv_succ']; exact hM t ht)
  simpa only [← iteratedDeriv_succ'] using h

/-- Between the midpoint `1/2` and a point of `[0, 1]` one stays in `[0, 1]`. -/
lemma uIcc_half_subset {x : ℝ} (hx : x ∈ Icc (0 : ℝ) 1) : uIcc (1 / 2 : ℝ) x ⊆ Icc 0 1 :=
  uIcc_subset_Icc ⟨by norm_num, by norm_num⟩ hx

/-- **Simpson's rule for the path, with remainder.** If `g` is five times continuously differentiable and
`|g⁽⁵⁾| ≤ L` on `[0, 1]`, Simpson's combination `((g'(0) + g'(1))/2 + 2 g'(1/2))/3` of the two traces misses the
whole-input effect `T = g 1 - g 0` by at most `L/720`. -/
theorem simpson_remainder (g : ℝ → ℝ) (hg : ContDiff ℝ 5 g) (L : ℝ)
    (hL : ∀ t ∈ Icc (0 : ℝ) 1, |iteratedDeriv 5 g t| ≤ L) :
    |((deriv g 0 + deriv g 1) / 2 + 2 * deriv g (1 / 2)) / 3 - (g 1 - g 0)| ≤ L / 720 := by
  have h5 : ContDiff ℝ ((4 : ℕ) + 1) g := by exact_mod_cast hg
  have h5' : ContDiff ℝ ((3 : ℕ) + 1 + 1) g := by exact_mod_cast hg
  have e1 := taylor_abs_sub_le g 4 h5 (1 / 2) 1 L
    (fun t ht => hL t (uIcc_half_subset ⟨by norm_num, by norm_num⟩ ht))
  have e0 := taylor_abs_sub_le g 4 h5 (1 / 2) 0 L
    (fun t ht => hL t (uIcc_half_subset ⟨by norm_num, by norm_num⟩ ht))
  have d1 := taylor_deriv_abs_sub_le g 3 h5' (1 / 2) 1 L
    (fun t ht => hL t (uIcc_half_subset ⟨by norm_num, by norm_num⟩ ht))
  have d0 := taylor_deriv_abs_sub_le g 3 h5' (1 / 2) 0 L
    (fun t ht => hL t (uIcc_half_subset ⟨by norm_num, by norm_num⟩ ht))
  norm_num [Finset.sum_range_succ, Nat.factorial] at e1 e0 d1 d0
  rw [abs_le] at e1 e0 d1 d0 ⊢
  constructor <;> linarith [e1.1, e1.2, e0.1, e0.2, d1.1, d1.2, d0.1, d0.2]

/-- **Proposition 9 for a general path.** If `g` is five times continuously differentiable and `|g⁽⁵⁾| ≤ L` on
`[0, 1]`, the midpoint trace's error `g'(1/2) - T` is `-1/2` of the base-point (trapezoid) trace's error
`(g'(0) + g'(1))/2 - T` up to at most `L/480`. The left side is `3/2` times Simpson's error. -/
theorem midpoint_neg_half_remainder (g : ℝ → ℝ) (hg : ContDiff ℝ 5 g) (L : ℝ)
    (hL : ∀ t ∈ Icc (0 : ℝ) 1, |iteratedDeriv 5 g t| ≤ L) :
    |(deriv g (1 / 2) - (g 1 - g 0)) + (1 / 2) * ((deriv g 0 + deriv g 1) / 2 - (g 1 - g 0))| ≤ L / 480 := by
  have h := simpson_remainder g hg L hL
  have e : (deriv g (1 / 2) - (g 1 - g 0)) + (1 / 2) * ((deriv g 0 + deriv g 1) / 2 - (g 1 - g 0)) =
      (3 / 2) * (((deriv g 0 + deriv g 1) / 2 + 2 * deriv g (1 / 2)) / 3 - (g 1 - g 0)) := by ring
  rw [e, abs_mul, abs_of_pos (by norm_num : (0 : ℝ) < 3 / 2)]
  linarith

/-- **The trapezoid's error to leading order.** If `g` is four times continuously differentiable and `|g⁽⁴⁾| ≤ L₄` on
`[0, 1]`, the base-point (trapezoid) trace's error `(g'(0) + g'(1))/2 - T` is `g'''(1/2)/12` up to at most
`5 L₄/192`. -/
theorem trapezoid_leading_order (g : ℝ → ℝ) (hg : ContDiff ℝ 4 g) (L₄ : ℝ)
    (hL : ∀ t ∈ Icc (0 : ℝ) 1, |iteratedDeriv 4 g t| ≤ L₄) :
    |(deriv g 0 + deriv g 1) / 2 - (g 1 - g 0) - iteratedDeriv 3 g (1 / 2) / 12| ≤ 5 * L₄ / 192 := by
  have h4 : ContDiff ℝ ((3 : ℕ) + 1) g := by exact_mod_cast hg
  have h4' : ContDiff ℝ ((2 : ℕ) + 1 + 1) g := by exact_mod_cast hg
  have e1 := taylor_abs_sub_le g 3 h4 (1 / 2) 1 L₄
    (fun t ht => hL t (uIcc_half_subset ⟨by norm_num, by norm_num⟩ ht))
  have e0 := taylor_abs_sub_le g 3 h4 (1 / 2) 0 L₄
    (fun t ht => hL t (uIcc_half_subset ⟨by norm_num, by norm_num⟩ ht))
  have d1 := taylor_deriv_abs_sub_le g 2 h4' (1 / 2) 1 L₄
    (fun t ht => hL t (uIcc_half_subset ⟨by norm_num, by norm_num⟩ ht))
  have d0 := taylor_deriv_abs_sub_le g 2 h4' (1 / 2) 0 L₄
    (fun t ht => hL t (uIcc_half_subset ⟨by norm_num, by norm_num⟩ ht))
  norm_num [Finset.sum_range_succ, Nat.factorial] at e1 e0 d1 d0
  rw [abs_le] at e1 e0 d1 d0 ⊢
  constructor <;> linarith [e1.1, e1.2, e0.1, e0.2, d1.1, d1.2, d0.1, d0.2]

/-- **The midpoint's error to leading order.** If `g` is four times continuously differentiable and `|g⁽⁴⁾| ≤ L₄` on
`[0, 1]`, the midpoint trace's error `g'(1/2) - T` is `-g'''(1/2)/24` up to at most `L₄/192`. -/
theorem midpoint_leading_order (g : ℝ → ℝ) (hg : ContDiff ℝ 4 g) (L₄ : ℝ)
    (hL : ∀ t ∈ Icc (0 : ℝ) 1, |iteratedDeriv 4 g t| ≤ L₄) :
    |deriv g (1 / 2) - (g 1 - g 0) + iteratedDeriv 3 g (1 / 2) / 24| ≤ L₄ / 192 := by
  have h4 : ContDiff ℝ ((3 : ℕ) + 1) g := by exact_mod_cast hg
  have e1 := taylor_abs_sub_le g 3 h4 (1 / 2) 1 L₄
    (fun t ht => hL t (uIcc_half_subset ⟨by norm_num, by norm_num⟩ ht))
  have e0 := taylor_abs_sub_le g 3 h4 (1 / 2) 0 L₄
    (fun t ht => hL t (uIcc_half_subset ⟨by norm_num, by norm_num⟩ ht))
  norm_num [Finset.sum_range_succ, Nat.factorial] at e1 e0
  rw [abs_le] at e1 e0 ⊢
  constructor <;> linarith [e1.1, e1.2, e0.1, e0.2]

/-- **Remark 10, a mirrored set at a complete cut.** A pair and its mirror contribute `(g'(0) + g'(1))/2` to the trace
built with base-point gradients, which attribution patching reports, and `g'(1/2)` to the midpoint trace
(`Paper.mirror_terms`). If `g` is three times continuously differentiable and `|g'''| ≤ L` on `[0, 1]`, the two differ by
at most `L/8`; `g'''` carries three factors of `Δ`, so they agree to third order. `L/8` is sharp: a cubic gives
`g'''/12 + g'''/24 = g'''/8` (`Paper.trapezoid_error_cubic`, `Paper.midpoint_error_cubic`). -/
theorem atp_mirrored_third_order (g : ℝ → ℝ) (hg : ContDiff ℝ 3 g) (L : ℝ)
    (hL : ∀ t ∈ Icc (0 : ℝ) 1, |iteratedDeriv 3 g t| ≤ L) :
    |(deriv g 0 + deriv g 1) / 2 - deriv g (1 / 2)| ≤ L / 8 := by
  have h3 : ContDiff ℝ ((1 : ℕ) + 1 + 1) g := by exact_mod_cast hg
  have d1 := taylor_deriv_abs_sub_le g 1 h3 (1 / 2) 1 L
    (fun t ht => hL t (uIcc_half_subset ⟨by norm_num, by norm_num⟩ ht))
  have d0 := taylor_deriv_abs_sub_le g 1 h3 (1 / 2) 0 L
    (fun t ht => hL t (uIcc_half_subset ⟨by norm_num, by norm_num⟩ ht))
  norm_num [Finset.sum_range_succ, Nat.factorial] at d1 d0
  rw [abs_le] at d1 d0 ⊢
  constructor <;> linarith [d1.1, d1.2, d0.1, d0.2]

/-- **Remark 10, elsewhere.** Without its mirror (or inside the network, where the mirror's path is another function), a
pair contributes `g'(0)` to the base-point trace and `g'(1/2)` to the midpoint trace. If `g` is twice continuously
differentiable and `|g''| ≤ L₂` on `[0, 1]`, the two differ by at most `L₂/2`; `g''` carries two factors of `Δ`, so they
agree to second order. `L₂/2` is sharp (`g t = t²`). -/
theorem atp_single_second_order (g : ℝ → ℝ) (hg : ContDiff ℝ 2 g) (L₂ : ℝ)
    (hL : ∀ t ∈ Icc (0 : ℝ) 1, |iteratedDeriv 2 g t| ≤ L₂) :
    |deriv g 0 - deriv g (1 / 2)| ≤ L₂ / 2 := by
  have h2 : ContDiff ℝ ((0 : ℕ) + 1 + 1) g := by exact_mod_cast hg
  have d0 := taylor_deriv_abs_sub_le g 0 h2 (1 / 2) 0 L₂
    (fun t ht => hL t (uIcc_half_subset ⟨by norm_num, by norm_num⟩ ht))
  norm_num [Finset.sum_range_succ, Nat.factorial] at d0
  linarith

end Paper
