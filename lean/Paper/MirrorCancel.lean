import Paper.TrapezoidRemainder

set_option linter.style.header false
set_option linter.style.longLine false

/-!
# Appendix A.1, a pair and its mirror: mirrors cancel the second-order part of a rank-one effect's error

At a pair's swap `h_b → h_s = h_b + Δ`, a rank-one interchange along `d` moves the state by `p = (d·Δ) d`. Its first-order
value in the paper uses the readout's derivative at the midpoint `m = h_b + Δ/2` of the *whole* swap (the paper's `h̄`):
`L p` with `L = Dz(m)`, the gradient behind `M_mid`. Its exact effect is `z(h_b + p) - z(h_b)`, and the two differ
already at second order, because the partial swap's own midpoint is not `m`.

A mirrored set contains, with each pair, its mirror: base and source exchanged, so the change is `-Δ`, the rank-one move is
`-p`, the midpoint is the same `m`, and the answer read is the other label's, `-z`. The mirror's first-order value is
`(-L)(-p) = L p` and its exact effect is `-z(h_s - p) + z(h_s)`.

* `mirror_pair_error_eq`: the two errors (exact minus first order) add up to two central-difference remainders,
  `[z(m+u) - z(m-u) - 2 L u] + [z(m+v) - z(m-v) - 2 L v]` with `u = p - Δ/2` and `v = Δ/2`. Only the linearity of `L` is
  used; every even-order term of `z` about `m`, the second-order one included, has cancelled in it.
* `central_difference_le`: if `f : ℝ → ℝ` is three times continuously differentiable and `|f'''| ≤ M` on `[-1, 1]`, then
  `|f 1 - f (-1) - 2 f'(0)| ≤ M/3`.
* `hasDerivAt_line`: along the line `t ↦ z(m + t • x)` the derivative at `0` is `L x` when `L` is the derivative of `z` at
  `m`; so each bracket above is `f 1 - f (-1) - 2 f'(0)` for `f t = z(m + t • x)` (with `x = u` or `x = v`), third order in
  `x` by `central_difference_le`.
-/

namespace Paper

open Set

variable {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]

/-- **A pair and its mirror.** The errors of the first-order value `L p` for a rank-one move `p` at a pair (base
`m - Δ/2`, readout `z`) and at its mirror (base `m + Δ/2`, move `-p`, readout `-z`, first-order value `(-L)(-p) = L p`) add up
to two central-difference remainders about the shared midpoint `m`. -/
theorem mirror_pair_error_eq (z : E → ℝ) (L : E →L[ℝ] ℝ) (m Δ p : E) :
    ((z (m - (1 / 2 : ℝ) • Δ + p) - z (m - (1 / 2 : ℝ) • Δ)) - L p) +
        ((-z (m + (1 / 2 : ℝ) • Δ - p) - -z (m + (1 / 2 : ℝ) • Δ)) - (-L) (-p)) =
      (z (m + (p - (1 / 2 : ℝ) • Δ)) - z (m - (p - (1 / 2 : ℝ) • Δ)) - 2 * L (p - (1 / 2 : ℝ) • Δ)) +
        (z (m + (1 / 2 : ℝ) • Δ) - z (m - (1 / 2 : ℝ) • Δ) - 2 * L ((1 / 2 : ℝ) • Δ)) := by
  have e1 : m - (1 / 2 : ℝ) • Δ + p = m + (p - (1 / 2 : ℝ) • Δ) := by abel
  have e2 : m + (1 / 2 : ℝ) • Δ - p = m - (p - (1 / 2 : ℝ) • Δ) := by abel
  rw [e1, e2, map_sub, neg_apply, map_neg]
  ring

/-- **The central difference is third order.** If `|f'''| ≤ M` on `[-1, 1]`, then `|f 1 - f (-1) - 2 f'(0)| ≤ M/3`. -/
theorem central_difference_le (f : ℝ → ℝ) (hf : ContDiff ℝ 3 f) (M : ℝ)
    (hM : ∀ t ∈ Icc (-1 : ℝ) 1, |iteratedDeriv 3 f t| ≤ M) :
    |f 1 - f (-1) - 2 * deriv f 0| ≤ M / 3 := by
  have hf' : ContDiff ℝ ((2 : ℕ) + 1) f := by exact_mod_cast hf
  have hp := taylor_abs_sub_le f 2 hf' 0 1 M (fun t ht => hM t (by
    rw [uIcc_of_le zero_le_one] at ht; exact ⟨by linarith [ht.1], ht.2⟩))
  have hn := taylor_abs_sub_le f 2 hf' 0 (-1) M (fun t ht => hM t (by
    rw [uIcc_of_ge (by norm_num : (-1 : ℝ) ≤ 0)] at ht; exact ⟨ht.1, by linarith [ht.2]⟩))
  simp only [Finset.sum_range_succ, Finset.sum_range_zero, iteratedDeriv_zero, iteratedDeriv_one] at hp hn
  norm_num at hp hn
  rw [abs_le] at hp hn ⊢
  constructor <;> nlinarith [hp.1, hp.2, hn.1, hn.2]

/-- **The derivative along a line.** If `L` is the derivative of `z` at `m`, the path `t ↦ z(m + t • x)` has derivative
`L x` at `0`. -/
theorem hasDerivAt_line (z : E → ℝ) (L : E →L[ℝ] ℝ) (m x : E) (hz : HasFDerivAt z L m) :
    HasDerivAt (fun t : ℝ => z (m + t • x)) (L x) 0 := by
  have hl : HasDerivAt (fun t : ℝ => m + t • x) x 0 := by
    simpa using ((hasDerivAt_id (0 : ℝ)).smul_const x).const_add m
  have hz' : HasFDerivAt z L (m + (0 : ℝ) • x) := by simpa using hz
  exact hz'.comp_hasDerivAt (0 : ℝ) hl

end Paper
