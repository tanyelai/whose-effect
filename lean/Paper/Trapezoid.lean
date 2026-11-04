import Mathlib

set_option linter.style.header false
set_option linter.style.longLine false

/-!
# Proposition 9: the trace's error on a mirrored set, and cross-entropy's derivative in the base label's logit

* Along a pair's swap `g t = z (h_b + t Δ)`, `T = g 1 - g 0 = ∫₀¹ g'`. The base-point trace is the trapezoid rule
  `(g'(0) + g'(1))/2` (the pair contributes `g'(0)`, its mirror `g'(1)`), the midpoint trace is the midpoint rule `g'(1/2)`.
  For every `g` of degree at most four the midpoint's error is exactly `-1/2` of the trapezoid's and Simpson's combination
  `(trapezoid + 2 midpoint)/3` is exactly `T`; for a cubic the two errors are `g'''/12` and `-g'''/24`. So the paper's
  leading-order relation holds through fourth order, and the first term that breaks it is of fifth order.
* The mirror of a pair (base `h_s`, difference `-Δ`, measure `-z`) has the path `t ↦ -g (1 - t)`, whose derivative is
  `g'(1)` at its base point and `g'(1/2)` at its midpoint.
* Cross-entropy of the source label, `CE z = -z s + log Σⱼ exp (z j)` over the raw logits `z` (the paper's `o`, not its
  log odds), has derivative `softmax z b` in another label's logit `z b` (Section 3, the paragraph after Corollary 6).
For a general smooth `g` the remainders are bounded in `Paper/TrapezoidRemainder.lean` (the relation up to `L/480` when
`|g⁽⁵⁾| ≤ L` on `[0, 1]`; the leading-order errors up to the fourth derivative).
-/

namespace Paper

open Real

/-- A polynomial of degree at most four on the swap path. -/
def quartic (a : Fin 5 → ℝ) (t : ℝ) : ℝ := a 0 + a 1 * t + a 2 * t ^ 2 + a 3 * t ^ 3 + a 4 * t ^ 4

/-- Its derivative. -/
def quartic' (a : Fin 5 → ℝ) (t : ℝ) : ℝ := a 1 + 2 * a 2 * t + 3 * a 3 * t ^ 2 + 4 * a 4 * t ^ 3

theorem quartic_hasDerivAt (a : Fin 5 → ℝ) (t : ℝ) : HasDerivAt (quartic a) (quartic' a t) t := by
  have h1 := (hasDerivAt_id t).const_mul (a 1)
  have h2 := (hasDerivAt_pow 2 t).const_mul (a 2)
  have h3 := (hasDerivAt_pow 3 t).const_mul (a 3)
  have h4 := (hasDerivAt_pow 4 t).const_mul (a 4)
  have h := ((((hasDerivAt_const t (a 0)).add h1).add h2).add h3).add h4
  unfold quartic quartic'
  convert h using 1
  · ext x; simp
  · simp; ring_nf

/-- The whole-input effect `T = g 1 - g 0`, the trapezoid rule and the midpoint rule for it. -/
def wholeT (a : Fin 5 → ℝ) : ℝ := quartic a 1 - quartic a 0
noncomputable def trapezoid (a : Fin 5 → ℝ) : ℝ := (quartic' a 0 + quartic' a 1) / 2
noncomputable def midpoint (a : Fin 5 → ℝ) : ℝ := quartic' a (1 / 2)

/-- **Proposition 9, the relation.** For every `g` of degree at most four, the midpoint trace's error is exactly `-1/2`
of the base-point (trapezoid) trace's. -/
theorem midpoint_error_neg_half (a : Fin 5 → ℝ) :
    midpoint a - wholeT a = -(1 / 2) * (trapezoid a - wholeT a) := by
  unfold midpoint wholeT trapezoid quartic quartic'; ring

/-- **Proposition 9, Simpson's rule.** `(trapezoid + 2 midpoint)/3 = T` for every `g` of degree at most four. -/
theorem simpson_exact (a : Fin 5 → ℝ) : (trapezoid a + 2 * midpoint a) / 3 = wholeT a := by
  unfold midpoint wholeT trapezoid quartic quartic'; ring

/-- **Proposition 9, the errors.** For a cubic (`a 4 = 0`, so `g''' = 6 a 3`), the trapezoid's error is `g'''/12`. -/
theorem trapezoid_error_cubic (a : Fin 5 → ℝ) (h4 : a 4 = 0) : trapezoid a - wholeT a = 6 * a 3 / 12 := by
  unfold wholeT trapezoid quartic quartic'; rw [h4]; ring

/-- For a cubic the midpoint's error is `-g'''/24`. -/
theorem midpoint_error_cubic (a : Fin 5 → ℝ) (h4 : a 4 = 0) : midpoint a - wholeT a = -(6 * a 3) / 24 := by
  unfold midpoint wholeT quartic quartic'; rw [h4]; ring

/-- **The mirror's terms.** If `g` has derivative `g'` along the pair's path, the mirror's path `t ↦ -g (1 - t)` has
derivative `g' 1` at its base point `t = 0` and `g' (1/2)` at its midpoint `t = 1/2`; it moves the output by the same `T`. -/
theorem mirror_terms (g g' : ℝ → ℝ) (hg : ∀ t, HasDerivAt g (g' t) t) :
    HasDerivAt (fun t => -g (1 - t)) (g' 1) 0 ∧ HasDerivAt (fun t => -g (1 - t)) (g' (1 / 2)) (1 / 2) ∧
      ((-g (1 - 1)) - (-g (1 - 0)) = g 1 - g 0) := by
  have path (s : ℝ) : HasDerivAt (fun t => -g (1 - t)) (g' (1 - s)) s := by
    have hl : HasDerivAt (fun t : ℝ => 1 - t) (-1) s := by
      simpa using (hasDerivAt_id s).const_sub 1
    have hc : HasDerivAt (fun t => g (1 - t)) (g' (1 - s) * (-1)) s := (hg (1 - s)).comp s hl
    have hn : HasDerivAt (fun t => -g (1 - t)) (-(g' (1 - s) * (-1))) s := hc.neg
    have e : -(g' (1 - s) * (-1)) = g' (1 - s) := by ring
    rw [e] at hn; exact hn
  refine ⟨?_, ?_, by simp only [sub_self, sub_zero]; ring⟩
  · simpa using path 0
  · have := path (1 / 2); norm_num at this ⊢; exact this

/-- Cross-entropy of the source label `s` over logits `z : Fin n → ℝ`. -/
noncomputable def crossEntropy {n : ℕ} (z : Fin n → ℝ) (s : Fin n) : ℝ := -z s + Real.log (∑ j, Real.exp (z j))

/-- **Cross-entropy's derivative in the base label's logit.** For a label `b ≠ s`, moving `z b` alone changes the loss at
rate `softmax z b = exp (z b) / Σⱼ exp (z j)`; the log odds `z s - z b` moves at rate `-1`. -/
theorem crossEntropy_deriv_base {n : ℕ} (z : Fin n → ℝ) (s b : Fin n) (hbs : b ≠ s) :
    HasDerivAt (fun x => crossEntropy (Function.update z b x) s)
      (Real.exp (z b) / ∑ j, Real.exp (z j)) (z b) := by
  classical
  have hsum : HasDerivAt (fun x => ∑ j, Real.exp (Function.update z b x j)) (Real.exp (z b)) (z b) := by
    have : ∀ j, HasDerivAt (fun x => Real.exp (Function.update z b x j))
        (if j = b then Real.exp (z b) else 0) (z b) := by
      intro j
      by_cases hj : j = b
      · subst hj; simpa using (Real.hasDerivAt_exp (z j))
      · simpa [Function.update_of_ne hj, hj] using hasDerivAt_const (z b) (Real.exp (z j))
    have h := HasDerivAt.fun_sum (u := Finset.univ) (fun j _ => this j)
    simpa using h
  have hpos : (∑ j, Real.exp (Function.update z b (z b) j)) ≠ 0 := by
    apply ne_of_gt; apply Finset.sum_pos (fun j _ => Real.exp_pos _) ⟨s, Finset.mem_univ _⟩
  have hlog := hsum.log hpos
  have hzs : HasDerivAt (fun x => -(Function.update z b x s)) 0 (z b) := by
    simpa [Function.update_of_ne (Ne.symm hbs)] using hasDerivAt_const (z b) (-(z s))
  have h := hzs.add hlog
  unfold crossEntropy
  refine h.congr_deriv ?_
  simp [Function.update_eq_self]

end Paper
