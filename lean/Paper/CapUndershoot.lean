import Mathlib

set_option linter.style.header false
set_option linter.style.longLine false

/-!
# Appendix C.4: a saturating cap makes first order undershoot (Gemma-2's soft cap)

Along a pair's swap, one label's raw logit runs from `o₀` to `o₁`; the readout passes it through a cap `g` (for Gemma-2,
`g(o) = c·tanh(o/c)`). The exact change of the capped logit is `g o₁ - g o₀`. A first-order value takes the cap's slope at the
swap's midpoint, where the final norm puts the raw logit at `oₘ`, not at the straight midpoint `(o₀ + o₁)/2`.

* `hermite_hadamard_deriv`: if `g'` is convex on `[o₀, o₁]`, the average slope is at least the slope at the straight midpoint,
  `g o₁ - g o₀ ≥ (o₁ - o₀) · g'((o₀ + o₁)/2)` (the left Hermite–Hadamard inequality for `g'`).
* `cap_undershoot`: if moreover the slope at the bent midpoint is no larger, `g'(oₘ) ≤ g'((o₀ + o₁)/2)` (for a positive logit
  of `c·tanh(·/c)`: the norm bends the midpoint outward, where `sech²` is smaller), then the exact change is at least the
  first-order value's slope times the raw change: `g o₁ - g o₀ ≥ (o₁ - o₀) · g'(oₘ)`. Read as ratios: with the cap, the exact
  change over the raw change, `(g o₁ - g o₀)/(o₁ - o₀)`, is at least `g'(oₘ)`, the factor the cap puts on the first-order value;
  so the cap raises exact effects relative to first order. The gap has two parts: the curvature of `g'` (the first inequality)
  and the bend (the second).

For `c·tanh(·/c)`, `g'` is convex where `3 tanh²(o/c) ≥ 1` (`|o| ≥ 0.66c`) and decreases in `|o|`; these facts and how often the
hypotheses hold in Gemma-2's pairs are measured, not proved here.
-/

namespace Paper

open Set

/-- **The left Hermite–Hadamard inequality for a slope.** If `g` is twice differentiable and `g'` is convex on `[o₀, o₁]`,
then `g o₁ - g o₀ ≥ (o₁ - o₀) · g'((o₀ + o₁)/2)`. -/
theorem hermite_hadamard_deriv (g : ℝ → ℝ) (hg : ContDiff ℝ 2 g) (o₀ o₁ : ℝ) (h01 : o₀ ≤ o₁)
    (hconv : ConvexOn ℝ (Icc o₀ o₁) (deriv g)) :
    (o₁ - o₀) * deriv g ((o₀ + o₁) / 2) ≤ g o₁ - g o₀ := by
  set m := (o₀ + o₁) / 2 with hm
  set d := (o₁ - o₀) / 2 with hd
  have hd0 : 0 ≤ d := by rw [hd]; linarith
  have hg1 : Differentiable ℝ g := hg.differentiable (by norm_num)
  -- φ(t) = g(m + t) - g(m - t) - 2 t g'(m), with φ(0) = 0 and φ' ≥ 0 on [0, d]
  set φ : ℝ → ℝ := fun t => g (m + t) - g (m - t) - 2 * t * deriv g m with hφ
  have hφd : ∀ t, HasDerivAt φ (deriv g (m + t) + deriv g (m - t) - 2 * deriv g m) t := by
    intro t
    have h1 : HasDerivAt (fun t => g (m + t)) (deriv g (m + t)) t := by
      have := (hg1 (m + t)).hasDerivAt.comp t ((hasDerivAt_id t).const_add m)
      simpa [Function.comp_def] using this
    have h2 : HasDerivAt (fun t => g (m - t)) (deriv g (m - t) * (-1)) t := by
      exact (hg1 (m - t)).hasDerivAt.comp t ((hasDerivAt_id t).const_sub m)
    have h3 : HasDerivAt (fun t => 2 * t * deriv g m) (2 * deriv g m) t := by
      simpa using ((hasDerivAt_id t).const_mul 2).mul_const (deriv g m)
    have := (h1.sub h2).sub h3
    convert this using 1
    ring
  have hmono : MonotoneOn φ (Icc 0 d) := by
    apply monotoneOn_of_deriv_nonneg (convex_Icc 0 d)
    · exact fun t _ => (hφd t).continuousAt.continuousWithinAt
    · exact fun t _ => (hφd t).differentiableAt.differentiableWithinAt
    · intro t ht
      rw [interior_Icc] at ht
      rw [(hφd t).deriv]
      have hp : m + t ∈ Icc o₀ o₁ := ⟨by rw [hm]; linarith [ht.1, ht.2], by rw [hm]; linarith [ht.1, ht.2]⟩
      have hn : m - t ∈ Icc o₀ o₁ := ⟨by rw [hm]; linarith [ht.1, ht.2], by rw [hm]; linarith [ht.1, ht.2]⟩
      have := hconv.2 hp hn (by norm_num : (0 : ℝ) ≤ 1 / 2) (by norm_num : (0 : ℝ) ≤ 1 / 2) (by norm_num)
      have hmid : (1 / 2 : ℝ) • (m + t) + (1 / 2 : ℝ) • (m - t) = m := by simp [smul_eq_mul]; ring
      rw [hmid, smul_eq_mul, smul_eq_mul] at this
      linarith
  have h0 : φ 0 = 0 := by simp [hφ]
  have hle := hmono ⟨le_refl 0, hd0⟩ ⟨hd0, le_refl d⟩ hd0
  rw [h0] at hle
  have e1 : m + d = o₁ := by rw [hm, hd]; ring
  have e2 : m - d = o₀ := by rw [hm, hd]; ring
  simp only [hφ, e1, e2] at hle
  have h2d : 2 * d * deriv g m = (o₁ - o₀) * deriv g m := by rw [hd]; ring
  linarith [hle, h2d]

/-- **The cap undershoots.** If `g'` is convex on `[o₀, o₁]` and the slope at the bent midpoint `oₘ` is at most the slope at
the straight midpoint, the exact change of the capped logit is at least the raw change times the slope at `oₘ`: the cap
raises the exact effect relative to the first-order value. -/
theorem cap_undershoot (g : ℝ → ℝ) (hg : ContDiff ℝ 2 g) (o₀ o₁ oₘ : ℝ) (h01 : o₀ ≤ o₁)
    (hconv : ConvexOn ℝ (Icc o₀ o₁) (deriv g)) (hbend : deriv g oₘ ≤ deriv g ((o₀ + o₁) / 2)) :
    (o₁ - o₀) * deriv g oₘ ≤ g o₁ - g o₀ := by
  have h := hermite_hadamard_deriv g hg o₀ o₁ h01 hconv
  have : (o₁ - o₀) * deriv g oₘ ≤ (o₁ - o₀) * deriv g ((o₀ + o₁) / 2) :=
    mul_le_mul_of_nonneg_left hbend (by linarith)
  linarith

end Paper
