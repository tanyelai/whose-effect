import Paper.TwoVector

set_option linter.style.header false
set_option linter.style.longLine false

/-!
# Proposition 8: a door the network leaves shut

One head at the readout position `n` reads a site at position `p`. With the other scores fixed, its weight on `p` is the
logistic function of the score on `p` minus the log of the other positions' total. Part (i), `door_max`, is the
two-vector lemma applied to the score's gradient `a` (`Paper.two_vector_isGreatest`); part (ii) is `door_shut`,
`door_open` and `door_unread`.
-/

namespace Paper

open Real
open scoped RealInnerProductSpace

/-- The logistic function. -/
noncomputable def sigm (x : ℝ) : ℝ := 1 / (1 + exp (-x))

theorem sigm_pos (x : ℝ) : 0 < sigm x := by unfold sigm; positivity

theorem sigm_lt_one (x : ℝ) : sigm x < 1 := by
  unfold sigm
  rw [div_lt_one (by positivity)]
  linarith [exp_pos (-x)]

theorem sigm_strictMono : StrictMono sigm := by
  intro x y hxy
  unfold sigm
  apply one_div_lt_one_div_of_lt (by positivity)
  have : exp (-y) < exp (-x) := exp_lt_exp.mpr (by linarith)
  linarith

/-- The softmax weight on one position, with the other positions' exponentials summing to `R > 0`, is the logistic
function of its score minus `log R`. -/
theorem softmax_eq_sigm (s R : ℝ) (hR : 0 < R) : exp s / (exp s + R) = sigm (s - log R) := by
  unfold sigm
  rw [neg_sub, exp_sub, exp_log hR]
  field_simp

/-- `x + log (1 / σ x) = -log (1 - σ x)`: the score that reaches the door from the base weight `α = σ x`. -/
theorem add_log_inv_sigm (x : ℝ) : x + log (1 / sigm x) = -log (1 - sigm x) := by
  have hpos : 0 < 1 + exp (-x) := by positivity
  have h1 : 1 / sigm x = 1 + exp (-x) := by unfold sigm; rw [one_div_one_div]
  have h2 : 1 - sigm x = exp (-x) / (1 + exp (-x)) := by
    unfold sigm; field_simp; ring
  rw [h1, h2, log_div (exp_pos _).ne' hpos.ne', log_exp]
  ring

/-- **Proposition 8(ii), the network's door stays shut.** With base weight `α = σ x` and `L = log (1/α)`, a score
change `c ≤ L - t` leaves the weight at most `σ (-t - log (1 - α))`. -/
theorem door_shut (x c t : ℝ) (h : c ≤ log (1 / sigm x) - t) :
    sigm (x + c) ≤ sigm (-t - log (1 - sigm x)) := by
  apply sigm_strictMono.monotone
  have := add_log_inv_sigm x
  linarith

/-- **Proposition 8(ii), a rank-one interchange opens it.** A score change `c ≥ L + t` raises the weight to at least
`σ t`. -/
theorem door_open (x c t : ℝ) (h : log (1 / sigm x) + t ≤ c) : sigm t ≤ sigm (x + c) := by
  apply sigm_strictMono.monotone
  have h1 := add_log_inv_sigm x
  have h2 : log (1 - sigm x) ≤ 0 := log_nonpos (by linarith [sigm_lt_one x]) (by linarith [sigm_pos x])
  linarith

/-- **Proposition 8(ii), a difference the head does not read.** With `⟪a, Δ⟫ = 0` and `t ≤ L`, the own
interchange's condition `⟪a, Δ⟫ ≤ L - t` holds, and the rank-one condition `(⟪a, Δ⟫ + ‖a‖‖Δ‖)/2 ≥ L + t` holds if and
only if `‖a‖‖Δ‖ ≥ 2 (L + t)`. -/
theorem door_unread {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] (a Δ : E) (L t : ℝ)
    (h0 : ⟪a, Δ⟫ = 0) (ht : t ≤ L) :
    ⟪a, Δ⟫ ≤ L - t ∧ (L + t ≤ (⟪a, Δ⟫ + ‖a‖ * ‖Δ‖) / 2 ↔ 2 * (L + t) ≤ ‖a‖ * ‖Δ‖) := by
  rw [h0]
  refine ⟨by linarith, ?_⟩
  constructor <;> intro h <;> linarith

/-- **Proposition 8(i).** Over unit directions, the largest first-order score change of a rank-one interchange,
`⟪a, d⟫⟪d, Δ⟫`, is `(⟪a, Δ⟫ + ‖a‖‖Δ‖)/2` (dimension at least 2). -/
theorem door_max {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] [FiniteDimensional ℝ E]
    (h2 : 2 ≤ Module.finrank ℝ E) (a Δ : E) :
    IsGreatest {r | ∃ d : E, ‖d‖ = 1 ∧ ⟪a, d⟫ * ⟪d, Δ⟫ = r} ((⟪a, Δ⟫ + ‖a‖ * ‖Δ‖) / 2) :=
  two_vector_isGreatest h2 a Δ

end Paper
