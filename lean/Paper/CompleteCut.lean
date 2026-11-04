import Mathlib

set_option linter.style.header false
set_option linter.style.longLine false

/-!
# Proposition 3: the complete-cut identity

A cut `C` for a pair is a site such that the output is a function of the states at `C` and of side inputs
(weights, position indices, attention mask) that agree between the base and the source run. Kinds (a) and (b) of
Definition 2 are the two situations the paper uses in which the side inputs agree; the identity is the statement
below for any network whose output factors in this way.
-/

namespace Paper

/-- **Proposition 3.** If the output factors as `z x = g (h x) (side x)` through the states `h x` at the cut and the
side inputs agree, `side x_b = side x_s`, then swapping in the source's states at the cut changes the base output by
exactly the input's whole effect `T = z x_s - z x_b`. -/
theorem complete_cut_identity {X H S : Type*} (h : X → H) (side : X → S) (g : H → S → ℝ) (xb xs : X)
    (hside : side xb = side xs) :
    g (h xs) (side xb) - g (h xb) (side xb) = g (h xs) (side xs) - g (h xb) (side xb) := by
  rw [hside]

/-- Kind (b), the readout, as the trivial case of `complete_cut_identity`: the output reads only the last state, so
there are no side inputs and both sides are the same expression. -/
theorem complete_cut_readout {X H : Type*} (h : X → H) (g : H → ℝ) (xb xs : X) :
    g (h xs) - g (h xb) = g (h xs) - g (h xb) := rfl

end Paper
