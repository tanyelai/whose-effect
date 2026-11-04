import Paper.Range

set_option linter.style.header false
set_option linter.style.longLine false

/-!
# Theorem 5, read in any orthonormal basis

The network's own interchange is the trace of `M` (Theorem 5(ii)), and the trace is the same in every orthonormal basis.
So along any complete set of orthogonal directions `b₁, …, b_D` the rank-one effects `⟪bⱼ, M bⱼ⟫` (Theorem 5(i)) sum to the
own interchange, and the effect along one of them exceeds it exactly when the others together sum below zero. This is
the reading of Theorem 5 that Section 3 states after it and the abstract uses; in the eigenbasis it is Corollary 6(i)'s
excess, `λ₁ - tr M = -∑_{j≥2} λⱼ`. Grouping the directions into parts of any rank gives the introduction's statement: the
effects of patching orthogonal parts sum to the effect of patching the whole state.
-/

namespace Paper

open scoped RealInnerProductSpace

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]
variable {ι : Type*} [Fintype ι]

/-- **Any complete set of orthogonal directions.** For an orthonormal basis `b`, the rank-one effects along its directions sum
to the network's own interchange, `∑ⱼ ⟪bⱼ, M bⱼ⟫ = ∑ᵢ pᵢ ⟪wᵢ, Δᵢ⟫`. -/
theorem sum_rank_one_effects_basis [FiniteDimensional ℝ E] {κ : Type*} [Fintype κ] (b : OrthonormalBasis κ ℝ E)
    (p : ι → ℝ) (w Δ : ι → E) :
    ∑ j, ⟪b j, effM p w Δ (b j)⟫ = ∑ i, p i * ⟪w i, Δ i⟫ := by
  rw [← trace_effM, LinearMap.trace_eq_sum_inner _ b]

/-- **One direction exceeds the whole exactly when the others push back.** Along an orthonormal basis `b`, the effect along
`b j₀` exceeds the network's own interchange if and only if the effects along the other directions sum below zero. -/
theorem rank_one_exceeds_iff_rest_negative [FiniteDimensional ℝ E] {κ : Type*} [Fintype κ] [DecidableEq κ]
    (b : OrthonormalBasis κ ℝ E) (p : ι → ℝ) (w Δ : ι → E) (j₀ : κ) :
    ∑ i, p i * ⟪w i, Δ i⟫ < ⟪b j₀, effM p w Δ (b j₀)⟫ ↔
      ∑ j ∈ Finset.univ.erase j₀, ⟪b j, effM p w Δ (b j)⟫ < 0 := by
  rw [← sum_rank_one_effects_basis b p w Δ, ← Finset.add_sum_erase _ _ (Finset.mem_univ j₀)]
  constructor <;> intro h <;> linarith

/-- **A part spanned by basis vectors.** Patching the part of each `Δᵢ` along the span of the basis vectors `b j`,
`j ∈ s` (its orthogonal projection `∑_{j ∈ s} ⟪b j, Δᵢ⟫ b j`), has mean first-order effect `∑_{j ∈ s} ⟪b j, M b j⟫`, the
rank-`|s|` effect `tr(PᵀMP)` of Theorem 5(iii) for `P` with columns `b j`. -/
theorem part_effect_eq_sum_rank_one {κ : Type*} [Fintype κ] (b : OrthonormalBasis κ ℝ E) (s : Finset κ)
    (p : ι → ℝ) (w Δ : ι → E) :
    ∑ i, p i * ⟪w i, ∑ j ∈ s, ⟪b j, Δ i⟫ • b j⟫ = ∑ j ∈ s, ⟪b j, effM p w Δ (b j)⟫ := by
  simp_rw [← rank_one_effect_eq, inner_sum, Finset.mul_sum]
  exact Finset.sum_comm

/-- **Orthogonal parts of any rank (the introduction).** Split the directions of an orthonormal basis `b` into parts by any
assignment `part`. The mean first-order effects of patching each part, the projection of `Δᵢ` onto the span of its
directions, sum to the network's own interchange `∑ᵢ pᵢ ⟪wᵢ, Δᵢ⟫`, whichever basis and split chose the parts. -/
theorem sum_part_effects_basis [FiniteDimensional ℝ E] {κ π : Type*} [Fintype κ] [Fintype π] [DecidableEq π]
    (b : OrthonormalBasis κ ℝ E) (part : κ → π) (p : ι → ℝ) (w Δ : ι → E) :
    ∑ a, ∑ i, p i * ⟪w i, ∑ j ∈ Finset.univ.filter (fun j => part j = a), ⟪b j, Δ i⟫ • b j⟫ =
      ∑ i, p i * ⟪w i, Δ i⟫ := by
  simp_rw [part_effect_eq_sum_rank_one, Finset.sum_fiberwise]
  exact sum_rank_one_effects_basis b p w Δ

end Paper
