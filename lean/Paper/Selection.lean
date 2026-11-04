import Paper.Range

set_option linter.style.header false
set_option linter.style.longLine false

/-!
# Corollary 6: what the objective selects

The theorems named `kind1_…` are about an effect-seeking objective in pure form, one strictly decreasing in the mean
effect. Those named `kind2_…` are about a target-aimed objective in pure form, one that depends on `d` only through the
mean effect and is uniquely minimised at a target `τ`. All statements are about the first-order mean effect
`⟪d, M d⟫` (Theorem 5(i)) on the unit sphere. Eigenvalues follow Mathlib's order: `λ₀` is the paper's `λ₁`, and
`λ_{n-1}` its `λ_D`.
-/

namespace Paper

open scoped RealInnerProductSpace

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] [FiniteDimensional ℝ E]
variable {n : ℕ} (hn : Module.finrank ℝ E = n) {T : E →ₗ[ℝ] E} (hT : T.IsSymmetric)

/-- A unit vector reaching the top of the Rayleigh form lies in the top eigenspace. -/
theorem top_eigenspace_of_rayleigh_eq_top (h0 : 0 < n) (d : E) (hd : ‖d‖ = 1)
    (htop : ⟪d, T d⟫ = hT.eigenvalues hn ⟨0, h0⟩) : T d = hT.eigenvalues hn ⟨0, h0⟩ • d := by
  set b := hT.eigenvectorBasis hn
  set lam := hT.eigenvalues hn
  set l0 := lam ⟨0, h0⟩
  have hsum := sum_sq_inner_eigenvector hn hT d
  rw [hd, one_pow] at hsum
  have hexp := rayleigh_expansion hn hT d
  -- ∑ (λ₀ - λᵢ) cᵢ² = 0 with nonnegative terms
  have hzero : ∑ i, (l0 - lam i) * ⟪b i, d⟫ ^ 2 = 0 := by
    have : ∑ i, (l0 - lam i) * ⟪b i, d⟫ ^ 2 = l0 * ∑ i, ⟪b i, d⟫ ^ 2 - ∑ i, lam i * ⟪b i, d⟫ ^ 2 := by
      rw [Finset.mul_sum, ← Finset.sum_sub_distrib]
      apply Finset.sum_congr rfl; intro i _; ring
    rw [this, hsum, ← hexp, htop]; ring
  have hnn : ∀ i ∈ Finset.univ, 0 ≤ (l0 - lam i) * ⟪b i, d⟫ ^ 2 := by
    intro i _
    apply mul_nonneg _ (sq_nonneg _)
    have := hT.eigenvalues_antitone hn (Fin.mk_le_of_le_val (Nat.zero_le i.val) : (⟨0, h0⟩ : Fin n) ≤ i)
    linarith
  have hterm := (Finset.sum_eq_zero_iff_of_nonneg hnn).mp hzero
  -- in the eigenbasis, every coordinate of `T d - λ₀ d` vanishes
  apply b.toBasis.ext_elem
  intro i
  simp only [OrthonormalBasis.coe_toBasis_repr_apply, map_smul, smul_eq_mul]
  rw [hT.eigenvectorBasis_apply_self_apply hn]
  have hi := hterm i (Finset.mem_univ i)
  rcases mul_eq_zero.mp hi with h | h
  · have : lam i = l0 := by linarith
    simp only [lam, l0] at this ⊢
    rw [this]
    simp [b, RCLike.ofReal_real_eq_id, OrthonormalBasis.coe_toBasis_repr_apply]
  · have hc : ⟪b i, d⟫ = 0 := pow_eq_zero_iff (by norm_num) |>.mp h
    have hrepr : b.repr d i = 0 := by rw [OrthonormalBasis.repr_apply_apply]; exact hc
    simp [b, hrepr]

/-- **Corollary 6(i).** For an objective `L` strictly decreasing in the mean effect, a unit vector minimises
`L (⟪d, M d⟫)` over the unit sphere if and only if it reports the top eigenvalue; such a vector lies in the top
eigenspace. -/
theorem kind1_selects_top (h0 : 0 < n) (L : ℝ → ℝ) (hL : StrictAnti L) (d : E) (hd : ‖d‖ = 1) :
    ((∀ d' : E, ‖d'‖ = 1 → L ⟪d, T d⟫ ≤ L ⟪d', T d'⟫) ↔ ⟪d, T d⟫ = hT.eigenvalues hn ⟨0, h0⟩) ∧
      (⟪d, T d⟫ = hT.eigenvalues hn ⟨0, h0⟩ → T d = hT.eigenvalues hn ⟨0, h0⟩ • d) := by
  refine ⟨⟨fun hmin => ?_, fun htop d' hd' => ?_⟩, top_eigenspace_of_rayleigh_eq_top hn hT h0 d hd⟩
  · have hle := rayleigh_le_top hn hT h0 d hd
    have he := hmin _ ((hT.eigenvectorBasis hn).orthonormal.1 ⟨0, h0⟩)
    rw [rayleigh_eigenvector hn hT] at he
    by_contra hne
    have hlt : ⟪d, T d⟫ < hT.eigenvalues hn ⟨0, h0⟩ := lt_of_le_of_ne hle hne
    exact absurd he (not_le.mpr (hL hlt))
  · rw [htop]
    exact hL.antitone (rayleigh_le_top hn hT h0 d' hd')

/-- **Corollary 6(i), the excess.** The top eigenvalue exceeds the trace by `-∑_{j ≥ 1} λ_j`, the paper's
`-∑_{j ≥ 2} λ_j`. -/
theorem top_sub_trace (h0 : 0 < n) :
    hT.eigenvalues hn ⟨0, h0⟩ - LinearMap.trace ℝ E T = -∑ j ∈ Finset.univ.erase ⟨0, h0⟩, hT.eigenvalues hn j := by
  rw [hT.trace_eq_sum_eigenvalues hn]
  simp only [RCLike.ofReal_real_eq_id, id_eq]
  rw [← Finset.add_sum_erase _ _ (Finset.mem_univ (⟨0, h0⟩ : Fin n))]
  ring

/-- **Corollary 6(ii).** In dimension at least 2, the level set `{d : ‖d‖ = 1, ⟪d, M d⟫ = τ}` is non-empty if and only
if `λ_{n-1} ≤ τ ≤ λ₀`; an objective minimised exactly at `τ` and depending on `d` only through `⟪d, M d⟫` selects it. -/
theorem kind2_selects_level (h2 : 2 ≤ n) (τ : ℝ) (f : ℝ → ℝ) (hf : ∀ x, x ≠ τ → f τ < f x) :
    ((∃ d : E, ‖d‖ = 1 ∧ ⟪d, T d⟫ = τ) ↔
        τ ∈ Set.Icc (hT.eigenvalues hn ⟨n - 1, by omega⟩) (hT.eigenvalues hn ⟨0, by omega⟩)) ∧
      ∀ d : E, ‖d‖ = 1 → (∃ d' : E, ‖d'‖ = 1 ∧ ⟪d', T d'⟫ = τ) →
        ((∀ d' : E, ‖d'‖ = 1 → f ⟪d, T d⟫ ≤ f ⟪d', T d'⟫) ↔ ⟪d, T d⟫ = τ) := by
  refine ⟨?_, ?_⟩
  · rw [← rayleigh_range hn hT h2]; rfl
  · rintro d hd ⟨d', hd', hτ⟩
    constructor
    · intro hmin
      by_contra hne
      have := hmin d' hd'
      rw [hτ] at this
      exact absurd this (not_le.mpr (hf _ hne))
    · intro heq d'' _
      rw [heq]
      by_cases h : ⟪d'', T d''⟫ = τ
      · rw [h]
      · exact (hf _ h).le

/-- **Corollary 6(iii).** A target-aimed objective with `τ = tr M` reaches the trace exactly when
`λ_{n-1} ≤ tr M ≤ λ₀`, that is, when `∑_{j ≥ 1} λ_j ≤ 0 ≤ ∑_{j ≤ n-2} λ_j`. This holds under one shared direction
(Proposition 7, `Paper.natural_trace_mem_range`), but not for every `M`: the two-pair example `Paper.counterexample_undershoot` has
`λ₀ < tr M`. -/
theorem kind2_reaches_trace_iff (h2 : 2 ≤ n) :
    (∃ d : E, ‖d‖ = 1 ∧ ⟪d, T d⟫ = LinearMap.trace ℝ E T) ↔
      LinearMap.trace ℝ E T ∈ Set.Icc (hT.eigenvalues hn ⟨n - 1, by omega⟩) (hT.eigenvalues hn ⟨0, by omega⟩) := by
  rw [← rayleigh_range hn hT h2]; rfl

end Paper
