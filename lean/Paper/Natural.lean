import Paper.Selection

set_option linter.style.header false
set_option linter.style.longLine false

/-!
# Proposition 7: one shared direction

When `𝔼[w Δᵀ] = w̄ mᵀ` (a shared readout direction `wᵢ = cᵢ w̄`, or a single code `Δᵢ = sᵢ Δ̄`), the first-order
effect matrix is that of the single pair `(w̄, m)`. The angle statements are proved for the parametrisation
`ŵ = cos θ m̂ + sin θ u` with `m̂ ⟂ u` unit vectors, which is how the angle `θ = ∠(w̄, m)` is defined in the span.
-/

namespace Paper

open scoped RealInnerProductSpace
open Real

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]
variable {ι : Type*} [Fintype ι]

/-- The single-pair matrix `sym(w̄ mᵀ)`. -/
noncomputable abbrev effM1 (wbar m : E) : E →ₗ[ℝ] E :=
  effM (fun _ : Unit => (1 : ℝ)) (fun _ => wbar) (fun _ => m)

/-- **Shared readout direction**: with `wᵢ = cᵢ w̄`, `M = sym(w̄ mᵀ)` for `m = ∑ pᵢ cᵢ Δᵢ`. -/
theorem effM_shared_readout (p c : ι → ℝ) (wbar : E) (Δ : ι → E) :
    effM p (fun i => c i • wbar) Δ = effM1 wbar (∑ i, (p i * c i) • Δ i) := by
  apply LinearMap.ext
  intro x
  rw [effM_apply, effM_apply]
  congr 1
  simp only [Finset.univ_unique, Finset.sum_singleton, one_smul, sum_inner, real_inner_smul_left,
    real_inner_smul_right, Finset.sum_smul, Finset.smul_sum, smul_add, ← Finset.sum_add_distrib, smul_smul]
  apply Finset.sum_congr rfl
  intro i _
  congr 1 <;> congr 1 <;> ring

/-- **Single code**: with `Δᵢ = sᵢ Δ̄`, `M = sym(w̄ Δ̄ᵀ)` for `w̄ = ∑ pᵢ sᵢ wᵢ`. -/
theorem effM_single_code (p s : ι → ℝ) (w : ι → E) (Δbar : E) :
    effM p w (fun i => s i • Δbar) = effM1 (∑ i, (p i * s i) • w i) Δbar := by
  apply LinearMap.ext
  intro x
  rw [effM_apply, effM_apply]
  congr 1
  simp only [Finset.univ_unique, Finset.sum_singleton, one_smul, sum_inner, real_inner_smul_left,
    real_inner_smul_right, Finset.sum_smul, Finset.smul_sum, smul_add, ← Finset.sum_add_distrib, smul_smul]
  apply Finset.sum_congr rfl
  intro i _
  congr 1 <;> congr 1 <;> ring

/-- **Proposition 7(i).** `e(d) = (dᵀw̄)(dᵀm)` and `tr M = w̄ᵀm`. -/
theorem natural_effect (wbar m d : E) : ⟪d, effM1 wbar m d⟫ = ⟪d, wbar⟫ * ⟪d, m⟫ := by
  rw [effM1, rayleigh_single, real_inner_comm wbar d]

theorem natural_trace [FiniteDimensional ℝ E] (wbar m : E) : LinearMap.trace ℝ E (effM1 wbar m) = ⟪wbar, m⟫ := by
  rw [effM1, trace_effM]; simp

/-- **Proposition 7(ii).** The readout direction and the input direction both land on the trace. -/
theorem natural_endpoints (wbar m : E) (hw : wbar ≠ 0) (hm : m ≠ 0) :
    ⟪‖wbar‖⁻¹ • wbar, effM1 wbar m (‖wbar‖⁻¹ • wbar)⟫ = ⟪wbar, m⟫ ∧
      ⟪‖m‖⁻¹ • m, effM1 wbar m (‖m‖⁻¹ • m)⟫ = ⟪wbar, m⟫ := by
  have hw0 : ‖wbar‖ ≠ 0 := norm_ne_zero_iff.mpr hw
  have hm0 : ‖m‖ ≠ 0 := norm_ne_zero_iff.mpr hm
  rw [natural_effect, natural_effect]
  simp only [real_inner_smul_left, real_inner_self_eq_norm_sq]
  constructor
  · field_simp
  · rw [real_inner_comm m wbar]; field_simp

/-- **Corollary 6(iii) under one shared direction.** The condition of `Paper.kind2_reaches_trace_iff` holds: the trace
`tr M = ⟪w̄, m⟫` lies in `[λ_{n-1}, λ₀]` (dimension at least 2), since it is the effect along `ŵ` (Proposition 7(ii)), or
`0`, the effect along every unit vector, when `w̄ = 0` or `m = 0`. -/
theorem natural_trace_mem_range [FiniteDimensional ℝ E] {n : ℕ} (hn : Module.finrank ℝ E = n) (h2 : 2 ≤ n)
    (wbar m : E) :
    LinearMap.trace ℝ E (effM1 wbar m) ∈
      Set.Icc ((effM_isSymmetric (fun _ : Unit => (1 : ℝ)) (fun _ => wbar) (fun _ => m)).eigenvalues hn ⟨n - 1, by omega⟩)
        ((effM_isSymmetric (fun _ : Unit => (1 : ℝ)) (fun _ => wbar) (fun _ => m)).eigenvalues hn ⟨0, by omega⟩) := by
  set hT := effM_isSymmetric (fun _ : Unit => (1 : ℝ)) (fun _ => wbar) (fun _ => m)
  rw [← kind2_reaches_trace_iff hn hT h2, natural_trace]
  by_cases h : wbar = 0 ∨ m = 0
  · refine ⟨_, (hT.eigenvectorBasis hn).orthonormal.1 ⟨0, by omega⟩, ?_⟩
    rw [natural_effect]
    rcases h with rfl | rfl <;> simp
  · rw [not_or] at h
    exact ⟨_, norm_smul_inv_norm h.1, (natural_endpoints wbar m h.1 h.2).1⟩

/-- **Proposition 7(iii), the arc.** With orthonormal `m̂, u`, `w̄ = A (cos θ m̂ + sin θ u)` and `m = B m̂`, the unit
vector `d = cos φ m̂ + sin φ u` has `e(d) = tr M · (1 + sin φ sin (θ - φ) / cos θ)`. -/
theorem natural_arc (mh u : E) (hmh : ‖mh‖ = 1) (hu : ‖u‖ = 1) (hmu : ⟪mh, u⟫ = 0) (A B θ φ : ℝ)
    (hθ : cos θ ≠ 0) :
    ‖cos φ • mh + sin φ • u‖ = 1 ∧
      ⟪cos φ • mh + sin φ • u, effM1 (A • (cos θ • mh + sin θ • u)) (B • mh) (cos φ • mh + sin φ • u)⟫ =
        ⟪A • (cos θ • mh + sin θ • u), B • mh⟫ * (1 + sin φ * sin (θ - φ) / cos θ) := by
  have hmm : ⟪mh, mh⟫ = 1 := by rw [real_inner_self_eq_norm_sq, hmh]; ring
  have huu : ⟪u, u⟫ = 1 := by rw [real_inner_self_eq_norm_sq, hu]; ring
  have hum : ⟪u, mh⟫ = 0 := by rw [real_inner_comm]; exact hmu
  constructor
  · have h : ⟪cos φ • mh + sin φ • u, cos φ • mh + sin φ • u⟫ = 1 := by
      simp only [inner_add_left, inner_add_right, real_inner_smul_left, real_inner_smul_right, hmm, huu, hmu, hum]
      nlinarith [sin_sq_add_cos_sq φ]
    rw [real_inner_self_eq_norm_sq] at h
    have := norm_nonneg (cos φ • mh + sin φ • u)
    nlinarith
  · rw [natural_effect]
    simp only [inner_add_left, inner_add_right, real_inner_smul_left, real_inner_smul_right, hmm, huu, hmu, hum]
    rw [sin_sub]
    field_simp
    linear_combination (A * B * cos θ ^ 2) * sin_sq_add_cos_sq φ

/-- **Proposition 7(iii), the top.** `λ₁ / tr M = (1 + 1/cos θ)/2` with `cos θ = ⟪w̄, m⟫ / (‖w̄‖‖m‖) > 0`; `λ₁` is
the top eigenvalue, Mathlib's index `0`. -/
theorem natural_top_ratio [FiniteDimensional ℝ E] {n : ℕ} (hn : Module.finrank ℝ E = n) (h2 : 2 ≤ n)
    (wbar m : E) (hcos : 0 < ⟪wbar, m⟫) :
    (effM_isSymmetric (fun _ : Unit => (1 : ℝ)) (fun _ => wbar) (fun _ => m)).eigenvalues hn ⟨0, by omega⟩ /
        LinearMap.trace ℝ E (effM1 wbar m) =
      (1 + 1 / (⟪wbar, m⟫ / (‖wbar‖ * ‖m‖))) / 2 := by
  rw [top_eigenvalue_single hn h2, natural_trace]
  have hw : wbar ≠ 0 := by rintro rfl; simp at hcos
  have hm : m ≠ 0 := by rintro rfl; simp at hcos
  have hw0 : ‖wbar‖ ≠ 0 := norm_ne_zero_iff.mpr hw
  have hm0 : ‖m‖ ≠ 0 := norm_ne_zero_iff.mpr hm
  field_simp

/-- **Proposition 7(iii), between.** Every direction strictly between `m̂` and `ŵ` (`0 < φ < θ < π/2`) exceeds the
trace. -/
theorem natural_between_exceeds (mh u : E) (hmh : ‖mh‖ = 1) (hu : ‖u‖ = 1) (hmu : ⟪mh, u⟫ = 0) (A B θ φ : ℝ)
    (hA : 0 < A) (hB : 0 < B) (hθ : θ < π / 2) (hφ0 : 0 < φ) (hφθ : φ < θ) :
    ⟪A • (cos θ • mh + sin θ • u), B • mh⟫ <
      ⟪cos φ • mh + sin φ • u, effM1 (A • (cos θ • mh + sin θ • u)) (B • mh) (cos φ • mh + sin φ • u)⟫ := by
  have hcos : 0 < cos θ := cos_pos_of_mem_Ioo ⟨by linarith, hθ⟩
  rw [(natural_arc mh u hmh hu hmu A B θ φ hcos.ne').2]
  have hmm : ⟪mh, mh⟫ = 1 := by rw [real_inner_self_eq_norm_sq, hmh]; ring
  have hum : ⟪u, mh⟫ = 0 := by rw [real_inner_comm]; exact hmu
  have htr : ⟪A • (cos θ • mh + sin θ • u), B • mh⟫ = A * B * cos θ := by
    simp only [inner_add_left, real_inner_smul_left, real_inner_smul_right, hmm, hum]; ring
  rw [htr]
  have hs1 : 0 < sin φ := sin_pos_of_pos_of_lt_pi hφ0 (by linarith)
  have hs2 : 0 < sin (θ - φ) := sin_pos_of_pos_of_lt_pi (by linarith) (by linarith)
  have hpos : 0 < A * B * cos θ := by positivity
  have : 0 < sin φ * sin (θ - φ) / cos θ := by positivity
  nlinarith

/-- **Proposition 7(iii), the top is the bisector, uniquely up to sign.** Among unit vectors, `d` reaches the top
`(⟪w̄, m⟫ + ‖w̄‖‖m‖)/2` of `e(d) = (dᵀw̄)(dᵀm)` if and only if `d = ± c/‖c‖` with `c = ‖m‖ w̄ + ‖w̄‖ m`, the bisector of
`ŵ` and `m̂`. -/
theorem natural_top_bisector (wbar m c d : E) (hcdef : c = ‖m‖ • wbar + ‖wbar‖ • m) (hc : c ≠ 0) (hd : ‖d‖ = 1) :
    ⟪d, effM1 wbar m d⟫ = (⟪wbar, m⟫ + ‖wbar‖ * ‖m‖) / 2 ↔ d = ‖c‖⁻¹ • c ∨ d = -(‖c‖⁻¹ • c) := by
  rw [natural_effect, real_inner_comm wbar d]
  exact two_vector_eq_iff wbar m c d hcdef hc hd

end Paper
