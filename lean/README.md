# The paper's theorems in Lean

The paper's proved statements are checked here in Lean 4 with Mathlib (versions pinned in `lean-toolchain` and
`lakefile.toml`). The few that are not, and why, are listed at the end.

```bash
cd lean
lake exe cache get      # once: Mathlib's prebuilt files
lake build              # builds everything in Paper/, with no sorry
```

Each theorem rests only on Lean's three standard axioms (`propext`, `Classical.choice`, `Quot.sound`). To see this for one,
put `import Paper` and `#print axioms Paper.<name>` in a file and run `lake env lean <file>` in `lean/` after the build.

**Reading the table.** Statement numbers are the paper's, and its Appendix A.6 lists every statement that is checked;
the table below adds their files and a few supporting lemmas. Mathlib numbers eigenvalues from 0 in decreasing order, so the
paper's `λ₁` and `λ_D` sit at indices `0` and `n - 1`.

**What "first order" means.** For pair `i`, a patch `v` changes the output by `⟪wᵢ, v⟫` to first order, with `wᵢ` the
gradient at the midpoint of the full swap. Lean checks the paper's first-order statements as exact statements about that
quantity. The step from the network to it is the expansion of Appendix A.1, checked too: the second-order term exactly
(`midpoint_second_order` and the lemmas after it), and the remainder bounded by the third derivative, along the segment for
the full swap (`full_swap_third_order`) and along the two segments from the midpoint for any other patch
(`general_patch_third_order`). Lean asks a little more than the paper, `F` differentiable everywhere and the paths `C³` on all
of ℝ; the networks measured are smooth everywhere.

| paper | statement | Lean (file in `Paper/`) |
|---|---|---|
| Prop. 3 | complete-cut identity, `G(C) = T` | `complete_cut_identity` (`CompleteCut`) |
| Thm. 5(i) | the rank-one mean effect is `dᵀMd`; it takes exactly the values in `[λ_D, λ₁]` (dimension ≥ 2) | `rank_one_effect_eq`, `rank_one_effect_range` (`Range`); `rayleigh_range` (`Rayleigh`) |
| Thm. 5(ii) | the own interchange has mean effect `tr M = ∑ λⱼ`; third order for the full swap | `trace_effM`, `trace_effM_eq_sum_eigenvalues` (`Range`); `full_swap_third_order` (`Expansion`) |
| Thm. 5(iii), App. A.3 | rank `k`: mean effect `tr(PᵀMP)`, between the sums of the `k` smallest and `k` largest eigenvalues (Ky Fan) | `rank_k_effect`, `kyfan_upper`, `kyfan_lower` (`KyFan`) |
| App. A.3, one shared direction | a larger rank does not raise the top: every rank-`k` interchange stays at or below the rank-one top, and for every `k < D` some rank-`k` interchange reaches it | `natural_rank_k_le`, `natural_rank_k_attained`, `natural_rank_k_exists` (`RankK`) |
| Thm. 5(iv) | single pair: `λ₁ = (wᵀΔ + ‖w‖‖Δ‖)/2 ≥ wᵀΔ`, equality iff parallel with `wᵀΔ ≥ 0` | `top_eigenvalue_single`, `top_ge_trace_single` (`Range`); `two_vector_isGreatest`, `two_vector_eq_iff` (`TwoVector`) |
| Thm. 5, any basis | along any orthonormal basis the rank-one effects sum to the trace, so one exceeds it exactly when the others sum below zero; grouped into orthogonal parts of any rank (Section 1), the parts' effects, each the sum of its rank-one effects, sum to the trace | `sum_rank_one_effects_basis`, `rank_one_exceeds_iff_rest_negative`, `part_effect_eq_sum_rank_one`, `sum_part_effects_basis` (`AnyBasis`) |
| Ex. 1 | two dimensions, `Δ = (1, 0)`, `w = (1, 1)`: trace `1`, the variable's direction `1`; the bisector and the orthogonal direction are eigenvectors with effects `(1 + √2)/2 ≈ 1.207` and `(1 - √2)/2 ≈ -0.207`, and move the state by `±√2/4 ≈ 0.354` along the second coordinate; with `w = (0, 1)` the trace is `0` and the bisector gives `1/2` | `two_dimensions_example` (`ExampleTwo`) |
| Cor. 6(i) | an effect-seeking objective in pure form selects the top eigenspace and reports `λ₁`; excess `-∑_{j≥2} λⱼ` | `kind1_selects_top`, `top_sub_trace` (`Selection`); `effM_shared_readout` (`Natural`) |
| Cor. 6(ii) | a target-aimed objective in pure form selects the level set when it is non-empty, which holds iff `λ_D ≤ τ ≤ λ₁` | `kind2_selects_level` (`Selection`) |
| Cor. 6(iii) | with `τ = T` it reaches `T` exactly when `T ∈ [λ_D, λ₁]` (always under Prop. 7, not in general) | `kind2_reaches_trace_iff` (`Selection`); `natural_trace_mem_range` (`Natural`); `counterexample_undershoot` (`Counterexamples`) |
| Section 3.2, whose effect the excess is | where the whole effect is positive, a part exceeds it only through its component perpendicular to `Δ`, which moves the output by at least the excess; the `1.207` example is `confined_example` | `part_exceeds_through_perpendicular`, `part_exceeds_only_perpendicular` (`Perpendicular`) |
| Section 3 | cross-entropy of the source label has derivative `p_b`, the base label's probability, in that label's logit | `crossEntropy_deriv_base` (`Trapezoid`) |
| Prop. 7 | one shared direction (a shared readout direction or a single code): `e(d) = (dᵀw̄)(dᵀm)`, `tr M = w̄ᵀm`; both natural directions land on the trace; the angle formulas `e(d)/tr M = 1 + sin φ sin(θ - φ)/cos θ` and `λ₁/tr M = (1 + 1/cos θ)/2`, every direction strictly between exceeds the trace, the top is the bisector | `effM_shared_readout`, `effM_single_code`, `natural_effect`, `natural_trace`, `natural_endpoints`, `natural_arc`, `natural_top_ratio`, `natural_between_exceeds`, `natural_top_bisector` (`Natural`) |
| Prop. 8 | the door: the largest rank-one score change, the logistic bounds on the attention weight, the unread difference | `door_max`, `softmax_eq_sigm`, `door_shut`, `door_open`, `door_unread` (`Door`) |
| Prop. 9 | the trace's error on a mirrored set: for every path of degree at most four the midpoint's error is exactly `-1/2` of the trapezoid's and Simpson's combination is `T`; the errors of a cubic; the mirror's terms | `midpoint_error_neg_half`, `simpson_exact`, `trapezoid_error_cubic`, `midpoint_error_cubic`, `mirror_terms` (`Trapezoid`) |
| Prop. 9, general path | with `|g⁽⁵⁾| ≤ L` the relation holds up to `L/480` and Simpson's rule up to `L/720`; the leading orders with `|g⁽⁴⁾| ≤ L₄` | `midpoint_neg_half_remainder`, `simpson_remainder`, `trapezoid_leading_order`, `midpoint_leading_order` (`TrapezoidRemainder`) |
| Rem. 10 | attribution patching's second-order term cancels on a mirrored pair; along the path, the base-point trace is within `L/8` of the midpoint trace on a mirrored pair at a complete cut (`L` bounds `g'''` on `[0, 1]`, third order) and a single pair's term within `L₂/2` (`L₂` bounds `g''`, second order) | `atp_mirrored` (`Expansion`); `atp_mirrored_third_order`, `atp_single_second_order` (`TrapezoidRemainder`) |
| App. A.1 | the midpoint expansion: second-order term `½ vᵀH(v-Δ)`, zero for the full swap, its rank-one form, cancelled by a mirrored pair | `midpoint_second_order`, `full_swap_second_order`, `rank_one_second_order`, `mirrored_second_order` (`Expansion`) |
| App. A.1, general patch | `F(h_b + v) - F(h_b) = ⟪w, v⟫ + ½ D²F(c)(v, v - Δ) + R`, `|R| ≤ (L_a + L_b)/6` | `general_patch_third_order` (`GeneralPatch`) |
| App. A.1, a pair and its mirror | the two errors of the midpoint's first-order value add up to two central-difference remainders | `mirror_pair_error_eq`, `central_difference_le`, `hasDerivAt_line` (`MirrorCancel`) |
| App. A.1, one dimension | `|φ(1) - φ(-1) - 2φ'(0)| ≤ L/3` when `|φ'''| ≤ L` on `[-1, 1]`; `1/3` is sharp | `symm_diff_first_order`, `symm_diff_first_order_sharp` (`SymmetricDifference`) |
| App. A.4 | confinement: one code makes the rank-one interchange the own one; the compression keeps the trace of `M`; the confined top (also "size alone"); the `1.207 tr M` example | `single_code_rank_one`, `confined_trace`, `confined_two_vector`, `confined_example` (`Confinement`) |
| App. A.5 | an indefinite `M` with the top below the trace; the first-order undershoot; a per-pair squared error not minimised where the mean effect is `T` | `counterexample_indefinite`, `counterexample_undershoot`, `counterexample_sq_error` (`Counterexamples`) |
| App. A.5 | the per-pair squared error's exact minimum and its minimisers' mean effect; with one readout direction for every pair the loss reaches `0` and every minimiser's mean effect is `T` | `sq_error_lower_bound`, `sq_error_attained`, `sq_error_minimiser_mean`, `sq_error_shared_readout` (`SqErrorMinimiser`) |
| Appendix C.4 | a saturating cap makes first order undershoot | `hermite_hadamard_deriv`, `cap_undershoot` (`CapUndershoot`) |

**Not formalised, and why.**

- Statements about trained models are measurements, not theorems. Appendix A.4's "size alone" values follow from
  `confined_two_vector` with the expected share of a random projection, which is not formalised.
- Prop. 7's angle statements are proved in the parametrisation `ŵ = cos θ m̂ + sin θ u` with orthonormal `m̂, u`, which is
  how the angle `θ = ∠(w̄, m)` is defined in their span (`u` is the normalised component of `ŵ` orthogonal to `m̂`).
- Prop. 3 is proved for any network whose output factors through the states at the cut and side inputs that agree; that
  the cuts of Definition 2 factor in this way is the content of the definition, not a theorem.
- Appendix C.4: the facts about `c·tanh(·/c)` (`g'` convex past `0.66c`, decreasing in `|o|`) and how often the hypotheses
  hold in Gemma-2's pairs are measured.
