# TODO — Gaps Worth Improving

Working backlog for the contextual-bandit fraud detection project. Reference docs:
`MATH_MODEL.md` (math formulation), `README.md` (current results).

Status legend: `[ ]` todo · `[~]` in progress · `[x]` done

## 1. Algorithmic gaps

- [x] **GLM-UCB / logistic-UCB variant** — implemented in `src/models/glmucb.py`: a single
  logistic GLM for `p(x) = P(y=1|x)` trained by online Newton/IRLS, optimism applied to the
  probability, action taken by comparing `p_bar(x)` against the reward-matrix threshold
  `p* = 1/11`. Result: reward -202 vs LinUCB -192 and RF -178 (PR-AUC 0.720). Follow-ups
  discovered while implementing, kept below as open items.
- [ ] **Drift handling: sliding window / forgetting factor** — `A_a` accumulates over
  200k+ steps with no forgetting, so the model becomes rigid and cannot track the
  well-documented temporal drift in this dataset. Add a discounted update
  (`A_a ← γ·A_a + x xᵀ`) or a rolling window, with `γ` exposed as a CLI flag.
- [ ] **Reward-scale normalization** — all rewards are ≤ 0 with a 10× asymmetry
  (`R_FP=-1`, `R_FN=-10`). The UCB width for arm 0 (mostly 0 reward) vs arm 1 is not
  normalized, so `alpha` is not interpretable across reward matrices. Normalize or
  report `alpha` in reward units.
- [x] **Probabilistic LinUCB score** — `LinUCB.predict_proba` inverts the reward matrix
  (`p = (q_a - R(a,0)) / (R(a,1) - R(a,0))`) on both arms and averages them. Result:
  LinUCB PR-AUC = 0.202 vs GLM-UCB 0.720, i.e. LinUCB's action choices are good but its
  fitted arm models rank frauds poorly.
- [ ] **Revisit the default policy given ranking quality** — LinUCB wins on cumulative
  reward (-192 vs -202) but its implied-probability PR-AUC is 3.5x worse than GLM-UCB's.
  Decide whether reward or ranking should drive the default, and whether a reward-aware
  threshold on GLM-UCB scores (tuned on validation, like the baselines) closes the gap.
- [ ] **Delayed / censored feedback** — `MATH_MODEL.md` §1 discusses labels revealed
  later, but the implementation assumes instant observation. Model a delayed-label
  queue (e.g. fraud reported after N transactions) and evaluate degradation.
- [ ] **Random tie-breaking** — `predict` uses `np.where(p_1 > p_0, 1, 0)`, i.e.
  deterministic prefer-approve. The documented random tie-break option is missing.
- [ ] **GLM-UCB: intercept term** — contexts have no constant feature, so the logistic model
  cannot fit the base rate (mean p̂ = 0.020 vs true test rate 0.0013). Adding an intercept
  improves PR-AUC (0.720 → 0.745) but *worsens* reward (-202 → -364) because better
  calibration means far fewer blocks. Needs a calibration/threshold study before adopting.
- [ ] **GLM-UCB: per-arm variant is degenerate** — Faury-style per-arm logistic models of the
  rescaled reward collapse to block-all (reward-matrix offset, not context, separates the
  arms). Documented in `README.md`; do not re-attempt without changing the reward encoding.

## 2. Evaluation rigor

- [ ] **Hyperparameter sweep** — results exist only for `alpha=0.1, lambda_reg=1.0,
  alpha_decay=1.0`. Add a sweep script over `alpha`, `lambda_reg`, `alpha_decay`
  (and later `gamma`) writing a comparison table.
- [ ] **Multiple seeds / variance** — single run, no confidence intervals. Add repeated
  runs (seeds + tie-breaking randomness) and report mean ± std per policy.
- [ ] **Offline counterfactual evaluation** — no IPS / doubly-robust estimate and no
  cumulative-regret trajectory over time, which is the bandit's main selling point.
- [ ] **Regret & reward trajectory plots** — nothing is persisted today (stdout only),
  so runs cannot be diffed. Save results + curves as artifacts (CSV/JSON + figures)
  under a `results/` or `reports/` directory convention.
- [ ] **Exact threshold grid** — `find_optimal_threshold` uses a coarse
  `linspace(0.01, 0.99, 100)` grid. Use the unique validation probabilities as candidate
  thresholds (RF's chosen 0.109 sits near grid resolution).
- [ ] **Prequential baseline counterpart** — baselines are trained batch on the full
  train split while LinUCB sees data sequentially. Add an incrementally trained
  (partial-fit) baseline to quantify this asymmetry.

## 3. Engineering / hygiene

- [ ] **Vectorize the stream reward** — `run_bandit_stream` calls
  `calculate_cumulative_reward` per sample, routing scalars through a vectorized
  function. Precompute the reward lookup for speed.
- [ ] **Numerical stability of Sherman–Morrison** — ~230k rank-1 inverse updates risk
  drift. Periodically re-factorize via Cholesky (as recommended in `MATH_MODEL.md` §3)
  or recompute `A_inv` on an interval; assert symmetry/positive-definiteness.
- [ ] **Package structure** — no `__init__.py` in `src/`, `src/models/`,
  `src/data_processing/`, `src/evaluation/` (works via namespace packages, but fragile).
- [ ] **Formatting & tooling** — no `black`/`isort` config, no lint or CI. Add config
  files and a minimal CI workflow.
- [ ] **Results directory convention** — define where artifacts/figures land, and
  `.gitignore` them appropriately.

## Notes

- Per `AGENTS.md`: do **not** write unit tests; do **not** read `data/raw/creditcard.csv`
  (always load `data/processed/creditcard_clean.parquet`).
- Commit style: `<type>(<scope>): <description>` on `main`, then `git push`.
