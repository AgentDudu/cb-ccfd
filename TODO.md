# TODO — Gaps Worth Improving

Working backlog for the contextual-bandit fraud detection project. Reference docs:
`MATH_MODEL.md` (math formulation), `README.md` (current results).

Status legend: `[ ]` todo · `[~]` in progress · `[x]` done

## 1. Algorithmic gaps

- [ ] **GLM-UCB / logistic-UCB variant** — LinUCB assumes a *linear* expected reward,
  but `P(y=1|x)` is logistic on the VSA features. `MATH_MODEL.md` §3 explicitly flags
  this misspecification. Implement a logistic-UCB (or GLM-UCB) arm model so the
  linearity assumption can be tested empirically against LinUCB.
- [ ] **Drift handling: sliding window / forgetting factor** — `A_a` accumulates over
  200k+ steps with no forgetting, so the model becomes rigid and cannot track the
  well-documented temporal drift in this dataset. Add a discounted update
  (`A_a ← γ·A_a + x xᵀ`) or a rolling window, with `γ` exposed as a CLI flag.
- [ ] **Reward-scale normalization** — all rewards are ≤ 0 with a 10× asymmetry
  (`R_FP=-1`, `R_FN=-10`). The UCB width for arm 0 (mostly 0 reward) vs arm 1 is not
  normalized, so `alpha` is not interpretable across reward matrices. Normalize or
  report `alpha` in reward units.
- [ ] **Probabilistic LinUCB score** — PR-AUC is currently `n/a` for LinUCB, so it
  cannot be compared on ranking quality. Emit a calibrated score, e.g.
  `sigma(theta_1^T x - theta_0^T x)`, and feed it to `calculate_pr_auc`.
- [ ] **Delayed / censored feedback** — `MATH_MODEL.md` §1 discusses labels revealed
  later, but the implementation assumes instant observation. Model a delayed-label
  queue (e.g. fraud reported after N transactions) and evaluate degradation.
- [ ] **Random tie-breaking** — `predict` uses `np.where(p_1 > p_0, 1, 0)`, i.e.
  deterministic prefer-approve. The documented random tie-break option is missing.

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
