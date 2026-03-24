# gsoc-pymc-pro

> **Predictive Model Checking for PyMC — GSoC 2026 Prototype**  
> Shashank Dubey · HEC Paris  
> Mentors: Chris Fonnesbeck, PyMC core team

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/gsoc-pymc-pro/blob/main/gsoc_pymc_pro_demo.ipynb)

---

## What This Is

A self-contained prototype for a GSoC 2026 project proposing to bring
principled, scoring-rule-grounded predictive model checking to PyMC.

The core deliverable is `PredictiveCheck` — a composable wrapper around any
fitted PyMC model that reduces a full predictive checking workflow to four
method calls:

```python
checker = PredictiveCheck(model=model, idata=idata, observed=data)

checker.prior_check()      # Are priors compatible with the data scale?
checker.posterior_check()  # Does the fitted model reproduce observed patterns?
checker.summary()          # Bayesian p-values for 7 built-in test statistics
checker.jensen_gap()       # How much predictive information is lost by posterior collapse?
```

---

## The Problem

Running a complete predictive check in PyMC today requires a practitioner to
manually orchestrate `pm.sample_prior_predictive`, `pm.sample_posterior_predictive`,
xarray extraction from InferenceData, custom plotting loops, and custom test-statistic
calculations — before any scientific insight is produced. The consequence is
predictable: most practitioners skip the checks entirely.

This project wraps that boilerplate into a single coherent interface, analogous to
how CausalPy abstracts causal inference model construction in PyMC.

---

## The Theoretical Grounding

This prototype goes beyond standard ArviZ tooling. The `jensen_gap()` method
implements a diagnostic directly from McLatchie et al. (2025),
*Predictively Oriented Posteriors* (arXiv:2510.01915).

The paper identifies a fundamental tension in Bayesian inference: standard
posteriors — including Gibbs posteriors — are designed to find the best
*single parameter*, then predict from it. When a model is misspecified,
however, *averaging predictions across parameters* beats predicting from any
one parameter. The **Jensen Gap** measures exactly how much predictive
information is lost by this collapse:

$$\text{JensenGap} = \underbrace{\frac{1}{n}\sum_{i=1}^n \int S(P_\theta, x_i)\,dQ(\theta)}_{\text{Gibbs/Bayes: score each } \theta \text{, then average}} \;-\; \underbrace{\frac{1}{n}\sum_{i=1}^n S\!\left(\int P_\theta\,dQ(\theta),\ x_i\right)}_{\text{PrO: average } \theta \text{ first, then score the mixture}}$$

By Jensen's inequality this gap is always ≥ 0. Crucially:

- **Gap ≈ 0** → model is well-specified; posterior collapse costs nothing
- **Gap >> 0** → model is non-trivially misspecified; the PrO posterior
  would outperform the Bayes posterior at prediction

This makes `jensen_gap()` a formal, quantitative diagnostic for model
misspecification — not just a visual heuristic.

---

## The Demo Notebook

**`gsoc_pymc_pro_demo.ipynb`** is fully self-contained. No cloning, no
external imports — the entire `PredictiveCheck` class is defined inline in
cell 1. Open in Colab and run top to bottom.

### What the notebook demonstrates

The demo fits a deliberately misspecified model — a single Gaussian applied
to bimodal data — and shows that:

1. **Convergence diagnostics alone cannot detect misspecification.** R̂ ≈ 1
   and ESS look healthy. The model appears to have sampled correctly.

2. **The posterior predictive check reveals the failure visually.** The
   fitted model produces unimodal predictive draws; the observed data is
   clearly bimodal.

3. **Bayesian p-values quantify which features fail.** The `mean` p-value
   is near 0.5 (the model gets the grand mean right). The `kurtosis` and
   `std` p-values are extreme — flagging the bimodal structure the model
   cannot capture.

4. **The Jensen Gap confirms non-trivial misspecification formally.** For
   the misspecified Gaussian, the gap is large and positive — every
   observation contributes. When the correct mixture model is fit instead,
   the gap collapses to near zero.

### Notebook structure

| Section | Content |
|---|---|
| 0 | Install dependencies |
| 1 | Full `PredictiveCheck` library — defined inline |
| 2 | Generate misspecified data (bimodal truth, Gaussian model) |
| 3 | Define model + prior predictive check |
| 4 | Fit model with NUTS |
| 5 | Posterior predictive check |
| 6 | Bayesian p-values for mean, std, skew, kurtosis |
| 7 | **Jensen Gap** — the PrO paper diagnostic |
| 8 | Sanity check: well-specified mixture model |
| 9 | Summary comparison table |

---

## Key Results

| Diagnostic | Single Gaussian (wrong) | Mixture (correct) |
|---|---|---|
| R̂ | ✅ ≈ 1 | ✅ ≈ 1 |
| Posterior predictive visual | ❌ Unimodal | ✅ Bimodal |
| p-value — mean | ✅ ~0.5 | ✅ ~0.5 |
| p-value — std | ❌ Extreme | ✅ ~0.5 |
| p-value — kurtosis | ❌ Extreme | ✅ ~0.5 |
| **Jensen Gap** | **❌ Large > 0** | **✅ ≈ 0** |

The Jensen Gap is the only diagnostic that directly quantifies *how wrong*
the model is in a theoretically grounded way — not just whether it looks
wrong visually.

---

## API Reference

### `PredictiveCheck(model, idata, observed, var_name=None, random_seed=42)`

Wraps a fitted PyMC model and its InferenceData.

| Method | Returns | Description |
|---|---|---|
| `prior_check(n_samples=50)` | `Figure` | Visual overlay of prior predictive draws vs observed |
| `posterior_check(n_samples=50)` | `Figure` | Visual overlay of posterior predictive draws vs observed |
| `score(stat, group='posterior')` | `dict` | Bayesian p-value for one test statistic |
| `summary(group='posterior')` | `dict` | All 7 built-in statistics at once |
| `jensen_gap(mu_param, sigma_param)` | `dict` | Jensen Gap diagnostic from McLatchie et al. (2025) |

Built-in statistics for `score()`: `mean`, `std`, `median`, `min`, `max`,
`skew`, `kurtosis`, or `custom` with a user-supplied function.

---

## GSoC 2026 Roadmap

- [x] `PredictiveCheck` core class — prior/posterior visual checks
- [x] Bayesian p-values for built-in and custom test statistics
- [x] Jensen Gap diagnostic grounded in McLatchie et al. (2025)
- [ ] CRPS, energy score, and MMD scoring rules (Phase 2)
- [ ] `PredictiveCheck.from_formula(formula, data)` constructor (Phase 3)
- [ ] `checker.compare(other_checker)` — scoring-rule model comparison (Phase 3)
- [ ] Hierarchical model support — group-level predictive checks (Phase 3)
- [ ] Full test suite, documentation, PyMC-examples PR (Phase 4)

---

## Dependencies

```
pymc >= 5.0
arviz >= 0.17
numpy >= 1.24
matplotlib >= 3.7
scipy >= 1.10
```

Install with:

```bash
pip install pymc arviz scipy
```

---

## References

- McLatchie, Y., Chérief-Abdellatif, B-E., Frazier, D.T., & Knoblauch, J. (2025).
  *Predictively Oriented Posteriors.* arXiv:2510.01915.
- Gelman, A., et al. (2020). *Bayesian Workflow.* arXiv:2011.01808.
- Gabry, J., Simpson, D., Vehtari, A., Betancourt, M., & Gelman, A. (2019).
  *Visualization in Bayesian workflow.* JRSS-A, 182(2), 389–402.
- Gneiting, T. & Raftery, A.E. (2007). *Strictly proper scoring rules, prediction,
  and estimation.* JASA, 102(477), 359–378.
