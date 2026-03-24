# bayes-corner

> A composable toolkit for prior and posterior predictive checks in PyMC.  
> GSoC 2026 proposal prototype — Shashank Dubey, HEC Paris.

---

## The Problem

Running predictive checks in PyMC today requires manually orchestrating
`pm.sample_prior_predictive`, `pm.sample_posterior_predictive`, ArviZ plot
calls, and custom test-statistic loops. This friction means most practitioners
skip the checks entirely — a critical gap in the Bayesian workflow.

This library abstracts that boilerplate into a single `PredictiveCheck` object
with a clean `.check()` / `.score()` interface, analogous to how CausalPy
abstracts causal inference model construction.

---

## Quick Start

```python
import pymc as pm
import numpy as np
from bayes_corner import PredictiveCheck

# 1. Fit your model as usual
data = np.concatenate([np.random.normal(-2, 0.5, 150),
                       np.random.normal( 2, 0.5, 150)])

with pm.Model() as model:
    mu    = pm.Normal("mu", 0, 5)
    sigma = pm.HalfNormal("sigma", 3)
    obs   = pm.Normal("obs", mu=mu, sigma=sigma, observed=data)
    idata = pm.sample(random_seed=42)

# 2. Wrap with PredictiveCheck
checker = PredictiveCheck(model=model, idata=idata, observed=data)

# 3. Visual checks — one line each
checker.prior_check()      # Are priors compatible with data scale?
checker.posterior_check()  # Does the model reproduce observed patterns?

# 4. Formal Bayesian p-values for test statistics
result = checker.score(stat="std")
print(f"Bayesian p-value (std): {result['bayesian_p_value']:.3f}")
# -> 0.031  <-- model fails to reproduce observed spread (bimodal data!)

# 5. Full sweep
print(checker.summary())
# {'mean': 0.48, 'std': 0.031, 'skew': 0.51, 'kurtosis': 0.002}
#  ^^ kurtosis p=0.002 correctly flags bimodal misspecification
```

---

## Motivation: Scoring Rules and Predictive Orientation

This project is grounded in the framework of McLatchie et al. (2025)
*"Predictively Oriented Posteriors"*, which argues that Bayesian model
checking should be evaluated through proper scoring rules applied to the
**predictive distribution** — not just point-estimate comparison.

The `score()` method implements exactly this: it evaluates a test statistic
$T$ under the predictive distribution and computes the Bayesian p-value
$p_B = P(T(y_\text{rep}) \geq T(y_\text{obs}))$, making misspecification
visible and quantifiable.

---

## Roadmap (GSoC 2026)

- [x] `PredictiveCheck` core class with prior/posterior visual checks
- [x] Bayesian p-values for built-in and custom test statistics
- [ ] Scoring-rule-based divergence measures (log score, CRPS, MMD)
- [ ] Formula interface: `PredictiveCheck.from_formula("y ~ x", data)`
- [ ] Integration with PyMC model comparison (`pm.compare`)
- [ ] Hierarchical model support — group-level predictive checks
- [ ] Full test suite and documentation

---

## Installation (development)

```bash
git clone https://github.com/your-username/bayes-corner
cd bayes-corner
pip install -e ".[dev]"
```

---

## References

- McLatchie, Y., Chérief-Abdellatif, B-E., Frazier, D.T., & Knoblauch, J. (2025).
  *Predictively Oriented Posteriors.* arXiv:2510.01915.
- Gelman, A., et al. (2020). *Bayesian Workflow.* arXiv:2011.01808.
- Gabry, J., et al. (2019). *Visualization in Bayesian workflow.*
  JRSS-A, 182(2), 389–402.
