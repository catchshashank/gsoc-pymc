"""
example_usage.py
-----------------
End-to-end demonstration of PredictiveCheck on a deliberately
misspecified model — a single Gaussian fitted to bimodal data.

This mirrors the kind of model criticism described in McLatchie et al.
(2025): the posterior concentrates on a "best single parameter" but
the predictive check reveals that the model fundamentally misses the
bimodal structure.

Run with:  python example_usage.py
"""

import numpy as np
import pymc as pm
import matplotlib.pyplot as plt

from bayes_corner.predictive_check import PredictiveCheck

# -----------------------------------------------------------------------
# 1. Generate deliberately misspecified data
#    Truth: bimodal mixture of two Gaussians
#    Model: single Gaussian (misspecified)
# -----------------------------------------------------------------------
rng = np.random.default_rng(42)
n = 300
# 50% from N(-2, 0.5), 50% from N(2, 0.5)
data = np.concatenate([
    rng.normal(-2, 0.5, n // 2),
    rng.normal(2, 0.5, n // 2),
])

print("Data shape:", data.shape)
print(f"Data mean: {data.mean():.3f}, std: {data.std():.3f}")
print("True distribution: bimodal — N(-2,0.5) and N(2,0.5)")
print()

# -----------------------------------------------------------------------
# 2. Fit the misspecified model: single Gaussian
# -----------------------------------------------------------------------
with pm.Model() as gaussian_model:
    mu = pm.Normal("mu", mu=0, sigma=5)
    sigma = pm.HalfNormal("sigma", sigma=3)
    obs = pm.Normal("obs", mu=mu, sigma=sigma, observed=data)
    idata = pm.sample(draws=1000, tune=1000, random_seed=42, progressbar=True)

print("\nPosterior summary:")
print(az_summary := __import__("arviz").summary(idata, var_names=["mu", "sigma"]))
print()

# -----------------------------------------------------------------------
# 3. Run predictive checks
# -----------------------------------------------------------------------
checker = PredictiveCheck(
    model=gaussian_model,
    idata=idata,
    observed=data,
    var_name="obs",
    random_seed=42,
)

# Prior predictive check
print("Running prior predictive check...")
fig_prior = checker.prior_check(n_samples=50)
fig_prior.savefig("prior_check.png", dpi=150, bbox_inches="tight")
print("Saved: prior_check.png")

# Posterior predictive check
print("Running posterior predictive check...")
fig_post = checker.posterior_check(n_samples=50)
fig_post.savefig("posterior_check.png", dpi=150, bbox_inches="tight")
print("Saved: posterior_check.png")

# Scoring: Bayesian p-values for key statistics
print("\nBayesian p-values (posterior):")
print("  A value near 0.5 = good fit for that statistic.")
print("  A value near 0 or 1 = model fails to capture that feature.\n")

for stat in ["mean", "std", "skew", "kurtosis"]:
    result = checker.score(stat=stat, group="posterior")
    p = result["bayesian_p_value"]
    flag = " <-- MODEL FAILURE" if p < 0.05 or p > 0.95 else ""
    print(f"  {stat:12s}: p = {p:.3f}{flag}")
    result["fig"].savefig(f"score_{stat}.png", dpi=120, bbox_inches="tight")
    plt.close(result["fig"])

# Full summary sweep
print("\nFull summary sweep:")
summary = checker.summary(group="posterior")
for k, v in summary.items():
    print(f"  {k:12s}: {v:.3f}" if v is not None else f"  {k:12s}: FAILED")

print("\nDone. Check the saved .png files.")
