"""
bayes_corner.predictive_check
------------------------------
A lightweight, composable interface for prior and posterior predictive
checks in PyMC, inspired by the scoring-rule framework of McLatchie et al.
(2025) and the Bayesian workflow of Gelman et al. (2020).

The core idea: rather than requiring users to manually call
pm.sample_prior_predictive / pm.sample_posterior_predictive and then
wire up ArviZ plots by hand, PredictiveCheck wraps a fitted PyMC model
and exposes a clean .check() interface with sensible defaults and
composable test statistics.

Usage
-----
    import pymc as pm
    from bayes_corner.predictive_check import PredictiveCheck

    with pm.Model() as model:
        mu = pm.Normal("mu", 0, 1)
        sigma = pm.HalfNormal("sigma", 1)
        obs = pm.Normal("obs", mu=mu, sigma=sigma, observed=data)
        idata = pm.sample()

    checker = PredictiveCheck(model=model, idata=idata, observed=data)
    checker.prior_check()          # visual prior predictive check
    checker.posterior_check()      # visual posterior predictive check
    checker.score(stat="mean")     # Bayesian p-value for mean
    checker.score(stat="std")      # Bayesian p-value for std
    checker.score(stat="custom", fn=my_fn)  # user-defined statistic
"""

from __future__ import annotations

import warnings
from typing import Callable, Literal, Optional

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm
from arviz import InferenceData


# ---------------------------------------------------------------------------
# Supported built-in test statistics
# ---------------------------------------------------------------------------
_BUILTIN_STATS: dict[str, Callable[[np.ndarray], float]] = {
    "mean": np.mean,
    "std": np.std,
    "median": np.median,
    "min": np.min,
    "max": np.max,
    "skew": lambda x: float(
        np.mean(((x - np.mean(x)) / (np.std(x) + 1e-8)) ** 3)
    ),
    "kurtosis": lambda x: float(
        np.mean(((x - np.mean(x)) / (np.std(x) + 1e-8)) ** 4) - 3
    ),
}


class PredictiveCheck:
    """
    Wraps a PyMC model and its InferenceData to expose a clean interface
    for prior and posterior predictive checks.

    Parameters
    ----------
    model : pm.Model
        The compiled PyMC model (used to draw new predictive samples
        if they are not already in idata).
    idata : InferenceData
        ArviZ InferenceData object. Should contain at minimum a
        ``posterior`` group; prior_predictive and
        posterior_predictive groups are generated on demand if absent.
    observed : np.ndarray
        The raw observed data array used to fit the model.
    var_name : str, optional
        Name of the observed variable in the model. If None, the first
        observed variable in the model is used.
    random_seed : int, optional
        Seed for reproducibility when drawing predictive samples.
    """

    def __init__(
        self,
        model: pm.Model,
        idata: InferenceData,
        observed: np.ndarray,
        var_name: Optional[str] = None,
        random_seed: int = 42,
    ) -> None:
        self.model = model
        self.idata = idata
        self.observed = np.asarray(observed)
        self.random_seed = random_seed

        # Resolve observed variable name
        observed_vars = [v.name for v in model.observed_RVs]
        if not observed_vars:
            raise ValueError("Model has no observed random variables.")
        if var_name is None:
            self.var_name = observed_vars[0]
        elif var_name in observed_vars:
            self.var_name = var_name
        else:
            raise ValueError(
                f"'{var_name}' not found among observed variables: {observed_vars}"
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_prior_predictive(self) -> None:
        """Draw prior predictive samples if not already in idata."""
        if "prior_predictive" not in self.idata.groups():
            with self.model:
                prior_pred = pm.sample_prior_predictive(
                    random_seed=self.random_seed
                )
            self.idata.extend(prior_pred)

    def _ensure_posterior_predictive(self) -> None:
        """Draw posterior predictive samples if not already in idata."""
        if "posterior_predictive" not in self.idata.groups():
            with self.model:
                post_pred = pm.sample_posterior_predictive(
                    self.idata, random_seed=self.random_seed
                )
            self.idata.extend(post_pred)

    def _extract_samples(self, group: str) -> np.ndarray:
        """
        Return a flat 2-D array (n_samples, n_obs) from the given
        InferenceData group for self.var_name.
        """
        group_data = getattr(self.idata, group)
        arr = group_data[self.var_name].values  # (chain, draw, *obs_shape)
        # Flatten chain × draw into a single samples dimension
        n_chains, n_draws = arr.shape[:2]
        return arr.reshape(n_chains * n_draws, -1)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def prior_check(
        self,
        n_samples: int = 50,
        figsize: tuple[int, int] = (10, 4),
        title: Optional[str] = None,
    ) -> plt.Figure:
        """
        Visual prior predictive check.

        Overlays `n_samples` draws from the prior predictive distribution
        against the observed data histogram, making it easy to see
        whether the prior is broadly compatible with the data scale.

        Parameters
        ----------
        n_samples : int
            Number of prior predictive draws to overlay.
        figsize : tuple
            Figure size passed to matplotlib.
        title : str, optional
            Custom title for the plot.

        Returns
        -------
        matplotlib.Figure
        """
        self._ensure_prior_predictive()
        samples = self._extract_samples("prior_predictive")  # (S, N)

        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # Left: overlay of prior predictive draws
        ax = axes[0]
        idx = np.random.default_rng(self.random_seed).choice(
            len(samples), size=min(n_samples, len(samples)), replace=False
        )
        for i in idx:
            ax.hist(
                samples[i],
                bins=30,
                alpha=0.05,
                color="steelblue",
                density=True,
            )
        ax.hist(
            self.observed,
            bins=30,
            alpha=0.6,
            color="black",
            density=True,
            label="Observed",
            histtype="step",
            linewidth=2,
        )
        ax.set_title("Prior Predictive Draws vs. Observed")
        ax.set_xlabel(self.var_name)
        ax.legend()

        # Right: ArviZ plot_ppc for the prior group
        az.plot_ppc(self.idata, group="prior", ax=axes[1], num_pp_samples=n_samples)
        axes[1].set_title("Prior Predictive Check (ArviZ)")

        fig.suptitle(title or f"Prior Predictive Check — {self.var_name}", y=1.02)
        fig.tight_layout()
        return fig

    def posterior_check(
        self,
        n_samples: int = 50,
        figsize: tuple[int, int] = (10, 4),
        title: Optional[str] = None,
    ) -> plt.Figure:
        """
        Visual posterior predictive check.

        Parameters
        ----------
        n_samples : int
            Number of posterior predictive draws to overlay.
        figsize : tuple
            Figure size.
        title : str, optional
            Custom title.

        Returns
        -------
        matplotlib.Figure
        """
        self._ensure_posterior_predictive()
        samples = self._extract_samples("posterior_predictive")  # (S, N)

        fig, axes = plt.subplots(1, 2, figsize=figsize)

        ax = axes[0]
        idx = np.random.default_rng(self.random_seed).choice(
            len(samples), size=min(n_samples, len(samples)), replace=False
        )
        for i in idx:
            ax.hist(
                samples[i],
                bins=30,
                alpha=0.05,
                color="darkorange",
                density=True,
            )
        ax.hist(
            self.observed,
            bins=30,
            alpha=0.6,
            color="black",
            density=True,
            label="Observed",
            histtype="step",
            linewidth=2,
        )
        ax.set_title("Posterior Predictive Draws vs. Observed")
        ax.set_xlabel(self.var_name)
        ax.legend()

        az.plot_ppc(self.idata, group="posterior", ax=axes[1], num_pp_samples=n_samples)
        axes[1].set_title("Posterior Predictive Check (ArviZ)")

        fig.suptitle(title or f"Posterior Predictive Check — {self.var_name}", y=1.02)
        fig.tight_layout()
        return fig

    def score(
        self,
        stat: Literal["mean", "std", "median", "min", "max", "skew", "kurtosis", "custom"] = "mean",
        fn: Optional[Callable[[np.ndarray], float]] = None,
        group: Literal["prior", "posterior"] = "posterior",
        figsize: tuple[int, int] = (6, 4),
    ) -> dict:
        """
        Compute a Bayesian p-value for a test statistic T(y).

        The Bayesian p-value is defined as:
            p_B = P(T(y_rep) >= T(y_obs))
        where y_rep are draws from the predictive distribution.

        A value close to 0 or 1 signals that the observed statistic
        is in the tail of the predictive distribution — evidence of
        model misfit with respect to that statistic.

        Parameters
        ----------
        stat : str
            One of the built-in statistics or "custom".
        fn : callable, optional
            Required when stat="custom". Takes a 1-D array and returns
            a scalar.
        group : str
            Whether to use "prior" or "posterior" predictive samples.
        figsize : tuple
            Figure size for the diagnostic plot.

        Returns
        -------
        dict with keys:
            - "statistic": name of the test statistic
            - "observed_value": T(y_obs)
            - "predictive_values": array of T(y_rep) for each draw
            - "bayesian_p_value": scalar in [0, 1]
            - "fig": matplotlib Figure
        """
        if stat == "custom":
            if fn is None:
                raise ValueError("Must provide fn= when stat='custom'.")
            stat_fn = fn
            stat_name = "custom"
        elif stat in _BUILTIN_STATS:
            stat_fn = _BUILTIN_STATS[stat]
            stat_name = stat
        else:
            raise ValueError(
                f"Unknown stat '{stat}'. Choose from {list(_BUILTIN_STATS)} or 'custom'."
            )

        # Ensure samples exist
        if group == "prior":
            self._ensure_prior_predictive()
        else:
            self._ensure_posterior_predictive()

        samples = self._extract_samples(f"{group}_predictive")  # (S, N)

        # Compute T for each draw
        t_rep = np.array([stat_fn(samples[i]) for i in range(len(samples))])
        t_obs = stat_fn(self.observed)
        p_val = float(np.mean(t_rep >= t_obs))

        # Plot
        fig, ax = plt.subplots(figsize=figsize)
        ax.hist(t_rep, bins=40, color="steelblue" if group == "prior" else "darkorange",
                alpha=0.7, label=f"T(y_rep): {stat_name}")
        ax.axvline(t_obs, color="black", linewidth=2.5, linestyle="--",
                   label=f"T(y_obs) = {t_obs:.3f}")
        ax.set_xlabel(f"T = {stat_name}")
        ax.set_ylabel("Frequency")
        ax.set_title(
            f"{group.capitalize()} Predictive Check\n"
            f"Bayesian p-value = {p_val:.3f}",
            fontsize=12,
        )
        ax.legend()
        fig.tight_layout()

        return {
            "statistic": stat_name,
            "observed_value": t_obs,
            "predictive_values": t_rep,
            "bayesian_p_value": p_val,
            "fig": fig,
        }

    def summary(self, group: Literal["prior", "posterior"] = "posterior") -> dict:
        """
        Run all built-in statistics at once and return a summary dict
        of Bayesian p-values. Useful for a quick diagnostic sweep.

        Returns
        -------
        dict mapping statistic name -> Bayesian p-value
        """
        results = {}
        for stat_name in _BUILTIN_STATS:
            try:
                result = self.score(stat=stat_name, group=group)
                results[stat_name] = result["bayesian_p_value"]
            except Exception as e:
                warnings.warn(f"Statistic '{stat_name}' failed: {e}")
                results[stat_name] = None
        return results
