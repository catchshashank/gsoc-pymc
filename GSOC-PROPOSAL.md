# GSoC 2026 Proposal
## Predictive Model Checking for PyMC: A Scoring-Rule-Based Framework

**Applicant:** Shashank Dubey  
**Institution:** HEC Paris (PhD Candidate, Quantitative Marketing / Decision Sciences)  
**Email:** [shashank.dubey@hec.edu](shashank.dubey@hec.edu)

---

## 1. About Me

I am a PhD candidate at HEC Paris interested in conversational dynamics, ambiguity, 
and decision making using deep learning models.
I am currently researching buyer-dealer sales negotiations, specifically 
belief-formation and updating and their effect on negotiation outcome. 
In negotiations, priors over hidden states are difficult to elicit since belief is ambiguous 
and hard to check. I have felt the absence of easy-to-use predictive checking tools
directly. 

For example. When I fit a Gaussian prior over a dealer's
"concession span" and want to ask whether that prior is compatible with
what dealers actually say, there is no clean way to do this in PyMC
without writing substantial boilerplate code by hand. This naturally draws me to this
project since it forms an integral part of my research.

Additionally, I also completed the PyMC Applied Bayesian Modeling course last month.
I got a hands-on-practice implementing hierarchical models, GLMs, prior and posterior
predictive checks, and causal inference models in PyMC. Therefore, I am familiar to
the PyMC coding environment.

---

## 2. The Problem

### 2.1 What Predictive Checks Are and Why They Matter

A prior predictive check asks: before seeing data, does my model
generate plausible observations? A posterior predictive check asks:
after fitting, does my model reproduce the key features of the observed
data?

Both checks are central to the Bayesian workflow (Gelman et al., 2020;
Gabry et al., 2019). They are the primary mechanism by which we 
discover model misspecification — where no single parameter value can fully
describe the data-generating process.

### 2.2 The Current State in PyMC

Running a full predictive check in PyMC today requires a practitioner to:

1. Call `pm.sample_prior_predictive()` and extend the InferenceData
2. Call `pm.sample_posterior_predictive()` and extend again
3. Manually extract the right variable arrays from the nested xarray
   structure
4. Write a plotting loop or navigate ArviZ's `plot_ppc` arguments
5. Implement test statistics and Bayesian p-value calculations by hand
6. Repeat for every model iteration

This is five steps of boilerplate before any scientific insight is
produced. The consequence is that users may skip the checks.

### 2.3 The Gap This Proposal Fills

The specific gap is: **there is no object in the PyMC ecosystem that
takes a fitted model and exposes a composable, scoring-rule-grounded
interface for checking it**. ArviZ provides excellent low-level
visualisation. PyMC provides sampling. Nothing connects them into a
coherent, easy-to-use checking workflow.

### 2.4 Connection to Scoring Rules (McLatchie et al., 2025)

The theoretical foundation for this project is the scoring-rule
framework formalised in McLatchie et al. (2025), which argues that
predictive model assessment should be evaluated through proper scoring
rules applied to the **mixture predictive distribution** — not to
individual parameter draws.

This has a direct implication. The standard Bayesian
p-value computes a test statistic on individual posterior predictive
draws and asks how often they exceed the observed statistic. A
scoring-rule-grounded check goes further: it evaluates how much
predictive information is lost by the model's structural limitations —
i.e., it can detect the kind of "non-trivial misspecification" that
standard checks miss because they only probe one statistic at a time (McLatchie et al., 2025).

This project implements both layers: the standard visual and p-value
interface that practitioners need today, and a scoring-rule layer that
implements the more powerful divergence-based checks motivated by the
paper.

---

## 3. Proposed Solution

### 3.1 Core API Design

The primary deliverable is a `PredictiveCheck` class that wraps any
fitted PyMC model:

```python
from bayes_corner import PredictiveCheck

checker = PredictiveCheck(model=model, idata=idata, observed=data)

# Visual checks
checker.prior_check()           # overlay prior predictive vs observed
checker.posterior_check()       # overlay posterior predictive vs observed

# Formal test statistics — Bayesian p-values
checker.score(stat="mean")      # built-in statistics
checker.score(stat="std")
checker.score(stat="kurtosis")  # catches multimodality
checker.score(stat="custom", fn=my_fn)  # user-defined

# Scoring-rule-based divergence (Phase 2)
checker.divergence(rule="log")  # log predictive score
checker.divergence(rule="crps") # continuous ranked probability score

# Full sweep
checker.summary()               # all statistics at once
```

### 3.2 Key Design Principles

**Lazy evaluation.** Predictive samples are drawn on demand and cached.
A user who only wants a posterior check does not pay the cost of
drawing prior samples.

**Non-destructive.** The checker never modifies the user's InferenceData
or model. It extends idata internally with a copy.

**Composable.** Every method returns either a matplotlib Figure or a
plain Python dict, so results can be combined, saved, or passed to
other tools without lock-in.

### 3.3 Prototype

A working prototype is already available at [gsoc-pymc]
(github.com/[catchshashank]/gsoc-pymc)

The prototype implements the full `PredictiveCheck` class with prior
and posterior visual checks, Bayesian p-values for seven built-in
statistics, and a `summary()` sweep. The example script demonstrates
the API on a deliberately misspecified model (single Gaussian fitted
to bimodal data), where the kurtosis and std p-values correctly
identify the misspecification while the mean p-value does not — exactly
the pattern that scoring-rule in McLatchie et al. (2025) predicts.

---

## 4. Implementation Plan

### Phase 1 — Weeks 1–4 (~80 hours): Core foundation

**Goal:** A stable, tested `PredictiveCheck` class that handles the
common case correctly.

- Finalise the `PredictiveCheck` class with full lazy evaluation and
  caching
- Support multivariate models
- Handle the three common InferenceData configurations: (a) posterior
  only, (b) posterior + prior_predictive, (c) full
- Implement all seven built-in test statistics with documented
  interpretations
- Write unit tests for each method using synthetic models with known
  misspecification signatures
- First milestone: `checker.prior_check()`, `checker.posterior_check()`,
  and `checker.score()` all passing tests on linear regression,
  logistic regression, and hierarchical models on PyMC.

### Phase 2 — Weeks 5–9 (~120 hours): Scoring-rule divergences

**Goal:** Implement the scoring-rule-grounded divergence layer motivated
by McLatchie et al. (2025).

- Implement `checker.divergence(rule="log")`: log predictive score
  $S(P_Q, y) = -\log P_Q(y)$, averaged over the posterior
- Implement `checker.divergence(rule="crps")`: continuous ranked
  probability score, which is proper and sensitive to both calibration
  and sharpness
- Implement `checker.divergence(rule="energy")`: energy score for
  multivariate observations
- Implement the Jensen gap diagnostic: computes
  $\int S(P_\theta, y) dQ(\theta) - S(P_Q, y)$, which quantifies how
  much predictive information is lost by posterior concentration. A
  large Jensen gap signals non-trivial misspecification in the sense of
  McLatchie et al. (2025).
- Second milestone: divergence methods tested on the three
  misspecification regimes from the paper (trivial, non-trivial,
  convex recovery) with documented expected behaviour

### Phase 3 — Weeks 10–13 (~100 hours): Integration and formula interface

**Goal:** Make the library usable in real workflows without friction.

- Add `PredictiveCheck.from_formula(formula, data, family)` constructor
  that builds a standard PyMC model from a Bambi-style formula and
  immediately wraps it — the most common use case in applied work
- Add `checker.compare(other_checker)` for side-by-side model
  comparison using scoring rules — directly enabling the model selection
  workflow described in Vehtari et al. (2017)
- Add hierarchical model support: `checker.group_check(group_var)` runs
  the posterior predictive check separately for each group level and
  flags groups where the model fits poorly
- Third milestone: end-to-end notebook demonstrating the full workflow
  on the Palmer penguins dataset from the course materials, showing
  prior check → fit → posterior check → divergence → model comparison

### Phase 4 — Weeks 14–16 (~50 hours): Documentation and polish

**Goal:** The library is usable by someone who has never seen the source
code.

- Full API documentation with docstrings, type hints, and usage examples
- Three example notebooks: (1) simple regression, (2) hierarchical
  model, (3) model comparison using scoring rules
- Contribution guide and test coverage report
- Pull request to PyMC-examples repository with a canonical predictive
  checking tutorial using the library

---

## 5. Why I Will Succeed

### What I bring

My research will require me to check latent-variable models with non-unique
priors over sequential negotiation states. I have a genuine, sustained
need for exactly the tooling I am proposing to build. I will use this library in my own PhD work as I build it,
which means every design decision will be tested against a real
scientific use case.

My course with PyMC gave me hands-on experience with the
full PyMC workflow: prior elicitation, MCMC sampling, posterior
diagnostics, and predictive checks. I am comfortable
with ArviZ's InferenceData structure, which is the central data object
this library wraps.

The prototype on GitHub demonstrates that I can translate the
theoretical ideas from McLatchie et al. (2025) into working,
documented Python code.

### What I do not yet know and how I will learn it

I have not contributed to a large open source codebase before. My
first action in week 1 will be to file a small documentation PR to
PyMC or ArviZ — not to demonstrate skill, but to learn the contribution
workflow (review cycles, CI, commit conventions) before the main
work begins.

I have not implemented CRPS or energy scores from scratch. I will
study the properscoring library and the scoring rules literature
(Gneiting & Raftery, 2007) in the first two weeks to ensure my
implementations are correct before building the user-facing interface
on top of them.

## Final Thoughts

I am a quick learner and I will swiftly cover up my shortcomings. 
I always believe in delivering on time and fully determined to do that in the assigned 350 hours.
However, if computation or API design takes longer than expected, Phase 4 documentation can be
trimmed without compromising the core deliverable. I will not sacrifice
test coverage to hit a feature count.

---

## 6. References

- Gelman, A., et al. (2020). Bayesian Workflow. arXiv:2011.01808.
- Gabry, J., Simpson, D., Vehtari, A., Betancourt, M., & Gelman, A.
  (2019). Visualization in Bayesian workflow. JRSS-A, 182(2), 389–402.
- Gneiting, T. & Raftery, A.E. (2007). Strictly proper scoring rules,
  prediction, and estimation. JASA, 102(477), 359–378.
- McLatchie, Y., Chérief-Abdellatif, B-E., Frazier, D.T., &
  Knoblauch, J. (2025). Predictively Oriented Posteriors.
  arXiv:2510.01915.
- Vehtari, A., Gelman, A., & Gabry, J. (2017). Practical Bayesian
  model evaluation using leave-one-out cross-validation and WAIC.
  Statistics and Computing, 27(5), 1413–1432.
