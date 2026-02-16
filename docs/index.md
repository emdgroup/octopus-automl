<div align="center">
  <br/>

<div>
<a href="https://github.com/emdgroup/octopus-automl/actions/workflows/test-package.yml?query=branch%3Amain">
   <img src="https://img.shields.io/github/actions/workflow/status/emdgroup/octopus-automl/test-package.yml?branch=main&style=flat-square&label=Test%20Suite&labelColor=0f69af&color=ffdcb9" alt="Test Suite">
</a>
<a href="https://github.com/emdgroup/octopus-automl/actions/workflows/ruff.yml?query=branch%3Amain">
   <img src="https://img.shields.io/github/actions/workflow/status/emdgroup/octopus-automl/ruff.yml?branch=main&style=flat-square&label=Code%20Quality&labelColor=0f69af&color=ffdcb9" alt="Code Quality">
</a>
<a href="https://github.com/emdgroup/octopus-automl/actions/workflows/docs.yml?query=branch%3Amain">
   <img src="https://img.shields.io/github/actions/workflow/status/emdgroup/octopus-automl/docs.yml?branch=main&style=flat-square&label=Docs&labelColor=0f69af&color=ffdcb9" alt="Docs">
</a>
</div>

<div>
<a href="https://pypi.org/project/octopus-automl/">
   <img src="https://img.shields.io/pypi/pyversions/octopus-automl?style=flat-square&label=Supports%20Python&labelColor=96d7d2&color=ffdcb9" alt="Supports Python">
</a>
<a href="https://pypi.org/project/octopus-automl/">
   <img src="https://img.shields.io/pypi/v/octopus-automl.svg?style=flat-square&label=PyPI%20Version&labelColor=96d7d2&color=ffdcb9" alt="PyPI version">
</a>
<a href="https://pypistats.org/packages/octopus-automl">
   <img src="https://img.shields.io/pypi/dm/octopus-automl?style=flat-square&label=Downloads&labelColor=96d7d2&color=ffdcb9" alt="Downloads">
</a>
<a href="https://github.com/emdgroup/octopus-automl/issues/">
   <img src="https://img.shields.io/github/issues/emdgroup/octopus-automl?style=flat-square&label=Issues&labelColor=96d7d2&color=ffdcb9" alt="Issues">
</a>
<a href="https://github.com/emdgroup/octopus-automl/pulls/">
   <img src="https://img.shields.io/github/issues-pr/emdgroup/octopus-automl?style=flat-square&label=PRs&labelColor=96d7d2&color=ffdcb9" alt="PRs">
</a>
<a href="http://www.apache.org/licenses/LICENSE-2.0">
   <img src="https://shields.io/badge/License-Apache%202.0-green.svg?style=flat-square&labelColor=96d7d2&color=ffdcb9" alt="License">
</a>
</div>

<div>
<a href="https://github.com/emdgroup/octopus-automl/">
   <img src="assets/logo.png" alt="Logo" width="200">
</a>
</div>

</div>

# Octopus

Octopus is a lightweight AutoML framework specifically designed for small datasets (<1k samples) and with high dimensionality (number of features). The goal of Octopus is to speed up machine learning projects and to increase the reliability of results in the context of small datasets.

---

## Why Octopus?

| | |
|---|---|
| **Nested cross-validation** | Separates hyperparameter tuning from performance estimation, giving you honest metrics even on 100-sample datasets. [Learn more](concepts/nested_cv.md) |
| **No information leakage** | Feature selection, imputation, and scaling happen *inside* each CV fold. Correlated observations are automatically grouped. |
| **Multi-step workflows** | Chain feature-selection modules (ROC, MRMR, Boruta) with ML modules (Octo, AutoGluon) into pipelines that progressively refine the feature set. [Learn more](concepts/workflow/index.md) |
| **Ensembling for small data** | Combines models across inner CV splits and Optuna trials into robust ensembles, optimized for the nested CV setting. |
| **Classification, regression & survival** | Supports binary/multiclass classification, regression, and time-to-event analysis out of the box. |

---

## Where to go from here?

- **[Getting Started](getting_started.md)** — Install Octopus and run your first study in five minutes.
- **[User Guide](userguide/userguide.md)** — Hands-on, step-by-step guides that show you *how* to configure and run each task type (classification, regression, survival analysis) with all available options.
- **[Concepts](concepts/concepts.md)** — Understand *why* Octopus works the way it does: nested CV, workflows, information leakage prevention, and feature importance.
- **[Examples](examples/index.md)** — Runnable end-to-end workflows from basic to advanced.
- **[API Reference](reference/reference.md)** — Auto-generated reference for all public classes and functions.
