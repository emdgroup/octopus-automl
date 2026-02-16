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
   <img src="https://raw.githubusercontent.com/emdgroup/octopus-automl/main/docs/assets/logo.png" alt="Logo" width="200">
</a>
</div>

<div>
<a href="https://emdgroup.github.io/octopus-automl/">Documentation<a/>
&nbsp;•&nbsp;
<a href="https://emdgroup.github.io/octopus-automl/userguide/userguide/">User Guide<a/>
&nbsp;•&nbsp;
<a href="https://emdgroup.github.io/octopus-automl/reference/reference/">API Reference<a/>
&nbsp;•&nbsp;
<a href="https://emdgroup.github.io/octopus-automl/contributing/">Contribute<a/>
</div>

</div>


# Octopus

Octopus is a lightweight AutoML framework specifically designed for small datasets (<1k samples) and with high dimensionality (number of features). The goal of Octopus is to speed up machine learning projects and to increase the reliability of results in the context of small datasets.

What distinguishes Octopus from others

* Nested cross-validation (CV)
* Performance on small datasets
* No information leakage
* No data split mistakes
* Constrained regularization
* Ensembling, optimized for (nested) CV
* Simplicity
* Time to event
* Testing system (branching workflows)
* Reporting based on nested CV
* Test predictions over all samples


## Hardware

For maximum speed it is recommended to run Octopus on a compute node with $n\times m$ CPUS for a $n \times m$ nested cross validation. Octopus development is done, for example, on a c5.9xlarge EC2 instance.

## Installation

Package Installation works via `pip` or any other standard Python package manager:

```bash
# Install with recommended dependencies (includes optional packages such as AutoGluon)
pip install "octopus-automl[recommended]"

# Explicitly specify optional dependencies
pip install "octopus-automl[autogluon]"     # AutoGluon
pip install "octopus-automl[boruta]"        # Boruta feature selection
pip install "octopus-automl[survival]"      # Support time-to-event / survival analysis
pip install "octopus-automl[examples]"      # Dependencies for running examples

# Install with more than one extras, e.g.
pip install "octopus-automl[autogluon,examples]"
```

For contributors / octopus developers, a specific dependency group exists.
It contains code sanitization and quality tools.

```bash
pip install "octopus-automl[dev]"
```
