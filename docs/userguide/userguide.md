# User Guide

!!! note "Backwards Compatibility and Deprecations"

    Octopus is in a constant state of development. As part of this, interfaces and objects
    might change in ways breaking existing code. We aspire to provide backwards **support
    for deprecated code of the last three minor versions**. After this time, old code will
    generally be removed. Both the moment of deprecation and full removal (deprecation
    expiration) will be noted in the [changelog](../changelog.md).

This section contains hands-on, step-by-step guides that show you how to
configure, run, and interpret each type of Octopus study.
The central interface for all study types is the
[`OctoStudy`](../reference/study.md) object.

## What You'll Learn

- **[Data Health Check](health_check.md)** — Automatic dataset validation that runs before every training. Detects missing values, class imbalance, potential leakage, and more.
- **Task-specific guides** — How to set up, configure, and run each study type:
    - **[Classification](classification.md)** — Binary and multiclass classification with `OctoClassification`
    - **[Regression](regression.md)** — Continuous target prediction with `OctoRegression`
    - **[Time to Event](time_to_event.md)** — Survival analysis with censored observations using `OctoTimeToEvent`
- **[Understanding the Output](output_structure.md)** — What Octopus writes to disk after a study completes: directory structure, result files, and how to load them programmatically.
