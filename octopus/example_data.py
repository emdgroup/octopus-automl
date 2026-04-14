"""Example data sets for use in Octopus examples."""

from typing import cast

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_diabetes, load_wine
from sklearn.utils import Bunch


def load_breast_cancer_data() -> tuple[pd.DataFrame, list[str], list[str]]:
    """Load the breast cancer dataset and return pandas dataframe, feature list, and target list."""
    breast_cancer = cast("Bunch", load_breast_cancer(as_frame=True))

    df = breast_cancer["frame"].reset_index()
    df.columns = df.columns.str.replace(" ", "_")
    features = [feature.replace(" ", "_") for feature in breast_cancer["feature_names"]]
    targets = [str(target) for target in breast_cancer["target_names"]]

    return df, features, targets


def load_diabetes_data() -> tuple[pd.DataFrame, list[str], list[str]]:
    """Load the diabetes dataset and return pandas dataframe, feature list, and target list."""
    diabetes = cast("Bunch", load_diabetes(as_frame=True))

    df = diabetes["frame"].reset_index()
    features = [str(feature) for feature in diabetes["feature_names"]]
    targets = ["target"]

    return df, features, targets


def load_wine_data() -> tuple[pd.DataFrame, list[str], list[str]]:
    """Load the wine dataset and return pandas dataframe, feature list, and target list."""
    wine = cast("Bunch", load_wine(as_frame=True))

    df = wine["frame"].reset_index()
    df.columns = df.columns.str.replace(" ", "_")
    features = [feature.replace(" ", "_") for feature in wine["feature_names"]]
    targets = [str(target) for target in wine["target_names"]]

    return df, features, targets


def load_survival_data(
    n_samples: int = 200,
    seed: int = 42,
) -> tuple[pd.DataFrame, list[str]]:
    """Generate a synthetic survival dataset for time-to-event examples.

    Create a dataset with 8 features where the first three are informative
    (influence the hazard) and the rest are noise. Event times are drawn from
    an exponential model with Cox proportional hazards, and independent
    censoring is applied so that roughly 30% of observations are censored.

    Args:
        n_samples: Number of samples to generate.
        seed: Random seed for reproducibility.

    Returns:
        A tuple of ``(df, features)`` where *df* is a DataFrame with an
        ``index`` column (sample ID), feature columns, ``duration`` (time to
        event or censoring), and ``event`` (1 = event observed, 0 = censored),
        and *features* is the list of feature column names.
    """
    rng = np.random.default_rng(seed)

    features = [
        "biomarker_a",
        "biomarker_b",
        "biomarker_c",
        "age_scaled",
        "lab_value_1",
        "lab_value_2",
        "lab_value_3",
        "lab_value_4",
    ]
    x = rng.standard_normal((n_samples, len(features)))

    risk_score = 0.8 * x[:, 0] + 0.5 * x[:, 1] - 0.3 * x[:, 2]
    baseline_hazard = 0.1
    duration = rng.exponential(scale=1.0 / (baseline_hazard * np.exp(risk_score)))

    censoring_time = rng.exponential(scale=np.median(duration) * 2.0, size=n_samples)
    observed_time = np.minimum(duration, censoring_time)
    event = (duration <= censoring_time).astype(int)

    df = pd.DataFrame(x, columns=features)
    df["duration"] = observed_time
    df["event"] = event
    df = df.reset_index()

    return df, features
