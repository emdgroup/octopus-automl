"""Example data sets for use in Octopus examples."""

from typing import cast

import pandas as pd
from sklearn.datasets import fetch_openml, load_breast_cancer, load_diabetes, load_wine
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


def load_tecator_data() -> tuple[pd.DataFrame, list[str], list[str]]:
    """Load the Tecator dataset (240 samples, 124 features, regression).

    Near-infrared absorbance spectra of meat samples used to predict fat content.
    A classic high-dimensional, small-sample regression benchmark (p >> n).
    Source: OpenML dataset ID 505.
    """
    data = fetch_openml(data_id=505, as_frame=True)

    df = data["data"].copy()
    df["target"] = data["target"].astype(float)
    df.columns = df.columns.str.replace(" ", "_")
    df = df.reset_index()
    features = [col.replace(" ", "_") for col in data["data"].columns]
    targets = ["target"]

    return df, features, targets


def load_sonar_data() -> tuple[pd.DataFrame, list[str], list[str]]:
    """Load the Sonar dataset (208 samples, 60 features, binary classification).

    Sonar signals bounced off a metal cylinder (Mine) vs. rocks (Rock).
    With only 208 samples and 60 features, results are highly sensitive to
    the random seed used for train/test splitting.
    Source: OpenML dataset ID 40.
    """
    data = fetch_openml(data_id=40, as_frame=True)

    df = data["data"].copy()
    # Encode target: Mine=1, Rock=0
    df["target"] = (data["target"] == "Mine").astype(int)
    df = df.reset_index()
    features = list(data["data"].columns)
    targets = ["Rock", "Mine"]

    return df, features, targets


def load_spectf_data() -> tuple[pd.DataFrame, list[str], list[str]]:
    """Load the SPECTF Heart dataset (349 samples, 44 features, binary classification).

    Cardiac Single Proton Emission Computed Tomography (SPECT) images.
    Each patient classified as normal (0) or abnormal (1).
    Imbalanced classes (254 abnormal vs. 95 normal).
    Source: OpenML dataset ID 337.
    """
    data = fetch_openml(data_id=337, as_frame=True)

    df = data["data"].copy()
    df["target"] = data["target"].astype(int)
    df = df.reset_index()
    features = list(data["data"].columns)
    targets = ["normal", "abnormal"]

    return df, features, targets


def load_promoters_data() -> tuple[pd.DataFrame, list[str], list[str]]:
    """Load the Molecular Biology Promoters dataset (106 samples, binary classification).

    DNA sequences classified as promoter (+) or non-promoter (-).
    The 57 categorical nucleotide positions (a/c/g/t) are one-hot encoded,
    resulting in 228 numeric features for only 106 samples — an extreme
    high-dimensional, small-sample scenario.
    Source: OpenML dataset ID 164.
    """
    data = fetch_openml(data_id=164, as_frame=True)

    df_raw = data["data"].copy()
    # One-hot encode the categorical nucleotide features (a/c/g/t)
    df = pd.get_dummies(df_raw, dtype=int)
    df["target"] = (data["target"] == "+").astype(int)
    df = df.reset_index()
    features = [col for col in df.columns if col not in ("index", "target")]
    targets = ["non-promoter", "promoter"]

    return df, features, targets
