"""Example data sets for use in Octopus examples."""

from typing import cast

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
