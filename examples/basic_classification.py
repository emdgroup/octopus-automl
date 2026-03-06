"""Basic example for using Octopus Classification."""

# This example demonstrates how to use Octopus to create a machine learning classification model.
# We will use the breast cancer dataset for this purpose.
# Please ensure your dataset is clean, with no missing values (`NaN`),
# and that all features are numeric.

### Necessary imports for this example
import os

from sklearn.datasets import load_breast_cancer

from octopus.modules import Octo
from octopus.study import OctoClassification

### Load and Preprocess Data
breast_cancer = load_breast_cancer(as_frame=True)

df = breast_cancer["frame"].reset_index()
df.columns = df.columns.str.replace(" ", "_")
features = list(breast_cancer["feature_names"])
features = [feature.replace(" ", "_") for feature in features]

### Create and run OctoClassification
study = OctoClassification(
    name="basic_classification",
    path=os.environ.get("STUDIES_PATH", "./studies"),
    target_metric="AUCROC",
    feature_cols=features,
    target_col="target",
    sample_id_col="index",
    stratification_col="target",
    workflow=[
        Octo(
            description="step1_octo_full",
            task_id=0,
            depends_on=None,  # First task, depends on input
            models=["ExtraTreesClassifier"],
            n_trials=100,  # 100 trials for hyperparameter optimization
            n_folds_inner=5,  # 5 inner folds
            max_features=30,  # Use all 30 features
            ensemble_selection=True,  # Enable ensemble selection
        ),
    ],
)

study.fit(data=df)

print("Workflow completed")
