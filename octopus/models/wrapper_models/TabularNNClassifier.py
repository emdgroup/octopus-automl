"""Enhanced Tabular Neural Network Classifier with Categorical Embeddings."""

from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_is_fitted
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from octopus.logger import get_logger
from octopus.models.config import OctoArrayLike, OctoMatrixLike

logger = get_logger()


class TabularNNClassifier(ClassifierMixin, BaseEstimator):
    """Enhanced neural network for binary and multiclass classification with categorical embeddings.

    This classifier automatically detects whether the problem is binary or multiclass
    and adjusts its architecture and loss function accordingly.

    Args:
        hidden_sizes: Sizes of hidden layers. Defaults to [200, 100].
        dropout: Dropout probability. Defaults to 0.1.
        learning_rate: Learning rate for optimizer. Defaults to 0.001.
        batch_size: Training batch size. Defaults to 256.
        epochs: Number of training epochs. Defaults to 100.
        weight_decay: L2 regularization strength. Defaults to 1e-5.
        activation: Activation function ('relu' or 'elu'). Defaults to 'relu'.
        optimizer: Optimizer type ('adam' or 'adamw'). Defaults to 'adam'.
        random_state: Random seed. Defaults to None.
        n_threads: Number of threads for PyTorch. Defaults to 1 (set to >1 with caution
            due to potential deadlocks). If set to 0, number of PyTorch threads will not be limited.
    """

    _estimator_type = "classifier"

    def __init__(
        self,
        hidden_sizes: list[int] | None = None,
        dropout: float = 0.1,
        learning_rate: float = 0.001,
        batch_size: int = 256,
        epochs: int = 100,
        weight_decay: float = 1e-5,
        activation: str = "relu",
        optimizer: str = "adam",
        random_state: int | None = None,
        n_threads: int = 1,
    ) -> None:
        self.hidden_sizes = hidden_sizes if hidden_sizes is not None else [200, 100]
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.weight_decay = weight_decay
        self.activation = activation
        self.optimizer = optimizer
        self.random_state = random_state
        self.n_threads = n_threads

        if self.n_threads < 0:
            raise ValueError(f"n_threads must be non-negative, got {self.n_threads}.")
        elif self.n_threads > 0:
            if self.n_threads > 1:
                logger.warning(
                    f"Using {self.n_threads} threads for PyTorch. This may lead to deadlocks in some environments, "
                    "see https://github.com/pytorch/pytorch/issues/91547#issuecomment-1370011188."
                )

            torch.set_num_threads(self.n_threads)

    def _detect_categorical_columns(self, X: Any) -> tuple[list[str], list[str] | list[int]]:
        """Detect categorical columns from DataFrame.

        Args:
            X: Input features.

        Returns:
            A tuple containing (categorical_columns, numerical_columns).
        """
        if isinstance(X, pd.DataFrame):
            # Use pandas dtypes to detect categorical columns
            cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
            numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
            return cat_cols, numerical_cols
        else:
            # If numpy array, no categorical columns
            return [], list(range(X.shape[1]))

    def fit(self, X: OctoMatrixLike, y: OctoArrayLike) -> "TabularNNClassifier":
        """Fit the model.

        Args:
            X: Training features.
            y: Target values.

        Returns:
            Fitted estimator.

        Raises:
            ValueError: If fewer than 2 classes are provided.
        """
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)

        # Store classes and determine if binary or multiclass
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        self.is_binary_ = self.n_classes_ == 2

        if self.n_classes_ < 2:
            raise ValueError(f"Classifier expects at least 2 classes, got {self.n_classes_}")

        # Encode target labels to 0, 1, 2, ... for both binary and multiclass
        self.target_encoder_ = LabelEncoder()
        y_encoded = self.target_encoder_.fit_transform(y)

        # Convert to DataFrame if needed and store feature names
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=[f"col_{i}" for i in range(X.shape[1])])

        # Store original feature names for sklearn compatibility
        self.feature_names_in_ = np.array(X.columns)
        self.n_features_in_ = len(self.feature_names_in_)

        # Detect categorical and numerical columns
        self.cat_cols_, self.numerical_cols_ = self._detect_categorical_columns(X)

        # Encode categorical features
        self.label_encoders_: dict[str, LabelEncoder] = {}
        self.embedding_sizes_: dict[str, tuple[int, int]] = {}

        X_cat_encoded = []
        for cat_col in self.cat_cols_:
            le = LabelEncoder()
            # Handle NaN by adding a special category
            X_col = X[cat_col].fillna("__NAN__")
            encoded = le.fit_transform(X_col)
            self.label_encoders_[cat_col] = le

            # Improved embedding size: min(50, max(3, (cardinality + 1) // 2))
            cardinality = len(le.classes_)
            emb_dim = min(50, max(3, (cardinality + 1) // 2))
            self.embedding_sizes_[cat_col] = (cardinality, emb_dim)

            X_cat_encoded.append(encoded)

        # Enhanced missing value handling for numerical features
        self.numerical_medians_ = {}
        self.missing_indicators_ = []

        X_numerical_list = []
        for numerical_col in self.numerical_cols_:
            col_data = X[numerical_col]
            is_missing = col_data.isna()

            # Store median for this column
            median_val = col_data.median()
            self.numerical_medians_[numerical_col] = median_val if not pd.isna(median_val) else 0.0

            # Fill missing with median
            filled_data = col_data.fillna(self.numerical_medians_[numerical_col])
            X_numerical_list.append(filled_data.to_numpy())

            # Add missing indicator if there are any missing values
            if is_missing.any():
                self.missing_indicators_.append(numerical_col)
                X_numerical_list.append(is_missing.astype(np.float32).to_numpy())

        X_numerical = (
            np.column_stack(X_numerical_list).astype(np.float32) if X_numerical_list else np.zeros((len(X), 0))
        )
        X_cat = np.column_stack(X_cat_encoded) if X_cat_encoded else np.zeros((len(X), 0), dtype=np.int64)

        # Build model
        self.model_ = self._build_model()

        # Convert to tensors
        X_cat_tensor = torch.LongTensor(X_cat)
        X_numerical_tensor = torch.FloatTensor(X_numerical)

        criterion: nn.BCEWithLogitsLoss | nn.CrossEntropyLoss
        if self.is_binary_:
            y_array = y.values if isinstance(y, pd.Series) else y
            y_binary = (y_array == self.classes_[1]).astype(np.float32)
            y_tensor = torch.FloatTensor(y_binary).unsqueeze(1)
            criterion = nn.BCEWithLogitsLoss()
        else:
            y_tensor = torch.LongTensor(y_encoded)
            criterion = nn.CrossEntropyLoss()

        # Training
        dataset = TensorDataset(X_cat_tensor, X_numerical_tensor, y_tensor)

        # Use generator for reproducible shuffling if random_state is set
        if self.random_state is not None:
            generator = torch.Generator()
            generator.manual_seed(self.random_state)
            dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, generator=generator)
        else:
            dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Select optimizer
        optimizer: torch.optim.AdamW | torch.optim.Adam
        if self.optimizer == "adamw":
            optimizer = torch.optim.AdamW(
                self.model_.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
            )
        else:  # adam
            optimizer = torch.optim.Adam(
                self.model_.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
            )

        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)

        self.model_.train()
        for _epoch in range(self.epochs):
            epoch_loss = 0.0
            for X_cat_batch, X_numerical_batch, y_batch in dataloader:
                optimizer.zero_grad()
                outputs = self.model_(X_cat_batch, X_numerical_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model_.parameters(), max_norm=1.0)

                optimizer.step()
                epoch_loss += loss.item()

            # Update learning rate based on epoch loss
            avg_loss = epoch_loss / len(dataloader)
            scheduler.step(avg_loss)

        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        """Predict class probabilities.

        Args:
            X: Input features.

        Returns:
            Predicted class probabilities.
        """
        check_is_fitted(self, "model_")

        # Convert to DataFrame if needed, using stored feature names
        if not isinstance(X, pd.DataFrame):
            # Use feature_names_in_ if available, otherwise generate column names
            if hasattr(self, "feature_names_in_"):
                X = pd.DataFrame(X, columns=self.feature_names_in_)
            else:
                X = pd.DataFrame(X, columns=[f"col_{i}" for i in range(X.shape[1])])

        # Encode categorical features
        X_cat_encoded = []
        for cat_col in self.cat_cols_:
            le = self.label_encoders_[cat_col]
            X_col = X[cat_col].fillna("__NAN__")
            # Handle unseen categories
            encoded = np.array([le.transform([val])[0] if val in le.classes_ else 0 for val in X_col])  # type: ignore[index]
            X_cat_encoded.append(encoded)

        # Prepare numerical features with same missing value handling as fit
        X_numerical_list = []
        for numerical_col in self.numerical_cols_:
            col_data = X[numerical_col]
            is_missing = col_data.isna()

            # Fill missing with stored median
            filled_data = col_data.fillna(self.numerical_medians_[numerical_col])
            X_numerical_list.append(filled_data.values)

            # Add missing indicator if this column had missing values during training
            if numerical_col in self.missing_indicators_:
                X_numerical_list.append(is_missing.astype(np.float32).values)

        X_numerical = (
            np.column_stack(X_numerical_list).astype(np.float32) if X_numerical_list else np.zeros((len(X), 0))
        )
        X_cat = np.column_stack(X_cat_encoded) if X_cat_encoded else np.zeros((len(X), 0), dtype=np.int64)

        # Convert to tensors and predict
        X_cat_tensor = torch.LongTensor(X_cat)
        X_numerical_tensor = torch.FloatTensor(X_numerical)

        self.model_.eval()
        with torch.no_grad():
            logits = self.model_(X_cat_tensor, X_numerical_tensor)

            if self.is_binary_:
                probs_class1 = torch.sigmoid(logits).numpy().flatten()
                probs_class0 = 1 - probs_class1
                return np.column_stack([probs_class0, probs_class1])
            else:
                probs = torch.softmax(logits, dim=1).numpy()
                return probs  # type: ignore[no-any-return]

    def predict(self, X: Any) -> np.ndarray:
        """Predict class labels.

        Args:
            X: Input features.

        Returns:
            Predicted class labels.
        """
        proba = self.predict_proba(X)
        y_pred_encoded = np.argmax(proba, axis=1)

        if self.is_binary_:
            return self.classes_[y_pred_encoded]  # type: ignore
        else:
            return self.target_encoder_.inverse_transform(y_pred_encoded)

    def _build_model(self) -> "TabularNNClassificationModel":
        """Build the neural network.

        Returns:
            The constructed PyTorch model.
        """
        # Calculate actual number of numerical features (including missing indicators)
        n_num_features = len(self.numerical_cols_) + len(self.missing_indicators_)

        # Determine output size: 1 for binary, n_classes for multiclass
        output_size = 1 if self.is_binary_ else self.n_classes_

        return TabularNNClassificationModel(
            cat_cols=self.cat_cols_,
            embedding_sizes=self.embedding_sizes_,
            n_num_features=n_num_features,
            output_size=output_size,
            hidden_sizes=self.hidden_sizes,
            dropout=self.dropout,
            activation=self.activation,
        )


class TabularNNClassificationModel(nn.Module):
    """PyTorch model for binary and multiclass classification with batch normalization and configurable activation."""

    def __init__(
        self, cat_cols, embedding_sizes, n_num_features, output_size, hidden_sizes, dropout, activation="relu"
    ):
        super().__init__()

        # Embeddings for categorical features
        self.embeddings = nn.ModuleList(
            [nn.Embedding(num_classes, emb_dim) for num_classes, emb_dim in embedding_sizes.values()]
        )

        # Calculate input dimension
        total_emb_dim = sum(emb_dim for _, emb_dim in embedding_sizes.values())
        input_dim = total_emb_dim + n_num_features

        # Build hidden layers with batch normalization
        layers = []
        prev_size = input_dim
        for hidden_size in hidden_sizes:
            # Create new activation instance for each layer
            activation_fn: nn.Module
            if activation == "elu":
                activation_fn = nn.ELU()
            else:
                activation_fn = nn.ReLU()

            layers.extend(
                [
                    nn.Linear(prev_size, hidden_size),
                    nn.BatchNorm1d(hidden_size),
                    activation_fn,
                    nn.Dropout(dropout),
                ]
            )
            prev_size = hidden_size

        # Output layer
        layers.append(nn.Linear(prev_size, output_size))

        self.network = nn.Sequential(*layers)

    def forward(self, X_cat: torch.Tensor, X_numerical: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            X_cat: Categorical features.
            X_numerical: Numerical features.

        Returns:
            Model output.
        """
        # Embed categorical features
        if X_cat.shape[1] > 0:
            embedded = [emb(X_cat[:, i]) for i, emb in enumerate(self.embeddings)]
            cat_features = torch.cat(embedded, dim=1)
        else:
            cat_features = torch.empty(X_cat.shape[0], 0)

        # Concatenate with numerical features
        x = torch.cat([cat_features, X_numerical], dim=1)

        return self.network(x)
