"""Enhanced Tabular Neural Network Regressor with Categorical Embeddings."""

from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_is_fitted
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from octopus.models.config import OctoArrayLike, OctoMatrixLike


class TabularNNRegressor(RegressorMixin, BaseEstimator):
    """Enhanced neural network for tabular regression with categorical embeddings.

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
    """

    _estimator_type = "regressor"

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
            num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
            return cat_cols, num_cols
        else:
            # If numpy array, no categorical columns
            return [], list(range(X.shape[1]))

    def fit(self, X: OctoMatrixLike, y: OctoArrayLike) -> "TabularNNRegressor":
        """Fit the model.

        Args:
            X: Training features.
            y: Target values.

        Returns:
            Fitted estimator.
        """
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)

        # Convert to DataFrame if needed and store feature names
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=[f"col_{i}" for i in range(X.shape[1])])

        # Store original feature names for sklearn compatibility
        self.feature_names_in_ = np.array(X.columns)
        self.n_features_in_ = len(self.feature_names_in_)

        # Detect categorical and numerical columns
        self.cat_cols_, self.num_cols_ = self._detect_categorical_columns(X)

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
        self.num_medians_ = {}
        self.missing_indicators_ = []

        X_num_list = []
        for num_col in self.num_cols_:
            col_data = X[num_col]
            is_missing = col_data.isna()

            # Store median for this column
            median_val = col_data.median()
            self.num_medians_[num_col] = median_val if not pd.isna(median_val) else 0.0

            # Fill missing with median
            filled_data = col_data.fillna(self.num_medians_[num_col])
            X_num_list.append(filled_data.to_numpy())

            # Add missing indicator if there are any missing values
            if is_missing.any():
                self.missing_indicators_.append(num_col)
                X_num_list.append(is_missing.astype(np.float32).to_numpy())

        X_num = np.column_stack(X_num_list).astype(np.float32) if X_num_list else np.zeros((len(X), 0))
        X_cat = np.column_stack(X_cat_encoded) if X_cat_encoded else np.zeros((len(X), 0), dtype=np.int64)

        # Build model
        self.model_ = self._build_model()

        # Convert to tensors
        X_cat_tensor = torch.LongTensor(X_cat)
        X_num_tensor = torch.FloatTensor(X_num)
        y_tensor = torch.FloatTensor(y.values if isinstance(y, pd.Series) else y).unsqueeze(1)

        # Training
        dataset = TensorDataset(X_cat_tensor, X_num_tensor, y_tensor)

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

        criterion = nn.MSELoss()

        self.model_.train()
        for _epoch in range(self.epochs):
            epoch_loss = 0.0
            for X_cat_batch, X_num_batch, y_batch in dataloader:
                optimizer.zero_grad()
                outputs = self.model_(X_cat_batch, X_num_batch)
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

    def predict(self, X: Any) -> np.ndarray:
        """Predict using the model.

        Args:
            X: Input features.

        Returns:
            Predicted values.
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
        X_num_list = []
        for num_col in self.num_cols_:
            col_data = X[num_col]
            is_missing = col_data.isna()

            # Fill missing with stored median
            filled_data = col_data.fillna(self.num_medians_[num_col])
            X_num_list.append(filled_data.values)

            # Add missing indicator if this column had missing values during training
            if num_col in self.missing_indicators_:
                X_num_list.append(is_missing.astype(np.float32).values)

        X_num = np.column_stack(X_num_list).astype(np.float32) if X_num_list else np.zeros((len(X), 0))
        X_cat = np.column_stack(X_cat_encoded) if X_cat_encoded else np.zeros((len(X), 0), dtype=np.int64)

        # Convert to tensors and predict
        X_cat_tensor = torch.LongTensor(X_cat)
        X_num_tensor = torch.FloatTensor(X_num)

        self.model_.eval()
        with torch.no_grad():
            predictions = self.model_(X_cat_tensor, X_num_tensor)

        return predictions.numpy().flatten()  # type: ignore[no-any-return]

    def _build_model(self) -> "TabularNNModel":
        """Build the neural network.

        Returns:
            The constructed PyTorch model.
        """
        # Calculate actual number of numerical features (including missing indicators)
        n_num_features = len(self.num_cols_) + len(self.missing_indicators_)

        return TabularNNModel(
            cat_cols=self.cat_cols_,
            embedding_sizes=self.embedding_sizes_,
            n_num_features=n_num_features,
            hidden_sizes=self.hidden_sizes,
            dropout=self.dropout,
            activation=self.activation,
        )


class TabularNNModel(nn.Module):
    """PyTorch model for tabular data with batch normalization and configurable activation."""

    def __init__(self, cat_cols, embedding_sizes, n_num_features, hidden_sizes, dropout, activation="relu"):
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
        activation_fn: nn.ELU | nn.ReLU
        for hidden_size in hidden_sizes:
            # Create new activation instance for each layer
            if activation == "elu":
                activation_fn = nn.ELU()
            else:  # default to relu
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
        layers.append(nn.Linear(prev_size, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, X_cat: torch.Tensor, X_num: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            X_cat: Categorical features.
            X_num: Numerical features.

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
        x = torch.cat([cat_features, X_num], dim=1)

        return self.network(x)
