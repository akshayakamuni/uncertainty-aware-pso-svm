
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_preprocessor(numeric_features, categorical_features=None):
    """
    Build a preprocessing pipeline.

    Numeric features:
        1. Fill missing values using the median.
        2. Standardize values.

    Categorical features:
        1. Fill missing values using the most frequent value.
        2. One-hot encode categories.
    """

    if categorical_features is None:
        categorical_features = []

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    transformers = [
        ("numeric", numeric_pipeline, numeric_features)
    ]

    if categorical_features:
        from sklearn.preprocessing import OneHotEncoder

        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "encoder",
                    OneHotEncoder(handle_unknown="ignore"),
                ),
            ]
        )

        transformers.append(
            (
                "categorical",
                categorical_pipeline,
                categorical_features,
            )
        )

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )
