from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Protocol, Sequence, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Protocols & Types
class SklearnModel(Protocol):
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs: Any) -> Any: ...
    def predict(self, X: np.ndarray) -> np.ndarray: ...
    def predict_proba(self, X: np.ndarray) -> np.ndarray: ...


# Data Handling
def load_and_prep_data(
    input_file: str,
    embedding_col: str,
    label_col: str,
    max_rows: Optional[int] = None,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Loads parquet, drops nulls, and samples data. Returns DataFrame, X, and y.
    """
    path = Path(input_file)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    logger.info(f"Loading data from {input_file}...")
    df = pd.read_parquet(path)

    required = {embedding_col, label_col}
    if not required.issubset(df.columns):
        raise ValueError(f"Missing columns: {required - set(df.columns)}")

    # Drop nulls
    initial_len = len(df)
    df = df.dropna(subset=[embedding_col, label_col])
    if len(df) < initial_len:
        logger.warning(f"Dropped {initial_len - len(df)} rows with missing values.")

    # Sampling
    if max_rows and len(df) > max_rows:
        logger.info(f"Sampling {max_rows} rows from {len(df)}...")
        df = df.sample(n=max_rows, random_state=random_state)

    # Reset index to ensure alignment later
    df = df.reset_index(drop=True)

    # validate embedding shape
    try:
        X = np.stack(df[embedding_col].values)
    except ValueError as e:
        raise ValueError(
            "Failed to stack embeddings. Ensure all vectors have the same dimension."
        ) from e

    y = df[label_col].astype(int).to_numpy()

    return df, X, y


# Model Factories
def create_rf_model(
    n_estimators: int = 500,
    max_depth: Optional[int] = None,
    random_state: int = 42,
    **kwargs: Any,
) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )


def create_xgb_model(
    n_classes: int,
    n_estimators: int = 800,
    learning_rate: float = 0.05,
    max_depth: int = 6,
    random_state: int = 42,
    **kwargs: Any,
) -> Any:
    from xgboost import XGBClassifier

    objective = "multi:softprob" if n_classes > 2 else "binary:logistic"
    eval_metric = "mlogloss" if n_classes > 2 else "logloss"

    return XGBClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        objective=objective,
        eval_metric=eval_metric,
        random_state=random_state,
        n_jobs=-1,
        tree_method="hist",
        early_stopping_rounds=30,  # Built-in for XGB > 3.0
        **kwargs,
    )


MODEL_DISPATCH: Dict[str, Callable] = {
    "rf": create_rf_model,
    "xgb": create_xgb_model,
}


# Evaluation
def print_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str,
    *,
    labels: Optional[Sequence[Union[int, str]]] = None,
    cm: Optional[np.ndarray] = None,
) -> None:
    print(f"\n{'=' * 10} {title} {'=' * 10}")
    print(f"Accuracy:    {accuracy_score(y_true, y_pred):.4f}")
    print(
        f"F1 (Macro):  {f1_score(y_true, y_pred, average='macro', zero_division=0):.4f}"
    )
    print(
        f"F1 (Wtd):    {f1_score(y_true, y_pred, average='weighted', zero_division=0):.4f}"
    )
    print(
        "\nReport:\n",
        classification_report(
            y_true,
            y_pred,
            labels=None if labels is None else list(labels),
            digits=4,
            zero_division=0,
        ),
    )
    if cm is None:
        cm = confusion_matrix(y_true, y_pred, labels=None if labels is None else list(labels))
    print("Confusion Matrix:\n", cm)


def show_confusion_matrix(
    cm: np.ndarray,
    labels: Sequence[Union[int, str]],
    *,
    title: str = "Confusion Matrix",
    cmap: str = "Blues",
) -> None:
    """
    Display a confusion matrix using matplotlib.
    """
    import matplotlib.pyplot as plt

    labels_str = [str(x) for x in labels]
    n = len(labels_str)
    figsize = (max(6, int(1.2 * n)), max(4, int(1.0 * n)))

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm, interpolation="nearest", cmap=cmap)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set(
        title=title,
        xlabel="Predicted label",
        ylabel="True label",
        xticks=np.arange(n),
        yticks=np.arange(n),
        xticklabels=labels_str,
        yticklabels=labels_str,
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.0 if cm.size else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(int(cm[i, j])),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    fig.tight_layout()
    plt.show()


# Main Logic
def run_pipeline(args: argparse.Namespace) -> None:
    # 1. Load Data
    df, X, y = load_and_prep_data(
        args.input, args.embedding_col, args.label_col, args.max_rows, args.random_state
    )

    logger.info(f"Data ready. X: {X.shape}, y: {y.shape}. Labels: {np.unique(y)}")

    # 2. Split Data
    # We split indices to keep dataframe alignment simple
    indices = np.arange(len(df))
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X,
        y,
        indices,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )

    X_val, y_val = None, None
    if args.algo == "xgb" and args.val_size > 0:
        X_train, X_val, y_train, y_val = train_test_split(
            X_train,
            y_train,
            test_size=args.val_size,
            random_state=args.random_state,
            stratify=y_train,
        )
        logger.info(
            f"Split: Train {len(y_train)}, Val {len(y_val)}, Test {len(y_test)}"
        )
    else:
        logger.info(f"Split: Train {len(y_train)}, Test {len(y_test)}")

    # 3. Instantiate & Train Model
    factory = MODEL_DISPATCH.get(args.algo)
    if not factory:
        raise ValueError(
            f"Unknown algorithm '{args.algo}'. Available: {list(MODEL_DISPATCH.keys())}"
        )

    logger.info(f"Training {args.algo.upper()}...")

    # Prepare fit arguments
    fit_params = {}
    if args.algo == "xgb" and X_val is not None:
        fit_params = {"eval_set": [(X_val, y_val)], "verbose": False}

    model = factory(n_classes=len(np.unique(y)), random_state=args.random_state)
    model.fit(X_train, y_train, **fit_params)

    # 4. Evaluate
    y_pred = model.predict(X_test)
    labels = np.unique(np.concatenate([y_test, y_pred]))
    cm = confusion_matrix(y_test, y_pred, labels=list(labels))
    print_metrics(
        y_test,
        y_pred,
        title=f"{args.algo.upper()} Test Results",
        labels=labels,
        cm=cm,
    )

    if args.show_cm:
        show_confusion_matrix(
            cm,
            labels,
            title=f"{args.algo.upper()} Confusion Matrix",
        )

    # 5. Save Artifacts
    if args.save_model:
        try:
            joblib.dump(model, args.save_model)
            logger.info(f"Model saved to {args.save_model}")
        except Exception as e:
            logger.error(f"Failed to save model: {e}")

    if args.pred_output:
        # Reconstruct dataframe with predictions
        df["split"] = "train"
        df.loc[idx_test, "split"] = "test"
        df["prediction"] = np.nan
        df.loc[idx_test, "prediction"] = y_pred

        try:
            df.to_parquet(args.pred_output)
            logger.info(f"Predictions saved to {args.pred_output}")
        except Exception as e:
            logger.error(f"Failed to save predictions: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Embedding Classification Pipeline")
    parser.add_argument(
        "--input", default="embeddings.parquet", help="Input parquet path"
    )
    parser.add_argument(
        "--embedding_col", default="embedding", help="Embedding column name"
    )
    parser.add_argument(
        "--label_col", default="true_label", help="Target label column name"
    )
    parser.add_argument("--algo", default="rf", choices=["rf", "xgb"], help="Algorithm")
    parser.add_argument("--test_size", type=float, default=0.2, help="Test split ratio")
    parser.add_argument(
        "--val_size", type=float, default=0.1, help="Validation split ratio (XGB only)"
    )
    parser.add_argument(
        "--max_rows", type=int, default=None, help="Subsample data limit"
    )
    parser.add_argument("--random_state", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--save_model", type=str, default=None, help="Path to save model"
    )
    parser.add_argument(
        "--pred_output", type=str, default=None, help="Path to save predictions"
    )
    parser.add_argument(
        "--show_cm",
        action="store_true",
        help="Display confusion matrix with matplotlib",
    )

    try:
        run_pipeline(parser.parse_args())
    except Exception as e:
        logger.critical(f"Execution failed: {e}")
        sys.exit(1)
