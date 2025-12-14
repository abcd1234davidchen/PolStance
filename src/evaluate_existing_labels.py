import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Sequence, Union
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    fbeta_score,
    roc_auc_score,
)
from sklearn.preprocessing import LabelBinarizer

def plot_confusion_matrix(
    cm: np.ndarray,
    labels: Sequence[Union[int, str]],
    *,
    title: str = "Confusion Matrix",
    cmap: str = "Blues",
    output_file: Optional[str] = None,
) -> None:
    """
    Display or save a confusion matrix using matplotlib.
    """
    labels_str = [str(x) for x in labels]
    n = len(labels_str)
    # Dynamic figsize
    figsize = (max(8, int(1.0 * n) + 2), max(6, int(0.8 * n) + 2))

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
    plt.setp(ax.get_xticklabels(), ha="right", rotation_mode="anchor")

    # Threshold for text color
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
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {output_file}")
    else:
        plt.show()
    
    plt.close(fig)

def evaluate_labels(input_file: str, true_col: str, pred_col: str, cm_output: Optional[str] = None, method_name: str = "Model"):
    print(f"Loading data from {input_file}...")
    try:
        df = pd.read_parquet(input_file)
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    if true_col not in df.columns or pred_col not in df.columns:
        print(f"Error: Columns '{true_col}' and/or '{pred_col}' not found in file.")
        print(f"Available columns: {df.columns.tolist()}")
        return

    # Drop nulls
    df = df.dropna(subset=[true_col, pred_col])
    
    y_true = df[true_col].astype(int).to_numpy()
    y_pred = df[pred_col].astype(int).to_numpy()
    
    labels = np.unique(np.concatenate([y_true, y_pred]))
    
    print(f"\n{'=' * 10} Evaluation Results: {method_name} {'=' * 10}")
    print(f"Total Samples: {len(y_true)}")
    
    # 1. Metrics
    print(f"Accuracy:    {accuracy_score(y_true, y_pred):.4f}")
    print(f"F1 (Macro):  {f1_score(y_true, y_pred, average='macro', zero_division=0):.4f}")
    print(f"F1 (Wtd):    {f1_score(y_true, y_pred, average='weighted', zero_division=0):.4f}")
    print(f"F2 (Wtd):    {fbeta_score(y_true, y_pred, beta=2, average='weighted', zero_division=0):.4f}")
    
    # 2. ROC AUC (Hard Labels)
    try:
        lb = LabelBinarizer()
        y_true_bin = lb.fit_transform(y_true)
        # Use hard predictions as scores (0 vs 1)
        y_score = lb.transform(y_pred)
        
        if y_true_bin.shape[1] > 1:
            # Multi-class
            auc = roc_auc_score(y_true_bin, y_score, multi_class='ovr', average='weighted')
            print(f"ROC AUC (Wtd, OvR, Hard Labels): {auc:.4f}")
        else:
            # Binary
            if y_true_bin.shape[1] == 1:
                # Ensure 1D for binary
                auc = roc_auc_score(y_true, y_pred)
                print(f"ROC AUC (Hard Labels): {auc:.4f}")
    except Exception as e:
        print(f"Could not calculate ROC AUC: {e}")

    # 3. Report
    print(
        "\nReport:\n",
        classification_report(
            y_true,
            y_pred,
            labels=labels,
            digits=4,
            zero_division=0,
        ),
    )
    
    # 4. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    print("Confusion Matrix:\n", cm)
    
    # Handle auto filename
    final_output = cm_output
    if final_output and final_output.lower() == "auto":
        final_output = f"{method_name}_confusion_matrix.png"
    
    if final_output:
        plot_confusion_matrix(
            cm,
            labels,
            title=f"{method_name} Confusion Matrix",
            output_file=final_output
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate existing predicted labels against true labels")
    parser.add_argument("--input", type=str, default="embeddings.parquet", help="Input parquet file")
    parser.add_argument("--true_col", type=str, default="true_label", help="Column name for GROUND TRUTH")
    parser.add_argument("--pred_col", type=str, default="predicted_label", help="Column name for PREDICTIONS")
    parser.add_argument("--cm_output", type=str, default=None, help="Path to save confusion matrix image. Use 'auto' to generate filename from method name.")
    parser.add_argument("--method", type=str, default="Model", help="Name of the method/model for titles and filenames")
    
    args = parser.parse_args()
    
    evaluate_labels(args.input, args.true_col, args.pred_col, args.cm_output, args.method)
