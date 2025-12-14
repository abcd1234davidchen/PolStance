import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering
import argparse
from typing import Optional
from collections import Counter

def get_majority_label(df, cluster_col, label_col):
    """
    For each cluster, find the most common label in the label_col.
    Returns a dictionary mapping cluster_id -> majority_label.
    """
    cluster_to_label = {}
    unique_clusters = df[cluster_col].unique()
    
    for cluster in unique_clusters:
        if cluster == -1: # Noise in DBSCAN
            continue
            
        subset = df[df[cluster_col] == cluster]
        if label_col in subset.columns:
            # Count labels in this cluster
            counts = subset[label_col].value_counts()
            if not counts.empty:
                majority_label = counts.idxmax()
                cluster_to_label[cluster] = majority_label
    
    return cluster_to_label

def plot_confusion_matrix(
    cm: pd.DataFrame,
    *,
    title: str = "Confusion Matrix",
    cmap: str = "Blues",
    output_file: Optional[str] = None,
    xlabel: str = "Predicted Label",
    ylabel: str = "True Label",
) -> None:
    """
    Display or save a confusion matrix using matplotlib (for Cluster Analysis).
    """
    import matplotlib.pyplot as plt
    
    # Convert dataframe to numpy for plotting, but keep labels for axes
    # The dataframe index is the Y axis (rows), columns are X axis (cols)
    row_labels = [str(x) for x in cm.index]
    col_labels = [str(x) for x in cm.columns]
    
    matrix = cm.values
    n_rows, n_cols = matrix.shape
    
    figsize = (max(8, int(1.0 * n_cols) + 2), max(6, int(0.8 * n_rows) + 2))
    
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(matrix, interpolation="nearest", cmap=cmap)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    ax.set(
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        xticks=np.arange(n_cols),
        yticks=np.arange(n_rows),
        xticklabels=col_labels,
        yticklabels=row_labels,
    )
    
    plt.setp(ax.get_xticklabels(), ha="right", rotation_mode="anchor")

    thresh = matrix.max() / 2.0 if matrix.size else 0.0
    for i in range(n_rows):
        for j in range(n_cols):
            ax.text(
                j,
                i,
                format(int(matrix[i, j])),
                ha="center",
                va="center",
                color="white" if matrix[i, j] > thresh else "black",
            )
            
    fig.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {output_file}")
    else:
        plt.show()
    
    plt.close(fig)

def cluster_embeddings(input_file="embeddings.parquet", output_file="clustered_embeddings.parquet", 
                       method="kmeans", n_clusters=3, eps=0.3, min_samples=10, cm_output=None):
    print(f"Loading embeddings from {input_file}...")
    try:
        df = pd.read_parquet(input_file)
    except FileNotFoundError:
        print(f"File {input_file} not found. Please run visualize.py first.")
        return

    # Convert the 'embedding' column (list of floats) into a 2D numpy array
    print("Preparing embeddings matrix...")
    embeddings_matrix = np.stack(df["embedding"].values)
    
    if method == "dbscan":
        print(f"Running DBSCAN clustering with eps={eps}, min_samples={min_samples}, metric='cosine'...")
        clusterer = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine', n_jobs=-1)
        clusters = clusterer.fit_predict(embeddings_matrix)
        
    elif method == "kmeans":
        print(f"Running KMeans clustering with n_clusters={n_clusters}...")
        clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
        clusters = clusterer.fit_predict(embeddings_matrix)
    elif method == "agg":
        clusterer = AgglomerativeClustering(n_clusters)
        clusters = clusterer.fit_predict(embeddings_matrix)
    else:
        print(f"Unknown method details: {method}")
        return

    # Assign raw cluster IDs to dataframe
    df["cluster"] = clusters
    
    # Statistics
    n_clusters_found = len(set(clusters)) - (1 if -1 in clusters else 0)
    n_noise_ = list(clusters).count(-1)
    
    print(f"\nClustering Result ({method}):")
    print(f"Estimated number of clusters: {n_clusters_found}")
    if method == "dbscan":
        print(f"Estimated number of noise points: {n_noise_}")
    
    print("\nCluster counts:")
    print(df["cluster"].value_counts().head(10)) 
    
    # Determine majority labels if ground truth 'label' or 'true_label' exists
    ground_truth_col = None
    if "true_label" in df.columns:
        ground_truth_col = "true_label"
    elif "label" in df.columns:
        ground_truth_col = "label"

    if ground_truth_col:
        print(f"\nMapping clusters to majority labels using '{ground_truth_col}'...")
        cluster_map = get_majority_label(df, "cluster", ground_truth_col)
        
        # Map the cluster ID to the predicted label
        df["cluster_predicted_label"] = df["cluster"].map(cluster_map)
        
        # Display mapping
        print("Cluster -> Majority Label Mapping:")
        for c, l in sorted(cluster_map.items()):
            print(f"  Cluster {c}: {l}")
            
        # Evaluation
        valid_points = df[df["cluster"] != -1].copy()
        
        if not valid_points.empty:
            y_true = valid_points[ground_truth_col]
            y_pred = valid_points["cluster_predicted_label"]
            
            # Accuracy
            accuracy = (y_true == y_pred).mean()
            print(f"\nMethod: {method}")
            print(f"Agreement with ground truth (Accuracy): {accuracy:.4f}")
            
            # Classification Report (Precision, Recall, F1)
            from sklearn.metrics import classification_report, f1_score, fbeta_score, roc_auc_score
            from sklearn.preprocessing import LabelBinarizer
            
            print("\nClassification Report:")
            print(classification_report(y_true, y_pred, digits=4, zero_division=0))
            
            # F1 and F2 scores (Weighted)
            f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
            f2 = fbeta_score(y_true, y_pred, beta=2, average='weighted', zero_division=0)
            print(f"F1 Score (weighted): {f1:.4f}")
            print(f"F2 Score (weighted): {f2:.4f}")
            
            # ROC AUC
            try:
                lb = LabelBinarizer()
                y_true_bin = lb.fit_transform(y_true)
                y_pred_bin = lb.transform(y_pred)
                
                if y_true_bin.shape[1] > 1:
                    # Multi-class
                    auc = roc_auc_score(y_true_bin, y_pred_bin, multi_class='ovr', average='weighted')
                    print(f"ROC AUC Score (weighted, OvR, Hard Labels): {auc:.4f}")
                else:
                    # Binary case
                    if y_true_bin.shape[1] == 1:
                        # Ensure we pass 1D array for binary
                        auc = roc_auc_score(y_true, y_pred) 
                        print(f"ROC AUC Score (Hard Labels): {auc:.4f}")
            except Exception as e:
                print(f"Could not calculate ROC AUC: {e}")

            # Confusion Matrix
            # Switch to True Label vs Predicted Label
            print("\nConfusion Matrix (True Label vs Predicted Label):")
            # Rows: True Label, Cols: Predicted Label (Standard Convention)
            cm_df = pd.crosstab(valid_points[ground_truth_col], valid_points["cluster_predicted_label"])
            print(cm_df)
            
            if cm_output:
                # Handle auto filename
                final_cm_output = cm_output
                if final_cm_output.lower() == "auto":
                    final_cm_output = f"{method}_cluster_confusion_matrix.png"
                
                plot_confusion_matrix(
                    cm_df,
                    title=f"Cluster Confusion Matrix ({method})",
                    output_file=final_cm_output,
                    xlabel="Predicted Label",
                    ylabel="True Label"
                )
    
    # Save result
    df.to_parquet(output_file)
    print(f"\nSaved clustered data to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cluster embeddings using DBSCAN or KMeans")
    parser.add_argument("--input", type=str, default="embeddings.parquet", help="Path to input parquet file")
    parser.add_argument("--output", type=str, default="clustered_embeddings.parquet", help="Path to output parquet file")
    
    # Method arg
    parser.add_argument("--method", type=str, default="kmeans", choices=["kmeans", "dbscan", "agg"], help="Clustering method to use")
    
    # KMeans args
    parser.add_argument("--n_clusters", type=int, default=3, help="Number of clusters for KMeans or Agglomerative Clustering")
    
    # DBSCAN args
    parser.add_argument("--eps", type=float, default=0.1, help="DBSCAN eps")
    parser.add_argument("--min_samples", type=int, default=5, help="DBSCAN min_samples")
    
    # Output CM arg
    parser.add_argument("--cm_output", type=str, default=None, help="Path to save confusion matrix image (e.g., cm_cluster.png)")
    
    args = parser.parse_args()
    
    cluster_embeddings(
        input_file=args.input, 
        output_file=args.output, 
        method=args.method, 
        n_clusters=args.n_clusters,
        eps=args.eps, 
        min_samples=args.min_samples,
        cm_output=args.cm_output
    )
