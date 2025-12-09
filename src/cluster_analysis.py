import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN, KMeans
import argparse
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

def cluster_embeddings(input_file="embeddings.parquet", output_file="clustered_embeddings.parquet", 
                       method="kmeans", n_clusters=3, eps=0.3, min_samples=10):
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
        # Run DBSCAN
        print(f"Running DBSCAN clustering with eps={eps}, min_samples={min_samples}, metric='cosine'...")
        # Note: DBSCAN with cosine metric expects distance, but if normalized, euclidean is related. 
        # Ideally we use metric='cosine', but scikit-learn DBSCAN with precomputed or cosine can be slow if not careful.
        # But for this size it's likely fine.
        clusterer = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine', n_jobs=-1)
        clusters = clusterer.fit_predict(embeddings_matrix)
        
    elif method == "kmeans":
        print(f"Running KMeans clustering with n_clusters={n_clusters}...")
        clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
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

    # Determine majority labels if ground truth 'label' exists
    if "label" in df.columns:
        print("\nMapping clusters to majority labels...")
        cluster_map = get_majority_label(df, "cluster", "label")
        
        # Map the cluster ID to the predicted label
        # For noise (-1), we might keep it as -1 or unassigned
        df["predicted_label"] = df["cluster"].map(cluster_map)
        
        # Display mapping
        print("Cluster -> Majority Label Mapping:")
        for c, l in sorted(cluster_map.items()):
            print(f"  Cluster {c}: {l}")
            
        # Optional: Calculate agreement/accuracy of this mapping
        # Only for non-noise points
        valid_points = df[df["cluster"] != -1]
        if not valid_points.empty:
            accuracy = (valid_points["label"] == valid_points["predicted_label"]).mean()
            print(f"\nAgreement with ground truth (majority vote accuracy): {accuracy:.4f}")
    
    # Save result
    df.to_parquet(output_file)
    print(f"\nSaved clustered data to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cluster embeddings using DBSCAN or KMeans")
    parser.add_argument("--input", type=str, default="embeddings.parquet", help="Path to input parquet file")
    parser.add_argument("--output", type=str, default="clustered_embeddings.parquet", help="Path to output parquet file")
    
    # Method arg
    parser.add_argument("--method", type=str, default="kmeans", choices=["kmeans", "dbscan"], help="Clustering method to use")
    
    # KMeans args
    parser.add_argument("--n_clusters", type=int, default=3, help="Number of clusters for KMeans")
    
    # DBSCAN args
    parser.add_argument("--eps", type=float, default=0.1, help="DBSCAN eps")
    parser.add_argument("--min_samples", type=int, default=5, help="DBSCAN min_samples")
    
    args = parser.parse_args()
    
    cluster_embeddings(
        input_file=args.input, 
        output_file=args.output, 
        method=args.method, 
        n_clusters=args.n_clusters,
        eps=args.eps, 
        min_samples=args.min_samples
    )
