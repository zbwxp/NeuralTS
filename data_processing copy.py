import numpy as np
import scipy as sp
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def plot_tsne_and_save(data, filename='tmp1.jpg'):
    """
    Perform t-SNE on the provided data and save the plot to the specified filename.

    Parameters:
    - data: numpy array of shape (n_samples, n_features)
    - filename: string, the name of the file to save the plot
    """
    # Perform t-SNE dimensionality reduction
    tsne = TSNE(n_components=2, random_state=42)  # Setting random_state for reproducibility
    reduced_data = tsne.fit_transform(data)

    # Plotting the result of t-SNE
    plt.figure(figsize=(10, 6))
    plt.scatter(reduced_data[:, 0], reduced_data[:, 1], alpha=0.5)
    plt.title('t-SNE visualization')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    
    # Save the plot to a file
    plt.savefig(filename)
    plt.close()  # Close the plot window

def main():
    X_path = "/home/bowen/dataset/wild_arena/gemini_judge_full.npz"

    with np.load(X_path, allow_pickle=True) as data:
        qa_data = data['qa_data']
        scores = data['scores']
        failed_index = data['all_fall_index']
        embeddings = data['embeddings']

    true_index = np.array([i for i in range(len(scores)) if i not in failed_index])
    scores = scores[true_index]
    scores = np.mean(scores, axis=1)
    embeddings = embeddings[true_index]
    qa_data = qa_data[true_index]
    X = scores * 10

    
    
    
    
    # Step 1: Cluster your data
    
    
    n_clusters = 200  # You may need to adjust this value
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(X)

    # Step 2 & 3: Calculate the average distance to the cluster centers and rank clusters
    keep_indices = []
    for i in range(n_clusters):
        cluster_points_indices = np.where(cluster_labels == i)[0]  # Indices of points in cluster i
        cluster_points = X[cluster_points_indices]
        if len(cluster_points) == 0:
            continue  # Skip empty clusters
        _, distances = pairwise_distances_argmin_min(cluster_points, kmeans.cluster_centers_[i].reshape(1, -1))
        # only keep the first 25% closet points and keep its index
        index, value = zip(*sorted(enumerate(distances), key=lambda x: x[1]))
        cluster_points_indices = cluster_points_indices[list(index[:int(len(index)*0.25)])]
        keep_indices.extend(cluster_points_indices)









if __name__ == '__main__':
    main()