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
        reasoning = data['reasoning']

    true_index = np.array([i for i in range(len(scores)) if i not in failed_index])
    scores = scores[true_index]
    scores = np.mean(scores, axis=1)
    # scores[:, 1] -= 0.05
    embeddings = embeddings[true_index]
    qa_data = qa_data[true_index]
    reasoning = reasoning[true_index]
    mean_score = np.mean(scores, axis=1)

    max_score = np.argmax(scores, axis=1)
    idx_0 = [i for i, val in enumerate(max_score) if val ==0 ]
    idx_1 = [i for i, val in enumerate(max_score) if val ==1 ]
    idx_2 = [i for i, val in enumerate(max_score) if val ==2 ]
    idx_3 = [i for i, val in enumerate(max_score) if val ==3 ]
    idx_4 = [i for i, val in enumerate(max_score) if val ==4 ]
    idx_5 = [i for i, val in enumerate(max_score) if val ==5 ]
    idx_6 = [i for i, val in enumerate(max_score) if val ==6 ]

    # # pick the idx that has the largest gap to the second largest
    # top2_gap0 = [max(s) - np.sort(s)[-2] for s in scores[idx_0]]
    # sort_indices = np.argsort(top2_gap0)[::-1]
    # keep_indices_gap0 = np.array(idx_0)[sort_indices[:int(0.2*len(idx_0))]]

    # top2_gap1 = [max(s) - np.sort(s)[-2] for s in scores[idx_1]]
    # sort_indices = np.argsort(top2_gap1)[::-1]
    # keep_indices_gap1 = np.array(idx_1)[sort_indices[:int(0.1*len(idx_1))]]

    # top2_gap2 = [max(s) - np.sort(s)[-2] for s in scores[idx_2]]
    # sort_indices = np.argsort(top2_gap2)[::-1]
    # keep_indices_gap2 = np.array(idx_2)[sort_indices[:int(0.2*len(idx_2))]]

    # top2_gap3 = [max(s) - np.sort(s)[-2] for s in scores[idx_3]]
    # sort_indices = np.argsort(top2_gap3)[::-1]
    # keep_indices_gap3 = np.array(idx_3)[sort_indices[:int(0.2*len(idx_3))]]

    # top2_gap4 = [max(s) - np.sort(s)[-2] for s in scores[idx_4]]
    # sort_indices = np.argsort(top2_gap4)[::-1]
    # keep_indices_gap4 = np.array(idx_4)[sort_indices[:int(0.2*len(idx_4))]]

    # top2_gap5 = [max(s) - np.sort(s)[-2] for s in scores[idx_5]]
    # sort_indices = np.argsort(top2_gap5)[::-1]
    # keep_indices_gap5 = np.array(idx_5)[sort_indices[:int(0.2*len(idx_5))]]

    # top2_gap6 = [max(s) - np.sort(s)[-2] for s in scores[idx_6]]
    # sort_indices = np.argsort(top2_gap6)[::-1]
    # keep_indices_gap6 = np.array(idx_6)[sort_indices[:int(0.2*len(idx_6))]]


    top2_gap0 = [max(s) - np.sort(s)[-2] for s in scores[idx_0]]
    sort_indices = np.argsort(top2_gap0)[::-1]
    keep_indices_gap0 = np.array(idx_0)[sort_indices[:100]]

    top2_gap1 = [max(s) - np.sort(s)[-2] for s in scores[idx_1]]
    sort_indices = np.argsort(top2_gap1)[::-1]
    keep_indices_gap1 = np.array(idx_1)[sort_indices[:100]]

    top2_gap2 = [max(s) - np.sort(s)[-2] for s in scores[idx_2]]
    sort_indices = np.argsort(top2_gap2)[::-1]
    keep_indices_gap2 = np.array(idx_2)[sort_indices[:100]]

    top2_gap3 = [max(s) - np.sort(s)[-2] for s in scores[idx_3]]
    sort_indices = np.argsort(top2_gap3)[::-1]
    keep_indices_gap3 = np.array(idx_3)[sort_indices[:100]]

    top2_gap4 = [max(s) - np.sort(s)[-2] for s in scores[idx_4]]
    sort_indices = np.argsort(top2_gap4)[::-1]
    keep_indices_gap4 = np.array(idx_4)[sort_indices[:100]]

    top2_gap5 = [max(s) - np.sort(s)[-2] for s in scores[idx_5]]
    sort_indices = np.argsort(top2_gap5)[::-1]
    keep_indices_gap5 = np.array(idx_5)[sort_indices[:100]]

    top2_gap6 = [max(s) - np.sort(s)[-2] for s in scores[idx_6]]
    sort_indices = np.argsort(top2_gap6)[::-1]
    keep_indices_gap6 = np.array(idx_6)[sort_indices[:100]]

    keep_indices = np.concatenate([keep_indices_gap0, keep_indices_gap1, keep_indices_gap2, keep_indices_gap3, keep_indices_gap4, keep_indices_gap5, keep_indices_gap6])

    embeddings = embeddings[keep_indices]
    scores = scores[keep_indices]

    new_data = {
        'selected_vectors': embeddings,
        'selected_scores': scores,
    }
    save_path = "/home/bowen/dataset/wild_arena/gemini_judge_full_selected_100_7.npz"
    np.savez(save_path, **new_data)
    print()


    # Step 1: Cluster your data
    n_clusters = 200  # You may need to adjust this value
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(X)

    # Step 2 & 3: Calculate the average distance to the cluster centers and rank clusters
    avg_distances = []
    for i in range(n_clusters):
        cluster_points_indices = np.where(cluster_labels == i)[0]  # Indices of points in cluster i
        cluster_points = X[cluster_points_indices]
        if len(cluster_points) == 0:
            continue  # Skip empty clusters
        _, distances = pairwise_distances_argmin_min(cluster_points, kmeans.cluster_centers_[i].reshape(1, -1))
        avg_distance = np.mean(distances)
        avg_distances.append((i, avg_distance, cluster_points_indices.shape, cluster_points_indices))

    # Sort clusters by their average distances (ascending order)
    sorted_clusters_by_distance = sorted(avg_distances, key=lambda x: x[1])

    # Step 4: Select approximately 400 vectors & their indices from the closest clusters
    selected_vectors = []
    selected_indices = []
    for cluster_id, _, _, indices in sorted_clusters_by_distance:
        cluster_points = embeddings[indices]
        selected_vectors.extend(cluster_points)
        selected_indices.extend(indices)
        if len(selected_vectors) >= 1500:
            break


    # Convert to a numpy array for consistency
    selected_vectors = np.array(selected_vectors)
    selected_indices = np.array(selected_indices)

    selected_scores = scores[selected_indices]

    print(selected_vectors.shape)
    plot_tsne_and_save(selected_vectors, filename='tmp1.jpg')


    save_path = "/home/bowen/dataset/wild_arena/gemini_judge_full_selected1.npz"

    new_data = {
        'selected_vectors': selected_vectors,
        'selected_scores': selected_scores,
    }

    np.savez(save_path, **new_data)




if __name__ == '__main__':
    main()