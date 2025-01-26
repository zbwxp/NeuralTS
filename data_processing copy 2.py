import numpy as np
import scipy as sp
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

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

def plot_grouped_tsne_and_save(data1, data2, filename='tmp1.jpg'):
    """
    Perform t-SNE on the combination of two datasets and save the plot to the specified filename.
    
    Parameters:
    - data1: numpy array of shape (n_samples1, n_features)
    - data2: numpy array of shape (n_samples2, n_features)
    - filename: string, the name of the file to save the plot
    """
    # Combine data1 and data2 into a single dataset
    combined_data = np.concatenate((data1, data2))

    # Fit t-SNE on the combined dataset
    tsne = TSNE(n_components=2, random_state=42)
    reduced_data = tsne.fit_transform(combined_data)

    # Extract transformed data separately for data1 and data2
    reduced_data1 = reduced_data[:data1.shape[0]]
    reduced_data2 = reduced_data[data1.shape[0]:]

    # Plot the result of t-SNE
    plt.figure(figsize=(10, 6))
    plt.scatter(reduced_data1[:, 0], reduced_data1[:, 1], color='red', alpha=0.7, label='Data 1')
    plt.scatter(reduced_data2[:, 0], reduced_data2[:, 1], color='blue', alpha=0.7, label='Data 2')
    plt.title('t-SNE visualization')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.legend(loc='best')
    
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
        label = data['label']


    true_index = np.array([i for i in range(len(scores)) if i not in failed_index])
    scores = scores[true_index]
    scores = np.mean(scores, axis=1)
    embeddings = embeddings[true_index]
    qa_data = qa_data[true_index]
    label = label[true_index]
    y = scores
    # y[:, 1] -= 0.06
    X = embeddings
    # convert X to torch mac os tensor
    X = torch.from_numpy(X).float().cuda()

    # calculate the distance between every two points of X
    distances = torch.cdist(X, X)

    min_dist = []
    top2_dist = []
    for i in range(len(distances)):
        # the distance between the point and itself is 0, so we need to remove it
        min_dist.append(torch.min(distances[i][distances[i] != 0]))
        top2_dist.append(torch.topk(distances[i][distances[i] != 0], 2, largest=False).values[1])

    min_dist = torch.stack(min_dist)
    # find the first 1000 most clustered points
    val, idx = torch.sort(min_dist)

    top2_dist = torch.stack(top2_dist)
    # find the first 1000 most clustered points
    val, idx = torch.sort(top2_dist)
    keep_indices1 = idx[:1000]
    keep_indices2 = idx[-1000:]

    plot_grouped_tsne_and_save(X[idx[:1000]].cpu().numpy(), X[idx[1000:]].cpu().numpy(), filename='plot2.jpg')
    plot_grouped_tsne_and_save(X[idx[-1000:]].cpu().numpy(), X[idx[:-1000]].cpu().numpy(), filename='plot3.jpg')
    
    plot_tsne_and_save(X[keep_indices].cpu().numpy(), filename='plot1.jpg')


    X, y = shuffle(X, y, random_state=42)

    train_X = X[:int(len(X)*0.7)]
    train_y = y[:int(len(y)*0.7)]
    test_X = X[int(len(X)*0.7):]
    test_y = y[int(len(y)*0.7):]

    results = []
    distances = []
    for x, y in zip(test_X, test_y):
        # nearest train x
        # calculate the distance between x and all train_X
        dist = np.linalg.norm(train_X - x, axis=1)
        # select the nearest 2 points
        # pred_y = np.mean(train_y[np.argsort(dist)[:2]], axis=0)
        pred_y = train_y[np.argmin(dist)]
        results.append(np.argmax(pred_y) == np.argmax(y))
        distances.append(np.min(dist))

    print()
    # get the ranking of the distance and calculate the corresponding accuracy rate of the results
    index, value = zip(*sorted(enumerate(distances), key=lambda x: x[1]))
    results = np.array(results)[list(index)]   
    # conver true false to 1, 0
    results = results.astype(int) 

    acc = []
    # get the mean accuracy of every next 10 points
    for i in range(0, len(results)-10):
        acc.append(np.mean(results[i:i+10]))


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