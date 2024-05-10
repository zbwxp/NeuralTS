from sklearn.datasets import fetch_openml
from sklearn.utils import shuffle
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import normalize
import numpy as np


class Bandit_multi:
    def __init__(self, name, is_shuffle=True, seed=None, custom=None):
        # Fetch data
        if name == 'mnist':
            X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
            # avoid nan, set nan as -1
            X[np.isnan(X)] = - 1
            X = normalize(X)
        elif name == 'mushroom':
            X, y = fetch_openml('mushroom', version=1, return_X_y=True)
            # avoid nan, set nan as -1
            X[np.isnan(X)] = - 1
            X = normalize(X)
        elif name == 'adult':
            X, y = fetch_openml('adult', version=2, return_X_y=True)
            # avoid nan, set nan as -1
            X[np.isnan(X)] = - 1
            X = normalize(X)
        elif name == 'covertype':
            X, y = fetch_openml('covertype', version=3, return_X_y=True)
            # avoid nan, set nan as -1
            X[np.isnan(X)] = - 1
            X = normalize(X)
        elif name == 'isolet':
            X, y = fetch_openml('isolet', version=1, return_X_y=True)
            # avoid nan, set nan as -1
            X[np.isnan(X)] = - 1
            X = normalize(X)
        elif name == 'letter':
            X, y = fetch_openml('letter', version=1, return_X_y=True)
            # avoid nan, set nan as -1
            X[np.isnan(X)] = - 1
            X = normalize(X)
        elif name == 'MagicTelescope':
            X, y = fetch_openml('MagicTelescope', version=1, return_X_y=True)
            # avoid nan, set nan as -1
            X[np.isnan(X)] = - 1
            X = normalize(X)
        elif name == 'shuttle':
            X, y = fetch_openml('shuttle', version=1, return_X_y=True)
            # avoid nan, set nan as -1
            X[np.isnan(X)] = - 1
            X = normalize(X)
        elif name == 'wild_arena': # rewrite this later
            with np.load(custom['X_path'], allow_pickle=True) as data:
                scores = data['scores']
                failed_index = data['all_fall_index']
            with np.load(custom['y_path'], allow_pickle=True) as data:
                embeddings = data['embeddings']

            true_index = np.array([i for i in range(len(scores)) if i not in failed_index])
            scores = scores[true_index]
            embeddings = embeddings[true_index]
            X = (embeddings - np.mean(embeddings, axis=1, keepdims=True)) / (np.std(embeddings, axis=1, keepdims=True) + 1e-8)
            y = np.mean(scores, axis=1)
            print(X.shape, y.shape)
            price = [5, 10, 20, 5, 3, 2, 1.8]  # gold policy
            y /= price
            # y[:, 1] -= 0.05
            y *= 10
        elif name == 'wild_arena_selected':
            with np.load(custom['X_path'], allow_pickle=True) as data:
                X = data['selected_vectors']
                y = data['selected_scores']

            X = (X - np.mean(X, axis=1, keepdims=True)) / (np.std(X, axis=1, keepdims=True) + 1e-8)
            print(X.shape, y.shape)
            # price = [5, 10, 20, 5, 3, 2, 1.8]  # gold policy
            # y /= price
            y[:, 1] -= 0.05
            y *= 10


        else:
            raise RuntimeError('Dataset does not exist')
        # Shuffle data
        if is_shuffle:
            self.X, self.y = shuffle(X, y, random_state=seed)
        else:
            self.X, self.y = X, y
        # generate one_hot coding:
        if custom:
            self.y_arm = np.argmax(self.y, axis=1).reshape((-1, 1))
            self.y_rwd = self.y
        else:
            self.y_arm = OrdinalEncoder(
                dtype=np.int32).fit_transform(self.y.to_numpy().reshape((-1, 1)))
        # cursor and other variables
        self.cursor = 0
        self.size = self.y.shape[0]
        self.n_arm = np.max(self.y_arm) + 1
        self.dim = self.X.shape[1] * self.n_arm
        self.act_dim = self.X.shape[1]

    def step(self, float_reward=False):
        assert self.cursor < self.size
        X = np.zeros((self.n_arm, self.dim))
        for a in range(self.n_arm):
            X[a, a * self.act_dim:a * self.act_dim +
                self.act_dim] = self.X[self.cursor]
        arm = self.y_arm[self.cursor][0]
        if float_reward:
            rwd = self.y_rwd[self.cursor]
        else:
            rwd = np.zeros((self.n_arm,))
            rwd[arm] = 1
        self.cursor += 1
        return X, rwd

    def finish(self):
        return self.cursor == self.size

    def reset(self):
        self.cursor = 0


if __name__ == '__main__':
    b = Bandit_multi('mushroom')
    x_y, a = b.step()
    # print(x_y[0])
    # print(x_y[1])
    # print(np.linalg.norm(x_y[0]))
