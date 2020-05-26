import numpy as np
import scipy as sp
from sklearn.metrics.pairwise import rbf_kernel
import torch

class KernelTS:
    def __init__(self, dim, lamdba=1, nu=1, style='ts'):
        self.dim = dim
        self.lamdba = lamdba
        self.nu = nu
        self.history_context = []
        self.history_reward = []
        self.history_len = 0
        self.scale = self.lamdba * self.nu
        self.style = style
    
    def select(self, context):
        a, f = context.shape
        if self.history_len == 0:
            mu = np.zeros((a,))
            sigma = self.scale * np.ones((a,))
        else:
            X_history = np.array(self.history_context)
            R_history = np.array(self.history_reward)
            if self.history_len >= 1000:
                k_t = torch.from_numpy(rbf_kernel(context, X_history)).cuda()
                r_t = torch.from_numpy(R_history).cuda()
                # (K_t + \lambda I)^{-1}
                K_t = torch.from_numpy(rbf_kernel(X_history, X_history)).cuda()
                U_t = torch.inverse(K_t + self.lamdba * torch.eye(self.history_len, device=torch.device('cuda'))) 
                mu_t = k_t.matmul(U_t.matmul(r_t))
                sigma_t = torch.diag(torch.ones((a,), device=torch.device('cuda')) - k_t.matmul(U_t.matmul(k_t.T)))
                mu = mu_t.cpu().numpy()
                sigma = sigma_t.cpu().numpy() * self.scale
            else:
                K_t = rbf_kernel(X_history, X_history)
                k_t = rbf_kernel(context, X_history)
                zz , _ = sp.linalg.lapack.dpotrf((self.lamdba * np.eye(self.history_len) + K_t), False, False)
                Linv, _ = sp.linalg.lapack.dpotri(zz)
                U_t = np.triu(Linv) + np.triu(Linv, k=1).T
                mu = np.dot(k_t, np.dot(U_t, R_history))
                sigma = np.zeros((a,))
                for i in range(a):
                    sigma[i] = self.scale * (1 - np.dot(k_t[i], U_t @ k_t[i]))

        if self.style == 'ts':
            r = np.random.multivariate_normal(mu, np.diag(sigma))
        elif self.style == 'ucb':
            r = mu + np.sqrt(sigma)
        return np.argmax(r), 1, np.mean(sigma), np.mean(r)

    def train(self, context, reward):
        if self.history_len <= 4000:
            self.history_context.append(context)
            self.history_reward.append(reward)
            self.history_len += 1
        return 0
