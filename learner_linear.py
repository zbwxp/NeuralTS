import numpy as np
import scipy as sp
import torch
import torch.linalg as linalg

class LinearTS:
    # Brute-force Linear TS with full inverse
    def __init__(self, dim, lamdba=1, nu=1, style='ts'):
        self.dim = dim
        self.U = lamdba * np.eye(dim)
        self.Uinv = 1 / lamdba * np.eye(dim)
        self.nu = nu
        self.jr = np.zeros((dim, ))
        self.mu = np.zeros((dim, ))
        if torch.cuda.is_available():
            self.jr = torch.tensor(self.jr, dtype=torch.float32)
            self.U = torch.tensor(self.U, dtype=torch.float32)
            self.jr = self.jr.cuda()
            self.U = self.U.cuda()


        self.lamdba = lamdba
        self.style = style

    def select_numpy(self, context):
        if self.style == 'ts':
            theta = np.random.multivariate_normal(self.mu, self.lamdba * self.nu * self.Uinv)
            r = np.dot(context, theta)
            return np.argmax(r), np.linalg.norm(theta), np.linalg.norm(theta - self.mu), np.mean(r)
        elif self.style == 'ucb':
            sig = np.diag(np.matmul(np.matmul(context, self.Uinv), context.T))
            r = np.dot(context, self.mu) + np.sqrt(self.lamdba * self.nu) * sig
            return np.argmax(r), np.linalg.norm(self.mu), np.mean(sig), np.mean(r)
        
    def select(self, context):
        context = torch.tensor(context, dtype=torch.float32)  # Or torch.cuda.FloatTensor if on GPU
        mu = torch.tensor(self.mu, dtype=torch.float32)  # Convert to a torch tensor
        Uinv = torch.tensor(self.Uinv, dtype=torch.float32)  # Convert to a torch tensor
        
        if torch.cuda.is_available():
            context = context.cuda()
            mu = mu.cuda()
            Uinv = Uinv.cuda()

        if self.style == 'ts':
            # For multivariate normal, ensure lambda, nu are floats and convert Uinv accordingly
            lamdba_nu_Uinv = self.lamdba * self.nu * Uinv
            theta = torch.distributions.MultivariateNormal(mu, lamdba_nu_Uinv).sample()
            r = torch.mv(context, theta)
            return torch.argmax(r).item(), torch.norm(theta).item(), torch.norm(theta - mu).item(), torch.mean(r).item()
        elif self.style == 'ucb':
            sig = torch.sqrt(torch.diagonal(torch.mm(torch.mm(context, Uinv), context.t())))
            r = torch.mv(context, mu) + torch.sqrt(torch.tensor(self.lamdba * self.nu)) * sig

            # returns should be converted to numpy arrays
            return torch.argmax(r).item(), torch.norm(mu).item(), torch.mean(sig).item(), torch.mean(r).item()
        
    
    def train(self, context, reward, lr=1e-2, epoch=100):
        self.jr += reward * context
        self.U += np.matmul(context.reshape((-1, 1)), context.reshape((1, -1)))
        # fast inverse for symmetric matrix
        zz , _ = sp.linalg.lapack.dpotrf(self.U, False, False)
        Linv, _ = sp.linalg.lapack.dpotri(zz)
        self.Uinv = np.triu(Linv) + np.triu(Linv, k=1).T
        self.mu = np.dot(self.Uinv, self.jr)
        return 0