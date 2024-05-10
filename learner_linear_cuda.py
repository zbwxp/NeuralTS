import numpy as np
import torch
import torch.linalg as linalg

class LinearTS:
    def __init__(self, dim, lamdba=1, nu=1, style='ts'):
        self.dim = dim
        self.lamdba = lamdba
        self.nu = nu
        self.style = style

        # Initialize tensors on GPU if available
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device  # Store the device for later use

        self.U = lamdba * torch.eye(dim, device=device)
        self.Uinv = (1 / lamdba) * torch.eye(dim, device=device)
        self.jr = torch.zeros(dim, device=device)
        self.mu = torch.zeros(dim, device=device)

    def select(self, context):
        context = torch.tensor(context, dtype=torch.float32, device=self.device)

        if self.style == 'ts':
            lamdba_nu_Uinv = self.lamdba * self.nu * self.Uinv
            theta = torch.distributions.MultivariateNormal(self.mu, scale_tril=torch.linalg.cholesky(lamdba_nu_Uinv)).sample()
            r = torch.mv(context, theta)
            return torch.argmax(r).item(), torch.norm(theta).item(), torch.norm(theta - self.mu).item(), torch.mean(r).item()
        elif self.style == 'ucb':
            sig = torch.sqrt(torch.diagonal(torch.matmul(torch.matmul(context, self.Uinv), context.T)))
            r = torch.mv(context, self.mu) + torch.sqrt(torch.tensor(self.lamdba * self.nu, device=self.device)) * sig
            return torch.argmax(r).item(), torch.norm(self.mu).item(), torch.mean(sig).item(), torch.mean(r).item()

    def train(self, context, reward, lr=1e-2, epoch=100):
        context = torch.tensor(context, dtype=torch.float32, device=self.device)
        reward = torch.tensor(reward, dtype=torch.float32, device=self.device)

        self.jr += reward * context
        self.U += torch.ger(context, context)
        # Utilize efficient PyTorch operations for inverse and Cholesky decomposition
        try:
            self.Uinv = torch.linalg.inv(self.U)
        except RuntimeError as e:
            print(f"Error inverting U: {e}")
        self.mu = torch.matmul(self.Uinv, self.jr)
        return 0