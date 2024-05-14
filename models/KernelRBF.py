import torch
from .AbstractKernel import AbstractKernel

class KernelRBF(AbstractKernel):
    def __init__(self, jitter=1e-5):
        self.jitter = jitter

    def pred_kernel(self, X1, X2, kernel_parameters):
        ls = torch.exp(kernel_parameters) # length scale (which is defined as the log of the length scale to avoid constraints)            
        D = self.pairwise_euclidean_distance_squared(X1, X2)
        K = torch.exp(-1.0 * D / (2.0*ls**2))
        return K

    def pairwise_euclidean_distance_squared(self, X1, X2):
        if X1.dim() == 2:
            s1 = (X1 ** 2).sum(1).view((-1, 1))
            s2 = (X2 ** 2).sum(1).view((-1, 1))
            D = s1 - 2 * X1 @ X2.T + s2.T
        elif X1.dim() == 3:
            s1 = (X1 ** 2).sum(2)
            s2 = (X2 ** 2).sum(2)
            D = s1 - 2 * (X1 * X2).sum(2) + s2
        else:
            raise ValueError("Input dimension not supported")

        return D

    
