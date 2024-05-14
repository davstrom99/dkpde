import torch

class AbstractKernel:
    """Abstract class for defining kernels in PyTorch"""

    def __init__(self, jitter=1e-5):
        self.jitter = jitter

    def pred_kernel(self, X1, X2,kernel_parameters):
        raise NotImplementedError("Abstract method - implement in subclass")

    def Gram(self, X,kernel_parameters):
        if self.jitter > 0:
            K = self.pred_kernel(X, X,kernel_parameters) + self.jitter * torch.eye(X.shape[0], dtype=X.dtype, device=X.device)
        else:
            K = self.pred_kernel(X, X,kernel_parameters)
        return K 


