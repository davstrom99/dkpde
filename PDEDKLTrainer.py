
import torch
from torch.autograd import grad
from BaseTrainer import BaseTrainer
import numpy as np
from omegaconf import OmegaConf
import os
import time
from utils.config_helpers import get_NN_min_max
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE" 

class PDEDKLTrainer(BaseTrainer):
    def loss(self, data,target):
        """
        Computes the log marginal likelihood with PDE regularization of the model given the data and the target.
        """
        # Generate some random collocation points
        X_collocation = torch.tensor(self.data_object.get_collocation_points(self.config.model.pde_settings.N_collocation_time,self.config.model.pde_settings.N_collocation_space),dtype=self.dtype, device=self.device, requires_grad=True)
        X_data = data.requires_grad_(True)

        N_collocation = X_collocation.shape[0]
        N_data = X_data.shape[0]
        N = N_data + N_collocation

        # Compute the Gram matrix
        K11,K12,K21,K22 = self.get_Gram(X_collocation, X_data)
        K11 = K11*torch.exp(self.log_scaling)
        K12 = K12*torch.exp(self.log_scaling)
        K21 = K21*torch.exp(self.log_scaling)
        K22 = K22*torch.exp(self.log_scaling)

        K11hat = K11 + torch.exp(self.log_epsilon) * torch.eye(N_data, dtype=self.dtype, device=self.device)
        K22hat = K22 + torch.tensor(self.config.model.kernel_jitter) * torch.eye(N_collocation, dtype=self.dtype, device=self.device)

        # Compute the determinant term
        low_block_input = K11hat - K12@torch.linalg.solve(K22hat,K21)
        loss_ll_det_11 = 0.5 * torch.logdet(K22hat) # marginal log likelihood
        loss_ll_det_22 = 0.5 * torch.logdet(low_block_input) # marginal log likelihood
        loss_ll_det = 0.5*N*np.log(2.0*np.pi) + loss_ll_det_11+loss_ll_det_22
    
        # Compute the quadratic term    
        block_11_inv = K11hat - K12 @ torch.linalg.solve(K22hat , K21)
        loss_ll_quad = 0.5 * target.T @ torch.linalg.solve(block_11_inv, target) 

        # Marginal log likelihood
        loss_ll = loss_ll_det +loss_ll_quad # marginal log likelihood

        return loss_ll, loss_ll_det, loss_ll_quad, loss_ll_det_11


    def predict(self,X, train_data = None, train_data_y = None):
        """
        Predicts the output of the model for a given input X using the mean of the predictive distribution.

        Args:
            X: Input data
            train_data: Grid of observation
            train_data_y: Values of the observations
        """
        
        if train_data is None:
            train_data = self.train_data[:][0]
        if train_data_y is None:
            train_data_y = self.train_data[:][1]

        N_train = self.train_data[:][0].shape[0]
        
        # Check if the deep part of the kernel is used
        if self.NN is not None:
            phi_train = self.NN(train_data)
            phi_pred = self.NN(X)
        else:
            phi_train = train_data
            phi_pred = X

        K = self.kernel.Gram(phi_train,self.kernel_parameters)*torch.exp(self.log_scaling)

        # Predictive mean        
        K_pred = self.kernel.pred_kernel(phi_pred,phi_train,self.kernel_parameters)*torch.exp(self.log_scaling)
        S = K + torch.exp(self.log_epsilon) * torch.eye(N_train, dtype=self.dtype, device=self.device)
        return K_pred @ torch.linalg.solve(S, train_data_y)


    def get_Gram(self,X_collocation, X_data):
        N_collocation = X_collocation.shape[0]
        N_data = X_data.shape[0]
        N = N_data + N_collocation

        # ----- Upper left block of the Gram matrix
        if self.NN is not None:
            phi_data = self.NN(X_data)
        else:
            phi_data = X_data
        K11 = self.kernel.Gram(phi_data,self.kernel_parameters) # Kernel matrix

        # ----- Off-diagonal blocks of the Gram matrix
        X12_collocation = X_collocation.unsqueeze(0).expand(N_data,-1,-1).reshape(N_collocation*N_data,4)
        X12_data = X_data.unsqueeze(1).expand(-1,N_collocation,-1).reshape(N_collocation*N_data,4)

        if self.NN is not None:
            phi12_collocation = self.NN(X12_collocation).view(N_data,N_collocation,self.NN.output_size)
            phi12_data = self.NN(X12_data).view(N_data,N_collocation,self.NN.output_size)
        else:
            phi12_collocation = X12_collocation.view(N_data,N_collocation,4)
            phi12_data = X12_data.view(N_data,N_collocation,4)

        K_data_collocation = self.kernel.pred_kernel(phi12_data,phi12_collocation,self.kernel_parameters) # Kernel matrix
        dK_data_collocation = torch.autograd.grad(K_data_collocation.sum(),X12_collocation,create_graph = True)[0]
        ddK_data_collocation_x = torch.autograd.grad(dK_data_collocation[:,0].sum(),X12_collocation,create_graph = True)[0]
        ddK_data_collocation_y = torch.autograd.grad(dK_data_collocation[:,1].sum(),X12_collocation,create_graph = True)[0]
        ddK_data_collocation_z = torch.autograd.grad(dK_data_collocation[:,2].sum(),X12_collocation,create_graph = True)[0]
        ddK_data_collocation_t = torch.autograd.grad(dK_data_collocation[:,3].sum(),X12_collocation,create_graph = True)[0]

        K_xx = ddK_data_collocation_x.view(N_data,N_collocation,4)[:,:,[0]]
        K_yy = ddK_data_collocation_y.view(N_data,N_collocation,4)[:,:,[1]]
        K_zz = ddK_data_collocation_z.view(N_data,N_collocation,4)[:,:,[2]]
        K_tt = ddK_data_collocation_t.view(N_data,N_collocation,4)[:,:,[3]]

        K12 = (K_xx+K_yy+K_zz-1/(self.sound_speed**2)*K_tt).squeeze(2)
        K21 = K12.T     

        # ----- Lower right block of the Gram matrix
        X22_cols = X_collocation.unsqueeze(0).expand(N_collocation,-1,-1).reshape(N_collocation*N_collocation,4)
        X22_rows = X_collocation.unsqueeze(1).expand(-1,N_collocation,-1).reshape(N_collocation*N_collocation,4)

        if self.NN is not None:
            phi22_cols = self.NN(X22_cols).view(N_collocation,N_collocation,self.NN.output_size)
            phi22_rows = self.NN(X22_rows).view(N_collocation,N_collocation,self.NN.output_size)
        else:
            phi22_cols = X22_cols.view(N_collocation,N_collocation,4)
            phi22_rows = X22_rows.view(N_collocation,N_collocation,4)

        K_22 = self.kernel.pred_kernel(phi22_rows,phi22_cols,self.kernel_parameters) # Kernel matrix
        dK_22 = torch.autograd.grad(K_22.sum(),X22_cols,create_graph = True)[0]
        ddK22_x = torch.autograd.grad(dK_22[:,0].sum(),X22_cols,create_graph = True)[0]
        ddK22_y = torch.autograd.grad(dK_22[:,1].sum(),X22_cols,create_graph = True)[0]
        ddK22_z = torch.autograd.grad(dK_22[:,2].sum(),X22_cols,create_graph = True)[0]
        ddK22_t = torch.autograd.grad(dK_22[:,3].sum(),X22_cols,create_graph = True)[0]

        K_xx = ddK22_x.view(N_collocation,N_collocation,4)[:,:,[0]]
        K_yy = ddK22_y.view(N_collocation,N_collocation,4)[:,:,[1]]
        K_zz = ddK22_z.view(N_collocation,N_collocation,4)[:,:,[2]]
        K_tt = ddK22_t.view(N_collocation,N_collocation,4)[:,:,[3]]

        z_22 = K_xx+K_yy+K_zz-1/(self.sound_speed**2)*K_tt

        # Then apply the PDE to the other input of the kernel
        dz_22 = torch.autograd.grad(z_22.sum(),X22_rows,create_graph = True)[0]
        ddz_22_x = torch.autograd.grad(dz_22[:,0].sum(),X22_rows,create_graph = True)[0]
        ddz_22_y = torch.autograd.grad(dz_22[:,1].sum(),X22_rows,create_graph = True)[0]
        ddz_22_z = torch.autograd.grad(dz_22[:,2].sum(),X22_rows,create_graph = True)[0]
        ddz_22_t = torch.autograd.grad(dz_22[:,3].sum(),X22_rows,create_graph = True)[0]

        z_xx = ddz_22_x.view(N_collocation,N_collocation,4)[:,:,[0]]
        z_yy = ddz_22_y.view(N_collocation,N_collocation,4)[:,:,[1]]
        z_zz = ddz_22_z.view(N_collocation,N_collocation,4)[:,:,[2]]
        z_tt = ddz_22_t.view(N_collocation,N_collocation,4)[:,:,[3]]

        K22 = (z_xx+z_yy+z_zz-1/(self.sound_speed**2)*z_tt).squeeze(2)

        return K11, K12,K21,K22

if __name__ == "__main__":
    # Load configuration
    config = OmegaConf.load("config.yaml")

    # Train the network
    trainer = PDEDKLTrainer(config)
    trainer.train()
