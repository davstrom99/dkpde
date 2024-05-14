import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from omegaconf import OmegaConf
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE" 
from utils.config_helpers import get_model, get_dataset, get_kernel,get_kernel_parameters,get_dtype,get_run_name
import wandb

class BaseTrainer:
    def __init__(self, config):
        self.config = config
        self.dtype = get_dtype(config)
        torch.set_default_dtype(self.dtype)
        self.device = torch.device(config.training.device)
        self.frequency = config.data.frequency
        self.sound_speed = config.model.sound_speed
        self.class_type = self.__class__.__name__
        torch.manual_seed(config.training.seed)
        np.random.seed(config.training.seed)

        # Initialize Wandb project
        self.use_wandb = config.wandb.use_wandb
        if self.use_wandb:
            wandb.init(project=self.config.wandb.get("project"), entity=self.config.wandb.get("entity"),
                       config=OmegaConf.to_container(self.config, resolve=True),name = get_run_name(self.config,self.class_type), 
                       group = self.config.wandb.group_name)

        # Define loss function
        self.criterion = self.loss

        # Initialize the model
        self.NN = get_model(self.config)
        self.kernel = get_kernel(self.config)
        self.kernel_parameters = get_kernel_parameters(self.config,self.dtype,self.device)

        self.log_epsilon = torch.tensor([0.], dtype=self.dtype, device=self.device, requires_grad=True) # log of noise variance
        self.log_epsilon_z = torch.tensor([0.], dtype=self.dtype, device=self.device, requires_grad=True) # log of noise variance
        self.log_scaling = torch.tensor([0.], dtype=self.dtype, device=self.device, requires_grad=True) # log of scaling factor for kernel

        # Parameters to optimize
        if self.NN is not None:
            self.parameters = [self.kernel_parameters, self.log_epsilon,self.log_epsilon_z, self.log_scaling] + list(self.NN.parameters())
        elif self.NN is None:
            self.parameters = [self.kernel_parameters, self.log_epsilon, self.log_scaling]

        # Define optimizer
        self.optimizer = optim.Adam(self.parameters, lr=self.config.training.learning_rate)

        # Load data based on data type specified in config
        self.batch_size = self.config.training.batch_size*self.config.data.N_time
        self.train_data, self.val_data,self.data_object = get_dataset(self.config)
        self.train_loader = DataLoader(self.train_data, batch_size=self.batch_size,shuffle=True)
        self.val_loader = DataLoader(self.val_data, batch_size=self.batch_size, shuffle=False)


    def train(self):
        for epoch in range(self.config.training.epochs):
            # Generate new signals data but with same microphone positions
            self.train_data, self.val_data,self.data_object = get_dataset(self.config)
            self.train_loader = DataLoader(self.train_data, batch_size=self.batch_size,shuffle=True)
            self.val_loader = DataLoader(self.val_data, batch_size=self.batch_size, shuffle=False)

            # Train the model
            train_loss = self.gradient_step(self.NN, self.train_loader, self.criterion, self.optimizer)

            # Evaluate the model on validation set
            val_loss = self.evaluate(self.NN, self.val_loader, self.criterion)
            val_NMSE = self.NMSE_val(self.NN)
            train_NMSE = self.NMSE_train(self.NN)

            # Log metrics to Wandb
            if self.use_wandb:
                if self.class_type == "PDEDKLTrainer":
                    wandb.log({"NMSE_val": val_NMSE,"NMSE_train": train_NMSE,"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss[0], "kernel_parameters": self.kernel_parameters, 
                            "log_epsilon": self.log_epsilon,"log_scaling":self.log_scaling, "loss_ll_det": val_loss[1], "loss_ll_quad": val_loss[2], 
                            "loss_pde": val_loss[3], "learning_rate": self.optimizer.param_groups[0]['lr']})
                else:
                    wandb.log({"NMSE_val": val_NMSE,"NMSE_train": train_NMSE, "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss[0], "kernel_parameters": self.kernel_parameters, 
                           "log_epsilon": self.log_epsilon,"log_scaling":self.log_scaling, "loss_ll_det": val_loss[1], "loss_ll_quad": val_loss[2], 
                           "regularization_coefficient": self.regularization_coefficient,"learning_rate": self.optimizer.param_groups[0]['lr']})

            # Print progress
            print(f"Epoch {epoch}/{self.config.training.epochs}: Train loss: {train_loss}, Val loss: {val_loss}")

        run_name = ""
        if self.class_type == "PDEDKLTrainer" and self.config.model.pde_settings.N_collocation_time and self.config.model.pde_settings.N_collocation_space:
            run_name = "deep_kernel_pde"
        elif self.class_type == "PDEDKLTrainer" and (self.config.model.pde_settings.N_collocation_time==0 or self.config.model.pde_settings.N_collocation_space==0):
            run_name = "deep_kernel"
        elif self.class_type == "PDEDKLTrainer" and self.config.model.architecture == None and self.config.model.kernel == "RBF":
            run_name = "RBF" 

        # Save the final model
        path = os.path.join("models_trained",self.config.wandb.group_name,run_name+".pth")
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))

        if self.NN is not None:
            torch.save({'NN': self.NN.state_dict(),
                        'optimizer': self.optimizer.state_dict(),
                        'kernel_parameters': self.kernel_parameters,
                        'log_epsilon': self.log_epsilon,
                        'log_scaling': self.log_scaling,
                        'training_data': self.train_data[:],
                        'val_data':self.val_data[:],
                        'config':self.config}, path)
        else:
            torch.save({'optimizer': self.optimizer.state_dict(),
                        'kernel_parameters': self.kernel_parameters,
                        'log_epsilon': self.log_epsilon,
                        'log_scaling': self.log_scaling,
                        'training_data': self.train_data[:],
                        'val_data':self.val_data[:],
                        'config':self.config}, path)

        # Save a plt visualization of the result
        wandb.finish()


    def gradient_step(self, model, train_loader,criterion, optimizer):
        if model is not None:
            model.train()
        train_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad(set_to_none=True)

            loss = criterion(data, target)
            loss[0].backward()
            optimizer.step()

            train_loss += loss[0].item()

        return train_loss / len(train_loader)

    def evaluate(self, model, val_loader, criterion):
        if model is not None:
            model.eval()
        val_loss = 0

        # Compute the validation loss. If PDE regularization is used, the validation requires gradients
        if self.class_type == "PDEDKLTrainer":
            total_loss = 0
            loss_det = 0
            loss_data = 0
            loss_pde = 0
            for data, target in val_loader:
                data.requires_grad = True
                l = criterion(data, target)
                total_loss += l[0].item()
                loss_det += l[1].item()
                loss_data += l[2].item()
                loss_pde += l[3].item()
            val_loss = (total_loss/len(val_loader),loss_det/len(val_loader),loss_data/len(val_loader),loss_pde/len(val_loader))
        else:
            for data, target in val_loader:
                l = criterion(data, target)
                val_loss += l[0].item()
            val_loss /= len(val_loader)
        return val_loss

    def NMSE_val(self, model):
        # Compute the NMSE
        if model is not None:
            model.eval()
        with torch.no_grad():
            # Iterate over all self.val_data
            inputs = self.val_data[:][0]
            targets = self.val_data[:][1]

            u_hat = self.predict(inputs)
            NMSE = torch.norm(u_hat - targets)**2 / torch.norm(targets)**2

        return NMSE
    
    def NMSE_train(self, model):
        # Compute the NMSE
        if model is not None:
            model.eval()
        with torch.no_grad():
            # Iterate over all self.val_data
            inputs = self.train_data[:][0]
            targets = self.train_data[:][1]

            u_hat = self.predict(inputs)
            NMSE = torch.norm(u_hat - targets)**2 / torch.norm(targets)**2

        return NMSE

if __name__ == "__main__":
    # Load configuration (adapt based on your choice)
    config = OmegaConf.load("config.yaml")

    # Train the network
    trainer = BaseTrainer(config)
    trainer.train()
