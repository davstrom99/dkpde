import torch 
import os
from PDEDKLTrainer import PDEDKLTrainer

def get_deep_kernel_pde_model(path, seed = 0, signal = None, frequency = None, cutoff_low = None, cutoff_high = None):
    """
    This function loads the deep kernel PDE model from the specified path.
    """
    # Load the run
    run = torch.load(path, map_location="cpu")
    config = run["config"]
    config.training.device = "cpu"
    config.model.architecture = "SIRENNet"

    # Add some more validation points
    N_val = 100
    config.training.num_mics_validation = N_val
    config.data.simulated_data.array_size = config.data.simulated_data.array_size+N_val



    config.wandb.use_wandb = False
    config.training.seed = seed
    if signal is not None:
        config.data.source_signal = signal
        if signal == "sine":
            config.data.frequency = frequency
        elif signal == "white":
            config.data.cutoff_low = cutoff_low
            config.data.cutoff_high = cutoff_high

    # Create an empty model
    trainer = PDEDKLTrainer(config)
    trainer.use_wandb = False
    trainer.NN.load_state_dict(run["NN"])
    trainer.NN.eval()

    trainer.log_epsilon = run["log_epsilon"]
    trainer.log_scaling = run["log_scaling"]
    trainer.kernel_parameters = run["kernel_parameters"]



    return trainer
