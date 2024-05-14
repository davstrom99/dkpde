import torch
import numpy as np

def get_model(config):
    """
    This function dynamically imports and returns the model based on the architecture specified in the config.
    
    Args:
        config: Configuration object.
        
    Returns:
        The actual model object.
    """

    if config.model.architecture == "SIRENNet":
        from models import SIRENNet
        output_size = config.model.SIRENNet.hidden_size

        # Compile the model
        t_min,t_max,space_min,space_max = get_NN_min_max(config)

        model = SIRENNet.SIRENNet(input_size = 4, hidden_size = config.model.SIRENNet.hidden_size, output_size = output_size,
                                       num_hidden_layers=config.model.SIRENNet.num_hidden_layers,
                                       t_min = t_min,t_max = t_max,space_min = space_min, space_max = space_max).to(torch.device(config.training.device))
    elif config.model.architecture == 'None':
        model = None
    else:
        ValueError(f"Model architecture {config.model.architecture} not supported")
    return model


def get_dataset(config):
    """
    This function returns the dataset based on the dataset type specified in the config.

    Args:
        config: Configuration object.

    Returns:
        The training and validation dataset objects.
    """
    data_type = config.data.dataset_name 
    if data_type == "simulated_data_3d":
        from data import SimulatedDataset3D
        data_object = SimulatedDataset3D.SimulatedDataset3D(
            room_dim = config.data.simulated_data.room_dim, 
            fs = config.data.simulated_data.fs, 
            source_pos = config.data.simulated_data.source_pos, 
            array_size = config.data.simulated_data.array_size,
            source_signal=config.data.source_signal,
            N_time = config.data.N_time,
            mic_pos_center = config.data.simulated_data.mic_array_3d.mic_pos_cube_center,
            mic_pos_side = config.data.simulated_data.mic_array_3d.mic_pos_cube_side,
            rt60 = config.data.simulated_data.rt60,
            frequency = config.data.frequency,
            SNR = config.data.SNR,
            dtype=get_dtype(config),
            device = torch.device(config.training.device),
            cutoff_low = config.data.cutoff_low,
            cutoff_high = config.data.cutoff_high,
            seed = config.training.seed)
        
        train_size = config.training.num_mics_training*config.data.N_time
        val_size = config.training.num_mics_validation*config.data.N_time

        # Since the sensor positions are already shuffled, it's fine to just take the first train_size and val_size
        # Note that a random split would use some time samples from every sensor, which is not what we want.
        train_data = torch.utils.data.Subset(data_object, range(train_size))
        val_data = torch.utils.data.Subset(data_object, range(train_size,train_size+val_size))
    else:
        ValueError(f"Data type {data_type} not supported")

    return train_data, val_data, data_object

def get_kernel(config):
    """
    This function returns the kernel based on the kernel type specified in the config.
    """

    kernel_type = config.model.kernel  # Default fallback
    if kernel_type == "RBF":
        from models import KernelRBF
        kernel = KernelRBF.KernelRBF(jitter=config.model.kernel_jitter)
    elif kernel_type == "None":
        kernel = None
    else:
        ValueError(f"Kernel type {kernel_type} not supported")
    return kernel

def get_kernel_parameters(config, dtype, device):
    """
    This function returns the kernel parameters based on the kernel type specified in the config.
    """

    kernel_type = config.model.kernel  # Default fallback
    if kernel_type == "RBF":
        kernel_parameters = torch.tensor([0.], dtype=dtype, device=device, requires_grad=True) # log of length scale of the kernel
    elif kernel_type == "None":
        kernel_parameters = None
    else:
        ValueError(f"Kernel type {kernel_type} not supported")

    return kernel_parameters

def get_dtype(config):
    """
    This function returns the data type for the training and the model.
    """

    dtype = config.training.dtype  # Default fallback
    if dtype == "float32":
        return torch.float32
    elif dtype == "float64":
        return torch.float64
    elif dtype == "float16":
        return torch.float16
    else:
        ValueError(f"Data type {dtype} not supported")

def get_run_name(config,class_type):
    """
    This function returns a name for the run based on the configuration. (for wandb)
    """
    num_microphones = config.training.num_mics_training
    run_name = f"{class_type}_"

    # Information about the PDE regularization
    if (config.model.pde_settings.N_collocation_time>0 and config.model.pde_settings.N_collocation_space>0) and class_type == "PDEDKLTrainer":
        run_name = run_name + f"PDEreg_time{config.model.pde_settings.N_collocation_time}_space{config.model.pde_settings.N_collocation_space}_"
    # Information about the neural network 
    if config.model.architecture == "SIRENNet":
        run_name = run_name + f"NN_{config.model.SIRENNet.hidden_size}_{config.model.SIRENNet.num_hidden_layers}_"
    run_name = run_name + f"{config.model.kernel}_batch_size_{config.training.batch_size}_{config.data.dataset_name}_{config.data.source_signal}_{num_microphones}mics"
    return run_name

def get_NN_min_max(config):
    """
    This function returns the min and max values for the normalization of the input to the neural network.
    """
    if config.data.dataset_name == "simulated_data_3d":
        t_min = torch.tensor(0).to(torch.device(config.training.device))
        t_max = torch.tensor(config.data.N_time/config.data.simulated_data.fs).to(torch.device(config.training.device))
        space_min = (torch.tensor(config.data.simulated_data.mic_array_3d.mic_pos_cube_center)-torch.tensor(config.data.simulated_data.mic_array_3d.mic_pos_cube_side)/2).to(torch.device(config.training.device))
        space_max = (torch.tensor(config.data.simulated_data.mic_array_3d.mic_pos_cube_center)+torch.tensor(config.data.simulated_data.mic_array_3d.mic_pos_cube_side)/2).to(torch.device(config.training.device))
    else:
        ValueError(f"Data type {config.data.dataset_name} not supported")

    return t_min,t_max,space_min,space_max
