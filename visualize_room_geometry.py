import matplotlib.pyplot as plt
import os
from models_trained.load_trained_models import get_deep_kernel_pde_model, 

path = os.path.join("models_trained", "final")  
path_deep_kernel_pde = os.path.join(path, "deep_kernel_pde_5.pth")

# Load the model with some arbitrary signal
DKPDE_model = get_deep_kernel_pde_model(path_deep_kernel_pde, seed = 0,signal = "white", frequency = 100, cutoff_low=50, cutoff_high=1000)

# Plot the geometry of the room
fig = DKPDE_model.data_object.plot_geometry()
plt.show()