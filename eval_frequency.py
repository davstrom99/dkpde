import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from models_trained.load_trained_models import get_deep_kernel_pde_model
from models.KernelHelmholtz import HelmholtzInterpolation

path = os.path.join("models_trained", "simulation_broadband")
path_deep_kernel_pde = os.path.join(path, "deep_kernel_pde_10.pth")
path_deep_kernel_pde_2 = os.path.join(path, "deep_kernel_pde_20.pth")
path_deep_kernel = os.path.join(path, "deep_kernel.pth")

# Load the model
seed = 0 
delta_freqs = 100
freq_low = 50
freq_high = 1000+delta_freqs
freqs = range(freq_low,freq_high,delta_freqs)

N_realizations = 10

NMSE_DKPDE = np.zeros((N_realizations,len(freqs)-1))
NMSE_DKPDE_2 = np.zeros((N_realizations,len(freqs)-1))
NMSE_DK = np.zeros((N_realizations,len(freqs)-1))
NMSE_Helmholtz = np.zeros((N_realizations,len(freqs)-1))
for n in range(N_realizations):
    for f_i in range(len(freqs)-1): # For a range of frequencies
        print(f_i, " frequency,   ", n, " realization")

        # PDE regularizer (proposed)
        DKPDE_model = get_deep_kernel_pde_model(path_deep_kernel_pde, seed = n,signal = "white", frequency = None, cutoff_low = freqs[f_i], cutoff_high = freqs[f_i+1])
        NMSE = DKPDE_model.NMSE_val(DKPDE_model.NN)
        NMSE_DKPDE[n,f_i] = NMSE.cpu().numpy()

        DKPDE_model_2 = get_deep_kernel_pde_model(path_deep_kernel_pde_2, seed = n,signal = "white", frequency = None, cutoff_low = freqs[f_i], cutoff_high = freqs[f_i+1])
        NMSE = DKPDE_model_2.NMSE_val(DKPDE_model_2.NN)
        NMSE_DKPDE_2[n,f_i] = NMSE.cpu().numpy()

        # Deep kernel (baseline)
        DK_model = get_deep_kernel_pde_model(path_deep_kernel, seed = n,signal = "white", frequency = None, cutoff_low = freqs[f_i], cutoff_high = freqs[f_i+1])
        NMSE = DK_model.NMSE_val(DK_model.NN)
        NMSE_DK[n,f_i] = NMSE.cpu().numpy()

        # Helmholtz kernel
        # Extract the signal from the dataset
        posMic = DK_model.train_data[:][0]
        sigMic = DK_model.train_data[:][1]
        posEval = DK_model.val_data[:][0]
        sigEval = DK_model.val_data[:][1].cpu()

        # Extract positions from input grid (not time)
        posMic = torch.cat((torch.unsqueeze(torch.unique(posMic[:,0],dim = 0),1),torch.unsqueeze(torch.unique(posMic[:,1],dim = 0),1),torch.unsqueeze(torch.unique(posMic[:,2],dim = 0),1)),1)
        posEval = torch.cat((torch.unsqueeze(torch.unique(posEval[:,0],dim = 0),1),torch.unsqueeze(torch.unique(posEval[:,1],dim = 0),1),torch.unsqueeze(torch.unique(posEval[:,2],dim = 0),1)),1)
        Mtrain = posMic.shape[0]
        Ntrain = int(DK_model.train_data[:][0].shape[0]/Mtrain)
        Mval = posEval.shape[0]
        Nval = int(DK_model.val_data[:][0].shape[0]/Mval)
        sigMic_reshape = sigMic.reshape((Mtrain,Ntrain))

        # Cross validation to find the best sigma
        sigma2s = [1e-2, 1e-1,1,10,100]
        NMSEs = np.zeros(len(sigma2s))
        for sigma2 in sigma2s:
            sigHat = HelmholtzInterpolation(sigMic_reshape.cpu(),posMic.cpu(),posEval.cpu(),c=DK_model.sound_speed,samplerate = DK_model.config.data.simulated_data.fs, sigma2 = sigma2, freq_low = freq_low, freq_high = freq_high)
            sigHat = torch.tensor(sigHat.flatten()).cpu()

            NMSEs[sigma2s.index(sigma2)] =  torch.norm(sigHat - sigEval)**2 / torch.norm(sigEval)**2

        # Choose the best sigma
        sigma2 = sigma2s[np.argmin(NMSEs)]
        print("NMSEs as function of sigmas : ",NMSEs)
        sigHat = HelmholtzInterpolation(sigMic_reshape.cpu(),posMic.cpu(),posEval.cpu(),c=DK_model.sound_speed,samplerate = DK_model.config.data.simulated_data.fs, sigma2 = sigma2)
        sigHat = torch.tensor(sigHat.flatten()).cpu()
        NMSE_Helmholtz[n,f_i] = torch.norm(sigHat - sigEval)**2 / torch.norm(sigEval)**2






# Convert to dB
NMSE_Helmholtz_plot = 10*np.log10(np.array(NMSE_Helmholtz.mean(0)))
NMSE_DK_plot = 10*np.log10(np.array(NMSE_DK.mean(0)))
NMSE_DKPDE_plot = 10*np.log10(np.array(NMSE_DKPDE.mean(0)))
NMSE_DKPDE_2_plot = 10*np.log10(np.array(NMSE_DKPDE_2.mean(0)))



print("NMSE_Helmholtz_plot", NMSE_Helmholtz_plot)
print("NMSE_DK_plot", NMSE_DK_plot)
print("DKPDE_plot", NMSE_DKPDE_plot)
print("DKPDE_2_plot", NMSE_DKPDE_2_plot)

# Construct x-axis
freqs_labels = []
for f in range(len(freqs)-1):
    freqs_labels.append(str(freqs[f]) + "-" + str(freqs[f+1]))


# plt.plot(freqs_labels,NMSE_SE_plot, '-', linewidth = 2)
plt.plot(freqs_labels,NMSE_Helmholtz_plot, '-', linewidth = 2)
plt.plot(freqs_labels,NMSE_DK_plot,'--', linewidth = 2)
plt.plot(freqs_labels,NMSE_DKPDE_plot,'-.', linewidth = 2)
plt.plot(freqs_labels,NMSE_DKPDE_2_plot, ':', linewidth = 2)


plt.xticks(rotation = 45)
plt.legend(["Bessel","DK","DKPDE10","DKPDE20"])
plt.xlabel("Frequency, Hz")
plt.ylabel("NMSE, dB")
# plt.ylim([NMSE_DKPDE_2_plot.min()-0.5,0.5])
plt.grid()
plt.tight_layout()

plt.show()
