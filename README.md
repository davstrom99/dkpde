# Sound Field Estimation Using Deep Kernel Learning Regularized by the Wave Equation

In this work, we introduce a spatio-temporal kernel for Gaussian process (GP) regression-based sound field estimation. Notably, GPs have the attractive property that the sound field is a linear function of the measurements, allowing the field to be estimated efficiently from distributed microphone measurements. However, to ensure analytical tractability, most existing kernels for sound field estimation have been formulated in the frequency domain, formed independently for each frequency. 
To address the analytical intractability of spatio-temporal kernels, we here propose to instead learn the kernel directly from data by the means of deep kernel learning.
Furthermore, to improve the generalization of the deep kernel, we propose a method for regularizing the learning process using the wave equation. The representational advantages of the proposed deep kernel and the improved generalization obtained by using the wave equation regularization are illustrated using numerical simulations.

# Usage
To train the network, set the parameters in the config.yaml and then run PDEDKLTrainer.py. To train the model without wave equation regularization, set N_collocation_time and N_collocation_space to 0.

To use a pre-trained network, see eval_frequency.py for an example.

# Citation
