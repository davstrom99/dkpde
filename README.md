# Sound Field Estimation Using Deep Kernel Learning Regularized by the Wave Equation:
This repository provides code for a wave equation-informed spatio-temporal deep kernel for Gaussian Process regression, enabling efficient sound field estimation from microphone data. 

# Key Features:
- Spatio-temporal kernel for GP regression-based sound field estimation
- Wave equation regularization for enhanced generalization

# Getting started:
This repository contains code used in the paper.

To train the network, set the parameters in the config.yaml and then run PDEDKLTrainer.py. To train the model without wave equation regularization, set N_collocation_time and N_collocation_space to 0.

To use a pre-trained network, see eval_frequency.py for an example.

# References
