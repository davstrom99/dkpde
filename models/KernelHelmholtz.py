import numpy as np
from scipy import signal
import scipy.spatial.distance as distfuncs
import scipy.special as special
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path


def kiFilterGen(k, posMic, posEst, filterLen=None, smplShift=None, reg=1e-1):
    """Kernel interpolation filter for estimating pressure distribution from measurements
    - N. Ueno, S. Koyama, and H. Saruwatari, “Kernel Ridge Regression With Constraint of Helmholtz Equation 
      for Sound Field Interpolation,” Proc. IWAENC, DOI: 10.1109/IWAENC.2018.8521334, 2018.
    - N. Ueno, S. Koyama, and H. Saruwatari, “Sound Field Recording Using Distributed Microphones Based on 
      Harmonic Analysis of Infinite Order,” IEEE SPL, DOI: 10.1109/LSP.2017.2775242, 2018.
    """
    numMic = posMic.shape[0]
    numEst = posEst.shape[0]
    numFreq = k.shape[0]
    fftlen = numFreq*2

    if filterLen is None:
        filterLen = numFreq+1
    if smplShift is None:
        smplShift = numFreq/2

    k = k[:, None, None]
    distMat = distfuncs.cdist(posMic, posMic)[None, :, :]
    K = special.spherical_jn(0, k * distMat)
    Kinv = np.linalg.inv(K + reg * np.eye(numMic)[None, :, :])
    distVec = np.transpose(distfuncs.cdist(posEst, posMic), (1, 0))[None, :, :]
    kappa = special.spherical_jn(0, k * distVec)
    kiTF = np.transpose(kappa, (0, 2, 1)) @ Kinv
    kiTF = np.concatenate((np.zeros((1, numEst, numMic)), kiTF, kiTF[int(fftlen/2)-2::-1, :, :].conj()))
    kiFilter = np.fft.ifft(kiTF, n=fftlen, axis=0).real
    kiFilter = np.concatenate((kiFilter[fftlen-smplShift:fftlen, :, :], kiFilter[:filterLen-smplShift, :, :]))

    return kiFilter



def HelmholtzInterpolation(sigMic,posMic,posEval,c=343,samplerate = 8000, sigma2 = 1e-3, freq_low = 50, freq_high = 1000):
    # Time 
    sigLen = sigMic.shape[1]
    t = np.arange(sigLen)/samplerate

    # FFT parameters
    fftlen = 16384 # Same as sigLen ???????????????????
    # fftlen = 5000

    # Filter parameters
    smplShift = 512
    filterLen = 1025
    # smplShift = 52
    # filterLen = 105
  
    
    freq = np.arange(1,fftlen/2+1)/fftlen*samplerate

    freq = freq[(freq>=freq_low) & (freq<=freq_high)]


    # Kernel interpolation filter
    k = 2 * np.pi * freq / c
    kiFilter = kiFilterGen(k, posMic, posEval, filterLen, smplShift, sigma2) # kiFilter contains (filterLen, numVal, numMic)

    # Convolution inerpolation filter
    specMic = np.fft.fft(sigMic.T, n=fftlen, axis=0)[:,:,None]
    specKiFilter = np.fft.fft(kiFilter, n=fftlen, axis=0)

    specEst = np.squeeze(specKiFilter @ specMic)
    sigEst = np.fft.ifft(specEst, n=fftlen, axis=0).real.T
    sigEst = sigEst[:,smplShift:sigLen+smplShift]
    return sigEst