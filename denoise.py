import fingerprinting
import pandas as pd
import os
import random
import matplotlib.pyplot as plt
import librosa
import numpy as np
from sklearn.neural_network import MLPClassifier
import models
import scipy
from scipy.fft import fft, fftfreq

class denoise:
    def __init__(self, signal):
        n_fft = 2048
        #ft = np.abs(scipy.fft(signal))
        #ft_mean = np.mean(ft)
        #print([i - ft_mean for i in signal])
        #fig, axs = plt.subplots(nrows= 2 , ncols= 1 )

        #axs[0].plot(signal, color = 'blue')
        new_s = []
        try:
            self.filtered_samples = scipy.signal.wiener(signal, 53)
        except:
            self.filtered_samples = signal
        #axs[1].plot(filtered_samples, color = 'red')

        #plt.show()
        #X = librosa.stft(signal)
        #s = librosa.amplitude_to_db(abs(X))
        #librosa.display.specshow(s, sr=22050, x_axis='time', y_axis='linear')
        #plt.colorbar()
        #lt.show()
