import librosa
import librosa.display as ld
import numpy as np
import scipy
import matplotlib.pyplot as plt

import denoise


class fingerprint:
    def __init__(self, name, bird, d_n):
        signal, sr = librosa.load(name, sr=22050)  # загружаем файл
        if (d_n):
            dn = denoise.denoise(signal)
            signal = dn.filtered_samples
        #plt.figure(figsize=(12, 4))
        #ld.waveshow(signal, sr=sr, color="blue")
        n_fft = 2048
        ft = np.abs(librosa.stft(signal[:n_fft], hop_length=n_fft + 1))
        """
        plt.plot(ft)
        plt.title('Spectrum')
        plt.xlabel('Frequency Bin')
        plt.ylabel('Amplitude')
        plt.show()"""

        X = librosa.stft(signal)
        s = librosa.amplitude_to_db(abs(X))
        """
        ld.specshow(s, sr=sr, x_axis='time', y_axis='linear')
        plt.colorbar()
        plt.show()
        """
        mfccs = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=40, hop_length=512)
        melspectrum = librosa.feature.melspectrogram(y=signal, sr=sr,
                                                     hop_length=512, n_mels=40)

        """
        ld.specshow(melspectrum, sr=sr, x_axis='time', y_axis='linear')
        plt.colorbar()
        plt.show()
        """
        cent = librosa.feature.spectral_centroid(y=signal, sr=sr)
        """
        plt.figure(figsize=(15, 5))
        plt.semilogy(cent.T, label='Spectral centroid')
        plt.ylabel('Hz')
        plt.legend()
        """

        rolloff = librosa.feature.spectral_rolloff(y=signal, sr=sr)
        """
        plt.figure(figsize=(15, 5))
        plt.semilogy(rolloff.T, label='Roll-off frequency')
        plt.ylabel('Hz')
        plt.legend()
        """
        zrate = librosa.feature.zero_crossing_rate(signal)
        """
        plt.figure(figsize=(14, 5))
        plt.semilogy(zrate.T, label='Fraction')
        plt.ylabel('Fraction per Frame')
        plt.legend()
        """
        self.mfcc_mean = np.mean(mfccs)
        self.mfcc_std = np.std(mfccs)
        self.cent_mean = np.mean(cent)
        self.cent_std = np.std(cent)
        self.roloff_mean = np.mean(rolloff)
        self.roloff_std = np.std(rolloff)
        self.ft_mean = np.mean(ft)
        self.ft_std = np.std(ft)
        self.zrate_mean = np.mean(zrate)
        self.zrate_std = np.std(zrate)


    def __init__(self, signal):
        sr = 22050
        mfccs = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=40, hop_length=512)
        cent = librosa.feature.spectral_centroid(y=signal, sr=sr)
        rolloff = librosa.feature.spectral_rolloff(y=signal, sr=sr)
        zrate = librosa.feature.zero_crossing_rate(signal)
        ft = np.abs(librosa.stft(signal))

        self.mfcc_mean = np.mean(mfccs)
        self.mfcc_std = np.std(mfccs)
        self.cent_mean = np.mean(cent)
        self.cent_std = np.std(cent)
        self.roloff_mean = np.mean(rolloff)
        self.roloff_std = np.std(rolloff)
        self.ft_mean = np.mean(ft)
        self.ft_std = np.std(ft)
        self.zrate_mean = np.mean(zrate)
        self.zrate_std = np.std(zrate)



