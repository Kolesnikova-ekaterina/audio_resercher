import librosa
import numpy as np

import matplotlib.pyplot as plt
import denoise
import fingerprinting

class parser:
    def __init__(self, signal):
        noisegate = np.mean ([abs(x) for x in signal]) * 1.5

        '''
        plt.figure(figsize=(12, 4))
        plt.plot(signal, color='blue')
        plt.axhline (y=noisegate, color = 'red')
        plt.axhline (y=-noisegate, color = 'red')
        plt.show()'''

        self.signals = []

        begin = 0
        end = 0
        flag = False
        pause = 0
        for i in range(len(signal)):
            if abs(signal[i]) > noisegate :
                pause = 0
                if(flag):
                    end = i
                else:
                    begin = i
                    end = i
                    flag = True
            else:
                pause +=1
                if (pause<=2):
                    continue
                if (flag):
                    self.signals.append(signal[begin:end+1])
                flag = False










def _parsingaudio(X, directory_name ):
    i = 0
    newX = []
    for xx in X:
        bird = xx[1]
        signal, sr= librosa.load(directory_name + '\\' + bird + '\\' + xx[0], sr=22050)
        dn = denoise.denoise(signal)
        denoisesignal = dn.filtered_samples
        p = parser(denoisesignal)
        signals = p.signals
        i += 1
        for s in signals:
            if len(s)<2048:
                continue
            newX.append([s, bird, i])
            '''
            plt.figure(figsize=(4, 4))
            plt.plot(s, color='blue')
            plt.show()
            '''
    return newX



def deletenoise(X, directory_name):
    newX = []
    for xx in X:
        bird = xx[1]
        signal, sr = librosa.load(directory_name + '\\' + bird + '\\' + xx[0], sr=22050)
        dn = denoise.denoise(signal)
        denoisesignal = dn.filtered_samples

        newsignal = []
        noisegate = np.mean([abs(x) for x in denoisesignal])*1.5
        for i in denoisesignal:
            if (abs(i) > noisegate):
                newsignal.append(i)
        newX.append([np.asarray(newsignal), bird, 0])
        '''
        plt.figure(figsize=(12, 4))
        plt.plot(denoisesignal, color='blue')
        plt.axhline(y=noisegate, color='red')
        plt.axhline(y=-noisegate, color='red')
        plt.show()
        plt.figure(figsize=(12, 4))
        plt.plot(newsignal, color='blue')
        plt.axhline(y=noisegate, color='red')
        plt.axhline(y=-noisegate, color='red')
        plt.show()
        '''
    return newX