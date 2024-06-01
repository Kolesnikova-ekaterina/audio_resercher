import matplotlib.pyplot as plt
import sys
import librosa
import numpy as np
import help

class craft_crepe:
    def __init__(self):
      self.table = self.matchingfrequencies()

    def matchingfrequencies(self):
        table = [[16.352, 18.354, 20.602, 21.827, 24.5, 27.5, 30.868]]
        for i in range (1,9):
            table.append([])
            for j in range(7):
                table[i].append(table[i-1][j] * 2)

        return table

    def note_to_audiovector(self, notes):
        res = []
        for i in range(1, len(notes)):
            oc, note = notes[i]
            oc_p, note_p = notes[i-1]
            if (oc > oc_p) :
                res.append(0)
                res.append(1)
                continue
            if oc == oc_p :
                if (note > note_p):
                    res.append(0 )
                    res.append(1)
                    continue
                if (note == note_p):
                    res.append(0 )
                    res.append(0)
                    continue
                if (note < note_p):
                    res.append(1 )
                    res.append(0)
                    continue
            if (oc < oc_p) :
                res.append(1 )
                res.append(0)
                continue
        return res
    def process_signal(self, signal):
        notes = []
        for s in abs(signal*1000):
            octave = -1
            note = -1
            min = sys.float_info.max
            for i in range(9):
                for j in range(7):
                    if abs(s - self.table[i][j]) < min :
                        octave = i
                        note = j
                        min = abs(s - self.table[i][j])
            notes.append([octave, note])
        return self.note_to_audiovector(notes)


def initaudiovectors(X):
    cr = craft_crepe()
    set = []
    for s in X :
      signal = cr.process_signal(s[0])
      #librosa.feature.chroma_stf
      set.append(signal)
    return set


def fromchronotovector(chromo):
    res = []
    notes = []
    print(chromo)
    for i in range(len(chromo[1,:])):
        max = np.max(chromo[:,i])
        note = chromo[:,i].tolist().index(max)
        notes.append(note)
    prevnote = notes[0]
    print(notes)
    for i in range( 1, len(notes)):
        if (notes[i-1] == notes[i]):
            res.append(0)
            res.append(0)
        else:
            if notes[i] > notes[i-1]:
                res.append(0)
                res.append(1)
            else:
                res.append(1)
                res.append(0)
    return res



def crepeomchromo(X):
    set = []
    for sx in X:
        print(sx[0])
        S = np.abs(librosa.stft(sx[0], n_fft=256))
        chr = librosa.feature.chroma_stft(S=S)
        set.append(fromchronotovector(chr))
        '''
        fig, ax = plt.subplots(nrows=2, sharex=True)
        img = librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max),
                                       y_axis='log', x_axis='time', ax=ax[0])
        fig.colorbar(img, ax=[ax[0]])
        ax[0].label_outer()
        img = librosa.display.specshow(chr, y_axis='chroma', x_axis='time', ax=ax[1])
        fig.colorbar(img, ax=[ax[1]])
        plt.show()'''
    return set





