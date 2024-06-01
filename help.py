import sys

import fingerprinting as fgp
import pandas as pd
import librosa
import matplotlib.pyplot as plt
import numpy as np

def init_set(Set, fn):
    birds = []
    ft_stds = []
    ft_means = []
    zrate_stds = []
    zrate_means = []
    roloff_stds = []
    roloff_means = []
    cent_stds = []
    cent_means = []
    mfcc_stds = []
    mfcc_means = []
    index = []
    i = 0
    for xx in Set:
        i += 1
        fp = fgp.fingerprint(xx[0])
        bird = xx[1]
        # print(bird, i)
        birds.append(bird),
        ft_stds.append(fp.ft_std)
        ft_means.append(fp.ft_mean)
        zrate_stds.append(fp.zrate_std)
        zrate_means.append(fp.zrate_mean)
        roloff_stds.append(fp.roloff_std)
        roloff_means.append(fp.roloff_mean)
        cent_stds.append(fp.cent_std)
        cent_means.append(fp.cent_mean)
        mfcc_stds.append(fp.mfcc_std)
        mfcc_means.append(fp.mfcc_mean)
        index.append(xx[2])
    data = {
        'birds': birds,
        'ft_stds': ft_stds,
        'ft_means': ft_means,
        'zrate_stds': zrate_stds,
        'zrate_means': zrate_means,
        'roloff_stds': roloff_stds,
        'roloff_means': roloff_means,
        'cent_stds': cent_stds,
        'cent_means': cent_means,
        'mfcc_stds': mfcc_stds,
        'mfcc_means': mfcc_means,
        'index': index
    }
    train_set = pd.DataFrame(data)

    train_set.to_csv(fn)
    return train_set


def correct_parsed_results(predict, indexes):
    print(len(predict), len(indexes))
    inds = {}
    for i in range(len(predict)):
        if indexes[i][0] in inds.keys() :
            inds[indexes[i][0]][predict[i]] += 1
        else :
            inds[indexes[i][0]] = [0 for i in range(5)]
            inds[indexes[i][0]][predict[i]] += 1
    res = [ it[1].index(max(it[1]))  for it in inds.items() ]
    return res

def score(list1, list2):
    res = 0.0
    for i in range(len(list1)):
        print(list1[i] )
        if list1[i] == list2[i]:
            res += 1.0

    return res / len(list1)


def normilise(signals, maxlen):

    for s in signals:
        for i in range(maxlen - len(s)):
            s.append(0)

    return signals


def themostsimiliar(X, Y):
    for yy in Y:

        signal = []
        fgp = yy[1]
        error = sys.float_info.max
        rightsong = ""
        for xx in X:
            signalx = xx[0]
            fgpx = xx[1]
            errortemp = 0
            for f in range(len(fgp)):
                errortemp += (fgp[f] - fgpx[f])**2
            print(errortemp, error)
            if errortemp<error :
                rightsong = xx[2]
                error = errortemp
                signal = signalx
        print(yy[2], rightsong)
        if (yy[2] != rightsong):
            fig, axs = plt.subplots(nrows= 2 , ncols= 1 )

            axs[0].set_title(yy[2])
            axs[1].set_title(rightsong)
            axs[0].plot(yy[0], color='blue')
            axs[1].plot(signal, color='red')
            plt.show()
        if (yy[2] == rightsong):
            fig, axs = plt.subplots(nrows= 2 , ncols= 1 )
            fig.suptitle(rightsong)
            axs[0].plot(yy[0], color='blue')
            axs[1].plot(signal, color='red')
            plt.show()
