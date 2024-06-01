#Обработка данных
import mycrepe
import denoise
import fingerprinting
import pandas as pd
import os
import random
import matplotlib.pyplot as plt
import librosa
import numpy as np
from sklearn.neural_network import MLPClassifier
import models
import parser
import help

X = []
Y = []
directory_name = "birdclef-2021\\train_short_audio"
print(librosa.__version__)
species = os.listdir(directory_name)

def init_dataset():
    birds_names = os.listdir(directory_name)
    for bird_name in birds_names[:5]:
        directory_name1 = directory_name + '\\' + bird_name
        songs = os.listdir(directory_name1)
        #train_part = random.choices(songs, k=15)
        #test_part = random.choices(songs, k=5)
        tp = random.choices(songs, k=30)
        train_part = tp[:25]
        test_part = tp[25:30]
        for t in train_part:
            X.append([t,bird_name])
        for t in test_part:
            Y.append([t,bird_name])
    return X,Y

def init_trainset(dn = False, fn = 'train.csv' ):

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
    print(len(X))
    i = 0
    for xx in X:
        i += 1
        fp = fingerprinting.fingerprint(directory_name + '\\' + xx[1] + '\\' + xx[0], xx[1], dn)
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
        'mfcc_means': mfcc_means
    }
    train_set = pd.DataFrame(data)

    train_set.to_csv(fn)
    return train_set

def init_testset(dn = False, fn = 'test.csv'):

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
    print(len(Y))
    i = 0

    for xx in Y:
        i += 1
        fp = fingerprinting.fingerprint(directory_name + '\\' + xx[1] + '\\' + xx[0], xx[1],dn)
        bird = xx[1]
        #print(bird, i)
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
        'mfcc_means': mfcc_means
    }
    test_set = pd.DataFrame(data)

    test_set.to_csv(fn)
    return test_set



'''
X,Y = init_dataset()
init_trainset()
init_testset()
trainset = pd.read_csv('train.csv')
testset = pd.read_csv('test.csv')
y_train = [  species.index(i) for i in trainset[['birds']].values ]
x_train = trainset[['ft_stds', 'ft_means', 'zrate_stds', 'zrate_means',
       'roloff_stds', 'roloff_means', 'cent_stds', 'cent_means', 'mfcc_stds',
       'mfcc_means']]

y_test =[ species.index(i) for i in testset[['birds']].values ]
x_test = testset[['ft_stds', 'ft_means', 'zrate_stds', 'zrate_means',
       'roloff_stds', 'roloff_means', 'cent_stds', 'cent_means', 'mfcc_stds',
       'mfcc_means']]

knn = models.KNN(x_train,[y[0] for y in y_train.values])
knn.predict(x_test)
ac = knn.accuracy_score([y[0] for y in y_test.values])
print('knn accuracy = ',ac) #0.04836272040302267

mlp = models.mlp(x_train,[y[0] for y in y_train.values])
mlp.predict(x_test)
ac0 = mlp.accuracy_score([y[0] for y in y_test.values])
print('mlp accuracy = ',ac0) #0.011083123425692695

birds_names = os.listdir(directory_name)
for bird_name in birds_names:
    directory_name1 = directory_name + '\\' + bird_name
    songs = os.listdir(directory_name1)
    train_part = random.choices(songs, k=15)
    test_part = random.choices(songs, k=5)

    for t in train_part:
        denoise.denoise(directory_name1 + '\\' + t)

xgb = models.xgb(x_train,y_train)
xgb.predict(x_test)
ac = xgb.accuracy_score(y_test)
print('xgb accuracy = ',ac) #0.102

X,Y = init_dataset()
init_trainset(True, 'train_denoise.csv')
init_testset(True, 'test_denoise.csv')

trainset_dn = pd.read_csv('train_denoise.csv')
testset_dn = pd.read_csv('test_denoise.csv')
y_train_dn = [species.index(i) for i in trainset_dn[['birds']].values ]
x_train_dn = trainset_dn[['ft_stds', 'ft_means', 'zrate_stds', 'zrate_means',
       'roloff_stds', 'roloff_means', 'cent_stds', 'cent_means', 'mfcc_stds',
       'mfcc_means']]

y_test_dn =[species.index(i) for i in testset_dn[['birds']].values ]
x_test_dn = testset_dn[['ft_stds', 'ft_means', 'zrate_stds', 'zrate_means',
       'roloff_stds', 'roloff_means', 'cent_stds', 'cent_means', 'mfcc_stds',
       'mfcc_means']]

knn = models.KNN(x_train_dn,y_train_dn)
knn.predict(x_test_dn)
ac = knn.accuracy_score(y_test_dn)
print('knn accuracy = ',ac) #0.28

mlp = models.mlp(x_train_dn,y_train_dn)
mlp.predict(x_test_dn)
ac0 = mlp.accuracy_score(y_test_dn)
print('mlp accuracy = ',ac0) #0.44

xgb = models.xgb(x_train_dn,y_train_dn)
xgb.predict(x_test_dn)
ac = xgb.accuracy_score(y_test_dn)
print('xgb accuracy = ',ac) #0.52

'''
X,Y = init_dataset()

parsed_X = parser._parsingaudio(X, directory_name)
parsed_Y = parser._parsingaudio(Y, directory_name)

train = help.init_set(parsed_X, 'parsed_train.csv')
test = help.init_set(parsed_Y, 'parsed_test.csv')

trainset_dn = pd.read_csv('parsed_train.csv')
testset_dn = pd.read_csv('parsed_test.csv')
y_train_dn = [species.index(i) for i in trainset_dn[['birds']].values ]
x_train_dn = trainset_dn[['ft_stds', 'ft_means', 'zrate_stds', 'zrate_means',
       'roloff_stds', 'roloff_means', 'cent_stds', 'cent_means', 'mfcc_stds',
       'mfcc_means']]

y_test_dn =[species.index(i) for i in testset_dn[['birds']].values ]
x_test_dn = testset_dn[['ft_stds', 'ft_means', 'zrate_stds', 'zrate_means',
       'roloff_stds', 'roloff_means', 'cent_stds', 'cent_means', 'mfcc_stds',
       'mfcc_means']]

knn = models.KNN(x_train_dn,y_train_dn)
knn.predict(x_test_dn)
ac = knn.accuracy_score(y_test_dn)
trcr = help.correct_parsed_results(y_test_dn, testset_dn[['index']].values)
crr = help.correct_parsed_results(knn.predictions, testset_dn[['index']].values )
print('knn accuracy = ',ac, '  correct results = ', help.score(trcr, crr))# knn accuracy =  0.85


mlp = models.mlp(x_train_dn,y_train_dn)
mlp.predict(x_test_dn)
ac0 = mlp.accuracy_score(y_test_dn)
crr = help.correct_parsed_results(mlp.predictions, testset_dn[['index']].values )
print('mlp accuracy = ',ac0, '  correct results = ', help.score(trcr, crr)) #mlp accuracy =  0.70

xgb = models.xgb(x_train_dn,y_train_dn)
xgb.predict(x_test_dn)
ac = xgb.accuracy_score(y_test_dn)
crr = help.correct_parsed_results(xgb.predictions, testset_dn[['index']].values )
print('xgb accuracy = ',ac, '  correct results = ', help.score(trcr, crr)) #xgb accuracy =  0.75
'''

X,Y = init_dataset()

parsed_X = parser.deletenoise(X, directory_name)
parsed_Y = parser.deletenoise(Y, directory_name)

train = help.init_set(parsed_X, 'nonoise_train.csv')
test = help.init_set(parsed_Y, 'nonoise_test.csv')

trainset_dn = pd.read_csv('nonoise_train.csv')
testset_dn = pd.read_csv('nonoise_test.csv')
y_train_dn = [species.index(i) for i in trainset_dn[['birds']].values ]
x_train_dn = trainset_dn[['ft_stds', 'ft_means', 'zrate_stds', 'zrate_means',
       'roloff_stds', 'roloff_means', 'cent_stds', 'cent_means', 'mfcc_stds',
       'mfcc_means']]

y_test_dn =[species.index(i) for i in testset_dn[['birds']].values ]
x_test_dn = testset_dn[['ft_stds', 'ft_means', 'zrate_stds', 'zrate_means',
       'roloff_stds', 'roloff_means', 'cent_stds', 'cent_means', 'mfcc_stds',
       'mfcc_means']]

knn = models.KNN(x_train_dn,y_train_dn)
knn.predict(x_test_dn)
ac = knn.accuracy_score(y_test_dn)
print('knn accuracy = ',ac)# knn accuracy =  0.52


mlp = models.mlp(x_train_dn,y_train_dn)
mlp.predict(x_test_dn)
ac0 = mlp.accuracy_score(y_test_dn)
print('mlp accuracy = ',ac0) #mlp accuracy =  0.45

xgb = models.xgb(x_train_dn,y_train_dn)
xgb.predict(x_test_dn)
ac = xgb.accuracy_score(y_test_dn)
print('xgb accuracy = ',ac) #xgb accuracy =  0.56



X,Y = init_dataset()

parsed_X = parser._parsingaudio(X, directory_name)
parsed_Y = parser._parsingaudio(Y, directory_name)



X_train_n = mycrepe.initaudiovectors(parsed_X)
X_test = [species.index(x[1]) for x in parsed_X]
Y_train_n = mycrepe.initaudiovectors(parsed_Y)
Y_test = [species.index(x[1]) for x in parsed_Y]
maxlen = max(np.max([len(x) for x in X_train_n]), np.max([len(x) for x in Y_train_n]))
X_train = help.normilise(X_train_n, maxlen)
Y_train = help.normilise(Y_train_n, maxlen)




knn = models.KNN(X_train,X_test)
knn.predict(Y_train)
ac = knn.accuracy_score(Y_test)
print('knn accuracy = ',ac)# knn accuracy =  0.42


mlp = models.mlp(X_train,X_test)
mlp.predict(Y_train)
ac0 = mlp.accuracy_score(Y_test)
print('mlp accuracy = ',ac0) #mlp accuracy =  0.45

xgb = models.xgb(X_train,X_test)
xgb.predict(Y_train)
ac = xgb.accuracy_score(Y_test)
print('xgb accuracy = ',ac) #xgb accuracy =  0.44
'''

'''

X_train_n = [x[0].tolist() for x in parsed_X]
X_test = [species.index(x[1]) for x in parsed_X]
Y_train_n = [x[0].tolist() for x in parsed_Y]
Y_test = [species.index(x[1]) for x in parsed_Y]
maxlen = max(np.max([len(x) for x in X_train_n]), np.max([len(x) for x in Y_train_n]))
X_train = help.normilise(X_train_n, maxlen)
Y_train = help.normilise(Y_train_n, maxlen)


knn = models.KNN(X_train,X_test)
knn.predict(Y_train)
ac = knn.accuracy_score(Y_test)
print('knn accuracy = ',ac)# knn accuracy =  0.35


mlp = models.mlp(X_train,X_test)
mlp.predict(Y_train)
ac0 = mlp.accuracy_score(Y_test)
print('mlp accuracy = ',ac0) #mlp accuracy =  0.38

xgb = models.xgb(X_train,X_test)
xgb.predict(Y_train)
ac = xgb.accuracy_score(Y_test)
print('xgb accuracy = ',ac) #xgb accuracy =  0.40






X_train_n = mycrepe.crepeomchromo(parsed_X)
X_test = [species.index(x[1]) for x in parsed_X]
Y_train_n = mycrepe.crepeomchromo(parsed_Y)
Y_test = [species.index(x[1]) for x in parsed_Y]
maxlen = max(np.max([len(x) for x in X_train_n]), np.max([len(x) for x in Y_train_n]))
X_train = help.normilise(X_train_n, maxlen)
Y_train = help.normilise(Y_train_n, maxlen)


tr1 = []
for i in range(len(X_test)):
    tr1.append([parsed_X[i][0],X_train[i],parsed_X[i][1]])
tr2 = []
for i in range(len(Y_test)):
    tr2.append([parsed_Y[i][0],Y_train[i],parsed_Y[i][1],])

help.themostsimiliar(tr1, tr2)
knn = models.KNN(X_train,X_test)
knn.predict(Y_train)
ac = knn.accuracy_score(Y_test)
print('knn accuracy = ',ac)# knn accuracy =  0.66


mlp = models.mlp(X_train,X_test)
mlp.predict(Y_train)
ac0 = mlp.accuracy_score(Y_test)
print('mlp accuracy = ',ac0) #mlp accuracy =  0.6111111111111112

xgb = models.xgb(X_train,X_test)
xgb.predict(Y_train)
ac = xgb.accuracy_score(Y_test)
print('xgb accuracy = ',ac) #xgb accuracy =  0.66667
'''

