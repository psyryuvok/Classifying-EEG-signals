# -*- coding: utf-8 -*-  
""" 
Created on Fri Mar 15 15:11:20 2019 
 
@author: Z 
"""  
  
import numpy as np  

import matplotlib.pyplot as plt  
  
#%matplotlib inline  
  
from collections import defaultdict  
rezultate=defaultdict(list)  
  
  
  
r=0;  
import glob  
file_list = glob.glob(r"C:\Users\Z\Downloads\baza de date\bci\phisyionet\test\aved\evenlarger\S*.mat")#filter\ 
 # Se citesc datele și se concateneaza într-un vector
for i in file_list:  
      
    import scipy.io  
    mat = scipy.io.loadmat(i)  
    X_RH=mat.get('RH');  
    size=len(X_train);  
    y_RH=np.ones(size);  
    X_LH=mat.get('LH');  
    size=len(X_test);  
    y_LH=np.zeros(size);  
#In caz ca se dorește antrenarea cu mai mult de 2 clase  
# =============================================================================  
#     X_BF=mat.get('BF');  
#     size=len(X_test1);  
#     y_BF=np.ones(size)*2;  
#     X_R1=mat.get('R1');  
#     size=len(X_test2);  
#     y_R1=np.ones(size)*3;  
# =============================================================================  
  
    X=np.concatenate((X_RH, X_LH), axis=0)  
    y=np.concatenate((y_RH, y_LH), axis=0)  
  
    from keras.utils import to_categorical  
    from sklearn.preprocessing import LabelEncoder  
    # codifică valoarea clasei ca și număr intreg 
    encoder = LabelEncoder()  
    y2=y;  
    encoder.fit(y2)  
    encoded_y = encoder.transform(y2)  
    # convertește numerele intregi in variabile dummy(i.e. one hot encoded)  
    y= to_categorical(encoded_y)  
      
    r=r+1;  
    if r<2:  
        X1=X;  
        y1=y;  
        y3=y2;  
  
    X1=np.concatenate((X1,X), axis=0)  
    y1=np.concatenate((y1,y), axis=0)  
    y3=np.concatenate((y3,y2), axis=0)  
          
#Inputul e scalat pentru a mări viteza de procesare          
from sklearn.preprocessing import StandardScaler  
scalers = {}  
for i in range(X1.shape[1]):  
    scalers[i] = StandardScaler()  
    X1[:, i, :] = scalers[i].fit_transform(X1[:, i, :])   

      
      
#Se redimensionează pentru a putea fi citit de rețea  
X1=X1.reshape(len(X1),64,640,1)  
from sklearn.model_selection import train_test_split  
# Împărțirea datelor in antrenament și test  
X_train, X_test, y_train, y_test = train_test_split(X1, y1, test_size=0.20)  
  
  
  
  
# import regularizer  
from keras.regularizers import l1  
reg = l1(0.001)  
  
from keras import Sequential  
from keras.strats import Dense  
from keras.strats import Dropout  
from keras.strats import Flatten  
from keras.strats.convolutional import Conv2D  
from keras.strats.convolutional import MaxPooling2D  
# definire model  
model = Sequential()  
model.add(Conv2D(filters=80, kernel_size=(1,30), activation='relu', input_shape=(64, 640, 1) ))  
model.add(Conv2D(filters=80, kernel_size=(64,1), activation='relu'))
model.add(MaxPooling2D(pool_size=(1,15)))  
model.add(Flatten())  
model.add(Dense(80, activation='relu', kernel_initializer='random_normal'))  
model.add(Dense(2, activation='softmax', kernel_initializer='random_normal')) 
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])  
model.summary()  
  
#Definire model pentru wrapper
def base_model(optimizer='rmsprop', init='glorot_uniform'):  
    model = Sequential()  
    model.add(Conv2D(filters=40, kernel_size=(1,30), activation='relu', input_shape=(64, 640, 1)))
    model.add(Conv2D(filters=40, kernel_size=(64,1), activation='relu' ))  
    model.add(MaxPooling2D(pool_size=(1,15)))  
    model.add(Flatten())  
    model.add(Dense(80, activation='relu', kernel_initializer=init))  
    model.add(Dense(2, activation='softmax', kernel_initializer=init))  
    model.compile( optimizer=optimizer,loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()  
    return model  
#Creare wrapper pentru a putea folosii librăria sk-learn  
from keras.wrappers.scikit_learn import KerasClassifier  
estimator = KerasClassifier(build_fn=base_model, epochs=100, batch_size=16, verbose=1)  
 
from sklearn.model_selection import KFold  
from sklearn.model_selection import cross_val_score  
kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=3)  
#Antrenare wrapper cu cross-validare
results = cross_val_score(estimator, X_train, y_train, cv=kfold)  
#Se printeazâ valoarea medie a acurateței și deviatia standard
print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))  
#######  
estimator.fit(X_train,y_train, validation_data=(X_test, y_test))  
predictions = estimator.predict(X_test)  
print(predictions)  
  
#Antrenarea unui model normal, fără wrapper
history=model.fit(X_train,y_train, validation_split=0.20, batch_size=16, epochs=20)  
  
# evaluare model
_, train_acc = model.evaluate(X_train, y_train, verbose=0)  
_, test_acc = model.evaluate(X_test, y_test, verbose=0)  
print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))  
# Plotat loss din timpul antrenamentului  
plt.subplot(211)  
plt.title('Loss')  
plt.plot(history.history['loss'], label='train')  
plt.plot(history.history['val_loss'], label='test')  
plt.legend()  
# Plotat acuratețe din timpul antrenamentului
plt.subplot(212)  
plt.title('Accuracy')  
plt.plot(history.history['acc'], label='train')  
plt.plot(history.history['val_acc'], label='test')  
plt.legend()  
plt.show()
  
# =============================================================================
#
#In caz că se dorește să se realizeze un grid search pentru obițnerea celor mai buni hyperparametri  
# model = KerasClassifier(build_fn=base_model, verbose=1)  
# # grid search face combinații de epochs, batch size, optimizer și init  
# optimizers = ['rmsprop', 'adam','adagrad','adadelta']  
# init = ['glorot_uniform', 'normal', 'uniform','lecun_normal']  
# epochs = [50] 
# batches = [16] 
# param_grid = dict( epochs=epochs, batch_size=batches , optimizer=optimizers, init=init)  
# grid = GridSearchCV(estimator=model, param_grid=param_grid,cv=5)  
# grid_result = grid.fit(X, y)  
# # summarizare resultate  
# print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))  
# means = grid_result.cv_results_['mean_test_score']  
# stds = grid_result.cv_results_['std_test_score']  
# params = grid_result.cv_results_['params']  
# for mean, stdev, param in zip(means, stds, params):  
#   print("%f (%f) with: %r" % (mean, stdev, param))  
# =============================================================================  
