# -*- coding: utf-8 -*-  
""" 
Created on Sun Mar  3 13:42:00 2019 
 
@author: Z 
"""  
def classify_functions(X,y):  
    import numpy as np  
    import seaborn as sns  
    import matplotlib.pyplot as plt  
    import pandas as pd  
    import types  
    from mpl_toolkits.mplot3d import Axes3D  
    fig = plt.Figura()  
    plt.style.use('ggplot')  
    plt.rcParams['image.cmap'] = 'RdBu'  
    import sklearn.datasets as datasets  
    from sklearn.preprocessing import PolynomialFeatures  
    import scipy.io as sio  
    from scipy.stats import pearsonr  
    from sklearn.neighbors import KNeighborsClassifier  
    import graphviz  
    from sklearn.tree import export_graphviz  
    from sklearn.tree import DecisionTreeClassifier  
    from sklearn.ensemble import RandomForestClassifier  
    from sklearn.svm import SVC  
    from sklearn.linear_model import LogisticRegression  
    import keras  
    from sklearn.preprocessing import StandardScaler  
    # %matplotlib inline  
    from sklearn.decomposition import PCA  
    def plot_history(h):  
        
      fig, (ax1, ax2) = plt.subplots(1, 2)  
        
      ax1.plot(h.history['loss'], label='Training Loss')  
      ax1.plot(h.history['val_loss'], label='Test Loss')  
      ax1.set_ylabel('Loss')  
      ax1.set_xlabel('Epoch')  
      ax1.legend(fontsize=24)  
      ax1.set_ylim(0, 1)  
        
      ax2.plot(h.history['acc'], label='Training Accuracy')  
      ax2.plot(h.history['val_acc'], label='Test Accuracy')  
      ax2.set_ylabel('Accuracy')  
      ax2.set_xlabel('Epoch')  
      ax2.legend(fontsize=24)  
      ax2.set_ylim(0, 1)  
        
      fig.suptitle('Evolution over epochs', fontsize=24)  
        
      fig.set_size_inches(21, 14)  
      
    # =============================================================================  
    # mat_contents = sio.loadmat('person1.mat')  
    # r1=mat_contents['x1']  
    # r2=mat_contents['x2']  
    # r3=mat_contents['x3']  
    # r4=mat_contents['x4']  
    # r5=mat_contents['x5']  
    # r6=mat_contents['x6']  
    # =============================================================================  
    trainAcc=[]  
    testAcc= []  
    # =============================================================================  
    # 0-polynomial features  
    # 1-logistic regresion  
    # 2-k neighbor  
    # 3-DecisionTreeClassifier  
    # 4-Random forest  
    # 5-SVM  
    # 6-Linear SVM  
    # 7-SVM gamma  
    # =============================================================================  
      
      
    X= StandardScaler().fit_transform(X)    
    #pca = PCA(n_components=2)  
    #pca.fit(X)  
    #polynomial features  
    from sklearn.model_selection import train_test_split  
    from sklearn.preprocessing import PolynomialFeatures  
      
    # degree = 3, because that looks the most appropriate  
    t = PolynomialFeatures(degree=3, include_bias=False)  
    X_poly = t.fit_transform(X)  
      
    # train / test split, as before; we want to be rigorous.  
    x_train, x_test, y_train, y_test = train_test_split(X_poly, y)  
      
      
      
    #from sklearn.model_selection import cross_val_score  
    #accuracy = cross_val_score(model, X, y, cv=10,scoring='accuracy')  
      
    model = LogisticRegression()  
    model.fit(x_train, y_train)  
    testAcc.append(model.score(x_test, y_test))  
    trainAcc.append(model.score(x_train, y_train))  
    print('Train accuracy', model.score(x_train, y_train))  
    print('Test accuracy', model.score(x_test, y_test))  
      
    #logistc regression  
      
      
    x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=0) # random state for reproducibility  
      
    model = LogisticRegression()  
    model.fit(x_train, y_train)  
      
    print('Train accuracy', model.score(x_train, y_train))  
    print('Test accuracy', model.score(x_test, y_test))  
    testAcc.append(model.score(x_test, y_test))  
    trainAcc.append(model.score(x_train, y_train))  
      
      
      
      
    # K neighbor graph  
      
    # put the test scores in this list  
    scorestrain = []  
    scorestest= []  
    ks = []  
      
    for k in range(1, 50):  
      #  Train your model with the current k  
      
      model = KNeighborsClassifier(k)  
      model.fit(x_train, y_train)  
        
      #  Check the test accuracy of your model  
      scoretrain = model.score(x_train, y_train)  
      scoretest = model.score(x_test, y_test)  
      #  Add the score to the list  
        
      scorestrain.append(scoretrain)  
      scorestest.append(scoretest)  
      ks.append(k)  
        
    best_k = ks[np.argmax(scorestest)]  
      
      
    knn = KNeighborsClassifier(n_neighbors=best_k)  
    knn.fit(x_train, y_train)  
    testAcc.append(knn.score(x_test, y_test))  
    trainAcc.append(knn.score(x_train, y_train))  
    print("Training accuracy", knn.score(x_train, y_train))  
    print("Test accuracy", knn.score(x_test, y_test))  
      
    plt.Figura(1)  
    plt.plot(range(1, len(scorestrain) + 1), scorestrain)  
    plt.plot(range(1, len(scorestest) + 1), scorestest)  
    plt.gcf().set_size_inches(21, 14)  
      
      
    # DecisionTreeClassifier  
    model = DecisionTreeClassifier()  
    model.fit(x_train, y_train)  
      
    print("Training accuracy", model.score(x_train, y_train))  
    print("Test accuracy", model.score(x_test, y_test))  
    testAcc.append(model.score(x_test, y_test))  
    trainAcc.append(model.score(x_train, y_train))  
    dot_data = export_graphviz(model,  
                               out_file=None,  
                               filled=True,  
                               rounded=True)  
    graph = graphviz.Source(dot_data)  
    graph  
      
      
      
    #random forest  
    # put the test scores in this list  
    scorestest = []  
    scorestrain = []  
    estimators = []  
      
    for estimator in range(1, 100):  
      # TODO train your model with the current k  
      
      model = RandomForestClassifier(n_estimators=estimator)  
      model.fit(x_train, y_train)  
      
      # TODO Check the test accuracy of your model  
      scoretest = model.score(x_test, y_test)  
      scoretrain = model.score(x_train, y_train)  
      # TODO add the score to the list  
      
      scorestest.append(scoretest)  
      scorestrain.append(scoretrain)  
      estimators.append(k)  
       
    best_estimator = estimators[np.argmax(scorestest)]  
    model = RandomForestClassifier(n_estimators=best_estimator)  
    model.fit(x_train, y_train)  
    testAcc.append(model.score(x_test, y_test))  
    trainAcc.append(model.score(x_train, y_train))  
    print("Training accuracy", model.score(x_train, y_train))  
    print("Test accuracy", model.score(x_test, y_test))  
      
    plt.Figura(2)  
    plt.plot(range(1, len(scorestrain) + 1), scorestrain)  
    plt.plot(range(1, len(scorestest) + 1), scorestest)  
    plt.gcf().set_size_inches(21, 14)  
      
    #SVM  
      
    model = SVC(kernel='rbf', probability=True,gamma='scale')  
    model.fit(x_train, y_train)  
      
    print("Training accuracy", model.score(x_train, y_train))  
    print("Test accuracy", model.score(x_test, y_test))  
      
    testAcc.append(model.score(x_test, y_test))  
    trainAcc.append(model.score(x_train, y_train))  
      
      
    #Linear SVM  
    model = SVC(kernel='linear', probability=True)  
    model.fit(x_train, y_train)  
      
    print("Training accuracy", model.score(x_train, y_train))  
    print("Test accuracy", model.score(x_test, y_test))  
    testAcc.append(model.score(x_test, y_test))  
    trainAcc.append(model.score(x_train, y_train))  
      
    # SVM with RBF kernel, multiple values for gamma  
      
    scorestest = []  
    scorestrain = []  
    estimators = []  
    for gamma in [0.00000000001,0.0000000001,0.000000001,0.00000001,0.0000001,0.000001,0.00001,0.0001,0.001, 0.01,0.1, 1, 10, 100]:  
        
      model = SVC(kernel='rbf', probability=True, gamma=gamma)  
      model.fit(x_train, y_train)  
        
       # TODO Check the test accuracy of your model  
      scoretest = model.score(x_test, y_test)  
      scoretrain = model.score(x_train, y_train)  
      # TODO add the score to the list  
      
      scorestest.append(scoretest)  
      scorestrain.append(scoretrain)  
      estimators.append(gamma)  
       
    best_estimator = estimators[np.argmax(scorestest)]  
    model = SVC(kernel='rbf', probability=True, gamma=best_estimator)  
    model.fit(x_train, y_train)  
    testAcc.append(model.score(x_test, y_test))  
    trainAcc.append(model.score(x_train, y_train))  
    print("Training accuracy", model.score(x_train, y_train))  
    print("Test accuracy", model.score(x_test, y_test))  
      
    plt.Figura(3)  
    plt.plot(range(1, len(scorestrain) + 1), scorestrain)  
    plt.plot(range(1, len(scorestest) + 1), scorestest)  
    plt.gcf().set_size_inches(21, 14)  
      
      
    #SVM quadratic 
     
    model =  SVC(kernel='poly', degree = 2, C=1.0, decision_function_shape = 'ovo',gamma='scale')  
    model.fit(x_train, y_train)  
      
    print("Training accuracy", model.score(x_train, y_train))  
    print("Test accuracy", model.score(x_test, y_test))  
      
    testAcc.append(model.score(x_test, y_test))  
    trainAcc.append(model.score(x_train, y_train))  
      
    # NN-Neural Network  
    from keras.utils import to_categorical  
    y=to_categorical(y);  
    x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=0)  
    from keras import Sequential  
    from keras.strats import Dense  
    from keras.strats import Dropout  
    model = Sequential()  
    model.add(Dense(200, activation='relu',input_dim=6, kernel_initializer='random_normal'))  
    model.add(Dense(100, activation='relu', kernel_initializer='random_normal'))  
    model.add(Dense(80, activation='relu', kernel_initializer='random_normal'))  
    model.add(Dense(80, activation='relu', kernel_initializer='random_normal'))  
    model.add(Dense(80, activation='relu', kernel_initializer='random_normal'))  
    model.add(Dense(2, activation='softmax', kernel_initializer='random_normal'))  
    model.compile( optimizer='adam',loss='categorical_crossentropy', metrics=['accuracy'])  
    model.summary()  
    history=model.fit(x_train,y_train, validation_split=0.10, batch_size=15, epochs=10)  
    _, train_acc = model.evaluate(x_train, y_train, verbose=0)  
    _, test_acc = model.evaluate(x_test, y_test, verbose=0)  
    testAcc.append(test_acc)  
    trainAcc.append(train_acc)  
      
    type_clasif=['polynomial features', 'logistic regresion', 'k neighbor', 'DecisionTreeClassifier', 'Random forest', 'SVM' ,'Linear SVM', 'SVM gamma','SVM quadratic','NN']  
    best_score_test=type_clasif[np.argmax(testAcc)]  
    best_score_train=type_clasif[np.argmax(trainAcc)]  
    # Salvarea valorilor într-un dicționar    
    dictpat=dict()  
    dictpat['type_clasif']=type_clasif;  
    dictpat['Best_score_test']=best_score_test;  
    dictpat['testAcc']=testAcc;  
    dictpat['Best_score_train']=best_score_train;  
    dictpat['trainAcc']=trainAcc;  
    return(dictpat)  
