import pandas as pd
from sklearn.svm import SVC
import numpy as np
from sklearn import metrics
from sklearn.decomposition import KernelPCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA

def get_data():
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")
    targets = train.Activity
    labels = test.Activity
    test.drop('Activity',1,inplace=True)
    train.drop('Activity',1,inplace=True)
    return train,targets,test,labels
    
def predict_svm(X_train, X_test, y_train, y_test):
    svc=SVC(kernel='linear') 
    svc.fit(X_train,y_train)
    y_pred=svc.predict(X_test)
    print('Accuracy Score with svm:')
    print(metrics.accuracy_score(y_test,y_pred))
    
def feature_engineering(train,test):
    train=train.append(test)    
    kpca = KernelPCA(kernel="rbf", fit_inverse_transform=True, gamma=10)
    train = kpca.fit_transform(train)
    #ipca = PCA(n_components=562)
    #ipca.fit(train)
    #train=ipca.transform(train)
    X1=train[0:7352]
    X2=train[7352:]
    return X1,X2
    
def feature_selection(train,test):
    train=np.concatenate((train,test))
    sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
    train=sel.fit_transform(train)
    X1=train[0:7352]
    X2=train[7352:]
    return X1,X2
    
train,targets,test,labels=get_data()
train,test=feature_engineering(train,test)
predict_svm(train,test,targets,labels)