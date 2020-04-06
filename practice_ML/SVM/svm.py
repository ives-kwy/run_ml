# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from scipy.io import loadmat

def LinearData():
    raw_data = loadmat('data/ex6data1.mat')

    data = pd.DataFrame(raw_data['X'], columns=['X1', 'X2'])
    data['y'] = raw_data['y']

    print(data.head())

    from sklearn import svm
    svc = svm.LinearSVC(C=1, loss='hinge', max_iter=500)
    svc.fit(data[['X1', 'X2']], data['y'])
    l_score_1 = svc.score(data[['X1', 'X2']], data['y'])
    print('For linear DataSet, C=1 the score of SVM is ' + str(l_score_1))

    svc2 = svm.LinearSVC(C=100, loss='hinge', max_iter=2000)
    svc2.fit(data[['X1', 'X2']], data['y'])
    l_score_2 = svc2.score(data[['X1', 'X2']], data['y'])
    print('For linear DataSet, C=100 the score of SVM is ' + str(l_score_2))

def NonlinearData():
    raw_data = loadmat('data/ex6data2.mat')

    data = pd.DataFrame(raw_data['X'], columns=['X1', 'X2'])
    data['y'] = raw_data['y']

    print(data.head())

    from sklearn import svm
    svc = svm.SVC(C=100, gamma=10, probability=True)
    svc.fit(data[['X1', 'X2']], data['y'])
    n_score = svc.score(data[['X1', 'X2']], data['y'])
    print('For non-linear DataSet, the score of SVM is ' + str(n_score))

if __name__ == '__main__':
    LinearData()
    NonlinearData()