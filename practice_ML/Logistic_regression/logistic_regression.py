# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
import scipy.optimize as opt

class LogisticRegression:
    def __init__(self, theta, regularization=None):
        self.regularization = regularization
        self.theta = theta

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def cost(self, theta, X, y):
        #print(X.shape, y.shape, theta.shape)
        theta.shape = [1,3]
        hypothesis = self._sigmoid(np.dot(X, theta.T))
        first = y * np.log(hypothesis)
        second = (1 - y) * np.log(1 - hypothesis)

        return np.sum((-first) + (-second)) / len(y)

    def gradient(self, theta, X, y):
        parameters = theta.shape[1]
        grad = np.zeros([1,X.shape[1]])
        hypothesis = self._sigmoid(np.dot(X, theta.T))
        length = len(y)

        error = hypothesis - y

        for i in range(parameters):
            term = error * X[:,i].reshape(length,1)
            grad[:,i] = np.sum(term,axis=0) / length

        return grad

    def optimize(self, X, y):
        result = opt.fmin_tnc(func=self.cost, x0=self.theta, fprime=self.gradient, args=(X,y))
        return result

if __name__ == "__main__":
    path = 'ex2data1.txt'
    data = pd.read_csv(path, header=None, names=['Exam 1', 'Exam 2', 'Admitted'])
    print(data.head())

    data.insert(0, 'Ones', 1)

    cols = data.shape[1]
    #X = data.iloc[:, 0:cols-1]
    #y = data.iloc[:, cols-1:cols]
    X = data.iloc[:, 0:-1]
    y = data.iloc[:, -1:]

    X = np.array(X.values)
    y = np.array(y.values)
    theta = np.zeros([1, cols-1])

    model = LogisticRegression(theta, None)
    result = model.optimize(X, y)
    print(result)

    print(model.cost(result[0], X, y))