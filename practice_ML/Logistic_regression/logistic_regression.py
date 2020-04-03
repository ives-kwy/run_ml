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
        theta.shape = [-1,X.shape[1]]
        hypothesis = self._sigmoid(np.dot(X, theta.T))
        first = y * np.log(hypothesis)
        second = (1 - y) * np.log(1 - hypothesis)
        length = len(y)

        if self.regularization is None:
            reg_term = 0
        else:
            reg_term = self.regularization / (2 * length) * np.sum(np.power(theta[:,1:], 2))
        return np.sum((-first) + (-second)) / length + reg_term

    def gradient(self, theta, X, y):
        parameters = theta.shape[1]
        grad = np.zeros([1,X.shape[1]])
        hypothesis = self._sigmoid(np.dot(X, theta.T))
        length = len(y)

        error = hypothesis - y

        for i in range(parameters):
            term = error * X[:,i].reshape(length,1)
            if self.regularization is None:
                grad[:,i] = np.sum(term,axis=0) / length
            else:
                if i == 0:
                    grad[:,i] = np.sum(term,axis=0) / length
                else:
                    grad[:,i] = np.sum(term,axis=0) / length + (self.regularization / length) * grad[:,i]

        return grad

    def predict(self, X):
        probability = self._sigmoid(np.dot(X, self.theta.T))
        return [1 if x >= 0.5 else 0 for x in probability]

    def optimize(self, X, y):
        result = opt.fmin_tnc(func=self.cost, x0=self.theta, fprime=self.gradient, args=(X,y))
        self.theta = result[0]
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