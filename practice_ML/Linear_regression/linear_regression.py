import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Linear_regression:

    def __init__(self, theta, alpha):
        self.theta = theta
        self.alpha = alpha

    def computeCost(self, X, y, theta=None):
        inner = np.power((np.dot(X, theta.T) - y), 2)
        return np.sum(inner) / (2 * len(X))

    def gradientDescent(self, X, y, iters):
        cost = np.zeros([iters,1])
        assert self.theta.shape[1] == X.shape[1], "The axis=1's shape of theta and X should be same"
        temp = np.zeros_like(self.theta)
        parameters = self.theta.shape[1]
        length = len(X)

        for i in range(iters):
            error = np.dot(X, self.theta.T) - y

            for j in range(parameters):
                term = error * X[:, j].reshape(length,1)
                temp[0,j] = self.theta[0,j] - self.alpha / length * np.sum(term)

            self.theta = temp
            cost[i] = self.computeCost(X, y, self.theta)

        return self.theta, cost

    def predict(self, X):
        return np.dot(X, self.theta.T)


if __name__ == "__main__":
    path = 'ex1data1.txt'
    print(os.getcwd())
    data = pd.read_csv(path, header=None, names=['Population', 'Profit'])
    print(data.head())
    print(data.describe())
    data.plot(kind='scatter', x='Population', y='Profit', figsize=(12,8))
    #plt.show()

    data.insert(0, 'Ones', 1)
    print(data.head())

    columns = data.shape[1]
    X = data.iloc[:,0:columns-1]#X是所有行，去掉最后一列
    y = data.iloc[:,columns-1:columns]#y是所有行，最后一列

    print(X.head(),y.head())

    X = np.array(X.values)
    y = np.array(y.values)

    alpha = 0.01
    iters = 1000
    theta = np.zeros([1, X.shape[1]])

    print(X.shape, theta.shape, y.shape)

    model = Linear_regression(theta, alpha)
    theta, cost = model.gradientDescent(X, y, iters)

    print(theta, cost[-1])