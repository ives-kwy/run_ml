# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def forward_propagates(X, theta):
    a = []
    z = []
    a.append(X) # a[0].shape = (m, n)

    for i in range(len(theta)):
        a[i] = np.insert(a[i], 0, values=1, axis=1) # a[0].shape = (m, n+1 or hidden_units + 1)
        z.append(np.dot(a[i], theta[i].T)) # z.shape = (m, hidden_units or outputs)
        a.append(sigmoid(z[-1])) # a.shape = (m, hidden_units or outputs)

    return z, a

def cost(params, input_size, hidden_size, num_labels, X, y, regularization):
    m = len(X)

    # reshape the parameter array into parameter matrices for each layer
    theta1 = np.reshape(params[:hidden_size * (input_size + 1)], (hidden_size, (input_size + 1)))
    theta2 = np.reshape(params[hidden_size * (input_size + 1):], (num_labels, (hidden_size + 1)))

    z, a = forward_propagates(X, [theta1, theta2])

    # compute the cost
    first_term = (-y) * np.log(a[-1])
    second_term = - (1 - y) * np.log(1 - a[-1])
    J = np.sum(first_term + second_term) / m

    # add the regularization cost term
    J += regularization / (2 * m) * (np.sum(np.power(z[0], 2)) + np.sum(np.power(z[1], 2)))

    return J

def sigmoid_gradient(z):
    return sigmoid(z) * (1 - sigmoid(z))

def backprop(params, input_size, hidden_size, num_labels, X, y, regularization):
    m = len(X)

    # reshape the parameter array into parameter matrices for each layer
    theta1 = np.reshape(params[:hidden_size * (input_size + 1)], (hidden_size, (input_size + 1)))
    theta2 = np.reshape(params[hidden_size * (input_size + 1):], (num_labels, (hidden_size + 1)))
    # initializations
    delta1 = np.zeros(theta1.shape)  # (25, 401)
    delta2 = np.zeros(theta2.shape)  # (10, 26)

    z, a = forward_propagates(X, [theta1, theta2])

    # compute the cost
    first_term = (-y) * np.log(a[-1])
    second_term = - (1 - y) * np.log(1 - a[-1])
    J = np.sum(first_term + second_term) / m

    # add the regularization cost term
    J += regularization / (2 * m) * (np.sum(np.power(z[0], 2)) + np.sum(np.power(z[1], 2)))

    # perform backpropagation
    for t in range(m):
        a1t = a[0][t,:].reshape(1,-1)  # (1, 401)
        z2t = z[0][t,:].reshape(1,-1)  # (1, 25)
        a2t = a[1][t,:].reshape(1,-1)  # (1, 26)
        ht = a[2][t,:].reshape(1,-1)  # (1, 10)
        yt = y[t,:].reshape(1,-1)  # (1, 10)

        d3t = ht - yt  # (1, 10)

        z2t = np.insert(z2t, 0, values=np.ones(1))  # (1, 26)
        d2t = np.dot(theta2.T, d3t.T).T * sigmoid_gradient(z2t)  # (1, 26)

        delta1 = delta1 + np.dot((d2t[:,1:]).T, a1t)
        delta2 = delta2 + np.dot(d3t.T, a2t)

    delta1 = delta1 / m
    delta2 = delta2 / m

    # add the gradient regularization term
    delta1[:,1:] = delta1[:,1:] + (theta1[:,1:] * regularization) / m
    delta2[:,1:] = delta2[:,1:] + (theta2[:,1:] * regularization) / m

    # unravel the gradient matrices into a single array
    grad = np.concatenate((np.ravel(delta1), np.ravel(delta2)))

    return J, grad


if __name__ == '__main__':
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from scipy.io import loadmat

    data = loadmat('ex4data1.mat')
    print(data)

    X = data['X']
    y = data['y']

    # Using One-hot for y
    from sklearn.preprocessing import OneHotEncoder
    encoder = OneHotEncoder(sparse=False)
    y_onehot = encoder.fit_transform(y)

    # inital the parameters
    input_size = 400
    hidden_size = 25
    num_labels = 10
    regularization = 0

    # random inital the theta
    params = (np.random.random(size=hidden_size * (input_size + 1) + num_labels * (hidden_size + 1)) - 0.5) * 0.25

    theta1 = np.reshape(params[:hidden_size * (input_size + 1)], (hidden_size, (input_size + 1)))
    theta2 = np.reshape(params[hidden_size * (input_size + 1):], (num_labels, (hidden_size + 1)))
    z, a = forward_propagates(X, (theta1, theta2))
    print(a[0].shape, z[0].shape, a[1].shape, z[1].shape, a[2].shape)

    cost = cost(params, input_size, hidden_size, num_labels, X, y_onehot, regularization)
    print(cost)

    J, grad = backprop(params, input_size, hidden_size, num_labels, X, y_onehot, regularization)
    print(J, grad.shape)

    from scipy.optimize import minimize

    # minimize the objective function
    fmin = minimize(fun=backprop, x0=params, args=(input_size, hidden_size, num_labels, X, y_onehot, regularization),
                    method='TNC', jac=True, options={'maxiter': 250})
    print(fmin)

    theta1_min = np.reshape(fmin.x[:hidden_size * (input_size + 1)], (hidden_size, (input_size + 1)))
    theta2_min = np.reshape(fmin.x[hidden_size * (input_size + 1):], (num_labels, (hidden_size + 1)))

    z, a = forward_propagates(X, [theta1_min, theta2_min])

    y_pred = np.argmax(a[-1], axis=1) + 1

    correct = [1 if a == b else 0 for (a, b) in zip(y_pred, y)]
    accuracy = (sum(map(int, correct)) / float(len(correct)))
    print ('accuracy = {0}%'.format(accuracy * 100))