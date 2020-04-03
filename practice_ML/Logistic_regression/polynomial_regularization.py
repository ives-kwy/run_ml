# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from logistic_regression import LogisticRegression

def Cal_accuracy(predictions, y):
    correct = [1 if ((a == 1 and b == 1) or (a == 0 and b == 0)) else 0 for (a, b) in zip(predictions, y)]
    accuracy = (sum(map(int, correct)) % len(correct))
    return accuracy

def Call_myLRmodel(X, y):
    theta = np.zeros([1,X.shape[1]])

    model = LogisticRegression(theta, regularization=1)
    result = model.optimize(X, y)
    cost = model.cost(result[0], X, y)
    print(result, cost)

    predictions = model.predict(X)
    accuracy = Cal_accuracy(predictions, y)

    return cost, accuracy

def Call_SklearnLR(X, y):
    model = linear_model.LogisticRegression(penalty='l2', C=1.0)

    model.fit(X, y.ravel())
    score = model.score(X, y.ravel())
    predictions = model.predict(X)
    accuracy = Cal_accuracy(predictions, y)

    return score, accuracy

if __name__ == '__main__':
    path = 'ex2data2.txt'
    data = pd.read_csv(path, header=None, names=['Test 1', 'Test 2', 'Accepted'])
    print(data.head())

    '''
    # base on the data to choose the model fo Logistic Regression
    positive = data[data['Accepted'].isin([1])]
    negative = data[data['Accepted'].isin([0])]

    fig, ax = plt.subplots(figsize=(12,8))
    ax.scatter(positive['Test 1'], positive['Test 2'], s=50, c='b', marker='o', label='Accepted')
    ax.scatter(negative['Test 1'], negative['Test 2'], s=50, c='r', marker='x', label='Rejected')
    ax.legend()
    ax.set_xlabel('Test 1 Score')
    ax.set_ylabel('Test 2 Score')
    plt.show()
    '''

    # According to the original data, a linear decision boundary cannot be found, so a polynomial is introduced.
    # Constructing polynomial features from raw data
    degree = 5
    x1 = data['Test 1']
    x2 = data['Test 2']

    data.insert(3, 'Ones', 1)

    for i in range(1, degree):
        for j in range(0, i):
            data['F' + str(i) + str(j)] = np.power(x1, i-j) * np.power(x2, j)

    data.drop('Test 1', axis=1, inplace=True)
    data.drop('Test 2', axis=1, inplace=True)

    print(data.head())

    # initial training data
    # set X and y (remember from above that we moved the label to column 0)
    cols = data.shape[1]
    X = data.iloc[:,1:]
    y = data.iloc[:,0:1]

    # convert to numpy arrays and initalize the parameter array theta
    X = np.array(X.values)
    y = np.array(y.values)
    
    score_1, accuracy_1 = Call_myLRmodel(X, y)
    score_2, accuracy_2 = Call_SklearnLR(X, y)

    dict = [{'Score':score_1, 'Accuracy':accuracy_1},
            {'Score':score_2, 'Accuracy':accuracy_2}]
    df = pd.DataFrame(dict)

    print(df)