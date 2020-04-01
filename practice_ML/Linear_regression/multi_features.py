import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from linear_regression import Linear_regression


def Call_myLRmodel(data):
    # add ones column
    data.insert(0, 'Ones', 1)

    # set X (training data) and y (target variable)
    cols = data.shape[1]
    X = data.iloc[:,0:cols-1]
    y = data.iloc[:,cols-1:cols]

    # convert to matrices and initialize theta
    X = np.array(X.values)
    y = np.array(y.values)
    theta = np.zeros([1,cols-1])
    #
    alpha = 0.01
    iters = 1000
    model = Linear_regression(theta, alpha)

    # perform linear regression on the data set
    g, cost = model.gradientDescent(X, y, iters)

    # get the cost (error) of the model
    all_cost = model.computeCost(X, y, g)

    '''
    # show the curve of Cost
    fig, ax = plt.subplots(figsize=(12,8))
    ax.plot(np.arange(iters), cost, 'r')
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Cost')
    ax.set_title('Error vs. Training Epoch')
    plt.show()
    '''
    return g, cost, all_cost

def Call_SklearnLR(data):
    # Using sklearn
    from sklearn import linear_model

    cols = data.shape[1]

    # the sklearn_parameters
    X_sk = data.iloc[:,0:cols-1]
    y_sk = data.iloc[:,cols-1:cols]

    X_sk = np.array(X_sk.values)
    y_sk = np.array(y_sk.values)

    model_sk = linear_model.LinearRegression()
    model_sk.fit(X_sk, y_sk)

    return model_sk.coef_, model_sk.score(X_sk, y_sk)

if __name__ == "__main__":
    path = 'ex1data2.txt'
    data = pd.read_csv(path, header=None, names=['Size', 'Bedrooms', 'Price'])
    print(data.head())

    #Make sure features are on a similar scale.
    #Which make the gradient descent faster
    data = (data - data.mean()) / data.std()
    print(data.head())

    Para_1, _, score_1 = Call_myLRmodel(data)
    Para_2, score_2    = Call_SklearnLR(data)

    dict = [{'Parameter':Para_1[:,1:], 'Score':score_1},
            {'Parameter':Para_2[:,1:], 'Score':score_2}]
    df = pd.DataFrame(dict)

    print(df)