import matplotlib.pyplot as plt
import numpy as np
from nn_utils import predict

def plot_planar_data(X, Y):
    plt.scatter(X[0, :], X[1, :], c=Y, s=40, cmap=plt.cm.Spectral);
    plt.show()

def plot_decision_boundary(model, X, y):
    # Set min and max values and give it some padding
    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole grid
    Z = model(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.scatter(X[0, :], X[1, :], c=y, cmap=plt.cm.Spectral)

def plot_linear_regression(model, X, Y):
    plot_decision_boundary(lambda x: model.predict(x), X, Y)
    plt.title("Logistic Regression")
    plt.show()

def plot_cost(costs, title) :
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title(title)
    plt.show()

def plot_plain_nn(parameters, activation, X, Y, title):
    plot_decision_boundary(lambda x: predict(parameters, x.T, activation), X, Y)
    plt.title(title)
    plt.show()

