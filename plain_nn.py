# Package imports
import sklearn.linear_model
from planar_utils import load_data
from nn_utils import predict, L_layer_model, layer_sizes
from plot_utils import *

# Load Dataset & Visualize
X, Y = load_data("planar")
plot_planar_data(X, Y)

# Get the training data size
shape_X = X.shape
shape_Y = Y.shape
m = np.size(X, axis = 1)

print ('The shape of X is: ' + str(shape_X))
print ('The shape of Y is: ' + str(shape_Y))
print ('I have m = %d training examples!' % (m))

# Train the logistic regression classifier
clf = sklearn.linear_model.LogisticRegressionCV();
clf.fit(X.T, Y.T);
# Print accuracy
LR_predictions = clf.predict(X.T)
print ('Accuracy of logistic regression: %d ' % float((np.dot(Y,LR_predictions) + np.dot(1-Y,1-LR_predictions))/float(Y.size)*100) +
       '% ' + "(percentage of correctly labelled datapoints)")
plot_linear_regression(clf, X, Y)

# Train with NN model
learning_rate = 1.2
activation = "tanh"
(n_x, n_y) = layer_sizes(X, Y)
n_h = 4
# no. of layers & corresponding layer sizes
layers_dims = [n_x, n_h, n_y]
parameters, costs = L_layer_model(X, Y, layers_dims, activation, learning_rate, num_iterations = 10000, print_cost = True)
# plot the cost
plot_cost(costs, "Learning rate =" + str(learning_rate))
# Print accuracy
predictions = predict(parameters, X, activationfunc="tanh")
print ('Accuracy: %d' % float((np.dot(Y,predictions.T) + np.dot(1-Y,1-predictions.T))/float(Y.size)*100) + '%')
# Plot the decision boundary
title = "Decision Boundary for hidden layer size " + str(4)
plot_plain_nn(parameters, activation, X, Y, title)


