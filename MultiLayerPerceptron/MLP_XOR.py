# Author: H. Benhabiles
# Version date: January 2022
""" MLP or fully connected neural network with dynamic Structure developped from scratch
This code example is provided in the frame of the Advanced Machine Learning JUNIA-M1 course.
It demonstrates how to build an MLP and code it from scratch. The code permits to solve logic functions but
can be easily adapted to solve non-linear separated data. It offers the possibility to build different MLP structures by
changing dynamically some hyperparameters: the depth of the MLP (number of hidden layers),
the size of each layer (number of neurones), activation functions, optimizers and kernel weights initializers.
"""
import numpy as np
import math


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def relu(x):
    if x < 0:
        return 0
    else:
        return 1


def tanh(x):
    return math.tanh(x)


sigmoid_v = np.vectorize(sigmoid)
relu_v = np.vectorize(relu)
tanh_v = np.vectorize(tanh)
sqrt_v = np.vectorize(math.sqrt)
log_v = np.vectorize(math.log)


def activation_fct(x, name):
    if name == 'sigmoid':
        return sigmoid_v(x)
    elif name == 'relu':
        return relu_v(x)
    elif name == 'tanh':
        return tanh_v(x)


def der_activation_fct(x, name):
    if name == 'sigmoid':
        return x * (1 - x)
    elif name == 'relu':
        return relu_v(x)  # np.full(x.shape,1)
    elif name == 'tanh':
        return 1 - (x * x)


# loss for classification problem with softmax
def entropy_fct(c, o):
    loss = 0
    for i in range(c.shape[0]):
        if c[i] == 0:
            loss = loss + abs(math.log((1 - o[i]) + 0.00000001))  # add epsilon to avoid undefined values
        else:
            loss = loss + abs(math.log(o[i] + 0.00000001))

    return loss / c.shape[0]


# change the order of appearance of the 4 examples
def toshuffle(X, C, shuffle):
    if shuffle:
        # shuffle X and C same way
        randomize = np.arange(len(C))
        np.random.shuffle(randomize)
        X = X[randomize]
        C = C[randomize]

    return X, C


def model_setinput(inputdim, dense, activation, initializer):
    model = []
    # first hidden layer (weights and baias)
    # W = np.random.uniform(-w_range,w_range,[inputdim,dense])

    if initializer == 'normal':
        W = np.random.normal(0, 1, [inputdim, dense])
        B = np.zeros([dense, 1])
    elif initializer == 'constant':
        W = np.full((inputdim, dense), 0.05)
        B = np.full((dense, 1), 0.05)
    elif initializer == 'uniform':
        W = np.random.uniform(-0.05, 0.05, [inputdim, dense])
        B = np.random.uniform(-0.05, 0.05, [dense, 1])
    # elif initializer == 'glorot': #### to do

    # associate weights to the dense layer
    dense_weight = []
    dense_weight.append(W)
    dense_weight.append(B)
    dense_weight.append(activation)

    model.append(dense_weight)

    return model


def model_addlayer(model, dense, activation, initializer):
    dense_previous = model[-1][1].shape[0]  # size of length of last hidden layer
    # hidden layer (weights and baias)
    # W = np.random.uniform(-w_range,w_range,[dense_previous,dense])
    if initializer == 'normal':
        W = np.random.normal(0, 1, [dense_previous, dense])
        B = np.zeros([dense, 1])
    elif initializer == 'constant':
        W = np.full((dense_previous, dense), 0.5)
        B = np.full((dense, 1), 0.5)
    elif initializer == 'uniform':
        W = np.random.uniform(-0.5, 0.5, [dense_previous, dense])
        B = np.random.uniform(-0.5, 0.5, [dense, 1])

    # associate weights to the dense layer
    dense_weight = []
    dense_weight.append(W)
    dense_weight.append(B)
    dense_weight.append(activation)

    model.append(dense_weight)

    return model


def model_sgd(model, gradient_list, activation_list, lr, x):
    # need to update first layer of weights alone since linked to input data.
    gradient = gradient_list[0].T * x.T
    model[0][0] = model[0][0] - lr * gradient  # weights
    model[0][1] = model[0][1] - lr * gradient_list[0]  # baias

    for layer_index in range(1, len(model)):
        gradient = gradient_list[layer_index].T * activation_list[layer_index - 1]
        model[layer_index][0] = model[layer_index][0] - lr * gradient  # weights
        model[layer_index][1] = model[layer_index][1] - lr * gradient_list[layer_index]  # baias

    return model


def model_adam(model, gradient_list, m, v, lr):
    beta_1 = 0.9
    beta_2 = 0.999
    epsilon = 1 / pow(10, 8)

    if not m and not v:
        # define m and v with same shape as gradient_list and set their units to 0
        for i in range(len(gradient_list)):
            m.append(np.zeros(gradient_list[i].shape))
            v.append(np.zeros(gradient_list[i].shape))

    # calculate moments and update weights
    for layer_index in range(len(gradient_list)):
        m[layer_index] = beta_1 * m[layer_index] + (1 - beta_1) * gradient_list[layer_index]
        v[layer_index] = beta_2 * v[layer_index] + (
                    (1 - beta_2) * gradient_list[layer_index] * gradient_list[layer_index])
        adam_gradient = (lr * m[layer_index]) / (sqrt_v(v[layer_index]) + epsilon)
        model[layer_index][0] = model[layer_index][0] - adam_gradient.T  # weights
        model[layer_index][1] = model[layer_index][1] - adam_gradient  # baias

    return model, m, v


def model_train(name, model, X, C, optimizer, lr, loss_fct, num_epochs, shuffle):
    solved = False

    # set initially to positive infinity, we suppose that the best error at this stage is too high
    best_error = float('inf')

    if optimizer == 'adam':
        m = []
        v = []
    for epochs in range(num_epochs):

        # We may shuffle the data if required
        X, C = toshuffle(X, C, shuffle)

        for example_index in range(len(C)):
            x = X[example_index].reshape(1, 2)

            # setup activation and gradient lists.
            activation_list = []
            gradient_list = []

            ######## feed pass or propagation #############
            W = model[0][0]
            B = model[0][1]
            transfer = x.dot(W).T + B  # first hidden layer
            activation = activation_fct(transfer, model[0][2])
            activation_list.append(activation)

            for layer_index in range(1, len(model)):  # next layers

                # calculate activation function
                W = model[layer_index][0]
                B = model[layer_index][1]
                transfer = activation_list[-1].T.dot(W).T + B  # next hidden layer
                activation = activation_fct(transfer, model[layer_index][2])
                activation_list.append(activation)

            output = activation_list[-1]
            if loss_fct == 'mse':
                loss = abs(C[example_index].reshape(1, 1) - output)
            elif loss_fct == 'entropy':
                if C[example_index] == 0:
                    loss = abs(math.log(1 - output + 0.00000001))
                else:
                    loss = abs(math.log(output + 0.00000001))

            gradient_output = der_activation_fct(output, model[-1][2]).dot(loss)

            gradient_list.append(gradient_output)

            ######## backpropagation #############
            for layer_index in range(len(model) - 1, 0, -1):
                W = model[layer_index][0]
                gradient = der_activation_fct(activation_list[layer_index - 1], model[layer_index - 1][2]) * (
                    W.dot(gradient_list[-1]))
                gradient_list.append(gradient)

            ######## update weights #############
            # gradient_list.reverse() to align with layers orders
            gradient_list.reverse()

            ######## optimizer call to update weights #############
            if optimizer == 'sgd':
                model = model_sgd(model, gradient_list, activation_list, lr, x)
            elif optimizer == 'adam':
                model, m, v = model_adam(model, gradient_list, m, v, lr)

        # prepare saving the best model
        O = model_predict(model, X, False)

        if loss_fct == 'mse':
            # calculate MSE the Mean Squared Error
            error = np.average(pow((C - O), 2))
        elif loss_fct == 'entropy':
            error = entropy_fct(C, O)

        if error < best_error:
            best_error = error
            # save model (weights and baias)
            best_model = model

        if error > 1:  # if big error stop iterating over epochs, it will not converge
            break

        print("epoch, ", loss_fct, ": ", epochs, error)

        if np.average(pow((C - np.round(O)), 2)) == 0:
            print(name, "solved!")
            print(epochs + 1, " epoch(s)")
            print("C,O ", C, O)
            solved = True
            break
    return solved, best_model


def model_predict(model, X, toround):
    O = np.empty(X.shape[0])
    for example_index in range(len(X)):

        x = X[example_index].reshape(1, 2)

        # setup activation
        activation_list = []

        ######## propagation #############
        W = model[0][0]
        B = model[0][1]
        transfer = x.dot(W).T + B  # first hidden layer
        activation = activation_fct(transfer, model[0][2])
        activation_list.append(activation)

        for layer_index in range(1, len(model)):  # hidden layers
            # calculate activation function
            W = model[layer_index][0]
            B = model[layer_index][1]
            transfer = activation_list[-1].T.dot(W).T + B  # next hidden layer
            activation = activation_fct(transfer, model[layer_index][2])
            activation_list.append(activation)

        O[example_index] = activation_list[-1]

    if (toround):
        return np.round(O)
    else:
        return O


############# call MLP train function #################

run = 0
X = np.array(([0, 0], [0, 1], [1, 0], [1, 1]), dtype=int)  # inputs
C = np.array((0, 1, 1, 0), dtype=int)  # resultat d'un XOR
lr = 0.001
num_epochs = 50
max_run = 0

solved = False
while (True):  # max_run < 25

    max_run = max_run + 1
    print("====Run {}====".format(max_run))
    # Build a new model
    model = model_setinput(2, 8, 'relu', initializer='normal')
    model = model_addlayer(model, 4, 'relu', initializer='normal')
    model = model_addlayer(model, 2, 'relu', initializer='normal')
    model = model_addlayer(model, 1, 'relu', initializer='normal')
    solved, best_model = model_train("XOR", model, X, C, 'sgd', lr, 'mse', num_epochs, shuffle=False)
    if solved:
        print("solved :) ---- Congratulation")
        break

print(model)

# Call prediction function of the model
prob = model_predict(best_model, X, False)
print('Prediction probs {}'.format(prob))
dec = model_predict(best_model, X, True)
print('Model decision {}'.format(dec))