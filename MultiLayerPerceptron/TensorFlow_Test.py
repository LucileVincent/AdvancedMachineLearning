# XOR exercice

# Import necessary packages
import tensorflow as tf
import numpy as np
from datetime import datetime

# Specify Training base S (X:x_train,Y:y_train)
x_train = np.array([[0,0],[0,1],[1,0],[1,1]])
y_train = np.array([[0],[1],[1],[0]])

# Create a sequential model (neural network architecture)
model = tf.keras.Sequential()

# Building the architecture of our model by adding layers

# Dense(5) is a fully-connected layer with 5 hidden units.
# in the first layer, you must specify the expected input data shape :
# here, vector of 2 inputs.
model.add(tf.keras.layers.Dense(5, activation='relu', input_dim=2))
model.add(tf.keras.layers.Dense(3, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

# relu : Rectified Linear Unit with default values, it returns element-wise max(x, 0)
# Specifying how to update the weights → optimizers

# we want to use stochastic gradient descent with learning rate = 0.02
sgd = tf.keras.optimizers.SGD(learning_rate=0.02)
# for each weight wij [forwardpropagation] :
#     wij=wij-lr*(c[i]-theta)*entry[j]

# Compile the model: Set loss function, optimizer and evaluation metrics
model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['mean_squared_error'])

# Set up TensorBoard
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# Train the model on data: training base S
model.fit(x_train, y_train, batch_size=1, epochs=20)

# Use predict and round functions of your model
print(model.predict(x_train))
print(model.predict(x_train).round())

