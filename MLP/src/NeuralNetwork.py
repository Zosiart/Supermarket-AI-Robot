# Note: you are free to organize your code in the way you find most convenient.
# However, make sure that when your main notebook is ran, it executes the steps indicated in the assignment.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def relu_activation_function(x):
    return 1 if x >= 0 else 0


# Perceptron for OR, AND
class Perceptron:
    def __init__(self, input_size, learning_rate, epochs):
        """
        Initialize a new perceptron with the given input size, learning rate, and number of epochs.

        Parameters:
            input_size (int): The number of input features.
            learning_rate (float): The learning rate to use during training (default 0.1).
            epochs (int): The maximum number of epochs to train for (default 100).

        Returns:
            None
        """
        self.weights = np.zeros(input_size + 1)
        self.learning_rate = learning_rate
        self.epochs = epochs

    def predict(self, x):
        """
        Predict the output for a given input vector x using the current weights of the perceptron.

        Parameters:
            x (ndarray): An input vector with shape (input_size,).

        Returns:
            float: The predicted output for the input vector.
        """
        z = np.dot(x, self.weights[1:]) + self.weights[0]
        a = relu_activation_function(z)
        return a

    def train(self, x_train, y_train):
        """
        Train the perceptron on the given training data.

        Parameters:
            x_train (ndarray): An array of shape (n_samples, input_size) containing the input features for each sample.
            y_train (ndarray): An array of shape (n_samples,) containing the true output values for each sample.

        Returns:
            list: A list containing the number of errors made by the perceptron during each epoch of training.
        """
        # Create an empty list to store the number of errors made during each epoch
        errors = []

        # Train the perceptron for the specified number of epochs
        for _ in range(self.epochs):
            # Initialize the error count for this epoch to 0
            error = 0

            # Iterate over each training sample and update the weights based on the perception's prediction
            for x, y in zip(x_train, y_train):
                y_pred = self.predict(x)
                delta = self.learning_rate * (y - y_pred)
                self.weights[1:] += delta * x
                self.weights[0] += delta
                error += int(delta != 0.0)

            # Add the number of errors made during this epoch to the errors list
            errors.append(error)

        # Return the list of errors made during each epoch
        return errors


# OR function
or_x_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
or_y_train = np.array([0, 1, 1, 1])
or_perceptron = Perceptron(input_size=2, learning_rate=0.1, epochs=5)
or_errors = or_perceptron.train(or_x_train, or_y_train)

# AND function
and_x_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
and_y_train = np.array([0, 0, 0, 1])
and_perceptron = Perceptron(input_size=2, learning_rate=0.1, epochs=5)
and_errors = and_perceptron.train(and_x_train, and_y_train)

# XOR function
xor_x_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
xor_y_train = np.array([0, 1, 1, 0])
xor_perceptron = Perceptron(input_size=2, learning_rate=0.1, epochs=5)
xor_errors = xor_perceptron.train(xor_x_train, xor_y_train)

plt.plot(or_errors)
plt.xlabel('Epochs')
plt.ylabel('Errors')
plt.title('Error over epochs - OR function')
plt.show()

plt.plot(and_errors)
plt.xlabel('Epochs')
plt.ylabel('Errors')
plt.title('Error over epochs - AND function')
plt.show()

plt.plot(xor_errors)
plt.xlabel('Epochs')
plt.ylabel('Errors')
plt.title('Error over epochs - XOR function')
plt.show()


# ACTIVATION FUNCTIONS

# Sigmoid function for activation function of hidden layer
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


# The sigmoid derivative
def sigmoid_derivative(x):
    return sigmoid(x) * (1.0 - sigmoid(x))


# Softmax function for activation function of output layer
def softmax(x):
    exps = np.exp(x - np.max(x))
    return exps / np.sum(exps)


# ARTIFICIAL NEURAL NETWORK CLASS
class ANN:

    def __init__(self, input_size, hidden_size, output_size, num_epochs, mini_batch_size, learning_rate):
        """
        Initializes ANN with given parameters
        :param input_size: dimensions of input data
        :param hidden_size: number of hidden neurons
        :param output_size: number of output neurons
        :param num_epochs: number of epochs
        :param mini_batch_size: size of one mini batch
        :param learning_rate: learning rate
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_epochs = num_epochs
        self.mini_batch_size = mini_batch_size
        self.learning_rate = learning_rate

        self.input_layer = None
        self.hidden_layer = None
        self.output_layer = None
        self.biases = None
        self.weights = None

        self.initialize_weight_bias()

    def initialize_weight_bias(self, seed=1):
        """
        Weights and biases initialization. Weights are assigned using samples from normal distribution with mean 0 and
        standard deviation equal to quare root of inverse of ANN's input size, biases are set to 0
        """
        self.weights = [np.random.randn(self.hidden_size, self.input_size) * np.sqrt(seed / self.input_size),
                        np.random.randn(self.output_size, self.hidden_size) * np.sqrt(seed / self.hidden_size)]
        self.biases = [np.zeros((self.hidden_size, 1)), np.zeros((self.output_size, 1))]

    def feedforward(self, input_data):
        """
        Feed forward function based on input_data
        :param input_data: data to feed forward with
        :returns output_layer
        """
        self.hidden_layer = sigmoid(np.dot(self.weights[0], input_data) + self.biases[0])
        self.output_layer = softmax(np.dot(self.weights[1], self.hidden_layer) + self.biases[1])
        self.input_layer = input_data
        return self.output_layer

    def train(self, training_set):
        """
        Training of ANN using training_set
        :param training_set: training set to use
        """
        for epoch in range(self.num_epochs):
            np.random.shuffle(training_set)
            mini_batches = []
            for index in range(0, len(training_set), self.mini_batch_size):
                mini_batch = training_set[index:index + self.mini_batch_size]
                mini_batches.append(mini_batch)

            for mini_batch in mini_batches:
                self.update_with_mini_batch(mini_batch)

    def update_with_mini_batch(self, mini_batch):
        """
        mini batching function
        :param mini_batch: mini batch to use
        """
        bias_gradients = [np.zeros(bias.shape) for bias in self.biases]
        weight_gradients = [np.zeros(weight.shape) for weight in self.weights]

        for input_data, target_data in mini_batch:
            batch_bias_gradients, batch_weight_gradients = self.backpropagation(input_data, target_data)

            bias_gradients = [bias_gradient + batch_bias_gradient
                              for bias_gradient, batch_bias_gradient
                              in zip(bias_gradients, batch_bias_gradients)]
            weight_gradients = [weight_gradient + batch_weight_gradient
                                for weight_gradient, batch_weight_gradient
                                in zip(weight_gradients, batch_weight_gradients)]

        self.biases = [bias - (self.learning_rate / len(mini_batch)) * bias_gradient for bias, bias_gradient in
                       zip(self.biases, bias_gradients)]
        self.weights = [weight - (self.learning_rate / len(mini_batch)) * weight_gradient for weight, weight_gradient in
                        zip(self.weights, weight_gradients)]

    def backpropagation(self, x, y):
        """
        Back propagation method
        :param x: input data
        :param y: target data
        :return: updated biases and weights
        """
        bias_gradients = [np.zeros(bias.shape) for bias in self.biases]
        weight_gradients = [np.zeros(weight.shape) for weight in self.weights]

        # Feedforward
        self.feedforward(x)

        # Calculate delta for output layer
        delta_output = (self.output_layer - y) * sigmoid_derivative(self.output_layer)

        # Backpropagate delta
        bias_gradients[1] = delta_output
        weight_gradients[1] = np.dot(delta_output, self.hidden_layer.T)

        delta_hidden = np.dot(self.weights[1].T, delta_output) * sigmoid_derivative(self.hidden_layer)
        bias_gradients[0] = delta_hidden
        if self.input_layer is not None:
            weight_gradients[0] = np.dot(delta_hidden, self.input_layer.transpose())

        return bias_gradients, weight_gradients

    def evaluate(self, validation_data):
        """
        Validates the network based on the validation data and return
        prediction accuracy.
        :param validation_data: data to validate model on
        :return: The prediction accuracy of the model as amount of correctly
        predicted data points / total data points.
        """
        total = len(validation_data)
        correct = 0

        for x, y in validation_data:
            output = self.feedforward(x)
            if np.argmax(output) == np.argmax(y):
                correct += 1

        return correct / total

    def predict(self, unknown_data):
        """
        Predicts the class of unknown data
        :param unknown_data: data to predict
        :return: predicted class
        """
        result = []
        for x in unknown_data:
            result.append(np.argmax(self.feedforward(x)) + 1)
        return result

    def cross_validation(self, training_data, k, validation_data):
        """
        Performs k-fold cross validation on the data.
        :param data: data to perform cross validation on
        :param k: number of folds
        :return: the average accuracy of the model
        """

        accuracies_mean = []
        accurancies_max = []
        hidden_sizes = []
        best_accuracy = 0
        best_hidden_size = 0

        for j in range(7, 30, 5):
            self.hidden_size = j
            self.initialize_weight_bias()
            hidden_sizes.append(j)
            acc = []
            for i in range(k):
                self.train(training_data)
                accuracy = self.evaluate(validation_data)
                acc.append(accuracy)
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_hidden_size = j
            accuracies_mean.append(np.mean(acc))
            accurancies_max.append(np.max(acc))

        plt.plot(hidden_sizes, accuracies_mean)
        plt.xlabel('Hidden Layer Size')
        plt.ylabel('Accuracy Mean')
        plt.title('Hidden Layer Size vs Accuracy Mean')
        plt.show()

        plt.plot(hidden_sizes, accurancies_max)
        plt.xlabel('Hidden Layer Size')
        plt.ylabel('Accuracy Max')
        plt.title('Hidden Layer Size vs Accuracy Max')
        plt.show()

        self.hidden_size = best_hidden_size
        self.initialize_weight_bias()
        self.train(data)
        return np.mean(accurancies_max)


def load_data():
    """
    Loads the input data.
    :return: (features, targets, unknown)
    """
    # read features data from csv file
    features_from_file = pd.read_csv("../data/features.txt", header=None)
    features_from_file = [np.array(row).reshape((10, 1)) for _, row in features_from_file.iterrows()]

    # read targets data from csv file
    targets_from_file = pd.read_csv("../data/targets.txt", header=None)
    targets_from_file = [vectorize_target(int(row), 7) for _, row in targets_from_file.iterrows()]

    # read unknown data from csv file
    unknown_from_file = pd.read_csv("../data/unknown.txt", header=None)
    unknown_from_file = [np.array(row).reshape((10, 1)) for _, row in unknown_from_file.iterrows()]

    return np.array(features_from_file), np.array(targets_from_file), np.array(unknown_from_file)


def vectorize_target(y, num_classes):
    """
    function for one-hoc encoding, vectorization
    :param y: target to vectorize
    :return: vectorized target format array([[0.],
       [0.],
       [1.],
       [0.],
       [0.]...])
    """
    vectorized_y = np.zeros((num_classes, 1))
    vectorized_y[y - 1] = 1.0
    return vectorized_y


def split_train_test(data, test_size, random=False):
    """
    Method that splits the data into train and test sets.
    :param data: data to split
    :param test_size: number between 0 and 1 indicating the proportion of the data to
    use for testing
    :param random: whether to shuffle the data before splitting.
    :return: (train_data, validate_data)
    """
    assert 0 <= test_size <= 1
    if random:
        np.random.shuffle(data)

    index = int(len(data) * test_size)
    return data[index:], data[:index]


# Load data
features, targets, unknown = load_data()
data = list(zip(features, targets))
# test data => 15% of total data
training_data, test_data = split_train_test(data, 0.15)
# training_data => 70.04% of total data, validation data => 14.96% of total data
training_data, validation_data = split_train_test(training_data, 0.176)

network = ANN(input_size=10, hidden_size=20, output_size=7, num_epochs=35, mini_batch_size=20, learning_rate=0.07)

# set to the data to which we are learning to 'data' because we are testing
# it on the unknown dataset and start training
network.train(training_data)

test_accuracy = network.evaluate(validation_data)
print(f"Accuracy on validation set= {test_accuracy}")

# # Cross validation
average = network.cross_validation(training_data, 10, test_data)
print(f"Average accuracy= {average}")

# # Test on test data
accuracy_test = network.evaluate(test_data)
print(f"Accuracy on test set= {accuracy_test}")

# # Test on unknown data
# network = ANN(input_size=10, hidden_size=29, output_size=7, num_epochs=35, mini_batch_size=20, learning_rate=0.07)
network.train(training_data)
unknown_data = network.predict(unknown)
print(f"Predicted classes= {unknown_data}")
# file = open("44_classes.txt", "w")
# for i in unknown_data:
#    file.write(str(i) + ",")

# max_accuracy = 0 hidden_neurons = [28, 29, 30] neurons_correct = [] number_of_neurons = -1 for i in hidden_neurons:
# sum = 0 for j in range(10): # Create the network with one hidden layer network = ANN(input_size=10, hidden_size=i,
# output_size=7, num_epochs=35, mini_batch_size=20, learning_rate=0.1)
#
#         # set to the data to which we are learning to 'data' because we are testing
#         # it on the unknown dataset and start training
#         network.train(training_data)
#
#         accuracy = network.evaluate(test_data)
#         sum += accuracy
#     averaged_accuracy = sum / 10.0
#     neurons_correct.append(averaged_accuracy)
#
#     print(f"averaged accuracy = {max_accuracy}")
#     print(f"neurons = {i}")
#     if averaged_accuracy > max_accuracy:
#         max_accuracy = averaged_accuracy
#         number_of_neurons = i

# TASK 1.3.10

# network = ANN(input_size=10, hidden_size=29, output_size=7, num_epochs=35, mini_batch_size=20, learning_rate=0.07)
# results = []
# weights_seeds = []
# for i in range(10):
#     network.initialize_weight_bias(i+1)
#     network.train(training_data)
#     accuracy = network.evaluate(test_data)
#     results.append(accuracy)
#     weights_seeds.append(i+1)
#     print(f"Accuracy on test set= {accuracy}")
#
# plt.plot(weights_seeds, results)
# plt.xlabel('weights seed: standard deviation equal to quare root of inverse of ANN\'s input size times seed')
# plt.ylabel('Accuracy')
# plt.title('Weights initialization seed vs Accuracy')
# plt.show()

# TASK 1.4.12

# network = ANN(input_size=10, hidden_size=29, output_size=7, num_epochs=35, mini_batch_size=20, learning_rate=0.07)
# results_validation = []
# results_training = []
# epochs = []
# for epoch in range(network.num_epochs):
#     np.random.shuffle(training_data)
#     mini_batches = []
#     for index in range(0, len(training_data), network.mini_batch_size):
#         mini_batch = training_data[index:index + network.mini_batch_size]
#         mini_batches.append(mini_batch)
#
#     for mini_batch in mini_batches:
#         network.update_with_mini_batch(mini_batch)
#     results_validation.append(network.evaluate(validation_data))
#     results_training.append(network.evaluate(training_data))
#     epochs.append(epoch)
#
# plt.plot(epochs, results_training)
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.title('Accuracy of ann over epochs on training data')
# plt.show()
#
# plt.plot(epochs, results_validation)
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.title('Accuracy of ann over epochs on validation data')
# plt.show()

# TASK 1.6.17
# We also made changes in the training, feedforward and backpropagation functions to use identity activation function
#
#
# network = ANN(input_size=10, hidden_size=29, output_size=7, num_epochs=35, mini_batch_size=200, learning_rate=0.01)
# results_validation = []
# results_training = []
# epochs = []
# for epoch in range(network.num_epochs):
#     np.random.shuffle(training_data)
#     mini_batches = []
#     for index in range(0, len(training_data), network.mini_batch_size):
#         mini_batch = training_data[index:index + network.mini_batch_size]
#         mini_batches.append(mini_batch)
#
#     for mini_batch in mini_batches:
#         network.update_with_mini_batch(mini_batch)
#     results_validation.append(network.evaluate(validation_data))
#     results_training.append(network.evaluate(training_data))
#     epochs.append(epoch)

# accuracy_test = network.evaluate(test_data)
# print(f"Accuracy on test set= {accuracy_test}")
# plt.plot(epochs, results_training)
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.title('Accuracy of ann over epochs on training data')
# plt.show()
#
# plt.plot(epochs, results_validation)
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.title('Accuracy of ann over epochs on validation data')
# plt.show()
