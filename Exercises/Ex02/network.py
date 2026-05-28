import numpy as np
from sklearn import datasets

####################################

class ReLULayer(object):
    def forward(self, input):
        # Remember the input for later backpropagation.
        # ReLU is applied element-wise: max(0, x).
        self.input = input
        relu = np.maximum(0.0, input)
        return relu

    def backward(self, upstream_gradient):
        # Chain rule:
        # d ReLU(x) / dx is 1 for x > 0 and 0 for x <= 0.
        downstream_gradient = upstream_gradient * (self.input > 0)
        return downstream_gradient

    def update(self, learning_rate):
        pass  # ReLU is parameter-free

####################################

class OutputLayer(object):
    def __init__(self, n_classes):
        self.n_classes = n_classes

    def forward(self, input):
        # Remember the input for later backpropagation.
        self.input = input

        # softmax:
        exp_values = np.exp(input)
        softmax = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        return softmax

    def backward(self, predicted_posteriors, true_labels):
        # For softmax + cross-entropy, the derivative w.r.t. the logits is:
        # predicted_posteriors - one_hot(true_labels).
        #
        # We divide by batch_size because the mini-batch loss is interpreted
        # as the average loss over the mini-batch.
        batch_size = predicted_posteriors.shape[0]

        true_labels = np.asarray(true_labels, dtype=int)
        one_hot = np.zeros_like(predicted_posteriors)
        one_hot[np.arange(batch_size), true_labels] = 1.0

        downstream_gradient = (predicted_posteriors - one_hot) / batch_size
        return downstream_gradient

    def update(self, learning_rate):
        pass  # softmax is parameter-free

####################################

class LinearLayer(object):
    def __init__(self, n_inputs, n_outputs):
        self.n_inputs  = n_inputs
        self.n_outputs = n_outputs

        # He initialization, suitable for ReLU networks.
        # The lecture used D_{l-1}+1 when the bias was absorbed into the
        # weight matrix by adding a constant input 1. Here the bias is separate,
        # but we use the same scale for consistency with that lecture notation.
        std = np.sqrt(2.0 / (self.n_inputs + 1))

        self.B = np.random.normal(0.0, std, size=(self.n_inputs, self.n_outputs))
        self.b = np.random.normal(0.0, std, size=(1, self.n_outputs))

        # Placeholders for gradients, filled during backward().
        self.grad_B = np.zeros_like(self.B)
        self.grad_b = np.zeros_like(self.b)

    def forward(self, input):
        # Remember the input for later backpropagation.
        self.input = input

        # Linear preactivation:
        # Z_tilde = Z_previous @ B + b
        preactivations = np.dot(input, self.B) + self.b
        return preactivations

    def backward(self, upstream_gradient):
        # upstream_gradient is dLoss / dZ_tilde for this layer.

        # Bias gradient:
        # Since b is added to every instance, we sum over the batch axis.
        self.grad_b = np.sum(upstream_gradient, axis=0, keepdims=True)

        # Weight gradient:
        # For a batch: dLoss/dB = input.T @ upstream_gradient
        self.grad_B = np.dot(self.input.T, upstream_gradient)

        # Downstream gradient for the preceding layer:
        # dLoss/dInput = upstream_gradient @ B.T
        downstream_gradient = np.dot(upstream_gradient, self.B.T)
        return downstream_gradient

    def update(self, learning_rate):
        # Gradient descent update for trainable parameters.
        self.B = self.B - learning_rate * self.grad_B
        self.b = self.b - learning_rate * self.grad_b

####################################

class MLP(object):
    def __init__(self, n_features, layer_sizes):
        # Construct a multi-layer perceptron with ReLU activations in the
        # hidden layers and softmax output.
        #
        # n_features: number of input features
        # len(layer_sizes): number of linear layers
        # layer_sizes[k]: number of neurons in linear layer k
        # layer_sizes[-1]: number of output classes
        self.n_layers = len(layer_sizes)
        self.layers   = []

        # Create interior layers: Linear + ReLU.
        n_in = n_features
        for n_out in layer_sizes[:-1]:
            self.layers.append(LinearLayer(n_in, n_out))
            self.layers.append(ReLULayer())
            n_in = n_out

        # Create last linear layer + output softmax.
        n_out = layer_sizes[-1]
        self.layers.append(LinearLayer(n_in, n_out))
        self.layers.append(OutputLayer(n_out))

    def forward(self, X):
        # X is a mini-batch of instances.
        batch_size = X.shape[0]

        # Flatten the other dimensions of X in case instances are images.
        X = X.reshape(batch_size, -1)

        # Compute the forward pass. Each layer stores what it needs for
        # subsequent backpropagation.
        result = X
        for layer in self.layers:
            result = layer.forward(result)
        return result

    def backward(self, predicted_posteriors, true_classes):
        # Start backpropagation at the output layer.
        upstream_gradient = self.layers[-1].backward(predicted_posteriors, true_classes)

        # Propagate the gradient through the remaining layers in reverse order.
        for layer in reversed(self.layers[:-1]):
            upstream_gradient = layer.backward(upstream_gradient)

    def update(self, X, Y, learning_rate):
        posteriors = self.forward(X)
        self.backward(posteriors, Y)

        for layer in self.layers:
            layer.update(learning_rate)

    def train(self, x, y, n_epochs, batch_size, learning_rate):
        N = len(x)
        n_batches = N // batch_size

        for i in range(n_epochs):
            # Reorder data for every epoch, i.e. sample mini-batches
            # without replacement within one epoch.
            permutation = np.random.permutation(N)

            for batch in range(n_batches):
                # Create mini-batch.
                start = batch * batch_size
                x_batch = x[permutation[start:start + batch_size]]
                y_batch = y[permutation[start:start + batch_size]]

                # Perform one forward pass, one backward pass,
                # and one parameter update.
                self.update(x_batch, y_batch, learning_rate)

##################################

if __name__ == "__main__":

    # Makes the experiment reproducible.
    np.random.seed(0)

    # Set training/test set size.
    N = 2000

    # Create training and test data.
    X_train, Y_train = datasets.make_moons(N, noise=0.05, random_state=0)
    X_test,  Y_test  = datasets.make_moons(N, noise=0.05, random_state=1)
    n_features = 2
    n_classes  = 2

    # Standardize features to be in [-1, 1], using training-set statistics.
    offset  = X_train.min(axis=0)
    scaling = X_train.max(axis=0) - offset
    X_train = ((X_train - offset) / scaling - 0.5) * 2.0
    X_test  = ((X_test  - offset) / scaling - 0.5) * 2.0

    # Set hyperparameters.
    n_epochs = 5
    batch_size = 200
    learning_rate = 0.05

    # Compare the four networks requested in the exercise sheet.
    architectures = [
        [2, 2, n_classes],
        [3, 3, n_classes],
        [5, 5, n_classes],
        [30, 30, n_classes],
    ]

    for layer_sizes in architectures:
        network = MLP(n_features, layer_sizes)

        # Train.
        network.train(X_train, Y_train, n_epochs, batch_size, learning_rate)

        # Test.
        predicted_posteriors = network.forward(X_test)

        # Winner-takes-all rule.
        predicted_classes = np.argmax(predicted_posteriors, axis=1)

        # Error rate.
        error_rate = np.mean(predicted_classes != Y_test)

        print("architecture:", layer_sizes, "error rate:", error_rate)
