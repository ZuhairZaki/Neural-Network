import numpy as np

class FullyConnected:
    def __init__(self, input_size, output_size, learning_rate=0.001):
        self.learning_rate = learning_rate
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2 / input_size)
        self.biases = np.zeros(output_size)

    def forward(self, input):
        self.input = input
        self.output = np.dot(input, self.weights) + self.biases
        return self.output

    def backward(self, dL_dout):
        self.dL_dinput = np.dot(dL_dout, self.weights.T)/dL_dout.shape[0]
        self.dL_dweights = np.dot(self.input.T, dL_dout)
        self.dL_dbiases = np.sum(dL_dout, axis=0)/dL_dout.shape[0]

        self.weights -= self.learning_rate * self.dL_dweights
        self.biases -= self.learning_rate * self.dL_dbiases

        return self.dL_dinput

    