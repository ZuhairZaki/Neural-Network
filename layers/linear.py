import numpy as np

class FullyConnected:
    def __init__(self, input_size, output_size, learning_rate=0.01):
        self.learning_rate = learning_rate
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2 / input_size)
        self.biases = np.zeros(output_size)

    def forward(self, input):
        self.input = input
        self.output = np.dot(input, self.weights) + self.biases
        return self.output

    def backward(self, dL_dout, outfile):
        self.dL_dinput = np.dot(dL_dout, self.weights.T)/dL_dout.shape[0]
        self.dL_dweights = np.dot(self.input.T, dL_dout)
        self.dL_dbiases = np.sum(dL_dout, axis=0)/dL_dout.shape[0]

        # outfile.write('FullyConnected backward\n')
        # outfile.write('self.input: ' + str(self.input.shape) + '\n')
        # outfile.write('self.output: ' + str(self.output.shape) + '\n')
        # outfile.write('dL_dout: ' + str(dL_dout.shape) + '\n')
        # outfile.write('self.dL_dinput: ' + str(self.dL_dinput.shape) + '\n')
        # outfile.write('self.dL_dweights: ' + str(self.dL_dweights.shape) + '\n')
        # outfile.write('self.dL_dbiases: ' + str(self.dL_dbiases.shape) + '\n')

        self.weights -= self.learning_rate * self.dL_dweights
        self.biases -= self.learning_rate * self.dL_dbiases

        return self.dL_dinput

    