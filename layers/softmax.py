import numpy as np

class Softmax:
    def __init__(self):
        pass

    def forward(self, input):
        input = input - np.max(input, axis=1, keepdims=True)
        self.output = np.exp(input) / np.sum(np.exp(input), axis=1, keepdims=True)
        return self.output

    def backward(self, Y, outfile):
        self.dL_dinput = self.output - Y
        outfile.write('Softmax backward\n')
        # outfile.write('self.input: ' + str(self.input) + '\n')
        # outfile.write('self.output: ' + str(self.output) + '\n')
        # outfile.write('Y: ' + str(Y.shape) + '\n')
        # outfile.write('self.output: ' + str(self.output.shape) + '\n')
        # outfile.write('self.dL_dinput: ' + str(self.dL_dinput.shape) + '\n')
        return self.dL_dinput
        