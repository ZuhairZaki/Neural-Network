import numpy as np

class Softmax:
    def __init__(self):
        pass

    def forward(self, input):
        self.output = np.exp(input) / np.sum(np.exp(input), axis=1, keepdims=True)
        return self.output

    def backward(self, Y):
        self.dL_dinput = self.output - Y
        return self.dL_dinput
        