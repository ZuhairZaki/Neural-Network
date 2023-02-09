import numpy as np

class ReLu:
    def __init__(self):
        pass

    def forward(self, input):
        self.input = input
        self.output = np.maximum(0, self.input)
        return self.output

    def backward(self, dL_dout):
        self.dL_dinput = dL_dout.copy()
        self.dL_dinput[self.input <= 0] = 0
        
        return self.dL_dinput