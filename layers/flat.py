import numpy as np

class Flattening:
    def __init__(self):
        pass

    def forward(self, input):
        self.input_shape = input.shape
        self.output = input.reshape(self.input_shape[0], -1)
        return self.output

    def backward(self, dL_dout):
        self.dL_dinput = dL_dout.reshape(self.input_shape)
        return self.dL_dinput