import numpy as np

class Flattening:
    def __init__(self):
        pass

    def forward(self, input):
        self.input_shape = input.shape
        self.output = input.reshape(self.input_shape[0], -1)
        return self.output

    def backward(self, dL_dout, outfile):
        self.dL_dinput = dL_dout.reshape(self.input_shape)

        # outfile.write('Flattening backward\n')
        # outfile.write('self.input: ' + str(self.input_shape) + '\n')
        # outfile.write('self.output: ' + str(self.output.shape) + '\n')
        # outfile.write('dL_dout: ' + str(dL_dout.shape) + '\n')
        # outfile.write('self.dL_dinput: ' + str(self.dL_dinput.shape) + '\n')
        
        return self.dL_dinput