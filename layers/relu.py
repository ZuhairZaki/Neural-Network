import numpy as np

class ReLu:
    def __init__(self):
        pass

    def forward(self, input):
        self.input = input
        self.output = np.maximum(0, self.input)
        return self.output

    def backward(self, dL_dout, outfile):
        self.dL_dinput = dL_dout.copy()
        self.dL_dinput[self.input <= 0] = 0

        # outfile.write('ReLu backward\n')
        # outfile.write('self.input: ' + str(self.input.shape) + '\n')
        # outfile.write('self.output: ' + str(self.output.shape) + '\n')
        # outfile.write('dL_dout: ' + str(dL_dout.shape) + '\n')
        # outfile.write('self.dL_dinput: ' + str(self.dL_dinput.shape) + '\n')
        
        return self.dL_dinput