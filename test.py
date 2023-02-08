from layers.conv import Convolutionlayer
from layers.pool import Pooling
from layers.relu import ReLu
from layers.flat import Flattening
from layers.linear import FullyConnected
from layers.softmax import Softmax
from tqdm import tqdm
import numpy as np

debug_file = open('debug.txt', 'w')

def convert_to_one_hot(Y):
    y = np.zeros(4)
    y[Y] = 1
    return y

class NeuralNetwork:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, x):
        x_forward = x
        for layer in self.layers:
            x_forward = layer.forward(x_forward)
        return x_forward

    def backward(self, y):
        y_backward = y
        for layer in reversed(self.layers):
            y_backward = layer.backward(y_backward,debug_file)



layers = [
    Convolutionlayer(5,3,6),
    ReLu(),
    Pooling(2,2),
    Flattening(),
    FullyConnected(6*12*12, 4),
    Softmax()
]

nn = NeuralNetwork(layers)

# with open('E:/ML/Toy Dataset/trainNN.txt','r') as f:
#     for line in f.readlines():
#         line = line.strip()
#         line = line.split()
#         line = [float(i) for i in line]
#         x = np.array(line[:-1])
#         y = convert_to_one_hot(int(line[-1])-1)
#         x_train = x[np.newaxis, :]
#         y_train = y[np.newaxis, :]


