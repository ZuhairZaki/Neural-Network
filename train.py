from layers.conv import Convolutionlayer
from layers.pool import Pooling
from layers.relu import ReLu
from layers.flat import Flattening
from layers.linear import FullyConnected
from layers.softmax import Softmax
from utils import DataLoader
from utils import load_test_set
from tqdm import tqdm
import numpy as np
import pickle

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
            y_backward = layer.backward(y_backward)

def main():
    train_set = ['training-a','training-b','training-c']

    epochs = 10
    batch_size = 64

    # layers = [
    #     Convolutionlayer(5,3,6),
    #     ReLu(),
    #     Pooling(2,2),
    #     Convolutionlayer(5,6,16),
    #     ReLu(),
    #     Pooling(2,2),
    #     Flattening(),
    #     FullyConnected(16*5*5, 120),
    #     ReLu(),
    #     FullyConnected(120, 84),
    #     ReLu(),
    #     FullyConnected(84, 10),
    #     Softmax()
    # ]

    layers = [
        Convolutionlayer(5,3,6),
        ReLu(),
        Pooling(2,2),
        Flattening(),
        FullyConnected(6*14*14, 10),
        Softmax()
    ]

    nn = NeuralNetwork(layers)

    training_data = [ 
        DataLoader(train_set[0]),
        DataLoader(train_set[1]),
        DataLoader(train_set[2])
    ]

    for epoch in tqdm(range(epochs)):
        print('\nEpoch: %d\n' % (epoch+1))
        for i in range(len(training_data)):
            print('\nTraining on %s\n' % train_set[i])
            for batch in tqdm(range(0, len(training_data[i].data), batch_size)):
                X_train, Y_train = training_data[i].load_minibatch(batch, batch_size)
                y_pred = nn.forward(X_train)
                nn.backward(Y_train)

    pickle.dump(nn, open('model.pkl', 'wb'))

if __name__ == '__main__':
    main()
