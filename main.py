from layers.conv import Convolutionlayer
from layers.pool import Pooling
from layers.relu import ReLu
from layers.flat import Flattening
from layers.linear import FullyConnected
from layers.softmax import Softmax
from utils import DataLoader
from tqdm import tqdm
import numpy as np

debug_file = open('debug.txt', 'w')

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


def main():
    train_set = 'training-a'

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

    data_loader = DataLoader(train_set)
    for epoch in tqdm(range(epochs)):
        debug_file.write('Epoch: ' + str(epoch) + '\n\n')
        for batch in tqdm(range(0, len(data_loader.data), batch_size)):
            X_train, Y_train = data_loader.load_minibatch(batch, batch_size)
            y_pred = nn.forward(X_train)
            nn.backward(Y_train)

    
    X_test, img_list = data_loader.load_test_set('E:/ML/testing-b/')
    y_pred = nn.forward(X_test)
    y_pred = np.argmax(y_pred, axis=1)

    for i in range(len(y_pred)):
        print(img_list[i], y_pred[i])

if __name__ == '__main__':
    main()
