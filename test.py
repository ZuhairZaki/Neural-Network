import os
import cv2
import pickle
import numpy as np
from train import NeuralNetwork

img_dim  =  32

def load_test_set(test_folder):
        X_test = []
        img_list = []
        for filename in os.listdir(test_folder):
            img = cv2.imread(os.path.join(test_folder,filename))
            if img is not None:
                img = cv2.resize(img, (img_dim, img_dim))
                img = img.transpose(2,0,1)/255.0
                X_test.append(img)
                img_list.append(filename)

        return np.array(X_test), img_list


def main():
    nn = pickle.load(open('model.pkl', 'rb'))

    X_test, img_list = load_test_set('E:/ML/testing-b/')
    y_pred = nn.forward(X_test)
    y_pred = np.argmax(y_pred, axis=1)

    for i in range(len(y_pred)):
        print(img_list[i], y_pred[i])


if __name__ == '__main__':
    main()


