import cv2
import pandas as pd
import numpy as np
import os

img_dim = 32
C = 10

def convert_to_one_hot(Y):
    y = np.zeros(C)
    y[Y] = 1
    return y

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

class DataLoader:
    def __init__(self, train_set, dir_path = 'E:/ML/'):
        self.train_path = dir_path + train_set + '/'
        self.data = pd.read_csv(dir_path + train_set + '.csv')

    def load_minibatch(self, batch_start, batch_size):
        batch_end = batch_start + batch_size if batch_start + batch_size < len(self.data) else len(self.data)
        batch = self.data[batch_start:batch_end]

        X_train = []
        Y_train = []
        for i in range(len(batch)):
            img = cv2.imread(self.train_path + batch['filename'].iloc[i])
            img = cv2.resize(img, (img_dim, img_dim))
            img = img.transpose(2,0,1)/255.0
            X_train.append(img)
            Y_train.append(convert_to_one_hot(batch['digit'].iloc[i]))

        return np.array(X_train), np.array(Y_train)


