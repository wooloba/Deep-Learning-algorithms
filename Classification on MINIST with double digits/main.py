import data_loader
import numpy as np
from training import train
from test import test


def main():
    print("Loading data ...")
    prefix = 'valid'

    x_train, y_train, x_valid, y_valid, train_bbox, valid_bbox ,xTest,yTest,Test_bbox= data_loader.dataloader(prefix)
    print("Shape:", x_train.shape, y_train.shape, x_valid.shape, y_valid.shape, train_bbox.shape, valid_bbox.shape)

    #task = "detection"
    task = 'classify'

    # Classification
    #train(x_train, y_train, x_valid, y_valid, train_bbox, valid_bbox, task)

    np.random.seed(0)
    acc = test(xTest,yTest,Test_bbox,task)
    np.random.seed()

    print("Test accuracy is : " + str(acc))


if __name__ == '__main__':
    main()
