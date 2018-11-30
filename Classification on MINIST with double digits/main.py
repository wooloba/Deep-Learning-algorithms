import timeit
import data_loader
import numpy as np
from training import train_classify,train_detection
from test import test_classify,test_detection


def main():
    print("Loading data ...")
    x_train, y_train, x_valid, y_valid,train_bbox,valid_bbox = data_loader.dataloader()

    print("Shape:", x_train.shape, y_train.shape, x_valid.shape, y_valid.shape)

    #Classification
    train_classify(x_train, y_train,x_valid,y_valid)
    time_start = timeit.default_timer()
    np.random.seed(0)
    acc = test_classify(x_valid, y_valid)
    np.random.seed()
    run_time = time_start - timeit.default_timer()

    #detection





    print("Test accuracy is : " + str(acc))

if __name__ == '__main__':
   main()
