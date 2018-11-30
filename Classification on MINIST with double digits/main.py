import timeit
import data_loader
import numpy as np
from training import train
from test import test


def main():
    print("Loading data ...")
    x_train, y_train, x_valid, y_valid,train_bbox,valid_bbox = data_loader.dataloader()

    print("Shape:", x_train.shape, y_train.shape, x_valid.shape, y_valid.shape)


    #task = "detection"
    task = 'classify'

    #Classification
    train(x_train, y_train,x_valid,y_valid,train_bbox,valid_bbox,task= task)
    time_start = timeit.default_timer()
    np.random.seed(0)
    acc = test(x_valid, y_valid,valid_bbox,task=task)
    np.random.seed()
    run_time = time_start - timeit.default_timer()


    print("Test accuracy is : " + str(acc))

if __name__ == '__main__':
   main()
