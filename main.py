from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
import numpy as np
import timeit
from collections import OrderedDict
from pprint import pformat
import tensorflow as tf
from knn import knn


def compute_score(acc, min_thres, max_thres):
    if acc <= min_thres:
        base_score = 0.0
    elif acc >= max_thres:
        base_score = 100.0
    else:
        base_score = float(acc - min_thres) / (max_thres - min_thres) \
            * 100
    return base_score


def run(algorithm, x_train, y_train, x_test, y_test):
    print('Running...')
    start = timeit.default_timer()
    np.random.seed(0)
    predicted_y_test = algorithm(x_train, y_train, x_test)
    np.random.seed()
    stop = timeit.default_timer()
    run_time = stop - start

    correct_predict = (y_test
                       == predicted_y_test).astype(np.int32).sum()
    incorrect_predict = len(y_test) - correct_predict
    accuracy = float(correct_predict) / len(y_test)

    print('Correct Predict: {}/{} total \tAccuracy: {:5f} \tTime: {:2f}'.format(correct_predict,
            len(y_test), accuracy, run_time))
    return (correct_predict, accuracy, run_time)


if __name__ == "__main__":
    min_thres = 0.84
    max_thres = 0.94

    mnist = read_data_sets('data', one_hot=False)
    result = [OrderedDict(first_name='Yaozhi',
                          last_name='Lu')]

    (x_train, y_train) = (mnist.train._images, mnist.train._labels)
    (x_valid, y_valid) = (mnist.test._images, mnist.test.labels)

    # You may want to use a smaller training set to save time when debugging
    # i.e.: Put something like:
    #(x_train, y_train) = (x_train[:5000], y_train[:5000])

    # For this assignment, we only test on the first 1000 samples of the test set
    (x_valid, y_valid) = (x_valid[:1000], y_valid[:1000])

    print("Dimension of dataset: ")
    print("Train:", x_train.shape, y_train.shape, "\nTest:", x_valid.shape, y_valid.shape)

    (correct_predict, accuracy, run_time) = run(knn, x_train, y_train, x_valid, y_valid)
    score = compute_score(accuracy, min_thres, max_thres)
    result = OrderedDict(correct_predict=correct_predict,
                         accuracy=accuracy, score=score,
                         run_time=run_time)

    with open('result.txt', 'w') as f:
        f.writelines(pformat(result, indent=4))

    print(pformat(result, indent=4))
