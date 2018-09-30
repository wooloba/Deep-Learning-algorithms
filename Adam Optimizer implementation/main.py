import numpy as np
import timeit
from collections import OrderedDict
from pprint import pformat
import tensorflow as tf
from adam_train import adam_train

def compute_score(acc, min_thres, max_thres):
    if acc <= min_thres:
        base_score = 0.0
    elif acc >= max_thres:
        base_score = 100.0
    else:
        base_score = float(acc - min_thres) / (max_thres - min_thres) \
                     * 100
    return base_score


def run(algorithm, x_train, y_train, x_valid, y_valid, x_test, y_test):
    start = timeit.default_timer()
    np.random.seed(0)
    predicted_y_test = algorithm(x_train, y_train, x_valid, y_valid, x_test)
    np.random.seed()
    stop = timeit.default_timer()
    run_time = stop - start

    y_test = y_test.flatten()
    predicted_y_test = np.asarray(predicted_y_test).flatten()

    correct_predict = (y_test == predicted_y_test).astype(np.int32).sum()
    incorrect_predict = len(y_test) - correct_predict
    accuracy = float(correct_predict) / len(y_test)

    return (correct_predict, accuracy, run_time)

def main():
    min_thres = 0.24
    max_thres = 0.34
    cifar10 = tf.keras.datasets.cifar10
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_valid, y_valid = x_train[-10000:], y_train[-10000:]
    x_train, y_train = x_train[:-10000], y_train[:-10000]

    correct_predict, accuracy, run_time = run(adam_train,
                                              x_train, y_train, x_valid, y_valid, x_test, y_test)
    score = compute_score(accuracy, min_thres, max_thres)

    result = OrderedDict(correct_predict=correct_predict,
                         accuracy=accuracy, score=score,
                         run_time=run_time)
    with open('result.txt', 'w') as f:
        f.writelines(pformat(result, indent=4))
    print("\nResult:\n", pformat(result, indent=4))

main()
