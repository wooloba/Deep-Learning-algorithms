from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
import numpy as np
import timeit
from collections import OrderedDict
from pprint import pformat
import tensorflow as tf
from logistic_regression import logistic_regression

def compute_score(acc, min_thres, max_thres):
    if acc <= min_thres:
        base_score = 0.0
    elif acc >= max_thres:
        base_score = 100.0
    else:
        base_score = float(acc - min_thres) / (max_thres - min_thres) \
                     * 100
    return base_score


def run(algorithm, dataset_name, x_train, y_train, x_valid, y_valid, x_test, y_test):
    start = timeit.default_timer()
    np.random.seed(0)
    predicted_y_test = algorithm(dataset_name, x_train, y_train, x_valid, y_valid, x_test)
    np.random.seed()
    stop = timeit.default_timer()
    run_time = stop - start

    y_test = y_test.flatten()
    predicted_y_test = np.asarray(predicted_y_test).flatten()

    correct_predict = (y_test == predicted_y_test).astype(np.int32).sum()
    incorrect_predict = len(y_test) - correct_predict
    accuracy = float(correct_predict) / len(y_test)

    return (correct_predict, accuracy, run_time)

def run_on_dataset(dataset_name):
    if dataset_name == "MNIST":
        min_thres = 0.82
        max_thres = 0.92
        mnist = read_data_sets('data', one_hot=False)
        x_train, y_train = (mnist.train._images, mnist.train._labels)
        x_test, y_test = (mnist.test._images, mnist.test.labels)
    elif dataset_name == "CIFAR10":
        min_thres = 0.28
        max_thres = 0.38
        cifar10 = tf.keras.datasets.cifar10
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_valid, y_valid = x_train[-10000:], y_train[-10000:]
    x_train, y_train = x_train[:-10000], y_train[:-10000]

    correct_predict, accuracy, run_time = run(logistic_regression, dataset_name,
                                              x_train, y_train, x_valid, y_valid, x_test, y_test)
    score = compute_score(accuracy, min_thres, max_thres)
    result = OrderedDict(correct_predict=correct_predict,
                         accuracy=accuracy, score=score,
                         run_time=run_time)
    return result, score


def main():
    result_all = OrderedDict()
    score_weights = [0.5, 0.5]
    scores = []
    for dataset_name in ["MNIST", "CIFAR10"]:
        result_all[dataset_name], this_score = run_on_dataset(dataset_name)
        scores.append(this_score)
    total_score = [score * weight for score, weight in zip(scores, score_weights)]
    total_score = np.asarray(total_score).sum().item()
    result_all['total_score'] = total_score
    with open('result.txt', 'w') as f:
        f.writelines(pformat(result_all, indent=4))
    print("\nResult:\n", pformat(result_all, indent=4))


main()
