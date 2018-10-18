import numpy as np
import timeit
from net import train, test
from cifar10 import Cifar10
from collections import OrderedDict
from pprint import pformat
import tensorflow as tf


min_thres=0.60
max_thres=0.80


def compute_score(acc, min_thres, max_thres):
    if acc <= min_thres:
        base_score = 0.0
    elif acc >= max_thres:
        base_score = 100.0
    else:
        base_score = float(acc - min_thres) / (max_thres - min_thres) \
            * 100
    return base_score


if __name__ == '__main__':
    TRAIN = True
    if TRAIN:
        train()
    cifar10_test = Cifar10(test=True, shuffle=False, one_hot=False)
    cifar10_test_images, cifar10_test_labels = cifar10_test._images, cifar10_test._labels

    start = timeit.default_timer()
    np.random.seed(0)
    predicted_cifar10_test_labels = test(cifar10_test_images)
    np.random.seed()
    stop = timeit.default_timer()
    run_time = stop - start
    correct_predict = (cifar10_test_labels.flatten() == predicted_cifar10_test_labels.flatten()).astype(np.int32).sum()
    incorrect_predict = len(cifar10_test_labels) - correct_predict
    accuracy = float(correct_predict) / len(cifar10_test_labels)
    print('Acc: {}. Testing took {}s.'.format(accuracy, stop - start))
    
    score = compute_score(accuracy, min_thres, max_thres)
    result = OrderedDict(correct_predict=correct_predict,
                     accuracy=accuracy, score=score,
                     run_time=run_time)

    with open('result.txt', 'w') as f:
        f.writelines(pformat(result, indent=4))
    print(pformat(result, indent=4))
