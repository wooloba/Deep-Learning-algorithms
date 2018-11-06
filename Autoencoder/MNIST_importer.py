from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import tensorflow as tf


class MNIST(object):
    def __init__(self, subset='train', batch_size=64, labeled_percent=0.2, shuffle=True):
        mnist = input_data.read_data_sets('data', one_hot=False)
        if subset == 'train':
            images = mnist.train._images
            labels = mnist.train._labels.reshape((-1, 1))
            np.random.seed(100)
            is_labeled = np.zeros((55000, 1))
            labeled_images = np.random.permutation(np.arange(55000))[:int(labeled_percent * 55000)]
            is_labeled[labeled_images] = 1
            np.random.seed()
        elif subset == 'valid':
            images = mnist.validation._images
            labels = mnist.validation._labels.reshape((-1, 1))
            is_labeled = np.ones((5000, 1))
        elif subset == 'test':
            images = mnist.test._images
            labels = mnist.test._labels.reshape((-1, 1))
            is_labeled = np.ones((10000, 1))
        else:
            raise NotImplementedError
        self._images = images
        self.images = self._images
        self._labels = labels
        self.labels = self._labels
        self._is_labeled = is_labeled
        self.is_labeled = self._is_labeled
        self.batch_size = batch_size
        self.num_samples = len(self.images)
        self.shuffle = shuffle
        if self.shuffle:
            self.shuffle_samples()
        self.next_batch_pointer = 0

    def shuffle_samples(self):
        image_indices = np.random.permutation(np.arange(self.num_samples))
        self.images = self._images[image_indices]
        self.labels = self._labels[image_indices]
        self.is_labeled = self._is_labeled[image_indices]

    def get_next_batch(self):
        num_samples_left = self.num_samples - self.next_batch_pointer
        if num_samples_left >= self.batch_size:
            x_batch = self.images[self.next_batch_pointer:self.next_batch_pointer + self.batch_size]
            y_batch = self.labels[self.next_batch_pointer:self.next_batch_pointer + self.batch_size]
            is_labeled_batch = self.is_labeled[self.next_batch_pointer:self.next_batch_pointer + self.batch_size]
            self.next_batch_pointer += self.batch_size
        else:
            x_partial_batch_1 = self.images[self.next_batch_pointer:self.num_samples]
            y_partial_batch_1 = self.labels[self.next_batch_pointer:self.num_samples]
            is_labeled_batch_1 = self.is_labeled[self.next_batch_pointer:self.num_samples]
            if self.shuffle:
                self.shuffle_samples()
            x_partial_batch_2 = self.images[0:self.batch_size - num_samples_left]
            y_partial_batch_2 = self.labels[0:self.batch_size - num_samples_left]
            is_labeled_batch_2 = self.is_labeled[0:self.batch_size - num_samples_left]
            x_batch = np.vstack((x_partial_batch_1, x_partial_batch_2))
            y_batch = np.vstack((y_partial_batch_1, y_partial_batch_2))
            is_labeled_batch = np.vstack((is_labeled_batch_1, is_labeled_batch_2))
            self.next_batch_pointer = self.batch_size - num_samples_left
        return x_batch, y_batch, is_labeled_batch