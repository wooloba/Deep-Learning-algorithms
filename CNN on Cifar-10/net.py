from ops import *
import timeit
from cifar10 import Cifar10


def net(input, is_training, dropout_kept_prob):
  # TODO: Write your network architecture here
  # Or you can write it inside train() function
  # Requirements:
  # - At least 5 layers in total
  # - At least 1 fully connected and 1 convolutional layer
  # - At least one maxpool layers
  raise NotImplementedError

def train():
  # Always use tf.reset_default_graph() to avoid error
  tf.reset_default_graph()
  # TODO: Write your training code here
  # - Create placeholder for inputs, training boolean, dropout keep probablity
  # - Construct your model
  # - Create loss and training op
  # - Run training
  # AS IT WILL TAKE VERY LONG ON CIFAR10 DATASET TO TRAIN
  # YOU SHOULD USE tf.train.Saver() TO SAVE YOUR MODEL AFTER TRAINING
  # AT TEST TIME, LOAD THE MODEL AND RUN TEST ON THE TEST SET
  raise NotImplementedError

def test(cifar10_test_images):
  # Always use tf.reset_default_graph() to avoid error
  tf.reset_default_graph()
  # TODO: Write your testing code here
  # - Create placeholder for inputs, training boolean, dropout keep probablity
  # - Construct your model
  # (Above 2 steps should be the same as in train function)
  # - Create label prediction tensor
  # - Run testing
  # DO NOT RUN TRAINING HERE!
  # LOAD THE MODEL AND RUN TEST ON THE TEST SET
  raise NotImplementedError
