from ops import *
import timeit
from cifar10 import Cifar10
from tensorflow.contrib.layers import flatten

def W_generator(shape):
    mu = 0
    sigma = 0.1
    W = tf.Variable(tf.truncated_normal(shape=shape,mean=mu,stddev=sigma))

    return W

def add_gradient_summaries(grads_and_vars):
    for grad, var in grads_and_vars:
        if grad is not None:
            tf.summary.histogram(var.op.name + "/gradient", grad)

def net(input, is_training, dropout_kept_prob):
  # TODO: Write your network architecture here
  # Or you can write it inside train() function
  # Requirements:
  # - At least 5 layers in total
  # - At least 1 fully connected and 1 convolutional layer
  # - At least one maxpool layers
  # - At least one batch norm
  # - At least one skip connection
  # - Use dropout

    #1.Convolution using 64 filters
    #using two 3x3 filter to replace one 5x5 filter
    with tf.name_scope("Layer1"):
        conv1_W1 = W_generator([3,3,3,64])
        conv1_b1 = tf.Variable(tf.zeros(64))
        conv1_1 = tf.nn.conv2d(input, conv1_W1,strides=[1, 1, 1, 1], padding='VALID') + conv1_b1

        #Dropout
        conv1_1 = tf.nn.dropout(conv1_1,keep_prob=0.7,is_training=is_training)

        conv1_W2 = W_generator([3,3,3,64])
        conv1_b2 = tf.Variable(tf.zeros(64))
        conv1_2 = tf.nn.conv2d(conv1_1, conv1_W2, strides=[1, 1, 1, 1], padding='VALID') + conv1_b2

        conv1_2 = tf.nn.relu(conv1_2)
        conv1_2 = tf.nn.max_pool(conv1_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],padding='VALID')

        # Dropout
        conv1_2 = tf.nn.dropout(conv1_2,keep_prob=0.7,is_training=is_training)

    #2.Convolution using 128 filters
    # using two 3x3 filter to replace one 5x5 filter
    with tf.name_scope("layer2"):
        conv2_W1 = W_generator([3,3,16,128])
        conv2_b1 = tf.Variable(tf.zeros(128))
        conv2_1 = tf.nn.conv2d(conv1_2, conv2_W1,strides=[1, 1, 1, 1], padding='VALID') + conv2_b1

        # Dropout
        conv2_1 = tf.nn.dropout(conv2_1,keep_prob=0.7,is_training=is_training)

        conv2_W2 = W_generator([3, 3, 16, 128])
        conv2_b2 = tf.Variable(tf.zeros(128))
        conv2_2 = tf.nn.conv2d(conv2_1, conv2_W2, strides=[1, 1, 1, 1], padding='VALID') + conv2_b2

        conv2_2 = tf.nn.relu(conv2_2)

        conv2_2 = tf.nn.max_pool(conv2_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

        # Dropout
        conv2_2 = tf.nn.dropout(conv2_2,keep_prob=0.7,is_training=is_training)

    # 3.Convolution using 256 filters
    # using three 3x3 filter to replace one 7x7 filter
    with tf.name_scope("layer3"):
        conv3_W1 = W_generator([3, 3, 8, 256])
        conv3_b1 = tf.Variable(tf.zeros(256))
        conv3_1 = tf.nn.conv2d(conv2_2, conv3_W1, strides=[1, 1, 1, 1], padding='VALID') + conv3_b1
        # Dropout
        conv3_1 = tf.nn.dropout(conv3_1,keep_prob=0.6,is_training=is_training)

        conv3_W2 = W_generator([3, 3, 8, 256])
        conv3_b2 = tf.Variable(tf.zeros(256))
        conv3_2 = tf.nn.conv2d(conv3_1, conv3_W2, strides=[1, 1, 1, 1], padding='VALID') + conv3_b2
        # Dropout
        conv3_2 = tf.nn.dropout(conv3_2,keep_prob=0.6,is_training=is_training)

        conv3_W3 = W_generator([3, 3, 8, 256])
        conv3_b3 = tf.Variable(tf.zeros(256))
        conv3_3 = tf.nn.conv2d(conv3_2, conv3_W3, strides=[1, 1, 1, 1], padding='VALID') + conv3_b3

        conv3_3 = tf.nn.relu(conv3_3)

        conv3_3 = tf.nn.max_pool(conv3_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
        # Dropout
        conv3_3 = tf.nn.dropout(conv3_3,keep_prob=0.6,is_training=is_training)

    with tf.name_scope("layer4"):
        conv4_W1 = W_generator([3, 3, 4, 512])
        conv4_b1 = tf.Variable(tf.zeros(512))
        conv4_1 = tf.nn.conv2d(conv3_3, conv4_W1, strides=[1, 1, 1, 1], padding='VALID') + conv4_b1
        # Dropout
        conv4_1 = tf.nn.dropout(conv4_1,keep_prob=0.6,is_training=is_training)

        conv4_W2 = W_generator([3, 3, 4, 512])
        conv4_b2 = tf.Variable(tf.zeros(512))
        conv4_2 = tf.nn.conv2d(conv4_1, conv4_W2, strides=[1, 1, 1, 1], padding='VALID') + conv4_b2

        conv4_2 = tf.nn.relu(conv4_2)

        conv4_2 = tf.nn.max_pool(conv4_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
        # Dropout
        conv4_2 = tf.nn.dropout(conv4_2,keep_prob=0.6,is_training=is_training)

    with tf.name_scope("layer5"):
        conv5_W1 = W_generator([3, 3, 2, 512])
        conv5_b1 = tf.Variable(tf.zeros(512))
        conv5_1 = tf.nn.conv2d(conv4_2, conv5_W1, strides=[1, 1, 1, 1], padding='VALID') + conv5_b1

        conv5_1 = tf.nn.avg_pool(conv5_1,ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
        #Dropout
        conv5_1 = tf.nn.dropout(conv5_1,keep_prob=0.5,is_training=is_training)

    fc0 = flatten(conv5_1)

    with tf.name_scope("layer6_fc"):
        fc1_W = W_generator([512, 120])
        fc1_b = tf.Variable(tf.zeros(120))
        fc1 = tf.matmul(fc0, fc1_W) + fc1_b

        fc1 = tf.nn.relu(fc1)

    with tf.name_scope("layer4_fc"):
        fc2_W = W_generator([120, 84])
        fc2_b = tf.Variable(tf.zeros(84))
        fc2 = tf.matmul(fc1, fc2_W) + fc2_b

        fc2 = tf.nn.relu(fc2)

    with tf.name_scope("layer5_fc"):
        fc3_W = W_generator([84, 10])
        fc3_b = tf.Variable(tf.zeros(10))
        logits = tf.matmul(fc2, fc3_W) + fc3_b

    return logits

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
