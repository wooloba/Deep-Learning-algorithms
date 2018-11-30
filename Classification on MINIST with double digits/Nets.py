import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import flatten

def W_generator(shape):
    mu = 0
    sigma = 0.1
    W = tf.Variable(tf.truncated_normal(shape=shape, mean=mu, stddev=sigma))
    return W

def net(X,is_training,task):
    with tf.name_scope("Layer1"):
        conv1_W = W_generator([5, 5, 1, 32])
        conv1_b = tf.Variable(tf.zeros(32))
        conv1 = tf.nn.conv2d(X, conv1_W, strides=[1, 1, 1, 1], padding='SAME') + conv1_b

        conv1 = tf.layers.batch_normalization(conv1, training=True)
        conv1 = tf.nn.relu(conv1)
        if is_training:
            conv1 = tf.nn.dropout(conv1, keep_prob=0.6)
        conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        # 2.Convolution using 128 filters
        # using two 3x3 filter to replace one 5x5 filter
    with tf.name_scope("layer2"):
        conv2_W = W_generator([5, 5, 32, 32])
        conv2_b = tf.Variable(tf.zeros(32))
        conv2 = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='SAME') + conv2_b

        # Batch_Normalization:
        conv2 = tf.layers.batch_normalization(conv2, center=True, scale=True, training=True)

        conv2 = tf.nn.relu(conv2)

        if is_training:
            conv2 = tf.nn.dropout(conv2, keep_prob=0.6)

        conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        # 3.Convolution using 256 filters
        # using three 3x3 filter to replace one 7x7 filter
    with tf.name_scope("layer3"):
        conv3_W = W_generator([7, 7, 32, 64])
        conv3_b = tf.Variable(tf.zeros(64))
        conv3 = tf.nn.conv2d(conv2, conv3_W, strides=[1, 1, 1, 1], padding='SAME') + conv3_b

        conv3 = tf.layers.batch_normalization(conv3, center=True, scale=True, training=True)
        conv3 = tf.nn.relu(conv3)

        if is_training:
            conv3 = tf.nn.dropout(conv3, keep_prob=0.6)

        conv3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    with tf.name_scope("layer4"):
        conv4_W = W_generator([5, 5, 64, 64])
        conv4_b = tf.Variable(tf.zeros(64))
        conv4 = tf.nn.conv2d(conv3, conv4_W, strides=[1, 1, 1, 1], padding='SAME') + conv4_b

        conv4 = tf.layers.batch_normalization(conv4, center=True, scale=True, training=True)

        if is_training:
            conv4 = tf.nn.dropout(conv4, keep_prob=0.6)

        conv4 = tf.nn.relu(conv4)
        conv4 = tf.nn.max_pool(conv4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    with tf.name_scope("layer5"):
        conv5_W = W_generator([3, 3, 64, 128])
        conv5_b = tf.Variable(tf.zeros(128))
        conv5 = tf.nn.conv2d(conv4, conv5_W, strides=[1, 1, 1, 1], padding='SAME') + conv5_b

        conv5 = tf.nn.relu(conv5)

        conv5 = tf.nn.avg_pool(conv5, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

        if is_training:
            conv5 = tf.nn.dropout(conv5, keep_prob=0.6)

    # shape = int(np.prod(conv5.get_shape()[1:]))
    # drop = tf.nn.dropout(shape, 0.6, name='drop')
    shape = int(np.prod(conv5.get_shape()[1:]))
    fc0 = tf.reshape(conv5, (-1, shape))

    #fc0 = flatten(conv5)
    #print(drop.get_shape())

    with tf.name_scope("layer6_fc"):
        fc1_W = W_generator([512, 128])
        fc1_b = tf.Variable(tf.zeros(128))
        fc1 = tf.matmul(fc0, fc1_W) + fc1_b

        if is_training:
            fc1 = tf.nn.dropout(fc1, keep_prob=0.6)

        fc1 = tf.nn.relu(fc1)

    with tf.name_scope("layer7_fc"):
        fc2_W = W_generator([128, 20])
        fc2_b = tf.Variable(tf.zeros(20))
        fc2 = tf.matmul(fc1, fc2_W) + fc2_b

    if task == 'classify':
        logits = tf.reshape(fc2,shape=[-1,2,10])
    elif task == 'detection':
        logits = tf.reshape(fc2,shape= [-1,2,4])

    return logits
