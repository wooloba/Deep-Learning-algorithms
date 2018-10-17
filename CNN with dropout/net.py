from ops import *
import timeit
from cifar10 import Cifar10
from tensorflow.contrib.layers import flatten
import tensorflow as tf


def W_generator(shape):
    mu = 0
    sigma = 0.1
    W = tf.Variable(tf.truncated_normal(shape=shape, mean=mu, stddev=sigma))

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

    # 1.Convolution using 64 filters
    # using two 3x3 filter to replace one 5x5 filter
    with tf.name_scope("Layer1"):
        conv1_W = W_generator([5, 5, 3, 32])
        conv1_b = tf.Variable(tf.zeros(32))
        conv1 = tf.nn.conv2d(input, conv1_W, strides=[1, 1, 1, 1], padding='SAME') + conv1_b

        conv1 = tf.layers.batch_normalization(conv1, training=True)
        conv1 = tf.nn.relu(conv1)
        if is_training:
            conv1 = tf.nn.dropout(conv1, keep_prob=dropout_kept_prob[0])

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
            conv2 = tf.nn.dropout(conv2, keep_prob=dropout_kept_prob[0])

        # conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    with tf.name_scope("skip_connection"):
        conv2 = tf.concat([conv1, conv2], 3)

        conv2 = tf.nn.relu(conv2)
        conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # 3.Convolution using 256 filters
    # using three 3x3 filter to replace one 7x7 filter
    with tf.name_scope("layer3"):
        conv3_W = W_generator([7, 7, 64, 128])
        conv3_b = tf.Variable(tf.zeros(128))
        conv3 = tf.nn.conv2d(conv2, conv3_W, strides=[1, 1, 1, 1], padding='SAME') + conv3_b

        conv3 = tf.layers.batch_normalization(conv3, center=True, scale=True, training=True)
        conv3 = tf.nn.relu(conv3)

        if is_training:
            conv3 = tf.nn.dropout(conv3, keep_prob=dropout_kept_prob[0])

        conv3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    with tf.name_scope("layer4"):
        conv4_W = W_generator([5, 5, 128, 256])
        conv4_b = tf.Variable(tf.zeros(256))
        conv4 = tf.nn.conv2d(conv3, conv4_W, strides=[1, 1, 1, 1], padding='SAME') + conv4_b

        conv4 = tf.layers.batch_normalization(conv4, center=True, scale=True, training=True)

        if is_training:
            conv4 = tf.nn.dropout(conv4, keep_prob=dropout_kept_prob[0])

        conv4 = tf.nn.relu(conv4)
        conv4 = tf.nn.max_pool(conv4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    with tf.name_scope("layer5"):
        conv5_W = W_generator([3, 3, 256, 512])
        conv5_b = tf.Variable(tf.zeros(512))
        conv5 = tf.nn.conv2d(conv4, conv5_W, strides=[1, 1, 1, 1], padding='SAME') + conv5_b

        conv5 = tf.nn.relu(conv5)

        conv5 = tf.nn.avg_pool(conv5, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

        if is_training:
            conv5 = tf.nn.dropout(conv5, keep_prob=dropout_kept_prob[0])

    fc0 = flatten(conv5)

    with tf.name_scope("layer6_fc"):
        fc1_W = W_generator([512, 128])
        fc1_b = tf.Variable(tf.zeros(128))
        fc1 = tf.matmul(fc0, fc1_W) + fc1_b

        if is_training:
            fc1 = tf.nn.dropout(fc1, keep_prob=dropout_kept_prob[1])

        fc1 = tf.nn.relu(fc1)

    with tf.name_scope("layer7_fc"):
        fc2_W = W_generator([128, 10])
        fc2_b = tf.Variable(tf.zeros(10))
        logits = tf.matmul(fc1, fc2_W) + fc2_b

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
    learning_rate = 0.001

    dropout_kept_prob = [0.6, 0.5, 0.6, 0.6, 0.5]

    X = tf.placeholder(tf.float32, shape=(None, 32, 32, 3), name="X")
    Y = tf.placeholder(tf.float32, shape=(None, 10), name="y")

    logits = net(X, True, dropout_kept_prob)

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y)
    loss_operation = tf.reduce_mean(cross_entropy)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    grads_and_vars = optimizer.compute_gradients(loss_operation, tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))
    training_operation = optimizer.apply_gradients(grads_and_vars)

    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name + "/histogram", var)

    add_gradient_summaries(grads_and_vars)
    tf.summary.scalar('loss_operation', loss_operation)
    merged_summary_op = tf.summary.merge_all()

    saver = tf.train.Saver()
    accuracy = 100.0 * tf.reduce_mean(
        tf.cast(tf.equal(tf.argmax(logits, axis=1), tf.argmax(Y, axis=1)), dtype=tf.float32))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        cifar10_train = Cifar10(batch_size=100, one_hot=True, test=False, shuffle=True)

        n_epochs = 100
        n_batches = 100

        for epoch in range(n_epochs):
            # compute model
            for iteration in range(n_batches):
                batch_x, batch_y = cifar10_train.get_next_batch()
                sess.run([training_operation, merged_summary_op], feed_dict={X: batch_x, Y: batch_y})

            acc = accuracy.eval(feed_dict={X: batch_x, Y: batch_y})
            loss = loss_operation.eval(feed_dict={X: batch_x, Y: batch_y})
            print(epoch, "Training batch accuracy:", acc, "loss is: ", loss)

        saver.save(sess, 'ckpt/', global_step=n_epochs)


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

    X = tf.placeholder(tf.float32, shape=cifar10_test_images.shape, name="X")

    logits = net(X, False, 0)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint('ckpt'))
        output = sess.run(logits, feed_dict={X: cifar10_test_images})

        return np.argmax(output, 1)


