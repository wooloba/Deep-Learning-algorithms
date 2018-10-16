from ops import *
import timeit
from cifar10 import Cifar10
from tensorflow.contrib.layers import flatten
import tensorflow as tf


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
        conv1_1 = tf.nn.conv2d(input, conv1_W1,strides=[1, 1, 1, 1], padding='SAME') + conv1_b1

        #Dropout
        #conv1_1 = tf.nn.dropout(conv1_1,keep_prob=dropout_kept_prob[0])

        conv1_W2 = W_generator([3,3,64,64])
        conv1_b2 = tf.Variable(tf.zeros(64))
        conv1_2 = tf.nn.conv2d(conv1_1, conv1_W2, strides=[1, 1, 1, 1], padding='SAME') + conv1_b2

        conv1_2 = tf.nn.relu(conv1_2)
        conv1_2 = tf.nn.max_pool(conv1_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],padding='SAME')
        conv1_2 = tf.nn.dropout(conv1_2, keep_prob=dropout_kept_prob[0])



        # Dropout

    #2.Convolution using 128 filters
    # using two 3x3 filter to replace one 5x5 filter
    with tf.name_scope("layer2"):
        # Batch_Normalization:
        conv1_2 = tf.layers.batch_normalization(conv1_2, center=True, scale=True, training=is_training)

        conv2_W1 = W_generator([3,3,64,128])
        conv2_b1 = tf.Variable(tf.zeros(128))
        conv2_1 = tf.nn.conv2d(conv1_2, conv2_W1,strides=[1, 1, 1, 1], padding='SAME') + conv2_b1

        # Dropout
        #conv2_1 = tf.nn.dropout(conv2_1,keep_prob=dropout_kept_prob[1])

        conv2_W2 = W_generator([3, 3, 128, 128])
        conv2_b2 = tf.Variable(tf.zeros(128))
        conv2_2 = tf.nn.conv2d(conv2_1, conv2_W2, strides=[1, 1, 1, 1], padding='SAME') + conv2_b2



        conv2_2 = tf.nn.relu(conv2_2)

        conv2_2 = tf.nn.max_pool(conv2_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        # Dropout
        conv2_2 = tf.nn.dropout(conv2_2,keep_prob=dropout_kept_prob[1])

    # 3.Convolution using 256 filters
    # using three 3x3 filter to replace one 7x7 filter
    with tf.name_scope("layer3"):
        # Batch_Normalization:
        conv2_2 = tf.layers.batch_normalization(conv2_2, center=True, scale=True, training=is_training)


        conv3_W1 = W_generator([3, 3, 128, 256])
        conv3_b1 = tf.Variable(tf.zeros(256))
        conv3_1 = tf.nn.conv2d(conv2_2, conv3_W1, strides=[1, 1, 1, 1], padding='SAME') + conv3_b1
        # Dropout
        #conv3_1 = tf.nn.dropout(conv3_1,keep_prob=dropout_kept_prob[2])

        conv3_W2 = W_generator([3, 3, 256, 256])
        conv3_b2 = tf.Variable(tf.zeros(256))
        conv3_2 = tf.nn.conv2d(conv3_1, conv3_W2, strides=[1, 1, 1, 1], padding='SAME') + conv3_b2
        # Dropout
        #conv3_2 = tf.nn.dropout(conv3_2,keep_prob=dropout_kept_prob[2])

        conv3_W3 = W_generator([3, 3, 256, 256])
        conv3_b3 = tf.Variable(tf.zeros(256))
        conv3_3 = tf.nn.conv2d(conv3_2, conv3_W3, strides=[1, 1, 1, 1], padding='SAME') + conv3_b3

        conv3_3 = tf.nn.relu(conv3_3)

        conv3_3 = tf.nn.max_pool(conv3_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        # Dropout
        conv3_3 = tf.nn.dropout(conv3_3,keep_prob=dropout_kept_prob[2])

    with tf.name_scope("layer4"):
        # Batch_Normalization:
        conv3_3 = tf.layers.batch_normalization(conv3_3, center=True, scale=True, training=is_training)


        conv4_W1 = W_generator([3, 3, 256, 512])
        conv4_b1 = tf.Variable(tf.zeros(512))
        conv4_1 = tf.nn.conv2d(conv3_3, conv4_W1, strides=[1, 1, 1, 1], padding='SAME') + conv4_b1
        # Dropout
        conv4_1 = tf.nn.dropout(conv4_1,keep_prob=dropout_kept_prob[3])

        conv4_W2 = W_generator([3, 3, 512, 512])
        conv4_b2 = tf.Variable(tf.zeros(512))
        conv4_2 = tf.nn.conv2d(conv4_1, conv4_W2, strides=[1, 1, 1, 1], padding='SAME') + conv4_b2

        conv4_2 = tf.nn.relu(conv4_2)

        conv4_2 = tf.nn.max_pool(conv4_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        # Dropout
        conv4_2 = tf.nn.dropout(conv4_2,keep_prob=dropout_kept_prob[3])

    with tf.name_scope("skip_connection"):
        conv4_2 = tf.nn.relu(conv4_2)

    with tf.name_scope("layer5"):
        conv5_W1 = W_generator([3, 3, 512, 1024])
        conv5_b1 = tf.Variable(tf.zeros(1024))
        conv5_1 = tf.nn.conv2d(conv4_2, conv5_W1, strides=[1, 1, 1, 1], padding='SAME') + conv5_b1

        conv5_1 = tf.nn.avg_pool(conv5_1,ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        #Dropout
        conv5_1 = tf.nn.dropout(conv5_1,keep_prob=dropout_kept_prob[4])
    fc0 = flatten(conv5_1)

    with tf.name_scope("layer6_fc"):
        fc1_W = W_generator([1024, 10])
        fc1_b = tf.Variable(tf.zeros(10))
        logits = tf.matmul(fc0, fc1_W) + fc1_b

        #fc1 = tf.nn.relu(fc1)

    # with tf.name_scope("layer4_fc"):
    #     fc2_W = W_generator([120, 84])
    #     fc2_b = tf.Variable(tf.zeros(84))
    #     fc2 = tf.matmul(fc1, fc2_W) + fc2_b
    #
    #     fc2 = tf.nn.relu(fc2)
    #
    # with tf.name_scope("layer5_fc"):
    #     fc3_W = W_generator([84, 10])
    #     fc3_b = tf.Variable(tf.zeros(10))
    #     logits = tf.matmul(fc2, fc3_W) + fc3_b
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

    dropout_kept_prob = [0.7,0.7,0.6,0.6,0.5]

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
    # accuracy = 100.0 * tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, axis=1), tf.argmax(Y, axis=1)),dtype=tf.float32))

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

    dropout_kept_prob = [0.7, 0.7, 0.6, 0.6, 0.5]

    logits = net(X, False, dropout_kept_prob)

    saver = tf.train.Saver()

    with tf.Session() as sess:
      saver.restore(sess, tf.train.latest_checkpoint('ckpt'))
      output = sess.run(logits, feed_dict={X: cifar10_test_images})

      return np.argmax(output, 1)
