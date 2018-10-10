from ops import *
import timeit
from cifar10 import Cifar10
from tensorflow.contrib.layers import flatten
def W_generator(shape):
    mu = 0
    sigma = 0.1

    W = tf.Variable(tf.truncated_normal(shape, mean= mu, stddev= sigma))

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

    #first layer: Convolutional layer Input :32x32x3, Output:
    with tf.name_scope("layer1"):
        conv1_W = W_generator([5,5,3,16])
        conv1_b = tf.Variable(tf.zeros(16))
        conv1 = tf.nn.conv2d(input, conv1_W, strides=[1,1,1,1], padding='VALID') + conv1_b

        #Activation
        conv1 = tf.nn.relu(conv1)

        # Max Pooling
        conv1 = tf.nn.max_pool(conv1, ksize=[1,3,3,1], strides=[1,2,2,1], padding='VALID')


    with tf.name_scope("layer2"):
        conv2_W = W_generator([5, 5, 16, 32])
        conv2_b = tf.Variable(tf.zeros(32))
        conv2 = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b

        # Activation
        conv2 = tf.nn.relu(conv2)

        #Max Pooling
        conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    fc0 = flatten(conv2)

    with tf.name_scope("layer3_fc"):
        fc1_W = W_generator([512,120])
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

    learning_rate = 0.001

    X = tf.placeholder(tf.float32, shape=(None,32,32,3), name="X")
    Y = tf.placeholder(tf.float32, shape=(None, 10), name="y")

    logits = net(X,True,None)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y)
    loss_operation = tf.reduce_mean(cross_entropy)
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
    grads_and_vars = optimizer.compute_gradients(loss_operation, tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))
    training_operation = optimizer.apply_gradients(grads_and_vars)

    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name + "/histogram", var)

    add_gradient_summaries(grads_and_vars)
    tf.summary.scalar('loss_operation', loss_operation)
    merged_summary_op = tf.summary.merge_all()
    #summary_writer = tf.summary.FileWriter('logs/')

    saver = tf.train.Saver()

    accuracy = 100.0 * tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, axis=1), tf.argmax(Y, axis=1)),
                                          dtype=tf.float32))

    with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            cifar10_train = Cifar10(batch_size=100, one_hot=True, test=False, shuffle=True)

            n_epochs = 100
            n_batches = 100

            for epoch in range(n_epochs):
                # compute model
                for iteration in range(n_batches):
                    batch_x, batch_y = cifar10_train.get_next_batch()
                    sess.run([training_operation,merged_summary_op],feed_dict={X:batch_x,Y:batch_y})
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

    X = tf.placeholder(tf.float32, shape=(None, cifar10_test_images.shape), name="X")
    #Y = tf.placeholder(tf.float32, shape=(None, 10), name="y")
    #prediction = tf.placeholder(tf.float32, shape=(None, 10), name="y")

    logits = net(X,True,0.1)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint('ckpt'))
        sess.run(logits, feed_dict={X: cifar10_test_images})

        return np.argmax(logits,1)


