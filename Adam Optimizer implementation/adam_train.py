import tensorflow as tf
import numpy as np


class DatasetIterator:
    def __init__(self, x, y, batch_size):
        assert len(x) == len(y)
        self.x = x
        self.y = y
        self.b_sz = batch_size
        self.b_pt = 0
        self.d_sz = len(x)
        self.idx = None
        self.randomize()

    def randomize(self):
        self.idx = np.random.permutation(self.d_sz)
        self.b_pt = 0

    def next_batch(self):
        start = self.b_pt
        end = self.b_pt + self.b_sz
        idx = self.idx[start:end]
        x = self.x[idx]
        y = self.y[idx]

        self.b_pt += self.b_sz
        if self.b_pt >= self.d_sz:
            self.randomize()

        return x, y


def one_hot(a, num_classes):
    return np.squeeze(np.eye(num_classes)[a.reshape(-1)])


def adam_train(x_train, y_train, x_valid, y_valid, x_test):
    # TODO: Make sure you set all the random seed properly at the beginning
    # of this function so your code won't
    # have different output every time you run

    x_train = x_train.reshape((len(x_train), -1))
    x_valid = x_valid.reshape((len(x_valid), -1))
    x_test = x_test.reshape((len(x_test), -1))
    y_train = one_hot(y_train, 10)
    y_valid = one_hot(y_valid, 10)

    n_inputs = 32 * 32 * 3
    n_hidden1 = 200
    n_hidden2 = 100
    n_outputs = 10

    X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
    y = tf.placeholder(tf.float32, shape=(None, n_outputs), name="y")

    def neuron_layer(X, n_neurons, name, activation=None):
        with tf.name_scope(name):
            n_inputs = int(X.get_shape()[1])
            stddev = 2 / np.sqrt(n_inputs)
            init = tf.truncated_normal((n_inputs, n_neurons), stddev=stddev)
            W = tf.Variable(init, name="kernel")
            b = tf.Variable(tf.zeros([1, n_neurons]), name="bias")
            Z = tf.matmul(X, W) + b
            if activation is not None:
                return activation(Z), W, b
            else:
                return Z, W, b

    with tf.name_scope("dnn"):
        hidden1, W1, b1 = neuron_layer(X, n_hidden1, name="hidden1", activation=tf.nn.relu)
        hidden2, W2, b2 = neuron_layer(hidden1, n_hidden2, name="hidden2", activation=tf.nn.relu)
        logits, W3, b3 = neuron_layer(hidden2, n_outputs, name="outputs")

    with tf.name_scope("loss"):
        yp = tf.nn.softmax(logits)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits))

    with tf.name_scope("backprop"):
        d_logits = yp - y

        d_hidden2 = tf.matmul(d_logits, tf.transpose(W3))
        d_W3 = tf.matmul(tf.transpose(hidden2), d_logits)
        d_b3 = tf.reduce_sum(d_logits, axis=0, keep_dims=True)

        d_2 = tf.to_float(tf.greater(tf.matmul(hidden1, W2) + b2, 0)) * d_hidden2
        d_hidden1 = tf.matmul(d_2, tf.transpose(W2))
        d_W2 = tf.matmul(tf.transpose(hidden1), d_2)
        d_b2 = tf.reduce_sum(d_2, axis=0, keep_dims=True)

        d_1 = tf.to_float(tf.greater(tf.matmul(X, W1) + b1, 0)) * d_hidden1
        d_W1 = tf.matmul(tf.transpose(X), d_1)
        d_b1 = tf.reduce_sum(d_1, axis=0, keep_dims=True)

    learning_rate = 0.0001  # this is a good learning rate. you can change

    update_ops = []  # contains the ops to update auxilary variables like
    # beta_power, m and v
    training_ops = []  # contains the ops to update variables
    eps = 1e-8

    # list of params and gradient
    Vs = [W1, b1, W2, b2, W3, b3]
    dVs = [d_W1, d_b1, d_W2, d_b2, d_W3, d_b3]
    # set betas
    beta1 = 0.9
    beta2 = 0.999

    # TODO: write all the code to update betas, m, v, compute m_hat, v_hat
    # and update all the variables here.
    # Add all tensorflow ops to update betas, m,v to updates_ops
    # Add all ops to update V to training_ops

    #minimize loss function
    theta = 0
    with tf.name_scope("adam"):


    with tf.name_scope("eval"):
        accuracy = 100.0 * tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, axis=1), tf.argmax(y, axis=1)),
                                                  dtype=tf.float32))

    init = tf.global_variables_initializer()

    n_epochs = 100
    batch_size = 200
    n_batches = len(x_train) // batch_size

    dataset_iterator = DatasetIterator(x_train, y_train, batch_size)

    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(n_epochs):
            # compute model
            for iteration in range(n_batches):
                x_batch, y_batch = dataset_iterator.next_batch()
                sess.run(update_ops, feed_dict={X: x_batch, y: y_batch})
                sess.run(training_ops, feed_dict={X: x_batch, y: y_batch})

            acc_train = accuracy.eval(feed_dict={X: x_batch, y: y_batch})
            acc_validation = accuracy.eval(feed_dict={X: x_valid, y: y_valid})
            print(epoch, "Training batch accuracy:", acc_train, "Validation set accuracy:", acc_validation)

        # Now that the model is trained, it is the test time!
        yp_test = sess.run(tf.argmax(logits, axis=1), feed_dict={X: x_test})

    return yp_test
