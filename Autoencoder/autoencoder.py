# TODO: (Optional)
# Define your convenient function for convolution, batch_norm,... (if applicable)
from MNIST_importer import MNIST
import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import flatten


def AutoEncoder(input_tensor, is_training):
    # input_tensor: image tensor, similar to previous assignment, size [N, 784]
    # is_training: boolean tensor indicate whether if in training phase or testing phase

    # TODO: Define the architecture of your autoencoder here

    n_inputs = 784
    hidden1 = 100
    hidden2 = 20
    hidden3 = 100
    n_outputs = 784

    # input_tensor = tf.placeholder(tf.float32,shape=(None,n_inputs),name="input_image")

    x_init = tf.contrib.layers.xavier_initializer()
    z_init = tf.zeros_initializer()

    with tf.name_scope("weights"):
        W1 = tf.get_variable(dtype=tf.float32, shape=(n_inputs, hidden1), initializer=x_init, name="W1")
        b1 = tf.get_variable(dtype=tf.float32, shape=(1, hidden1), initializer=z_init, name="b1")
        W2 = tf.get_variable(dtype=tf.float32, shape=(hidden1, hidden2), initializer=x_init, name="W2")
        b2 = tf.get_variable(dtype=tf.float32, shape=(1, hidden2), initializer=z_init, name="b2")
        W3 = tf.get_variable(shape=(hidden2, hidden3), initializer=x_init, name="W3")
        b3 = tf.get_variable(dtype=tf.float32, shape=(1, hidden3), initializer=z_init, name="b3")
        W4 = tf.get_variable(shape=(hidden3, n_outputs), initializer=x_init, name="W4")
        b4 = tf.get_variable(dtype=tf.float32, shape=(1, n_outputs), initializer=z_init, name="b4")

    with tf.name_scope("AE"):
        # encoding part
        # add gaussian noise
        # input_noise = input_tensor + tf.random_normal(tf.shape(input_tensor))
        # input_noise = tf.nn.dropout(input_tensor, keep_prob=0.6)

        hidden1 = tf.nn.elu(tf.matmul(input_tensor, W1) + b1)
        hidden1 = tf.layers.batch_normalization(hidden1, center=True, scale=True, training=True)

        hidden2 = tf.nn.elu(tf.matmul(hidden1, W2) + b2)
        hidden2 = tf.layers.batch_normalization(hidden2, center=True, scale=True, training=True)

        fc = flatten(hidden2)
        fc_W = tf.Variable(tf.truncated_normal(shape=(20, 10), mean=0, stddev=0.1))
        fc_b = tf.Variable(tf.zeros(10))
        logits = tf.matmul(fc, fc_W) + fc_b

        # decoding part
        hidden3 = tf.nn.elu(tf.matmul(hidden2, W3) + b3)
        recon = tf.nn.sigmoid(tf.matmul(hidden3, W4) + b4)  # sigmoid limits output within [0,1]

    # return:
    # recon: reconstruction of the image by the autoencoder
    # logits: logits of the classification branch from the bottleneck of the autoencoder
    return recon, logits

def run():
    # General setup
    EPOCHS = 10
    BATCH_SIZE = 64
    NUM_ITERS = int(55000 / BATCH_SIZE * EPOCHS)

    train_set = MNIST('train', batch_size=BATCH_SIZE)
    valid_set = MNIST('valid')
    test_set = MNIST('test', shuffle=False)

    tf.reset_default_graph()
    x = tf.placeholder(tf.float32, (None, 784))
    y = tf.placeholder(tf.int32, (None, 1))
    is_labeled = tf.placeholder(tf.float32, (None, 1))
    is_training = tf.placeholder(tf.bool, ())
    one_hot_y = tf.one_hot(y, 10)

    # create loss
    rate = 0.01
    recon, logits = AutoEncoder(x, is_training=is_training)
    prediction = tf.argmax(logits, axis=1)
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=one_hot_y) * is_labeled)
    recon_loss = tf.reduce_mean((recon - x) ** 2)
    loss_operation = cross_entropy + recon_loss
    optimizer = tf.train.AdamOptimizer(learning_rate=rate)
    grads_and_vars = optimizer.compute_gradients(loss_operation, tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))
    training_operation = optimizer.apply_gradients(grads_and_vars)

    def evaluation(images, true_labels):
        eval_batch_size = 100
        predicted_labels = []
        for start_index in range(0, len(images), eval_batch_size):
            end_index = start_index + eval_batch_size
            batch_x = images[start_index: end_index]
            batch_predicted_labels = sess.run(prediction, feed_dict={x: batch_x, is_training: False})
            predicted_labels += list(batch_predicted_labels)
        predicted_labels = np.vstack(predicted_labels).flatten()
        true_labels = true_labels.flatten()
        accuracy = float((predicted_labels == true_labels).astype(np.int32).sum()) / len(images)
        return predicted_labels, accuracy

    # train
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    print("Training...")
    for i in range(NUM_ITERS):
        batch_x, batch_y, batch_is_labeled = train_set.get_next_batch()

        _ = sess.run(training_operation, feed_dict={x: batch_x, y: batch_y, is_labeled: batch_is_labeled, is_training: True})
        if (i + 1) % 1000 == 0 or i == NUM_ITERS - 1:
            _, validation_accuracy = evaluation(valid_set._images, valid_set._labels)
            print("Iter {}: Validation Accuracy = {:.3f}".format(i, validation_accuracy))

    print('Evaluating on test set')
    _, test_accuracy = evaluation(test_set._images, test_set._labels)
    print("Test Accuracy = {:.3f}".format(test_accuracy))

    sess.close()
    return test_accuracy


if __name__ == '__main__':
    run()