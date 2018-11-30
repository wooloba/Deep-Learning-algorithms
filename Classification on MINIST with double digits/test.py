import numpy as np
import tensorflow as tf
import data_loader
from Nets import classify_net
from tensorflow.contrib.layers import flatten


def test_classify(x_test, y_test):
    tf.reset_default_graph()

    test_batch_size = 1000
    X = tf.placeholder(tf.float32, shape=[test_batch_size, 64, 64, 1], name='X')
    Y = tf.placeholder(tf.float32, shape=(test_batch_size, 2, 10), name="Y")

    logits = classify_net(X, False)

    pred = tf.argmax(logits, 2)

    acc = 100.0 * tf.reduce_mean(tf.cast(tf.equal(pred, tf.argmax(Y, 2)), dtype=tf.float32))

    saver = tf.train.Saver()

    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint('ckpt'))
        accuracy = 0
        for i in range(0, x_test.shape[0] // test_batch_size):
            sess.run(acc, feed_dict={X: x_test[i * test_batch_size:(i + 1) * test_batch_size],
                                     Y: y_test[i * test_batch_size:(i + 1) * test_batch_size]})

            accuracy += acc.eval(feed_dict={X: x_test[i * test_batch_size:(i + 1) * test_batch_size],
                                            Y: y_test[i * test_batch_size:(i + 1) * test_batch_size]})


        accuracy /= x_test.shape[0] // test_batch_size
        return accuracy

def test_detection(x_test,bbox_test):
    return acc