import numpy as np
import tensorflow as tf
import data_loader
from Nets import classify_net
from tensorflow.contrib.layers import flatten


def add_gradient_summaries(grads_and_vars):
    for grad, var in grads_and_vars:
        if grad is not None:
            tf.summary.histogram(var.op.name + "/gradient", grad)


def train_classify(x_train, y_train,x_valid,y_valid):
    tf.reset_default_graph()

    # params
    learning_rate = 0.01
    X = tf.placeholder(tf.float32, shape=(None, 64, 64, 1), name="images")
    Y = tf.placeholder(tf.float32, shape=(None, 2, 10), name="labels")

    logits = classify_net(X, is_training=True)

    prediction = tf.argmax(logits, 2)

    loss_function = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y)
    loss = tf.reduce_mean(loss_function)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    grads_and_vars = optimizer.compute_gradients(loss, tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))
    training_operation = optimizer.apply_gradients(grads_and_vars)

    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name + "/histogram", var)

    add_gradient_summaries(grads_and_vars)
    tf.summary.scalar("loss_operation", loss)
    merged_summary_op = tf.summary.merge_all()

    saver = tf.train.Saver()

    # calculate accuracy
    accuracy = 100.0 * tf.reduce_mean(tf.cast(tf.equal(prediction, tf.argmax(Y, 2)), dtype=tf.float32))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        batch_size = 100
        dataset = data_loader.dataIterator(x_train, y_train, batch_size)
        n_epochs = 2
        n_batches = 2

        for epoch in range(n_epochs):

            for iter in range(n_batches):
                batch_x, batch_y = dataset.next_batch()

                sess.run([training_operation, merged_summary_op], feed_dict={X: batch_x, Y: batch_y})

            valid_acc = 0
            for i in range(5):
                valid_acc += accuracy.eval(
                    feed_dict={X: x_valid[i * 1000:i * 1000 + 1000], Y: y_valid[i * 1000:i * 1000 + 1000]})
            valid_acc /= 5

            acc = accuracy.eval(feed_dict={X: batch_x, Y: batch_y})
            losses = loss.eval(feed_dict={X: batch_x, Y: batch_y})
            print(epoch, "Training batch accuracy:", acc, "loss is: ", losses, "valid accuracy is: ", valid_acc)

        saver.save(sess, 'ckpt/', global_step=n_epochs)

def train_detection(x_train, train_bboxes,x_valid,valid_bboxes):




    return