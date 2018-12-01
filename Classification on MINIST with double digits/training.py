import numpy as np
import tensorflow as tf
import data_loader
from Nets import net
from skimage.draw import polygon

from tensorflow.contrib.layers import flatten


def compute_classification_acc(pred, gt):
    # pred and gt are both
    assert pred.shape == gt.shape
    return (pred == gt).astype(int).sum() / gt.size


def compute_iou(b_pred, b_gt):
    # b_pred: predicted bounding boxes, shape=(n,2,4)
    # b_gt: ground truth bounding boxes, shape=(n,2,4)

    n = np.shape(b_gt)[0]
    L_pred = np.zeros((64, 64))
    L_gt = np.zeros((64, 64))
    iou = 0.0
    for i in range(n):
        for b in range(2):
            rr, cc = polygon([b_pred[i, b, 0], b_pred[i, b, 0], b_pred[i, b, 2], b_pred[i, b, 2]],
                             [b_pred[i, b, 1], b_pred[i, b, 3], b_pred[i, b, 3], b_pred[i, b, 1]], [64, 64])
            L_pred[rr, cc] = 1

            rr, cc = polygon([b_gt[i, b, 0], b_gt[i, b, 0], b_gt[i, b, 2], b_gt[i, b, 2]],
                             [b_gt[i, b, 1], b_gt[i, b, 3], b_gt[i, b, 3], b_gt[i, b, 1]], [64, 64])
            L_gt[rr, cc] = 1

            iou += (1.0 / (2 * n)) * (np.sum((L_pred + L_gt) == 2) / np.sum((L_pred + L_gt) >= 1))

            L_pred[:, :] = 0
            L_gt[:, :] = 0

    return iou


def add_gradient_summaries(grads_and_vars):
    for grad, var in grads_and_vars:
        if grad is not None:
            tf.summary.histogram(var.op.name + "/gradient", grad)


def train(x_train, y_train, x_valid, y_valid, train_bbox, valid_bbox, task):
    tf.reset_default_graph()

    # params
    learning_rate = 0.01
    X = tf.placeholder(tf.float32, shape=(None, 64, 64, 1), name="images")

    if task == 'classify':
        Y = tf.placeholder(tf.float32, shape=(None, 2, 10), name="labels")
        logits = net(X, is_training=True, task='classify')
        prediction = tf.argmax(logits, 2)
        loss_function = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y)
        loss = tf.reduce_mean(loss_function)
        # calculate accuracy
        accuracy = 100.0 * tf.reduce_mean(tf.cast(tf.equal(prediction, tf.argmax(Y, 2)), dtype=tf.float32))
        # accuracy = compute_classification_acc(prediction, tf.argmax(Y, 2))

    elif task == 'detection':
        Y = tf.placeholder(tf.float32, shape=(None, 2, 4), name="labels")
        prediction = net(X, is_training=True, task='detection')

        # loss = -compute_iou(prediction,Y)

        loss = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(Y, prediction)), axis=2))) + sum(
            tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

        # calculate accuracy

        # accuracy = 100.0 * tf.reduce_mean(tf.cast(tf.equal(prediction, Y), dtype=tf.float32))

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    grads_and_vars = optimizer.compute_gradients(loss, tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))
    training_operation = optimizer.apply_gradients(grads_and_vars)

    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name + "/histogram", var)

    add_gradient_summaries(grads_and_vars)
    tf.summary.scalar("loss_operation", loss)
    merged_summary_op = tf.summary.merge_all()

    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        batch_size = 100
        n_epochs = 300
        n_batches = 300

        if task == 'classify':
            dataset = dataIterator(x_train, y_train, batch_size)
        elif task == 'detection':
            dataset = dataIterator(x_train, train_bbox, batch_size)

        for epoch in range(n_epochs):
            for iter in range(n_batches):
                batch_x, batch_y = dataset.next_batch()
                sess.run([training_operation, merged_summary_op], feed_dict={X: batch_x, Y: batch_y})
            valid_acc = 0

            # validation
            for i in range(5):
                if task == 'classify':
                    valid_acc += accuracy.eval(
                        feed_dict={X: x_valid[i * 1000:i * 1000 + 1000], Y: y_valid[i * 1000:i * 1000 + 1000]})

                elif task == 'detection':
                    pred = prediction.eval(
                        feed_dict={X: x_valid[i * 1000:i * 1000 + 1000], Y: valid_bbox[i * 1000:i * 1000 + 1000]})
                    valid_acc += compute_iou(pred, valid_bbox[i * 1000:i * 1000 + 1000])

            valid_acc /= 5

            if task == 'classify':
                acc = accuracy.eval(feed_dict={X: batch_x, Y: batch_y})
                losses = loss.eval(feed_dict={X: batch_x, Y: batch_y})

                print(task + ": ", epoch, "Training batch accuracy:", acc, "loss is: ", losses, "valid accuracy is: ",
                      valid_acc)
            elif task == 'detection':
                losses = loss.eval(feed_dict={X: batch_x, Y: batch_y})
                print(task + ": ", epoch, "loss: ", losses, "valid iou is: ", valid_acc)

        saver.save(sess, 'ckpt/', global_step=n_epochs)
