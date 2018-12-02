import numpy as np
import tensorflow as tf
import data_loader
from Nets import net
from tensorflow.contrib.layers import flatten
from training import compute_classification_acc,compute_iou

def evaluation(pred, task,y_test):
    # pred_class: Your predicted labels for the 2 digits, shape [N, 2]
    # pred_bboxes: Your predicted bboxes for 2 digits, shape [N, 2, 4]

    if task == "classify":
        acc = compute_classification_acc(pred, y_test)
        # print(f"Classification Acc: {acc}")
    elif task == 'detection':
        acc = compute_iou(pred, y_test)
        # print(f"BBoxes IOU: {iou}")

    return acc


def test(x_test, y_test, test_bbox, task):
    tf.reset_default_graph()

    test_batch_size = x_test.shape[0]
    X = tf.placeholder(tf.float32, shape=[test_batch_size, 64, 64, 1], name='X')

    if task == "classify":
        Y = tf.placeholder(tf.float32, shape=(test_batch_size, 2, 10), name="Y")
        logits = net(X, False, task='classify')
        pred = tf.argmax(logits, 2)

        Y_test = y_test


    elif task == 'detection':
        Y = tf.placeholder(tf.float32, shape=(test_batch_size, 2, 4), name="Y")
        pred = net(X, False, task='detection')

        Y_test = test_bbox

    saver = tf.train.Saver()

    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint(task))

        prediction = pred.eval(feed_dict={X: x_test, Y: Y_test})

        accuracy = evaluation(prediction,task,Y_test)

        return accuracy