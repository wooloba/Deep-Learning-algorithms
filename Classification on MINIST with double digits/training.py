import numpy as np
import tensorflow as tf
import data_loader
from vgg import VGG_net
import timeit
from tensorflow.contrib.layers import flatten



def add_gradient_summaries(grads_and_vars):
    for grad, var in grads_and_vars:
        if grad is not None:
            tf.summary.histogram(var.op.name + "/gradient", grad)


def train(x_train,y_train):
    tf.reset_default_graph()

    #params
    learning_rate = 0.01

    X = tf.placeholder(tf.float32, shape=(None,64,64,1) , name="images")
    Y = tf.placeholder(tf.float32,shape=(None,10) , name = "labels")

    logits = VGG_net(X,is_training= True)

    prediction = tf.argmax(logits,2)

    loss_function = tf.nn.softmax_cross_entropy_with_logits(logits = logits,labels=Y)
    loss = tf.reduce_mean(loss_function)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    grads_and_vars = optimizer.compute_gradients(loss,tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))
    training_operation = optimizer.apply_gradients(grads_and_vars)

    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name + "/histogram", var)

    add_gradient_summaries(grads_and_vars)
    tf.summary.scalar("loss_operation",training_operation)
    merged_summary_op = tf.summary.merge_all()

    saver = tf.train.Saver()

    #calculate accuracy
    accuracy = 100.0 * tf.reduce_mean(tf.cast(tf.equal(prediction,tf.argmax(Y,2))))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        batch_size = 100
        dataset = data_loader.dataIterator(x_train,y_train,batch_size)
        n_epochs = 100
        n_batches = 100

        for epoch in range(n_epochs):

            for iter in range(n_batches):
                batch_x,batch_y = dataset.next_batch()
                sess.run([training_operation,merged_summary_op],feed_dict={X:batch_x,Y:batch_y})


            acc = accuracy.eval(feed_dict={X:batch_x,Y:batch_y})
            losses = loss.eval(feed_dict={X:batch_x,Y:batch_y})
            print(epoch, "Training batch accuracy:", acc, "loss is: ", losses)

        saver.save(sess, 'ckpt/', global_step=n_epochs)

def test(x_test):
    tf.reset_default_graph()

    X = tf.placeholder(tf.float32,shape= x_test.shape, name = 'X')

    logits = VGG_net(X,False)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        saver.restore(sess,tf.train.latest_checkpoint('ckpt'))

        output = sess.run(logits,feed_dict={X:x_test})

        return np.argmax(output,2)


if __name__ == '__main__':
    print("Loading data ...")
    x_train, y_train, x_valid, y_valid = data_loader.dataloader()
    # divide data into 45000 and 1000 for trainning data and test data
    x_train, y_train, x_test, y_test = data_loader.data_spliter(x_train, y_train)

    print("Shape:", x_train.shape, y_train.shape, x_test.shape, y_test.shape)


    train(x_train,y_train)

    time_start = timeit.default_timer()
    np.random.seed(0)
    pred = test(x_test)

    np.random.seed()
    run_time =time_start - timeit.default_timer()

    correct_pred = (y_test.flatten() == pred.flatten()).astype(np.int32).sum()

    accuracy = float(len(y_test) - correct_pred) / len(y_test)

    print("Test accuracy is : " + str(accuracy))