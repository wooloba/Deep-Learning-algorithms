import tensorflow as tf
import numpy as np

#Yaozhi Lu

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



# def cross_Validation(dataset_name, x_valid, y_valid):
#     y_valid = one_hot(y_valid, 10)
#     k = 100
#     lamda = tf.placeholder(dtype=tf.float32)
#
#     if dataset_name == "CIFAR10":
#
#         one_image_shape = 32*32*3
#         x_valid = x_valid.reshape(10000,one_image_shape)
#
#
#     elif dataset_name == "MNIST":
#         one_image_shape = 28 * 28
#
#
#     Xmean = np.mean(x_valid, axis=0).reshape(1, one_image_shape)  #
#     Ymean = np.mean(y_valid, axis=0)  # mean of training labels
#
#     X = tf.placeholder(dtype=tf.float32, shape=[None, one_image_shape])  # a single image 1 X 784(28*28) or 1 X 3072
#     y = tf.placeholder(dtype=tf.float32, shape=[None, 10])
#
#     W = tf.Variable(tf.zeros([one_image_shape, 10]))  # Parameters
#     score = tf.matmul((X - Xmean), W) + Ymean
#     Yp = tf.nn.softmax(score)
#     loss_function = -tf.reduce_mean(y * tf.log(Yp)) + lamda * tf.reduce_mean(tf.square(W))
#
#     optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
#     training_op = optimizer.minimize(loss_function)
#
#     y1 = tf.placeholder(tf.float32, [None, 10])
#     y2 = tf.placeholder(tf.float32, [None, 10])
#     acc = 100.0 * tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y1, axis=1), tf.argmax(y2, axis=1)), tf.float32))
#
#
#     with tf.Session() as sess:
#         sess.run(tf.global_variables_initializer())
#
#         bestLamda = -1
#         acc_test = 0
#         batchsize = 50
#
#         lamda_list = [e/1000.0 for e in range(500,1501)]
#         print(lamda_list)
#         for i in range(len(x_valid)/k):
#             lamda_test = lamda_list[i]
#
#             #print(len(x_train)/batchsize)
#             #[batch_x, batch_y] =  dataset.next_batch()
#             start = i*k
#             x_k_valid = x_valid[start:start+k]
#             y_k_valid = y_valid[start:start+k]
#
#             x_k_feed = tf.concat((x_valid[0:start],x_valid[start+k:len(x_valid)]),axis=0).eval()
#             y_k_feed = tf.concat((y_valid[0:start],y_valid[start+k:len(y_valid)]),axis=0).eval()
#
#
#             for j in range((len(x_valid)-k)/ batchsize):
#                 sess.run(training_op,feed_dict={X:x_k_feed ,y:y_k_feed, lamda:lamda_test})
#
#             W_valid = W.eval()
#
#             yp_train = Yp.eval(feed_dict={X: x_k_valid, W: W_valid})
#             acc_train = acc.eval(feed_dict={y1: y_k_valid, y2: yp_train})
#
#             if acc_train > acc_test:
#                 acc_test = acc_train
#                 bestLamda = lamda_test
#             print(dataset_name + '   ' + str(acc_test) + '   ' + str(bestLamda))
#         return bestLamda

def logistic_regression(dataset_name, x_train, y_train, x_valid, y_valid, x_test):
    #hyper parameters
    # MNIST 32.0  lambda: 0.063

    batchsize = 100
    #cross_Validation(dataset_name, x_valid, y_valid)

    #preprocessing
    y_train = one_hot(y_train,10)

    if dataset_name == "CIFAR10":
        run_loop = 400
        lamda = 1
        one_image_shape = 32*32*3
        x_train = x_train.reshape(40000,one_image_shape)
        x_test = x_test.reshape(10000,one_image_shape)
        print("MNIST done, Start CIFAR10.")
        #return 0

    elif dataset_name == "MNIST":
        #lamda = 0.0000000001    # 0.9134
        run_loop = 300
        lamda = 0.07
        #lamda = 0.00001
        one_image_shape = 28 * 28
        # return 0

    Xmean = np.mean(x_train,axis=0) #

    X = tf.placeholder(dtype=tf.float64, shape=[None, one_image_shape])  # a single image 1 X 784(28*28) or 1 X 3072
    y = tf.placeholder(dtype=tf.float64, shape=[None, 10])

    W = tf.Variable(tf.zeros([one_image_shape,10],dtype=tf.float64)) #Parameters

    #Si = (xi - X(mean))' * W + Y(mean)
    score = tf.matmul((X - Xmean), W)
    Yp = tf.nn.softmax(score)

    #loss function + L2 regularization
    loss_function = -tf.reduce_mean(y * tf.log(Yp + 0.001)) + lamda * tf.reduce_mean(tf.square(W))
    # add L1 regularization
    #loss_function += lamda* tf.reduce_mean(tf.abs(W))
    if dataset_name == "MNIST":
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
    else:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)

    training_op = optimizer.minimize(loss_function)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        dataset = DatasetIterator(x_train,y_train,batchsize)

        for i in range((len(x_train)/batchsize) * run_loop):
            [batch_x, batch_y] =  dataset.next_batch()
            sess.run(training_op,feed_dict={X:batch_x.reshape(batchsize,one_image_shape) ,y:batch_y})

        trained_W = W.eval()
        yp_test = tf.argmax(Yp.eval(feed_dict= {X:x_test.reshape(len(x_test),one_image_shape),W: trained_W}),1).eval()

        return  yp_test


#('\nResult:\n', "OrderedDict ([
# ('MNIST', OrderedDict([('run_time', 163.0433759689331), ('correct_predict', 9222), ('score', 100.0), ('accuracy', 0.9222)])),
#('CIFAR10', OrderedDict([('run_time', 480.2080068588257), ('correct_predict', 4048), ('score', 100.0), ('accuracy', 0.4048)])),
# ('total_score', 100.0)])")
