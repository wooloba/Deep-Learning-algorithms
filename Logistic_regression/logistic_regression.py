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


def logistic_regression(dataset_name, x_train, y_train, x_valid, y_valid, x_test):
    #x_train: 45000,784  /33,33,3
    #y_train: 45000,1
    #x_valid: 10000,784
    #y_valid: 10000,1

    #hyper parameters
    lamda = 1
    batchsize = 50

    #preprocessing
    y_train = one_hot(y_train,10)
    y_valid = one_hot(y_valid,10)


    if dataset_name == "MNIST":
        Xmean = np.mean(x_train,axis=0) #
        Ymean = np.mean(y_train,axis=0) #mean of training labels

        X = tf.placeholder(dtype= tf.float32,shape=[None,784]) #a single image 1 X 784(28*28)
        y = tf.placeholder(dtype= tf.float32,shape=[None,10])

        W = tf.Variable(tf.random_normal([784,10])) #Parameters

        #Si = (xi - X(mean))' * W + Y(mean)
        score = tf.matmul((X - Xmean), W) + Ymean

        Yp = tf.nn.softmax(score)

        #loss function
        loss_function = -tf.reduce_mean(y * tf.log(Yp))

        # add L1 regularization
        #loss_function += lamda* tf.reduce_mean(tf.abs(W))
        # L2 regularization
        loss_function += lamda * tf.reduce_mean(tf.square(W))

        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
        training_op = optimizer.minimize(loss_function)


        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            # If dump all data into memory, it will cause the system crashed unless you have lots of ram installed. To avoid this, the best way is to split the input into different batches, then read in and train each batch.
            dataset = DatasetIterator(x_train,y_train,batchsize)

            for i in range((len(x_train)*100)/batchsize):
                #print(len(x_train)/batchsize)
                [batch_x, batch_y] =  dataset.next_batch()
                sess.run(training_op,feed_dict={X:batch_x ,y:batch_y})

                c = sess.run(loss_function,feed_dict = {X:batch_x,y:batch_y})
                # if not i % 5:
                #     print(c)
                #
                #     W_training = W.eval()
                #     yp_training = Yp.eval(feed_dict={X:x_valid,W:W_training})


            trained_W = W.eval()
            yp_test = tf.argmax(Yp.eval(feed_dict= {X:x_test,W: trained_W}),1).eval()

            print(len(yp_test))
            return  yp_test

    elif dataset_name == "CIFAR10":
        pass


    # This is just a random function that return random label
    # TODO: implement logistic regression hyper-parameter tuning here
    #return np.random.randint(max(y_train) + 1, size=len(x_test))

