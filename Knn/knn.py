import tensorflow as tf
import numpy as np
import math

def knn(x_train, y_train, x_test):
    """
    x_train: 60000 x 784 matrix: each row is a flattened image of an MNIST digit
    y_train: 60000 vector: label for x_train
    x_test: 5000 x 784 testing images
    return: predicted y_test which is a 5000 vector
    """

    #run time: 184
    # k = 7
    # label_list = []
    # for i,e in enumerate(x_test):
    #     print(i)
    #     distance = abs(x_train - e)
    #     distance = np.sum(distance,axis=1)
    #
    #     dic = sorted(zip(distance,(e for e in range(len(distance)))))
    #     #dic = sorted(dic)
    #     pool = []
    #     for i in range(k):
    #         pool.append(y_train[dic[i][1]])
    #
    #     pool = sorted(pool,key = pool.count,reverse=True)
    #
    #     label_list.append(pool[0])
    # return label_list

    #runtime 107
    g = tf.Graph()
    with g.as_default() as g:
        k = 3
        tf_x_train = tf.constant(x_train,
                                 dtype=tf.float32,
                                 name='x_train')

        tf_x_test = tf.Variable(x_test,
                                dtype=tf.float32,
                                name= 'x_test')

        tf_i = tf.placeholder(dtype=tf.int32,name='i')
        distance = tf.placeholder(shape=(len(x_test),1),dtype=tf.float32,name='distance')
        distance = tf.reduce_sum((tf.abs(tf_x_train - tf_x_test[tf_i])),axis=1)
        distance = distance*-1

        topK = tf.nn.top_k(distance, k=k, sorted=True, name='topK')

    with tf.Session(graph=g) as sess:
        sess.run(tf.global_variables_initializer())
        label_list = []
        for i in range(len(x_test)):
            topK_sess = sess.run(topK, feed_dict = {tf_i : i})
            pool = []
            for j in topK_sess[1]:
                pool.append(y_train[abs(j)])
            pool = sorted(pool, key=pool.count, reverse=True)
            label_list.append(pool[0])
    return label_list