import numpy as np
import tensorflow as tf
import data_loader
from vgg import VGG_net

print("Loading data ...")
x_train,y_train,x_valid,y_valid = data_loader.dataloader()
#divide data into 45000 and 1000 for trainning data and test data
x_train,y_train,x_test,y_test = data_loader.data_spliter(x_train,y_train)

print("Shape:",x_train.shape, y_train.shape, x_test.shape, y_test.shape)


def train():
    tf.reset_default_graph()

    #params
    learning_rate = 0.01

    X = tf.placeholder(tf.float32, shape=(None,64,64,1) , name="images")
    Y = tf.placeholder(tf.float32,shape=(None,10) , name = "labels")

    logits = VGG_net(X,is_training= True)

