# -*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np
from numpy.random import RandomState
import matplotlib.pyplot as plt

# prepare data
rdm = RandomState(1)
dataset_size = 128
X = rdm.rand(dataset_size, 2)
Y = [[int(x1+x2 < 1)] for (x1, x2) in X]


# batch size
batch_size = 8
# ----------------------------------------

# Step:1
# 定义神经网络参数, 
w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1), name='w1')
w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1), name='w2')
bias1 = tf.Variable(tf.zeros([1]))  # important!!  
bias2 = tf.Variable(tf.zeros([1]))  # without bias can't learn a right model

# 定义输入
x = tf.placeholder(tf.float32, shape=[None, 2], name='x-input')
y_ = tf.placeholder(tf.float32, shape=[None, 1], name='y-input')

# forward
a = tf.add(tf.matmul(x, w1), bias1)
y = tf.add(tf.matmul(a, w2), bias2)

# output sigmoid
y = tf.sigmoid(y)


# Step:2
# cross-entropy loss
loss = tf.reduce_mean(-(y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0)) + (1-y_) * tf.log(tf.clip_by_value((1-y), 1e-10, 1.0))))

# optimize
optimizer = tf.train.GradientDescentOptimizer(0.001)
train_op = optimizer.minimize(loss)


# Step:3
init_op = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init_op)

    print('weights before training:')
    print(sess.run(w1))
    print(sess.run(w2))

    STEPS = 20000
    for i in range(STEPS):
        # get batch
        start = (i * batch_size) % dataset_size
        end = min(start+batch_size, dataset_size)
        x_batch = X[start:end]
        y_batch = Y[start:end]

        sess.run(train_op, feed_dict={x:x_batch, y_:y_batch})

        # print infos, such as: loss ...
        if i % 10000 == 0:
            total_cross_entropy = sess.run(loss, feed_dict={x:X, y_:Y})
            print("After %d training steps, cross entropy on all data is %g" % (i, total_cross_entropy))

    print('weights after training:')
    print(sess.run(w1))
    print(sess.run(w2))

    # show result
    for i in range(dataset_size):
        if Y[i][0] == 1:
            plt.scatter(X[i,0], X[i,1], c='b', marker='o')
        else:
            plt.scatter(X[i,0], X[i,1], c='r', marker='o')

    Y_res = sess.run(y, feed_dict={x:X})
    for i in range(dataset_size):
        if Y_res[i][0] >= 0.5:
            plt.scatter(X[i,0], X[i,1], c='b', marker='x')
        else:
            plt.scatter(X[i,0], X[i,1], c='r', marker='x')

    plt.show()
