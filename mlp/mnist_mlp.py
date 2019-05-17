# -*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np 
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt


# MNIST 数据集相关常数
INPUT_NODE = 784  # 28x28
OUTPUT_NODE = 10  # 0-9

# 配置神经网络的参数
LAYER1_NODE = 500 # hidden layer
BATCH_SIZE = 100  # BGD

LEARNING_RATE_BASE = 0.8    # 初始学习率
LEARNING_RATE_DECAY = 0.99  # 学习率衰减率

REGULARIZATION_RATE = 0.0001  # 正则参数 lambda
TRAINING_STEPS = 30000        # 总轮数
MOVING_AVERAGE_DECAY = 0.99   # 滑动平均衰减率

# 前向传播
def inference(input_tensor, avg_class, weights1, biases1, weights2, biases2):
    '''
    input_tensor, 输入图像
    avg_class, 用于滑动平均的类
    weights, biases, 网络参数
    '''
    if avg_class == None:
        # layer1
        layer1 = tf.nn.relu(tf.add(tf.matmul(input_tensor, weights1), biases1))
        output = tf.add(tf.matmul(layer1, weights2), biases2)
        return output

    else:
        # 在使用参数之前, 先计算滑动平均值
        weights1_avg = avg_class.average(weights1)
        biases1_avg = avg_class.average(biases1)
        weights2_avg = avg_class.average(weights2)
        biases2_avg = avg_class.average(biases2)

        layer1 = tf.nn.relu(tf.add(tf.matmul(input_tensor, weights1_avg), biases1_avg))  # 顺序倾向:relu->BN->dropout
        output = tf.add(tf.matmul(layer1, weights2_avg), biases2_avg)                    # 这里未用到BN和dropout
        return output


# 训练过程
def train(mnist):
    # STEP: 1, forward ----------------------------------------------------------------
    # 输入数据 placeholder
    x = tf.placeholder(tf.float32, [None, INPUT_NODE], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name='y-input')

    # 生成参数
    weights1 = tf.Variable(tf.truncated_normal([INPUT_NODE, LAYER1_NODE], stddev=0.1))
    biases1 = tf.Variable(tf.constant(0.1, shape=[LAYER1_NODE])) # tf.zeros([1])

    weights2 = tf.Variable(tf.truncated_normal([LAYER1_NODE, OUTPUT_NODE], stddev=0.1))
    biases2 = tf.Variable(tf.constant(0.1, shape=[OUTPUT_NODE]))

    # 前向传播
    y = inference(x, None, weights1, biases1, weights2, biases2)

    # global_step, 训练多少个batch
    global_step = tf.Variable(0, trainable=False)

    # 滑动平均类, (衰减系数, 训练轮数)
    ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)

    # 计算滑动平均, 对所有 GraphKeys.TRAINABLE_VARIABLES (维护影子变量)
    average_op = ema.apply(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))

    # 应用滑动平均参数的前向传播, (inference 内调用了 average 函数)
    y_average = inference(x, ema, weights1, biases1, weights2, biases2)

    # STEP: 2, loss -----------------------------------------------------------------
    # loss 计算
    cross_entropy = tf.reduce_mean(
        tf.reduce_sum(-y_ * tf.log(tf.nn.softmax(y)), axis=1))  # old-style: reduction_indices=[1]
    # cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1)))

    # regularization 计算
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    regularization = regularizer(weights1) + regularizer(weights2)

    # total loss
    loss = cross_entropy + regularization

    # STEP: 3, optimize/train_op -----------------------------------------------------
    # 设置学习率指数衰减
    lr = tf.train.exponential_decay(LEARNING_RATE_BASE, global_step, mnist.train.num_examples/BATCH_SIZE, LEARNING_RATE_DECAY)

    # optimization
    optimizer = tf.train.GradientDescentOptimizer(lr)
    train_step = optimizer.minimize(loss, global_step=global_step)

    # trian_op
    # train_op = train_step  # not EMA
    #train_op = tf.group(train_step, average_op)
    with tf.control_dependencies([train_step, average_op]):
        train_op = tf.no_op(name='train')


    # OPTIONAL: accuracy, prediction, ... --------------------------------------------
    # accuracy
    # correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))  # not EMA
    correct_prediction = tf.equal(tf.argmax(y_average, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


    # STEP: 4, session/ run training -------------------------------------------------
    # session & train
    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        # get validate and test dadaset
        validate_feed = {x: mnist.validation.images, y_: mnist.validation.labels}
        test_feed = {x: mnist.test.images, y_: mnist.test.labels}

        # training 
        for i in range(TRAINING_STEPS):
            # validate
            if i % 1000 == 0:
                validate_acc = sess.run(accuracy, feed_dict=validate_feed)
                print("After %d training step(s), validation accuracy using average model is %g " % (i, validate_acc))

            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            sess.run(train_op, feed_dict={x: xs, y_: ys})

        # test
        test_acc = sess.run(accuracy, feed_dict=test_feed)
        print("After %d training step(s), test accuracy using average model is %g " % (i, test_acc))


# 主程序入口
def main(argv=None):
    mnist = input_data.read_data_sets("/home/cluo/Workspace/Projects/TFLearning/mnist/MNIST_data/", one_hot=True)
    # print(mnist.test.num_examples, mnist.test.images.dtype, mnist.test.labels.dtype)
    # print('label', mnist.test.labels[0,:])
    # plt.imshow(np.reshape(mnist.test.images[0,:], (28, 28)))
    # plt.show()
    train(mnist)

if __name__ == '__main__'            :
    tf.app.run()

'''
结果, 0.98左右
------------------------
After 0 training step(s), validation accuracy using average model is 0.1274 
After 1000 training step(s), validation accuracy using average model is 0.9756 
After 2000 training step(s), validation accuracy using average model is 0.9806 
After 3000 training step(s), validation accuracy using average model is 0.9816 
After 4000 training step(s), validation accuracy using average model is 0.983 
After 5000 training step(s), validation accuracy using average model is 0.9828

有时候出现以下结果???还没明白
-------------------------
After 0 training step(s), validation accuracy using average model is 0.082 
After 1000 training step(s), validation accuracy using average model is 0.0958 
After 2000 training step(s), validation accuracy using average model is 0.0958 
After 3000 training step(s), validation accuracy using average model is 0.0958 
After 4000 training step(s), validation accuracy using average model is 0.0958 
After 5000 training step(s), validation accuracy using average model is 0.0958
'''    