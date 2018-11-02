# -*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Tensorflow in Action1: Linear Regression, 
# Y = WX + b or w1x1 + b

# 1. 定义超参数
learning_rate = 0.01
max_train_steps = 2000
log_step = 100


# 2. 输入数据
train_X = np.array([[3.3], [4.4], [5.5], [6.71], [6.93], [4.168], [9.779], [6.182], [7.59], [2.167], \
                    [7.042], [10.791], [5.313], [7.997], [5.654], [9.27], [3.1]], dtype=np.float32)
train_Y = np.array([[1.7], [2.76], [2.09], [3.19], [1.694], [1.573], [3.366], [2.596], [2.53], \
                    [1.221], [2.827], [3.465], [1.65], [2.904], [2.42], [2.94], [1.3]], dtype=np.float32)
total_samples = train_X.shape[0]

# 3. 构建模型
# 输入数据
X = tf.placeholder(tf.float32, [None, 1], name='X')
Y_ = tf.placeholder(tf.float32, [None, 1], name='Y_')

# 模型参数
W = tf.Variable(tf.random_normal([1, 1]), name='weight')
b = tf.Variable(tf.zeros([1]), name='bias')

# 推理值
Y = tf.matmul(X, W) + b

# 4. 定义损失函数
#loss = tf.reduce_sum(tf.pow(Y-Y_, 2))/(total_samples)
loss = tf.reduce_mean(tf.pow(Y-Y_, 2))

# 5. 创建优化器
optimizer = tf.train.GradientDescentOptimizer(learning_rate)

# 6. 定义单步训练操作
global_step = tf.Variable(0, trainable=False, name='global_step') #tf.constant(0)
train_op = optimizer.minimize(loss, global_step=global_step)

# 7. 创建会话
with tf.Session() as sess:
    # 初始化全局变量
    sess.run(tf.global_variables_initializer())

    # 8. 迭代训练
    print('start training ...')
    for step in range(max_train_steps):
        sess.run(train_op, feed_dict={X: train_X, Y_: train_Y})

        # 每隔log_step打印日志
        if step % log_step == 0:
            c = sess.run(loss, feed_dict={X: train_X, Y_: train_Y})
            print('Step: %d, loss=%.4f, W=%.4f, b=%.4f' % (step, c, sess.run(W), sess.run(b)))

    # 计算训练完毕的模型在训练集上的损失值, 并将其作为指标输出
    final_loss = sess.run(loss, feed_dict={X: train_X, Y_: train_Y})
    weight, bias = sess.run([W, b])
    print('Step: %d, loss=%.4f, W=%.4f, b=%.4f' % (max_train_steps, final_loss, weight, bias))
    print('Linear Regression Model: Y = %.4f * X + %.4f' % (weight, bias))
    

    # 9. 模型可视化
    plt.plot(train_X, train_Y, 'ro', label='Training data')
    plt.plot(train_X, weight * train_X + bias, 'b-', label='Fitted line')
    plt.legend()
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('show data & regression model')
    plt.show()
