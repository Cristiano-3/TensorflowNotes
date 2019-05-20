# coding: utf-8
import os
import numpy as np 
import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data

import mnist_inference

# training settings
BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99
REGULARIZATION_RATE = 0.0001
TRAINING_STEPS = 30000
MOVING_AVERAGE_DECAY = 0.99

# save setting
MODEL_SAVE_PATH = './checkpoints/'
MODEL_NAME = 'model.ckpt'


def train(mnist):
    # input placeholders
    x = tf.placeholder(
        tf.float32, 
        [None, 
        mnist_inference.IMAGE_SIZE, 
        mnist_inference.IMAGE_SIZE, 
        mnist_inference.NUM_CHANNELS], 
        name='x-inputs')
    y_ = tf.placeholder(tf.float32, [None, mnist_inference.NUM_LABELS], name='y-inputs')

    # inference
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    y = mnist_inference.inference(x, True, regularizer)

    # ema 
    global_step = tf.Variable(0, trainable=False)  # 随step衰减的地方会用到(ema, lr)
    ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    average_op = ema.apply(tf.trainable_variables())

    # loss
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=y, 
        labels=tf.argmax(y_, axis=1))

    cross_entropy_mean = tf.reduce_mean(cross_entropy)    
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))

    # learning rate
    lr = tf.train.exponential_decay(
        LEARNING_RATE_BASE, 
        global_step, 
        mnist.train.num_examples / BATCH_SIZE, 
        LEARNING_RATE_DECAY)

    # optimizer & train_op
    optimizer = tf.train.GradientDescentOptimizer(lr)
    train_step = optimizer.minimize(loss, global_step=global_step)

    with tf.control_dependencies([train_step, average_op]):
        train_op = tf.no_op("train")

    # saver
    # Saver(savable_variables, max_to_keep=n, keep_checkpoint_every_n_hours=m)
    saver = tf.train.Saver()
    
    # session & training
    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        for i in range(TRAINING_STEPS):
            # batch
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            reshaped_xs = np.reshape(xs, (BATCH_SIZE, 
                                          mnist_inference.IMAGE_SIZE,
                                          mnist_inference.IMAGE_SIZE,
                                          mnist_inference.NUM_CHANNELS))

            # train a step
            _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: reshaped_xs, y_: ys})

            # infos
            if i % 1000 == 0:
                print('After %d training step(s), loss on training batch is %g.' % (step, loss_value))
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)


def main(argv=None):
    mnist = input_data.read_data_sets('/home/cluo/Workspace/Projects/TFLearning/mnist/MNIST_data', one_hot=True)
    train(mnist)

if __name__ == "__main__":
    tf.app.run()
