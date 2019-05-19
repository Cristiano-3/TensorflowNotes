# coding: utf-8
import time 
import numpy as np 
import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data

import mnist_inference
import mnist_train

EVAL_INTERVAL_SECS = 10


def evaluate(mnist):
    with tf.Graph().as_default() as g:
        # ---------------------------------
        # input placeholders
        x = tf.placeholder(tf.float32, [None, mnist_inference.INPUT_NODE], name='x-input')
        y_ = tf.placeholder(tf.float32, [None, mnist_inference.OUTPUT_NODE], name='y-input')

        # 这里是测试的 inference, 不用regularizer
        y = mnist_inference.inference(x, None)

        # 计算正确率
        prediction = tf.argmax(y, 1)
        correct_prediction = tf.equal(prediction, tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # ---------------------------------
        # 用重命名方式, 使用滑动平均模型
        # 这样不需在 inference 中调用 average,
        # 完全共用 inference 函数
        ema = tf.train.ExponentialMovingAverage(mnist_train.MOVING_AVERAGE_DECAY)  # 这里就不需global_step了
        variables_to_restore = ema.variables_to_restore()

        # saver, 加载 EMA 模型参数
        saver = tf.train.Saver(variables_to_restore)
        
        # ---------------------------------
        # validate dataset
        validate_feed = {x: mnist.validation.images, 
                         y_: mnist.validation.labels}

        while True:
            with tf.Session() as sess:
                # 自动找到最新的模型文件名
                ckpt = tf.train.get_checkpoint_state(mnist_train.MODEL_SAVE_PATH)

                if ckpt and ckpt.model_checkpoint_path:
                    # restore 最新的模型参数
                    saver.restore(sess, ckpt.model_checkpoint_path)

                    # 运行测试    
                    accuracy_score, predict = sess.run([accuracy, prediction], feed_dict=validate_feed)

                    # 通过文件名得到模型保存时迭代的轮数
                    global_step = ckpt.model_checkpoint_path.split('/')[-1]\
                        .split('-')[-1]

                    print("After %s training step(s), validation accuracy = %g" % (global_step, accuracy_score))
                    print("Preds:", predict[:10], '\nLabel:', np.argmax(mnist.validation.labels[:10], 1))

                else:
                    print('No checkpoint file found')
                    return

            # 循环隔10s运行一次测试
            time.sleep(EVAL_INTERVAL_SECS)


def main(argv=None):
    mnist = input_data.read_data_sets('/home/cluo/Workspace/Projects/TFLearning/mnist/MNIST_data', one_hot=True)
    evaluate(mnist)


if __name__ == "__main__":
    tf.app.run()
