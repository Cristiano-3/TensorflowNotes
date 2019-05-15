# -*- coding:utf-8 -*-
import tensorflow as tf

a = tf.constant([1.0, 2.0], name='a')

# create session, 关键: 是否指定默认会话, 是否手动close()
# ---------------------------------------------------
# method1
# 需要手动close(), 且没有指定默认会话
sess = tf.Session()
print(sess.run(a))

# print(a.eval())  # 未指定默认会话, 无法使用Tensor.eval()
print(a.eval(session=sess))  # 但是可以这样指定

with sess.as_default():      # 还可以这样指定
    print(a.eval())

sess.close()  # 手动close()


# method2
# 用上下文管理来自动close(), 且已指定了默认会话
with tf.Session() as sess:
    print(sess.run(a))  # 
    print(a.eval())  # 


# method3
# 需要手动close(), 且已指定了默认会话
sess = tf.InteractiveSession()
print(a.eval())
sess.close()


# config session, 
# ----------------------------------------------------
config = tf.ConfigProto(allow_soft_placement=True, \
                        log_device_placement=True)
sess = tf.Session(config=config)
#sess = tf.InteractiveSession(config=config)

