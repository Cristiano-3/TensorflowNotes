# coding:utf-8
import tensorflow as tf
import numpy as np 

# # save -------------------
# v1 = tf.Variable(tf.constant(1.0, shape=[1]), name="v1")
# v2 = tf.Variable(tf.constant(2.0, shape=[1]), name="v2")
# result = v1 + v2 

# init_op = tf.global_variables_initializer()

# # 声明Saver用于模型保存和加载
# saver = tf.train.Saver()

# with tf.Session() as sess:
#     sess.run(init_op)
#     # save with sess and model_name
#     saver.save(sess, './checkpoints/model.ckpt')

# # restore method 1 ----------
# v1 = tf.Variable(tf.constant(1.0, shape=[1]), name="v1")
# v2 = tf.Variable(tf.constant(2.0, shape=[1]), name="v2")
# result = v1 + v2 

# saver = tf.train.Saver()  # 默认加载所有变量, 等价于指定加载列表 Saver([v1, v2])
# with tf.Session() as sess:
#     # save with sess and model_name
#     saver.restore(sess, './checkpoints/model.ckpt')
#     print(sess.run(result))

# # restore method 2 ------------
# saver = tf.train.import_meta_graph('./checkpoints/model.ckpt.meta')

# with tf.Session() as sess:
#     # save with sess and model_name
#     saver.restore(sess, './checkpoints/model.ckpt')
#     print(sess.run(tf.get_default_graph().get_tensor_by_name("add:0")))


# # restore to rename variable ---------------------------
# v1 = tf.Variable(tf.constant(1.0, shape=[1]), name="v1-other")
# v2 = tf.Variable(tf.constant(2.0, shape=[1]), name="v2-other")
# result = v1 + v2 

# saver = tf.train.Saver({'v1':v1, 'v2':v2})  # 重命名映射字典 {'v1':v1, 'v2':v2}
# with tf.Session() as sess:
#     # save with sess and model_name
#     saver.restore(sess, './checkpoints/model.ckpt')
#     print(sess.run(result))


# 重命名加载用于 滑动平均模型的加载 ----------------------------
# part1: save
# v = tf.Variable(0, dtype=tf.float32, name='v')
# for variable in tf.global_variables():
#     print(variable.name)  # 只有一个变量 'v'

# # 声明滑动平均模型
# ema = tf.train.ExponentialMovingAverage(0.99)
# maintain_averages_op = ema.apply(tf.global_variables())
# # 此时已自动生成影子变量 v/ExponentialMovingAverage
# for variable in tf.global_variables():
#     print(variable.name)  # 输出 'v:0' 和 'v/ExponentialMovingAverage:0'

# saver = tf.train.Saver()
# with tf.Session() as sess:
#     init_op = tf.global_variables_initializer()
#     sess.run(init_op)

#     sess.run(tf.assign(v, 10))     # 更新参数v
#     sess.run(maintain_averages_op) # 维护影子变量
#     print(sess.run([v, ema.average(v)]))

#     # save 会同时保存 v:0和v/EMA:0
#     saver.save(sess, './checkpoints/model_1.ckpt')

# part2: restore
v = tf.Variable(0, dtype=tf.float32, name='v')
# saver = tf.train.Saver({"v/ExponentialMovingAverage": v})  # 手动映射, 或
ema = tf.train.ExponentialMovingAverage(0.99)                # 应用ema类,自动映射
saver = tf.train.Saver(ema.variables_to_restore())

with tf.Session() as sess:
    saver.restore(sess, './checkpoints/model_1.ckpt')
    print(sess.run(v))

    