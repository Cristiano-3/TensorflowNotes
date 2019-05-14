# -*- coding:utf-8 -*-
# 使用计算图的形式来表示模型

import tensorflow as tf 

a = tf.constant([1.0, 2.0], name='a')
b = tf.constant([2.0, 3.0], name='b')
c = tf.Variable([3.0, 4.0], name='c')
result = a+b  # tf.add(a, b)  # tf.add 与 + 作用一样
#c = a     # 这里c会变成张量， 后面无法对其assign

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(result)) # constant可以不初始化就调用
    print(sess.run(c))  # Variable必须先初始化才能调用
    print(sess.run(tf.assign(c, a)))  # assign只对variable赋值，constant不可以； 且在graph中会有节点绘出

writer = tf.summary.FileWriter('./tensorboard/', tf.get_default_graph())
writer.close()

print(a.graph is tf.get_default_graph())

# ================ 以上未显式指定graph，所有操作在 default graph 上建图

g1 = tf.Graph()
with g1.as_default():    
    # 在graph g1中定义变量"v", 并零初始化
    v = tf.get_variable("v", shape=[1], initializer=tf.zeros_initializer) # 默认scope=""

g2 = tf.Graph()
with g2.as_default():
    with tf.variable_scope("bar"):
        # graph g2中定义变量"v", 并1初始化
        v = tf.get_variable("v", shape=[1], initializer=tf.ones_initializer) # scope="bar"

# 一个session只能执行一个特定graph， 一个graph可以被多个session加载
with tf.Session(graph=g1) as sess:
    sess.run(tf.global_variables_initializer())
    with tf.variable_scope("", reuse=True):      # reuse=True 使必须使用已有的，不存在则报错； 
        print(sess.run(tf.get_variable("v")))    # reuse=tf.AUTO_REUSE则创建或重用； 
                                                 # reuse=None或False，必须创建新的变量；

with tf.Session(graph=g2) as sess:
    sess.run(tf.global_variables_initializer())        
    with tf.variable_scope("bar", reuse=True):
        print(sess.run(tf.get_variable("v")))
