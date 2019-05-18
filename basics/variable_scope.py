# -*- coding:utf-8 -*-

import tensorflow as tf 
import numpy as np 

# tf.Variable, 允许重名, 会自动加 name_n
w = tf.Variable([1.0], name="w")  
print(w.name, w.value)

w1 = tf.Variable([1], name="w") 
print(w1.name, w1.value)


# tf.get_variable, 默认reuse=False, 不能重名
v1 = tf.get_variable("v", shape=[1])  # 未指定 initializer;
print(v1.name, v1.value)              # v:0, 当操作有多个输出便会有<op>:0, <op>::1, <op>:2, ...
#v2 = tf.get_variable("v", shape=[1]) , 会报错 ValueError: Variable v already exists, disallowed.

# 以上等价如下写法:
# with tf.variable_scope(""):
#     v1 = tf.get_variable("v", shape=[1])
#     print(v1.name)

# v1 未指定 initializer 但仍可以 eval
sess = tf.InteractiveSession()
sess.run(v1.initializer)      
print(v1.eval())
# If initializer is None (the default), the default initializer 
# passed in the variable scope will be used. If that one is None too, 
# a glorot_uniform_initializer will be used. (Xavier uniform initializer)


# 变量命名空间管理
with tf.variable_scope("foo"):
    v2 = tf.get_variable("v", shape=[1])
    print(v2.name)

with tf.variable_scope("foo"):
    with tf.variable_scope("bar"):
        v3 = tf.get_variable("v", shape=[1])
        print(v3.name)

    v4 = tf.get_variable("v1", [1])
    print(v4.name)

with tf.variable_scope("", reuse=True):
    v5 = tf.get_variable("foo/bar/v")
    print(v5 == v3)

    v6 = tf.get_variable("foo/v1")
    print(v6 == v4)
    