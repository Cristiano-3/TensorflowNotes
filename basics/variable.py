# -*- coding:utf-8 -*-
import tensorflow as tf 

# 常量生成函数
# 常量(用 Tensor 类表示)直接给定value, 而不是init_value
# -----------------------------------------------
# tf.constant(value, dtype, shape, name), value必须其他可选
# value可以是常量数值或list, dtype和shape可默认为value的, 也可以指定;
# name可指定, 也可自动命名为"Op:num"格式, 这里Op=Const
x = tf.constant([[0.7, 0.9]])     # Tensor
y = tf.ones([2,3], tf.int32)      # Tensor
# 类似常量还有 tf.zeros(), tf.fill()

print(x, type(x))
print(y, type(y))

# 不需初始化即可使用
sess = tf.InteractiveSession()
print(x.eval())
print(y.eval())
sess.close()



# 变量, 
# 用 Variable 类表示, 常用于保存/更新参数
# 需要指定初始化方法, 可以是:随机数, 常数, 其他变量的初始值
# 神经网络中的参数, 常用正态分布的随机数初始化, biases用常数初始化(如zeros)
# -------------------------------------------------
# tf.Variable(init_value, dtype, name, ... ...), init_value 必须其他可选
w1 = tf.Variable([1], name='a')    
w2 = tf.Variable(tf.random_normal([2, 3], stddev=2))  # tf.truncated_normal, tf.random_uniform, tf.random_gamma
w3 = tf.Variable(w2.initialized_value() * 2.0)

sess = tf.InteractiveSession()

# 执行变量初始化
#sess.run([w1.initializer, w2.initializer, w3.initializer])
#sess.run(tf.variables_initializer([w1, w2, w3]))
#sess.run(tf.variables_initializer(tf.global_variables()))
#sess.run(tf.variables_initializer(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)))
sess.run(tf.global_variables_initializer())
print(w1.eval())
print(w2.eval())
print(w3.eval(), type(w3.eval()))
sess.close()


# 无论变量还是常量, 要得到输出结果就要session运行, 结果输出张量的值(numpy.ndarray);