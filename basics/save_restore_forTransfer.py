# coding:utf-8
import tensorflow as tf 
from tensorflow.python.framework import graph_util


# 将计算图中的变量及其取值通过常量保存, 不必要的节点删除掉

# 示例：
# # part 1: save
# v1 = tf.Variable(tf.constant(1.0, shape=[1]), name="v1")
# v2 = tf.Variable(tf.constant(2.0, shape=[1]), name="v2")
# result = v1 + v2 

# init_op = tf.global_variables_initializer()
# with tf.Session() as sess:
#     sess.run(init_op)
#     graph_def = tf.get_default_graph().as_graph_def() # 获得 GraphDef 对象, 存储图的节点信息

#     # 获得要保存的节点, 将图中的变量及其取值转化为常量
#     output_graph_def = graph_util.convert_variables_to_constants(sess, graph_def, ['add'])
#     with tf.gfile.GFile('./checkpoints/combined_model.pb', 'wb') as f:
#         f.write(output_graph_def.SerializeToString()) 


# part 2: restore
with tf.Session() as sess:
    model_filename = './checkpoints/combined_model.pb'
    # 读取保存的模型文件, 并将文件解析成对应的 GraphDef Protocol Buffer
    with tf.gfile.FastGFile(model_filename, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # 将 graph_def 中保存的图加载到当前的图中
    # 保存的时候给的是 节点名称 "add", 加载是张量名称 "add:0"
    result = tf.import_graph_def(graph_def, return_elements=["add:0"])
    print(sess.run(result))
