# coding: utf-8
import tensorflow as tf 

# network-structure params
INPUT_NODE = 784
OUTPUT_NODE = 10
LAYER1_NODE = 500


# 复用到 'weights' 这个颗粒, 
# 未到 '层' 的复用, 
# get weights
def get_weight_variable(shape, regularizer):
    weights = tf.get_variable('weights', shape, 
                                initializer=tf.truncated_normal_initializer(stddev=0.1))
    
    # regularization
    if regularizer is not None:
        tf.add_to_collection('losses', regularizer(weights))
    
    return weights


# forward
def inference(input_tensor, regularizer):
    # layer1
    with tf.variable_scope('layer1'):
        weights = get_weight_variable([INPUT_NODE, LAYER1_NODE], regularizer)
        biases = tf.get_variable('biases', [LAYER1_NODE], initializer=tf.constant_initializer(0.0))
        layer1 = tf.nn.relu(tf.add(tf.matmul(input_tensor, weights), biases))

    # layer2
    with tf.variable_scope('layer2'):
        weights = get_weight_variable([LAYER1_NODE, OUTPUT_NODE], regularizer)
        biases = tf.get_variable('biases', [OUTPUT_NODE], initializer=tf.constant_initializer(0.0))
        layer2 = tf.add(tf.matmul(layer1, weights), biases)

    return layer2 
