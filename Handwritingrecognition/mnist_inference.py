# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 15:12:53 2018

@author: gemslab
"""
import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data

input_node=784
out_node=10
layer1_node=500
def get_weight_variable(shape,regularizer):
    weights=tf.get_variable('weights',shape,initializer=tf.truncated_normal_initializer(stddev=0.1),)
    if regularizer !=None:
        tf.add_to_collection('losses',regularizer(weights))
    return weights

def inference(input_tensor,regularizer):
    with tf.variable_scope('layer1'):
        weights=get_weight_variable([input_node,layer1_node],regularizer)
        biases=tf.get_variable('biases',[layer1_node],initializer=tf.constant_initializer(0.0))
        layer1=tf.nn.relu(tf.matmul(input_tensor,weights)+biases)
    with tf.variable_scope('layer2'):
        weights=get_weight_variable([layer1_node,out_node],regularizer)
        biases=tf.get_variable('biases',[out_node],initializer=tf.constant_initializer(0.0))
        layer2=tf.matmul(layer1,weights)+biases
        return layer2
        
        
        