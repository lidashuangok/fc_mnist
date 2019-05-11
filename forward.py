# !usr/bin/python
# Author:das
# -*-coding: utf-8 -*-
import tensorflow as tf
import numpy as np
INPUT_NODES =784
OUTPUT_NODES=10
FC1_NODES=800
FC2_NODES=300

def get_weight(shape,regularizer):
    w =tf.Variable( tf.random_normal(shape=shape,mean=0.0,stddev=1.0,dtype=tf.float32))
    if regularizer!=None:tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(regularizer)(w))
    return w
def get_bias(shape):
    b = tf.random_normal(shape=shape,mean=0.0,stddev=1.0,dtype=tf.float32)
    return tf.Variable(b)

def forward(x,regularizer):
    fc1_w = get_weight([INPUT_NODES,FC1_NODES],regularizer)
    fc1_b = get_bias([FC1_NODES])
    fc1 = tf.nn.relu(tf.matmul(x,fc1_w)+fc1_b)
    fc2_w = get_weight([FC1_NODES, FC2_NODES],regularizer)
    fc2_b = get_bias([FC2_NODES])
    fc2 = tf.nn.relu(tf.matmul(fc1, fc2_w) + fc2_b)
    fc3_w = get_weight([FC2_NODES, OUTPUT_NODES],regularizer)
    fc3_b = get_bias([OUTPUT_NODES])
    y=tf.matmul(fc2,fc3_w)+fc3_b
    return y