#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np

class modelRegularIInception(object):
    def __init__(self, is_training, num_classes):

        self.num_classes = num_classes
        self._is_training = is_training

    def preprocess(self, inputs):
        preprocessed_inputs = tf.to_float(inputs)
        preprocessed_inputs = tf.subtract(preprocessed_inputs, 128.0)
        preprocessed_inputs = tf.div(preprocessed_inputs, 128.0)
        return preprocessed_inputs

    def get_variable_with_l2_loss(self,shape,stddev,wl,name):
        var=tf.Variable(tf.truncated_normal(shape=shape,stddev=stddev),name=name)
        if wl is not None:
            var_l2_loss=tf.multiply(tf.nn.l2_loss(var), wl, name="l2_loss")
            tf.add_to_collection("Loss",var_l2_loss)
        return var

    def predict(self, preprocessed_inputs,keep_prob):
        shape = preprocessed_inputs.get_shape().as_list()
        height, width, num_channels = shape[1:]

        net = preprocessed_inputs

        #模块前的卷积
        conv1_weights = self.get_variable_with_l2_loss([3, 3, num_channels, 32],5e-2, None, 'conv1_weights')
        conv1_biases = tf.get_variable('conv1_biases', shape=[32], dtype=tf.float32)
        net = tf.nn.conv2d(net, conv1_weights, strides=[1, 1, 1, 1], padding='SAME')
        net = tf.nn.relu(tf.nn.bias_add(net, conv1_biases))

        #开始分支
        conv12_weights = self.get_variable_with_l2_loss([3, 3, 32, 16], 5e-2, None, 'conv12_weights')
        conv12_biases = tf.get_variable('conv12_biases', shape=[16], dtype=tf.float32)
        net12 = tf.nn.conv2d(net, conv12_weights, strides=[1, 2, 2, 1], padding='SAME')
        net12 = tf.nn.relu(tf.nn.bias_add(net12, conv12_biases))

        with tf.name_scope("Moudle1"):
            M1_branch1_weights=self.get_variable_with_l2_loss([3, 3, 32, 64],5e-2, None, 'M1_branch1_weights')
            M1_branch1_biases = tf.get_variable('M1_branch1_biases', shape=[64], dtype=tf.float32)
            M1_branch1=tf.nn.conv2d(net, M1_branch1_weights, strides=[1, 2, 2, 1], padding='SAME')
            M1_branch1=tf.nn.bias_add(M1_branch1, M1_branch1_biases)
            # M1_branch1 = tf.nn.relu(M1_branch1)

            M1_branch2_weights1 = self.get_variable_with_l2_loss([1, 1, 32, 48], 5e-2, None, 'M1_branch2_weights1')
            M1_branch2_biases1 = tf.get_variable('M1_branch2_biases1', shape=[48], dtype=tf.float32)
            M1_branch2_weights2 = self.get_variable_with_l2_loss([5, 5, 48, 64], 5e-2, None, 'M1_branch2_weights2')
            M1_branch2_biases2 = tf.get_variable('M1_branch2_biases2', shape=[64], dtype=tf.float32)
            M1_branch2 = tf.nn.conv2d(net, M1_branch2_weights1, strides=[1, 1, 1, 1], padding='SAME')
            M1_branch2=tf.nn.bias_add(M1_branch2, M1_branch2_biases1)
            # M1_branch2 = tf.nn.relu(M1_branch2)
            M1_branch2 = tf.nn.conv2d(M1_branch2, M1_branch2_weights2, strides=[1, 2, 2, 1], padding='SAME')
            M1_branch2=tf.nn.bias_add(M1_branch2, M1_branch2_biases2)
            # M1_branch2 = tf.nn.relu(M1_branch2)

            M1_branch3_weights1 = self.get_variable_with_l2_loss([1, 1, 32, 64], 5e-2, None, 'M1_branch3_weights1')
            M1_branch3_biases1 = tf.get_variable('M1_branch3_biases1', shape=[64], dtype=tf.float32)
            M1_branch3_weights2 = self.get_variable_with_l2_loss([3, 3, 64, 96], 5e-2, None, 'M1_branch3_weights2')
            M1_branch3_biases2 = tf.get_variable('M1_branch3_biases2', shape=[96], dtype=tf.float32)
            # M1_branch3_weights3 = self.get_variable_with_l2_loss([3, 3, 96, 96], 5e-2, None, 'M1_branch3_weights3')
            # M1_branch3_biases3 = tf.get_variable('M1_branch3_biases3', shape=[96], dtype=tf.float32)
            M1_branch3 = tf.nn.conv2d(net, M1_branch3_weights1, strides=[1, 1, 1, 1], padding='SAME')
            M1_branch3=tf.nn.bias_add(M1_branch3, M1_branch3_biases1)
            # M1_branch3 = tf.nn.relu(M1_branch3)
            M1_branch3 = tf.nn.conv2d(M1_branch3, M1_branch3_weights2, strides=[1, 2, 2, 1], padding='SAME')
            M1_branch3=tf.nn.bias_add(M1_branch3, M1_branch3_biases2)
            # M1_branch3 = tf.nn.relu(M1_branch3)
            # M1_branch3 = tf.nn.conv2d(M1_branch3, M1_branch3_weights3, strides=[1, 1, 1, 1], padding='SAME')
            # M1_branch3=tf.nn.bias_add(M1_branch3, M1_branch3_biases3)
            # M1_branch3 = tf.nn.relu(M1_branch3)

            M1_branch4 = tf.nn.avg_pool(net, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='SAME')
            M1_branch4_weights1 = self.get_variable_with_l2_loss([1, 1, 32, 32], 5e-2, None, 'M1_branch4_weights1')
            M1_branch4_biases1 = tf.get_variable('M1_branch4_biases1', shape=[32], dtype=tf.float32)
            # M1_branch4_weights2 = self.get_variable_with_l2_loss([3, 3, 64, 96], 5e-2, None, 'M1_branch4_weights2')
            # M1_branch4_biases2 = tf.get_variable('M1_branch4_biases2', shape=[96], dtype=tf.float32)
            M1_branch4 = tf.nn.conv2d(M1_branch4, M1_branch4_weights1, strides=[1, 2, 2, 1], padding='SAME')
            M1_branch4=tf.nn.bias_add(M1_branch4, M1_branch4_biases1)
            # M1_branch4 = tf.nn.relu(M1_branch4)

        net=tf.concat([M1_branch1,M1_branch2,M1_branch3,M1_branch4,net12],3)
        channals = net.get_shape().as_list()[-1]

        conv2_weights = self.get_variable_with_l2_loss([3, 3, channals, 64],5e-2, None, 'conv2_weights')
        conv2_biases = tf.get_variable('conv2_biases', shape=[64], dtype=tf.float32)
        net = tf.nn.conv2d(net, conv2_weights, strides=[1, 1, 1, 1], padding='SAME')
        net = tf.nn.relu(tf.nn.bias_add(net, conv2_biases))
        net = tf.nn.max_pool(net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


        conv3_weights = self.get_variable_with_l2_loss([3, 3, 64, 64],5e-2, None, 'conv3_weights')
        conv3_biases = tf.get_variable('conv3_biases', shape=[64], dtype=tf.float32)
        net = tf.nn.conv2d(net, conv3_weights, strides=[1, 1, 1, 1],padding='SAME')
        net = tf.nn.relu(tf.nn.bias_add(net, conv3_biases))
        # net = tf.nn.max_pool(net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],padding='SAME')

        (h, w, c) = net.get_shape().as_list()[1:]
        flat_size = h * w * c
        net = tf.reshape(net, shape=[-1, flat_size])

        fc8_weights = self.get_variable_with_l2_loss([flat_size, 64],5e-2, 0.1, 'fc8_weights')
        fc8_biases = tf.get_variable('f8_biases', shape=[64], dtype=tf.float32)
        net = tf.nn.relu(tf.add(tf.matmul(net, fc8_weights), fc8_biases))
        net=tf.nn.dropout(net,keep_prob=keep_prob)

        fc9_weights = self.get_variable_with_l2_loss([64, self.num_classes],5e-2, 0.1, 'fc9_weights')
        fc9_biases = tf.get_variable('f9_biases', shape=[self.num_classes], dtype=tf.float32)
        net = tf.add(tf.matmul(net, fc9_weights), fc9_biases)

        prediction_dict = {'logits': net}
        return prediction_dict

    def postprocess(self, prediction_dict):
        logits = prediction_dict['logits']
        logits = tf.nn.softmax(logits)
        classes = tf.cast(tf.argmax(logits, axis=1), dtype=tf.int32)
        postprecessed_dict = {'classes': classes}
        postprecessed_dict ['logits'] = logits  #   tf.clip_by_value(logits,1e-8,np.inf)
        return postprecessed_dict

    def loss(self, prediction_dict, groundtruth_lists):
        logits = prediction_dict['logits']
        class_loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=logits+1e-8, labels=groundtruth_lists))
        tf.add_to_collection("Loss",class_loss)
        loss_all=tf.add_n(tf.get_collection("Loss"),name="total_loss")
        loss_dict = {'loss': loss_all}
        return loss_dict
