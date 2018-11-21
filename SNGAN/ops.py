import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

import tensorflow as tf
import tensorflow.contrib as tf_contrib
from tensorflow.contrib.layers import xavier_initializer
from tensorflow.contrib.layers import variance_scaling_initializer


#  weight_init = tf.truncated_normal_initializer(mean=0.0, stddev=0.02)
#  weight_init = tf.random_normal_initializer(mean=0.0, stddev=0.02)
weight_init = xavier_initializer()
# weight_init = variance_scaling_initializer()


# weight_regularizer = tf_contrib.layers.l2_regularizer(scale=0.0001)
weight_regularizer = None

# pad = (k-1) // 2 = SAME !
# output = ( input - k + 1 + 2p ) // s


def conv2d(net, channels, kernel=3, stride=1, pad=0, pad_type='zero', use_bias=True, sn=False, data_format='NHWC', scope='conv_0'):
    with tf.variable_scope(scope):
        if pad_type == 'zero' :
            if data_format == 'NHWC':
                net = tf.pad(net, [[0, 0], [pad, pad], [pad, pad], [0, 0]])
            elif data_format == 'NCHW':
                net = tf.pad(net, [[0, 0], [0, 0], [pad, pad], [pad, pad]])
        if pad_type == 'reflect' :
            if data_format == 'NHWC':
                net = tf.pad(net, [[0, 0], [pad, pad], [pad, pad], [0, 0]], mode='REFLECT')
            elif data_format == 'NCHW':
                net = tf.pad(net, [[0, 0], [0, 0], [pad, pad], [pad, pad]], mode='REFLECT')

        if data_format == 'NHWC':
            in_channel = net.get_shape().as_list()[3]
        elif data_format == 'NCHW':
            in_channel = net.get_shape().as_list()[1]

        w = tf.get_variable("kernel", shape=[kernel, kernel, in_channel, channels],
                            initializer=weight_init, regularizer=weight_regularizer)
        if sn :
            #  net = tf.nn.conv2d(input=net, filter=spectral_normed_weight(w, update_collection=tf.GraphKeys.UPDATE_OPS), data_format=data_format,
                             #  strides=[1, stride, stride, 1], padding='VALID')
            net = tf.nn.conv2d(input=net, filter=spectral_norm(w), data_format=data_format,
                             strides=[1, stride, stride, 1], padding='VALID')
        else:
            net = tf.nn.conv2d(input=net, filter=w, data_format=data_format,
                             strides=[1, stride, stride, 1], padding='VALID')

        if use_bias:
            bias = tf.get_variable("bias", [channels], initializer=tf.constant_initializer(0.0))
            net = tf.nn.bias_add(net, bias, data_format=data_format)

        return net

def conv2d_transpose(net, channels, kernel=3, stride=1, use_bias=True, sn=False, data_format='NHWC', scope='deconv_0'):
    with tf.variable_scope(scope):
        net_shape = net.get_shape().as_list()
        output_shape = [net_shape[0], net_shape[1] * stride, net_shape[2] * stride, channels]

        w = tf.get_variable("kernel", shape=[kernel, kernel, channels, net.get_shape()[-1]],
                            initializer=weight_init, regularizer=weight_regularizer)
        if sn :
            #  net = tf.nn.conv2d_transpose(net, filter=spectral_normed_weight(w, update_collection=tf.GraphKeys.UPDATE_OPS), data_format=data_format,
                                         #  output_shape=output_shape, strides=[1, stride, stride, 1], padding='SAME')
            net = tf.nn.conv2d_transpose(net, filter=spectral_norm(w), data_format=data_format,
                                         output_shape=output_shape, strides=[1, stride, stride, 1], padding='SAME')
        else:
            net = tf.nn.conv2d_transpose(net, filter=w, data_format=data_format,
                                         output_shape=output_shape, strides=[1, stride, stride, 1], padding='SAME')
        if use_bias :
            bias = tf.get_variable("bias", [channels], initializer=tf.constant_initializer(0.0))
            net = tf.nn.bias_add(net, bias, data_format=data_format)

        return net

def max_pooling(x, kernel=2, stride=2) :
    return tf.layers.max_pooling2d(x, pool_size=kernel, strides=stride)

def avg_pooling(x, kernel=2, stride=2) :
    return tf.layers.average_pooling2d(x, pool_size=kernel, strides=stride)

def global_avg_pooling(x):
    """
    Incoming Tensor shape must be 4-D
    """
    gap = tf.reduce_mean(x, axis=[1, 2])
    return gap

def fully_connected(x, units, use_bias=True, sn=False, scope='fully_0'):
    with tf.variable_scope(scope):
        shape = x.get_shape().as_list()
        channels = shape[-1]

        if sn :
            w = tf.get_variable("kernel", [channels, units], tf.float32,
                                     initializer=weight_init, regularizer=weight_regularizer)
            if use_bias :
                bias = tf.get_variable("bias", [units],
                                       initializer=tf.constant_initializer(0.0))

                x = tf.matmul(x, spectral_norm(w)) + bias
            else :
                x = tf.matmul(x, spectral_norm(w))

        else :
            x = tf.layers.dense(x, units=units, kernel_initializer=weight_init, kernel_regularizer=weight_regularizer, use_bias=use_bias)

        return x


def l2_norm(v, eps=1e-12):
    return v / (tf.reduce_sum(v ** 2) ** 0.5 + eps)


def spectral_norm(w, iteration=1):
    w_shape = w.shape.as_list()
    w = tf.reshape(w, [-1, w_shape[-1]])

    u = tf.get_variable("u", [1, w_shape[-1]], initializer=tf.truncated_normal_initializer(), trainable=False)

    u_hat = u
    v_hat = None
    for i in range(iteration):
        """
        power iteration
        Usually iteration = 1 will be enough
        """
        v_ = tf.matmul(u_hat, tf.transpose(w))
        v_hat = l2_norm(v_)

        u_ = tf.matmul(v_hat, w)
        u_hat = l2_norm(u_)

    sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))
    w_norm = w / sigma

    with tf.control_dependencies([u.assign(u_hat)]):
        w_norm = tf.reshape(w_norm, w_shape)

    return w_norm
