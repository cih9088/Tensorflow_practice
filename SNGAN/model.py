import sys
import os
import six
import itertools
import pickle
import time
import math
import itertools
import ops
import matplotlib
import math

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import matplotlib.pyplot as plt
import seaborn as sns
import utils as utils

from tqdm import tqdm
from matplotlib.gridspec import GridSpec
from tf_utils.data.dataset import DataSet
import tf_utils.common as common
import tf_utils.inception_score as inception_score
import tensorflow.contrib.gan as gan


sns.set_style("darkgrid")

SEED = 6150
np.random.seed(SEED)


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=1)


def get_gradvars(target_gradvars):
    gradvars = []
    with tf.name_scope('gradient_averaging'):
        all_grads = {}
        for grad, var in itertools.chain(*target_gradvars):
            if grad is not None:
                all_grads.setdefault(var, []).append(grad)
        for var, grads in six.iteritems(all_grads):
            # Average gradients on the same device as the variables
            # to which they apply.
            with tf.device(var.device):
                if len(grads) == 1:
                    avg_grad = grads[0]
                else:
                    avg_grad = tf.multiply(tf.add_n(grads), 1. / len(grads))
            gradvars.append((avg_grad, var))
    return gradvars


class Model():
    def __init__(self, log_dir, num_gpus, name='model', **hparams):
        self.model_name = name

        self.dir = dict()
        self.dir['log'] = log_dir
        self.dir['imgs'] = os.path.join(log_dir, 'imgs')
        self.dir['imgs/train'] = os.path.join(self.dir['imgs'], 'train')
        self.dir['imgs/test'] = os.path.join(self.dir['imgs'], 'test')
        self.dir['imgs/valid'] = os.path.join(self.dir['imgs'], 'valid')
        self.dir['summary'] = os.path.join(log_dir, 'summary')
        self.dir['checkpoint'] = os.path.join(log_dir, 'checkpoint')
        for detailed_log_dir in self.dir.keys():
            self._prepare_directory(self.dir[detailed_log_dir])

        self.var = dict()

        self.num_gpus = num_gpus
        print('[*] The number of GPU is {}'.format(self.num_gpus))

        self._init_for_model(**hparams)
        self._build_model(**hparams)

    def _prepare_directory(self, log_dir=None, delete=False):
        if log_dir is None:
            print('[!] log_dir must be provided.')
        else:
            if delete:
                if os.path.exists(log_dir):
                    import shutil
                    shutil.rmtree(log_dir)
                os.makedirs(log_dir)
            else:
                if not os.path.exists(log_dir):
                    os.makedirs(log_dir)

    def _init_for_model(self, **hparams):

        self.z_dim       = hparams['z_dim']
        self.data_format = hparams['data_format']
        self.ps_strategy = hparams['ps_strategy']
        self.data        = hparams['data']
        self.tower       = dict()

        if self.data_format is None:
            if self.num_gpus == 0:
                self.data_format = 'NHWC'
            else:
                self.data_format = 'NCHW'

        consolidation_device = '/gpu:0' if self.ps_strategy == 'GPU' else '/cpu:0'
        with tf.device(consolidation_device):
            self.var['is_training'] = tf.placeholder(tf.bool, [], name='is_training')

        if hparams['data'] == 'imagenet64':
            self.height  = 64
            self.width   = 64
            self.channel = 3
        elif hparams['data'] == 'imagenet32' or hparams['data'] == 'cifar10':
            self.height  = 32
            self.width   = 32
            self.channel = 3
        else:
            raise ValueError('No support data type: {}'.format(hparams['data']))

    def _build_model(self, **hparams):
        self.var['input_data']   = None
        self.var['input_z']      = None
        self.var['D_p']          = None
        self.var['D_p_logit']    = None
        self.var['D_n']          = None
        self.var['D_n_logit']    = None
        self.var['G']            = None
        self.var['D_loss']       = None
        self.var['G_loss']       = None

        self.tower['input_data'] = []
        self.tower['input_z']    = []
        self.tower['D_p']        = []
        self.tower['D_p_logit']  = []
        self.tower['D_n']        = []
        self.tower['D_n_logit']  = []
        self.tower['G']          = []
        self.tower['D_loss']     = []
        self.tower['G_loss']     = []
        self.tower['D_gradvars'] = []
        self.tower['G_gradvars'] = []
        self.sum_list            = []

        if self.num_gpus == 0:
            num_devices = 1
            device_type = 'cpu'
        else:
            num_devices = self.num_gpus
            device_type = 'gpu'

        consolidation_device = '/gpu:0' if self.ps_strategy == 'GPU' else '/cpu:0'
        with tf.device(consolidation_device):
            if self.data_format == 'NHWC':
                self.var['input_data'] = tf.placeholder(tf.float32, [None, self.height, self.width, self.channel])
            elif self.data_format == 'NCHW':
                self.var['input_data'] = tf.placeholder(tf.float32, [None, self.channel, self.height, self.width])

            self.var['input_z'] = tf.placeholder(tf.float32, [None, self.z_dim])

            if self.num_gpus:
                self.tower['input_data'] = tf.split(self.var['input_data'], self.num_gpus)
                self.tower['input_z'] = tf.split(self.var['input_z'], self.num_gpus)
            else:
                self.tower['input_data'] = [self.var['input_dir']]
                self.tower['input_z'] = [self.var['input_z']]

        self.inception_score = gan.eval.classifier_score(gan.eval.preprocess_image(
            self.var['input_data'] if self.data_format == 'NHWC' else tf.transpose(self.var['input_data'], [0, 2, 3, 1])), gan.eval.run_inception, 10)

        for i in range(num_devices):
            worker_device = '/{}:{}'.format(device_type, i)
            if self.ps_strategy == 'CPU':
                device_setter = utils.local_device_setter(
                    worker_device=worker_device)
            elif self.ps_strategy == 'GPU':
                device_setter = utils.local_device_setter(
                    ps_device_type='gpu',
                    worker_device=worker_device,
                    ps_strategy=tf.contrib.training.GreedyLoadBalancingStrategy(
                        self.num_gpus, tf.contrib.training.byte_size_load_fn))
            with tf.name_scope('tower_%d' % i) as name_scope:
                with tf.device(device_setter):

                    tower_data = self.tower['input_data'][i]
                    tower_z = self.tower['input_z'][i]

                    self._build_model_per_tower(tower_data, tower_z, i != 0)
                    self._build_loss_per_tower(hparams['loss_type'])
                    self._build_extra_op_per_tower()

                    #  print(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, name_scope))
                    #  print(tf.get_collection(tf.GraphKeys.LOSSES, name_scope))
                    #  print(tf.get_collection(tf.GraphKeys.UPDATE_OPS, name_scope))
                    #  print(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))
                    #  print(tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES))
                    #  print(tf.get_collection(tf.GraphKeys.MODEL_VARIABLES))
                    #  print(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))

        self._build_train_op(**hparams)

    def _build_model_per_tower(self, data, z, reuse):
        D_p, D_p_logit = self.discriminator(data, reuse=reuse)
        G = self.generator(z, reuse=reuse)
        D_n, D_n_logit = self.discriminator(G, reuse=True)

        self.tower['D_p'].append(D_p)
        self.tower['D_p_logit'].append(D_p_logit)
        self.tower['D_n'].append(D_n)
        self.tower['D_n_logit'].append(D_n_logit)
        self.tower['G'].append(G)

    def _build_loss_per_tower(self, loss_type):
        D_p_logit = self.tower['D_p_logit'][-1]
        D_n_logit = self.tower['D_n_logit'][-1]
        D_p = self.tower['D_p'][-1]
        D_n = self.tower['D_n'][-1]

        if loss_type == 'hinge':
            D_loss = tf.reduce_mean(tf.nn.relu(1.0 - D_p_logit)) + tf.reduce_mean(tf.nn.relu(1.0 + D_n_logit))
            G_loss = -tf.reduce_mean(D_n_logit)
        elif loss_type == 'wasserstain':
            D_loss = -tf.reduce_mean(D_p_logit) + tf.reduce_mean(D_n_logit)
            G_loss = -tf.reduce_mean(D_n_logit)
        else:
            #  D_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(D_p_logit), logits=D_p_logit)) \
            #      + tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(D_n_logit), logits=D_n_logit))
            #  G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(D_n_logit), logits=D_n_logit))
            #  D_loss = tf.reduce_mean(tf.nn.softplus(-D_p_logit)) \
            #      + tf.reduce_mean(tf.nn.softplus(-D_n_logit) + D_n_logit)
            D_loss = tf.reduce_mean(tf.nn.softplus(-D_p_logit) + tf.nn.softplus(D_n_logit))
            G_loss = tf.reduce_mean(tf.nn.softplus(-D_n_logit))

        self.tower['D_loss'].append(D_loss)
        self.tower['G_loss'].append(G_loss)

    def _build_extra_op_per_tower(self):
        D_params = [param for param in tf.trainable_variables() if 'discriminator' in param.name]
        G_params = [param for param in tf.trainable_variables() if 'generator' in param.name]

        D_tower_grad = tf.gradients(self.tower['D_loss'][-1], D_params)
        G_tower_grad = tf.gradients(self.tower['G_loss'][-1], G_params)
        self.tower['D_gradvars'].append(zip(D_tower_grad, D_params))
        self.tower['G_gradvars'].append(zip(G_tower_grad, G_params))

    def discriminator(self, inputs, reuse=False):
        def res_blk_first(net, channel, data_format, down_sampling=False, scope='res_blk'):
            if data_format == 'NCHW':
                channel_origin = net.get_shape().as_list()[1]
                height_origin = net.get_shape().as_list()[2]
                width_origin = net.get_shape().as_list()[3]
            elif data_format == 'NHWC':
                channel_origin = net.get_shape().as_list()[3]
                height_origin = net.get_shape().as_list()[1]
                width_origin = net.get_shape().as_list()[2]

            identity = net

            net = ops.conv2d(net, channel, 3, 1, 1, data_format=self.data_format, sn=True, scope='{}_conv1'.format(scope))
            net = tf.nn.relu(net)
            net = ops.conv2d(net, channel, 3, 1, 1, data_format=self.data_format, sn=True, scope='{}_conv2'.format(scope))

            if down_sampling:
                net = slim.avg_pool2d(net, 2, 2, data_format=data_format)
                identity = slim.avg_pool2d(identity, 2, 2, data_format=data_format)

            if channel != channel_origin:
                identity = ops.conv2d(identity, channel, 1, 1, 0, data_format=self.data_format, sn=True, scope='{}_identity_conv'.format(scope))

            net = net + identity

            return net
        def res_blk(net, channel, data_format, down_sampling=False, scope='res_blk'):
            if data_format == 'NCHW':
                channel_origin = net.get_shape().as_list()[1]
                height_origin = net.get_shape().as_list()[2]
                width_origin = net.get_shape().as_list()[3]
            elif data_format == 'NHWC':
                channel_origin = net.get_shape().as_list()[3]
                height_origin = net.get_shape().as_list()[1]
                width_origin = net.get_shape().as_list()[2]

            identity = net

            net = tf.nn.relu(net)
            net = ops.conv2d(net, channel, 3, 1, 1, data_format=self.data_format, sn=True, scope='{}_conv1'.format(scope))
            net = tf.nn.relu(net)
            net = ops.conv2d(net, channel, 3, 1, 1, data_format=self.data_format, sn=True, scope='{}_conv2'.format(scope))

            if channel != channel_origin:
                identity = ops.conv2d(identity, channel, 1, 1, 0, data_format=self.data_format, sn=True, scope='{}_identity_conv'.format(scope))

            if down_sampling:
                net = slim.avg_pool2d(net, 2, 2, data_format=data_format)
                identity = slim.avg_pool2d(identity, 2, 2, data_format=data_format)

            net = net + identity

            return net

        with tf.variable_scope('discriminator', reuse=reuse) as scope:
            if self.height == 32:
                net = res_blk_first(inputs, 128, self.data_format, down_sampling=True, scope='res_blk1')
                net = res_blk(net, 128, self.data_format, down_sampling=True, scope='res_blk2')
                net = res_blk(net, 128, self.data_format, down_sampling=False, scope='res_blk3')
                net = res_blk(net, 128, self.data_format, down_sampling=False, scope='res_blk4')
                net = tf.nn.relu(net)
                if self.data_format == 'NHWC':
                    net = tf.reduce_sum(net, axis=(1, 2))
                elif self.data_format == 'NCHW':
                    net = tf.reduce_sum(net, axis=(2, 3))
                logit = ops.fully_connected(net, 1, sn=True)
            elif self.height == 64:
                net = res_blk_first(inputs, 64, self.data_format, down_sampling=True, scope='res_blk1')
                net = res_blk(net, 128, self.data_format, down_sampling=True, scope='res_blk2')
                net = res_blk(net, 256, self.data_format, down_sampling=True, scope='res_blk3')
                net = res_blk(net, 512, self.data_format, down_sampling=True, scope='res_blk4')
                net = res_blk(net, 1024, self.data_format, down_sampling=True, scope='res_blk5')
                net = tf.nn.relu(net)
                if self.data_format == 'NHWC':
                    net = tf.reduce_sum(net, axis=(1, 2))
                elif self.data_format == 'NCHW':
                    net = tf.reduce_sum(net, axis=(2, 3))
                logit = ops.fully_connected(net, 1, sn=True)
        return tf.nn.sigmoid(logit), logit

    def generator(self, inputs, reuse):
        def res_blk(net, channel, data_format, is_training, up_sampling=False, scope='res_blk'):
            if data_format == 'NCHW':
                channel_origin = net.get_shape().as_list()[1]
                height_origin = net.get_shape().as_list()[2]
                width_origin = net.get_shape().as_list()[3]
            elif data_format == 'NHWC':
                channel_origin = net.get_shape().as_list()[3]
                height_origin = net.get_shape().as_list()[1]
                width_origin = net.get_shape().as_list()[2]

            identity = net

            net = slim.batch_norm(net, is_training=is_training, data_format=data_format)
            net = tf.nn.relu(net)
            if up_sampling:
                if data_format == 'NCHW':
                    net = tf.transpose(net, [0, 2, 3, 1])
                net = tf.image.resize_images(net, (height_origin * 2, width_origin * 2))
                if data_format == 'NCHW':
                    net = tf.transpose(net, [0, 3, 1, 2])
            net = slim.conv2d(net, channel, [3, 3], 1, scope='{}_deconv1'.format(scope))
            net = slim.batch_norm(net, is_training=is_training, data_format=data_format)
            net = tf.nn.relu(net)
            net = slim.conv2d(net, channel, [3, 3], 1, scope='{}_deconv2'.format(scope))

            if up_sampling:
                if data_format == 'NCHW':
                    identity = tf.transpose(identity, [0, 2, 3, 1])
                identity = tf.image.resize_images(identity, (height_origin * 2, width_origin * 2))
                if data_format == 'NCHW':
                    identity = tf.transpose(identity, [0, 3, 1, 2])
            if channel != channel_origin:
                #  identity = slim.conv2d_transpose(identity, channel, [3, 3], 1, scope='{}_identity_deconv'.format(scope))
                identity = slim.conv2d(identity, channel, [1, 1], 1, scope='{}_identity_conv'.format(scope))

            net = net + identity

            return net

        with tf.variable_scope('generator', reuse=reuse) as scope:
            with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                                normalizer_fn=None,
                                #  weights_initializer=tf.variance_scaling_initializer(scale=2.0, mode='fan_in', distribution='uniform'),
                                #  weights_regularizer=slim.l2_regularizer(weight_decay),
                                data_format=self.data_format,
                                padding='SAME',
                                activation_fn=None):

                if self.height == 32:
                    net = slim.fully_connected(inputs, 4 * 4 * 256, activation_fn=tf.nn.relu)
                    if self.data_format == 'NCHW':
                        net = tf.reshape(net, [-1, 256, 4, 4])
                    elif self.data_format == 'NHWC':
                        net = tf.reshape(net, [-1, 4, 4, 256])
                    net = res_blk(net, 256, self.data_format, self.var['is_training'], up_sampling=True, scope='res_blk1')
                    net = res_blk(net, 256, self.data_format, self.var['is_training'], up_sampling=True, scope='res_blk2')
                    net = res_blk(net, 256, self.data_format, self.var['is_training'], up_sampling=True, scope='res_blk3')
                    net = slim.batch_norm(net, is_training=self.var['is_training'], data_format=self.data_format)
                    net = tf.nn.relu(net)
                    net = slim.conv2d(net, 3, [3, 3], 1, data_format=self.data_format, scope='last_conv')
                    net = tf.nn.tanh(net)

                elif self.height == 64:
                    net = slim.fully_connected(inputs, 4 * 4 * 1024, activation_fn=tf.nn.relu)
                    if self.data_format == 'NCHW':
                        net = tf.reshape(net, [-1, 1024, 4, 4])
                    elif self.data_format == 'NHWC':
                        net = tf.reshape(net, [-1, 4, 4, 1024])
                    net = res_blk(net, 512, self.data_format, self.var['is_training'], up_sampling=True, scope='res_blk1')
                    net = res_blk(net, 256, self.data_format, self.var['is_training'], up_sampling=True, scope='res_blk2')
                    net = res_blk(net, 128, self.data_format, self.var['is_training'], up_sampling=True, scope='res_blk3')
                    net = res_blk(net, 64, self.data_format, self.var['is_training'], up_sampling=True, scope='res_blk4')
                    net = slim.batch_norm(net, is_training=self.var['is_training'], data_format=self.data_format)
                    net = tf.nn.relu(net)
                    net = slim.conv2d(net, 3, [3, 3], 1, data_format=self.data_format, scope='last_conv')
                    net = tf.nn.tanh(net)

        return net

    def _build_train_op(self, **hparams):
        # Now compute global loss and gradients.
        D_gradvars = get_gradvars(self.tower['D_gradvars'])
        G_gradvars = get_gradvars(self.tower['G_gradvars'])

        consolidation_device = '/gpu:0' if self.ps_strategy == 'GPU' else '/cpu:0'
        with tf.device(consolidation_device):
            self.var['global_step'] = tf.get_variable('global_step', initializer=0, trainable=False)
            self.var['global_epoch'] = tf.get_variable('global_epoch', initializer=0, trainable=False)
            self.var['g_lr'] = tf.train.exponential_decay(
                hparams['g_lr'], self.var['global_epoch'], hparams['lr_decay_epoch'], hparams['lr_decay_rate'], staircase=True, name='g_decay')
            self.var['d_lr'] = tf.train.exponential_decay(
                hparams['d_lr'], self.var['global_epoch'], hparams['lr_decay_epoch'], hparams['lr_decay_rate'], staircase=True, name='d_decay')

            d_opt = tf.train.AdamOptimizer(self.var['g_lr'], hparams['beta1'], hparams['beta2'])
            g_opt = tf.train.AdamOptimizer(self.var['d_lr'], hparams['beta1'], hparams['beta2'])

            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                self.train_D = d_opt.apply_gradients(D_gradvars, global_step=self.var['global_step'])
                self.train_G = d_opt.apply_gradients(G_gradvars)

            self.global_epoch_incrementor = tf.assign(self.var['global_epoch'], self.var['global_epoch'] + 1)

            # Combine all of tower outputs
            self.var['D_p'] = tf.concat(self.tower['D_p'], axis=0)
            self.var['D_p_logit'] = tf.concat(self.tower['D_p_logit'], axis=0)
            self.var['D_n'] = tf.concat(self.tower['D_n'], axis=0)
            self.var['D_n_logit'] = tf.concat(self.tower['D_n_logit'], axis=0)
            self.var['G'] = tf.concat(self.tower['G'], axis=0)
            self.var['D_loss'] = tf.reduce_mean(self.tower['D_loss'])
            self.var['G_loss'] = tf.reduce_mean(self.tower['G_loss'])

            self.sum_list.append(tf.summary.scalar('D_loss', self.var['D_loss']))
            self.sum_list.append(tf.summary.scalar('G_loss', self.var['G_loss']))
            self.sum_list.append(tf.summary.scalar('D_lr', self.var['d_lr']))
            self.sum_list.append(tf.summary.scalar('G_lr', self.var['g_lr']))
            self.inception_score_summary = tf.summary.scalar('Inception_score', self.inception_score)

            # Merge all the summaries and write them out
            self.merged = tf.summary.merge(self.sum_list)
            self.saver = tf.train.Saver()

    def train(self, config, sess):
        dataset = DataSet(config.data, config.data_dir, self.data_format, config.normalise)

        train_writer = tf.summary.FileWriter(os.path.join(self.dir['summary'], 'train'),
                                                  sess.graph)
        init = tf.global_variables_initializer()
        sess.run(init)

        #  builder = tf.profiler.ProfileOptionBuilder
        #  opts = (tf.profiler.ProfilOptionBuilder.trainable_variables_parameter())
        #  _ = tf.profiler.profile(tf.get_default_graph(), options=opts)
        #
        #  tf.profiler.profile(sess.graph, run_meta, op_log, cmd, options)
        #  opts = tf.contrib.tfprof.model_analyzer.TRAINABLE_VARS_PARAMS_STAT_OPTIONS
        #  tf.contrib.tfprof.model_analyzer.print_model_analysis(sess.graph, tfprof_options=opts)

        counter = 0
        could_load, checkpoint_counter = self.load_model(sess, self.dir['checkpoint'])
        if could_load:
            counter = checkpoint_counter
            print('[*] Load SUCCESS')
        else:
            print('[!] Load failed...')

        logging_epoch = 1
        save_epoch = 5
        total_batch = int(np.floor(dataset.n_train / config.batch_size))
        start_time = time.time()
        with utils.std_out_err_redirect_tqdm() as orig_stdout:
            for epoch in tqdm(range(1, config.max_epoch + 1),
                              total=config.max_epoch,
                              desc='Training',
                              leave=True,
                              position=0,
                              file=orig_stdout.terminal):

                counter += 1
                sess.run(self.global_epoch_incrementor)
                d_total_loss = g_total_loss = 0

                for batch_data, batch_label in tqdm(dataset.iter(config.batch_size,
                                                                 which='train'),
                                                    total=total_batch,
                                                    desc='Epoch {}'.format(epoch),
                                                    leave=False,
                                                    position=1,
                                                    file=orig_stdout.terminal):
                    if batch_data.shape[0] != config.batch_size:
                        break

                    if config.z_dist == 'uniform':
                        bathc_z = np.random.uniform(-1., 1., [config.batch_size, config.z_dim])
                    elif config.z_dist == 'normal':
                        batch_z = np.random.normal(0, 1, [config.batch_size, config.z_dim])

                    _, d_losses = sess.run([self.train_D, self.var['D_loss']],
                                           feed_dict={self.var['input_data']: batch_data,
                                                      self.var['input_z']: batch_z,
                                                      self.var['is_training']: True})
                    _, g_losses = sess.run([self.train_G, self.var['G_loss']],
                                           feed_dict={self.var['input_z']: batch_z,
                                                      self.var['is_training']: True})
                    d_total_loss += d_losses
                    g_total_loss += g_losses

                d_total_loss /= total_batch
                g_total_loss /= total_batch

                batch_data, _ = dataset.get_data(config.batch_size,  which='train')
                if config.z_dist == 'uniform':
                    bathc_z = np.random.uniform(-1., 1., [config.batch_size, config.z_dim])
                elif config.z_dist == 'normal':
                    batch_z = np.random.normal(0, 1, [config.batch_size, config.z_dim])
                summary = sess.run(self.merged, feed_dict={self.var['input_data']: batch_data,
                                                           self.var['input_z']: batch_z,
                                                           self.var['is_training']: True})
                train_writer.add_summary(summary, counter)

                if epoch % logging_epoch == 0:
                    #  orig_stdout.terminal.write("\033[G\033[K") # go to the beginning of the line and clear to the end of line
                    orig_stdout.terminal.write("\033[G\033[K\033[s\033[B\033[K\033[u") # go to the beginning of the line and clear to the end of line
                    inception_size = 10000
                    split = 10
                    gen_imgs = []
                    for i in range(math.ceil(inception_size / config.batch_size)):
                        if config.z_dist == 'uniform':
                            bathc_z = np.random.uniform(-1., 1., [config.batch_size, config.z_dim])
                        elif config.z_dist == 'normal':
                            batch_z = np.random.normal(0, 1, [config.batch_size, config.z_dim])

                        gen_img = sess.run(
                            self.var['G'],
                            feed_dict={self.var['input_z']: batch_z, self.var['is_training']: False})
                        gen_imgs.append(gen_img)

                    gen_imgs = np.vstack(gen_imgs)
                    gen_imgs = gen_imgs[:inception_size]
                    inception_mean, inception_std = inception_score.get_inception_score(gen_imgs, session=sess)

                    #  scores = []
                    #  chunk = int(inception_size / split)
                    #  for i in range(split):
                    #      inception_score, inception_sum = \
                    #          sess.run([self.inception_score, self.inception_score_summary], feed_dict={self.var['input_data']: gen_imgs[i * chunk: i * chunk + chunk]})
                    #      scores.append(inception_score)
                    #  train_writer.add_summary(inception_sum, counter)
                    #  inception_mean, inception_std = np.mean(scores), np.std(scores)

                    if self.data_format == 'NCHW':
                        gen_imgs = np.transpose(gen_imgs, [0, 2, 3, 1])
                    gen_imgs = np.clip((gen_imgs + 1.0) * 127.5, 0.0, 255.0)
                    gen_tiled_imgs = common.img_tile(
                        gen_imgs[:100], border_color=1.0, stretch=True)
                    file_name = ''.join([self.dir['imgs/train'], '/generated_', str(counter).zfill(4), '.png'])
                    matplotlib.image.imsave(file_name, gen_tiled_imgs)

                    end_time = time.time()
                    training_spent = end_time - start_time
                    print('Epoch: {:>5d} ({:6<.2f}s){space}Counter: {:>5d}{space}D_train_loss: {:.4f}{space}G_train_loss: {:.4f}{space}Inception_score: {:.5f}, {:.5f}'.format(
                        epoch, training_spent, counter, d_total_loss, g_total_loss, inception_mean, inception_std, space='     '))
                    start_time = time.time()

                # Save model
                if epoch % save_epoch == 0:
                    self.save_model(sess, self.dir['checkpoint'], sess.run(self.var['global_step']))

    def test(self, config, sess):
        counter = 0
        could_load, checkpoint_counter = self.load_model(sess, self.dir['checkpoint'])
        if could_load:
            counter = checkpoint_counter
            print('[*] Load SUCCESS')
        else:
            print('[!] Load failed...')

        inception_size = 50000
        split = 50
        gen_imgs = []
        for i in range(math.ceil(inception_size / config.batch_size)):
            if config.z_dist == 'uniform':
                bathc_z = np.random.uniform(-1., 1., [config.batch_size, config.z_dim])
            elif config.z_dist == 'normal':
                batch_z = np.random.normal(0, 1, [config.batch_size, config.z_dim])

            gen_img = sess.run(
                self.var['G'],
                feed_dict={self.var['input_z']: batch_z, self.var['is_training']: False})
            gen_imgs.append(gen_img)

        gen_imgs = np.vstack(gen_imgs)
        gen_imgs = gen_imgs[:inception_size]
        mean, std = inception_score.get_inception_score(gen_imgs, session=sess)
        print(mean, std)

        #  gen_imgs = np.clip((gen_imgs + 1.0) * 127.5, 0.0, 255.0)
        #  scores = []
        #  chunk = int(inception_size / split)
        #  for i in range(split):
        #      inception_score = sess.run(self.inception_score, feed_dict={self.var['input_data']: gen_imgs[i * chunk: i * chunk + chunk]})
        #      scores.append(inception_score)
        #
        #  mean, std = np.mean(scores), np.std(scores)
        #  print(mean, std)


    def save_model(self, sess, model_dir, step):
        self.saver.save(sess, os.path.join(model_dir, self.model_name), global_step=step)

    def load_model(self, sess, model_dir):
        import re
        print('[*] Reading checkpoints...')

        ckpt = tf.train.get_checkpoint_state(model_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(sess, os.path.join(model_dir, ckpt_name))
            counter = int(next(re.finditer('(\d+)(?!.*d)', ckpt_name)).group(0))
            print('[*] Success to read {}'.format(ckpt_name))
            return True, counter
        else:
            print('[*] Failed to find a checkpoint')
            return False, 0
