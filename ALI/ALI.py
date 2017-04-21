import os
import time
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import matplotlib.pyplot as plt
import cv2

import tf_utils.common as common
from tf_utils.data.dataset import DataSet

from tqdm import tqdm


class ALI(object):
    def __init__(
            self, sess, log_dir, data, data_dir,
            batch_size=128, z_dim=100, z_dist='uniform'):
        self.sess = sess

        print('[*] Training start.... {}'.format(
            time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())))
        self.img_dir = os.path.join(log_dir, 'imgs')
        self.summary_dir = os.path.join(log_dir, 'summary')
        self.model_dir = os.path.join(log_dir, 'models')
        self.f = open(os.path.join(log_dir, 'training.log'), 'a')
        self.prepare_directory(self.img_dir)
        self.prepare_directory(self.summary_dir)
        self.prepare_directory(self.model_dir)

        self.is_training = tf.placeholder(tf.bool, [], name='is_training')

        self.data = data
        self.load_data(data_dir)

        self.batch_size = batch_size

        self.z_dim = z_dim
        self.z_dist = z_dist

        gpus = os.environ['CUDA_VISIBLE_DEVICES']
        self.n_gpu = len([int(s) for s in gpus.split(',') if s.isdigit()])

        self.build_model()

    def load_data(self, data_dir):
        self.data_set = DataSet(self.data, data_dir, normalise=True)

        self.height  = self.data_set.data_shape[0]
        self.width   = self.data_set.data_shape[1]
        self.channel = self.data_set.data_shape[2]

    def prepare_directory(self, log_dir=None, delete=False):
        if log_dir is None:
            print('[!] log_dir must be provided.')
        else:
            if delete:
                common.delete_and_create_directory(log_dir)
            else:
                common.create_directory(log_dir)

    def build_model(self):
        self.inputs = tf.placeholder(
            tf.float32, [self.batch_size * self.n_gpu, self.height, self.width, self.channel], name='inputs')

        # These are the lists for each tower
        self.tower_inputs     = []
        self.tower_encoded_z  = []
        self.tower_reconst_x  = []
        self.tower_D_p        = []
        self.tower_D_p_logits = []
        self.tower_D_n        = []
        self.tower_D_n_logits = []
        self.tower_D_loss     = []
        self.tower_G_loss     = []
        self.sum_list         = []

        # Define the network for each GPU
        for i in xrange(self.n_gpu):
            with tf.device('/gpu:%d' % i):
                with tf.name_scope('Tower_%d' % (i)) as scope:
                    # Grab this portion of the input
                    inputs_per_gpu = self.inputs[i * self.batch_size:(i + 1) * self.batch_size, :]

                    # Construct the model
                    D_p, D_p_logits, D_n, D_n_logits, encoded_z, reconst_x = \
                        self.build_model_per_gpu(inputs_per_gpu)

                    # Calculate the loss for this tower
                    D_loss, G_loss = \
                        self.get_loss(D_p_logits, D_n_logits)

                    # logging tensorboard
                    self.sum_list.append(tf.summary.scalar('tower_{}/discriminator_loss'.format(i), D_loss))
                    self.sum_list.append(tf.summary.scalar('tower_{}/generator_loss'.format(i), G_loss))

                    # Reuse variables for the next tower
                    # tf.get_variable_scope().reuse_variables()

                    # Keep track of models across all towers
                    self.tower_inputs.append(inputs_per_gpu)
                    self.tower_encoded_z.append(encoded_z)
                    self.tower_reconst_x.append(reconst_x)
                    self.tower_D_p.append(D_p)
                    self.tower_D_p_logits.append(D_p_logits)
                    self.tower_D_n.append(D_n)
                    self.tower_D_n_logits.append(D_n_logits)
                    self.tower_D_loss.append(D_loss)
                    self.tower_G_loss.append(G_loss)

        self.saver = tf.train.Saver()

    def build_model_per_gpu(self, inputs):
        if self.z_dist == 'normal':
            z = tf.random_normal([self.batch_size, 1, 1, self.z_dim])
            eps = tf.random_normal([self.batch_size, 1, 1, self.z_dim])
        else:
            print ('z_dist error! It must be normal.')
        squeeze_eps = tf.squeeze(eps)
        squeeze_z = tf.squeeze(z)

        z_mean, z_log_sigma_sq = self.encoder(inputs)
        # grab our actual z
        encoded_z = tf.add(z_mean, tf.multiply(tf.sqrt(tf.exp(z_log_sigma_sq)), squeeze_z))
        reconst_x = self.decoder(eps)
        D_p, D_p_logits = self.discriminator(inputs, encoded_z)
        D_n, D_n_logits = self.discriminator(reconst_x, squeeze_eps, reuse=True)

        return D_p, D_p_logits, D_n, D_n_logits, encoded_z, reconst_x

    def get_loss(self, D_p_logits, D_n_logits):
        D_loss = tf.reduce_mean(
            tf.nn.softplus(-D_p_logits) + tf.nn.softplus(-D_n_logits) + D_n_logits)
        G_loss = tf.reduce_mean(
            tf.nn.softplus(-D_p_logits) + D_p_logits + tf.nn.softplus(-D_n_logits))

        return D_loss, G_loss

    def discriminator(self, x_inputs, z_inputs, reuse=False):
        with tf.variable_scope('discriminator', reuse=reuse) as scope:

            batch_norm_params = {
                'is_training': self.is_training, 'updates_collections': None}
            with slim.arg_scope([slim.conv2d, slim.fully_connected],
                                padding='SAME',
                                normalizer_fn=slim.batch_norm,
                                normalizer_params=batch_norm_params,
                                activation_fn=common.leaky_relu):
                net = slim.conv2d(x_inputs, 32, [5, 5], 2, scope='conv1')
                net = slim.conv2d(net, 64, [5, 5], 2, scope='conv2')
                net = slim.conv2d(net, 128, [5, 5], 1, padding='VALID', scope='conv3')
                net = slim.dropout(net, 0.9, is_training=self.is_training, scope='dropout3')
                net = slim.flatten(net)
                net = tf.concat((net, z_inputs), 1)

            with slim.arg_scope([slim.fully_connected],
                                activation_fn=common.leaky_relu):
                net = slim.fully_connected(net, 128, scope='fc1')
                net = slim.fully_connected(net, 128, scope='fc2')
                logits = slim.fully_connected(net, 1, normalizer_fn=None, activation_fn=None, scope='fc3')

        return tf.nn.sigmoid(logits), logits

    def decoder(self, inputs, reuse=False):
        with tf.variable_scope('decoder', reuse=reuse) as scope:

            batch_norm_params = {
                'is_training': self.is_training, 'updates_collections': None}
            with slim.arg_scope([slim.conv2d_transpose],
                                padding='SAME',
                                normalizer_fn=slim.batch_norm,
                                normalizer_params=batch_norm_params,
                                activation_fn=common.leaky_relu):
                if self.data == 'mnist':
                    net = slim.conv2d_transpose(inputs, 128, [3, 3], 1, padding='VALID', scope='deconv1')
                    net = slim.conv2d_transpose(net, 64, [5, 5], 1, padding='VALID', scope='deconv2')
                    net = slim.conv2d_transpose(net, 32, [5, 5], 2, scope='deconv3')
                    net = slim.conv2d_transpose(net, 1, [5, 5], 2, normalizer_fn=None, activation_fn=tf.sigmoid, scope='deconv4')
                    return net

                elif self.data == 'cifar10':
                    net = slim.conv2d_transpose(inputs, 128, [4, 4], 1, padding='VALID', scope='deconv1')
                    net = slim.conv2d_transpose(net, 64, [5, 5], 1, padding='VALID', scope='deconv2')
                    net = slim.conv2d_transpose(net, 32, [5, 5], 2, scope='deconv3')
                    net = slim.conv2d_transpose(net, 3, [5, 5], 2, normalizer_fn=None, activation_fn=tf.sigmoid, scope='deconv4')
                    return net

    def encoder(self, inputs, reuse=False):
        with tf.variable_scope('encoder', reuse=reuse) as scope:

            batch_norm_params = {
                'is_training': self.is_training, 'updates_collections': None}
            with slim.arg_scope([slim.conv2d],
                                padding='SAME',
                                normalizer_fn=slim.batch_norm,
                                normalizer_params=batch_norm_params,
                                activation_fn=common.leaky_relu):
                net = slim.conv2d(inputs, 32, [5, 5], 2, scope='conv1')
                net = slim.conv2d(net, 64, [5, 5], 2, scope='conv2')
                net = slim.conv2d(net, 128, [5, 5], 1, padding='VALID', scope='conv3')
                net = slim.flatten(net)

            with slim.arg_scope([slim.fully_connected],
                                normalizer_fn=None,
                                normalizer_params=None,
                                activation_fn=None):
                z_mean = slim.fully_connected(net, self.z_dim, scope='fc1_mean')
                z_log_sigma_sq = slim.fully_connected(net, self.z_dim, scope='fc1_sigma')

        return z_mean, z_log_sigma_sq

    def train(self, config):

        d_opt = tf.train.AdamOptimizer(config.d_lr, beta1=config.beta1)
        g_opt = tf.train.AdamOptimizer(config.g_lr, beta1=config.beta1)

        # Merge all the summaries and write them out
        self.merged = tf.summary.merge(self.sum_list)
        self.board_writer = tf.summary.FileWriter(self.summary_dir, self.sess.graph)

        G_params = [param for param in tf.trainable_variables()
                    if 'encoder' in param.name or 'decoder' in param.name]
        D_params = [param for param in tf.trainable_variables()
                    if 'discriminator' in param.name]

        tower_gen_grads = []
        tower_disc_grads = []
        for i in xrange(self.n_gpu):
            # Calculate the gradients for the batch of data on this tower
            disc_grads = d_opt.compute_gradients(self.tower_D_loss[i], var_list=D_params)
            gen_grads = g_opt.compute_gradients(self.tower_G_loss[i], var_list=G_params)

            # Keep track of the gradients across all towers
            tower_disc_grads.append(disc_grads)
            tower_gen_grads.append(gen_grads)

        # Average the gradients
        disc_grads = common.average_gradient(tower_disc_grads)
        gen_grads = common.average_gradient(tower_gen_grads)

        # Apply the gradients with our optimizers
        D_train = d_opt.apply_gradients(disc_grads)
        G_train = g_opt.apply_gradients(gen_grads)

        # Train starts
        init = tf.global_variables_initializer()
        self.sess.run(init)

        counter = 0
        could_load, checkpoint_counter = self.load_model(self.model_dir)
        if could_load:
            counter = checkpoint_counter
            print('[*] Load SUCCESS')
        else:
            print('[!] Load failed...')

        total_batch = int(np.floor(self.data_set.n_train / (self.batch_size * self.n_gpu)))

        for epoch in xrange(config.max_epoch + 1):
            d_total_loss = g_total_loss = 0
            with tqdm(total=total_batch, leave=False) as pbar:
                for batch, label in self.data_set.iter(
                        self.batch_size * self.n_gpu, which='train'):
                    if batch.shape[0] != self.batch_size * self.n_gpu:
                        break

                    _, d_losses = self.sess.run(
                        [D_train, self.tower_D_loss],
                        feed_dict={self.inputs: batch,
                                   self.is_training: True})

                    _, g_losses = self.sess.run(
                        [G_train, self.tower_G_loss],
                        feed_dict={self.inputs: batch,
                                   self.is_training: True})

                    # Write Tensorboard log
                    summary = self.sess.run(
                        self.merged,
                        feed_dict={self.inputs: batch,
                                   self.is_training: True})
                    self.board_writer.add_summary(summary, counter)

                    d_total_loss += np.array(d_losses).mean()
                    g_total_loss += np.array(g_losses).mean()

                    pbar.set_description('Epoch {}'.format(epoch))
                    pbar.update()

                    # monitor generated data
                    if config.monitor:
                        gen_imgs = self.sess.run(
                            self.tower_reconst_x[0],
                            feed_dict={self.is_training: False})
                        gen_tiled_imgs = common.img_tile(
                            gen_imgs[0:100], border_color=1.0, stretch=True)
                        gen_tiled_imgs = gen_tiled_imgs[:, :, ::-1]
                        cv2.imshow('generated data', gen_tiled_imgs)
                        cv2.waitKey(1)

            # monitor training data
            if config.monitor:
                training_tiled_imgs = common.img_tile(
                    batch[0:100], border_color=1.0, stretch=True)
                training_tiled_imgs = training_tiled_imgs[:, :, ::-1]
                cv2.imshow('training data', training_tiled_imgs)
                cv2.waitKey(1)

            d_total_loss /= total_batch
            g_total_loss /= total_batch

            # Print display network output
            print('Counter: {}\t D_loss: {:.4f}\t G_loss: {:.4f}'.format(
                counter, d_total_loss, g_total_loss))
            self.f.flush()

            # Save model
            if counter % 100 == 0:
                self.save_model(self.model_dir, counter)

            # Save images
            if counter % 10 == 0:
                gen_imgs = self.sess.run(
                    self.tower_reconst_x[0],
                    feed_dict={self.is_training: False})
                gen_tiled_imgs = common.img_tile(
                    gen_imgs[0:100], border_color=1.0, stretch=True)
                gen_tiled_imgs = gen_tiled_imgs[:, :, ::-1]
                file_name = ''.join([self.img_dir, '/generated_', str(counter).zfill(4), '.jpg'])
                cv2.imwrite(file_name, gen_tiled_imgs * 255.)

            counter += 1

        cv2.destroyAllWindows()
        self.f.close()

    def save_model(self, model_dir, step):
        model_name = 'ALI.model'
        self.saver.save(self.sess, os.path.join(model_dir, model_name), global_step=step)

    def load_model(self, model_dir):
        import re
        print('[*] Reading checkpoints...')

        ckpt = tf.train.get_checkpoint_state(model_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(model_dir, ckpt_name))
            counter = int(next(re.finditer('(\d+)(?!.*d)', ckpt_name)).group(0))
            print('[*] Success to read {}'.format(ckpt_name))
            return True, counter
        else:
            print('[*] Failed to find a checkpoint')
            return False, 0
