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


class DCGAN_DFM(object):
    def __init__(
            self, sess, log_dir, data, data_dir, lambda_adv, lambda_denoise,
            batch_size=128, z_dim=100, z_dist='uniform',):
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
        self.data_dir = data_dir
        self.load_data(data_dir)

        self.batch_size = batch_size

        self.z_dim = z_dim
        self.z_dist = z_dist
        self.lambda_adv = lambda_adv
        self.lambda_denoise = lambda_denoise

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
        self.input_x = tf.placeholder(
            tf.float32,
            [self.batch_size * self.n_gpu, self.height, self.width, self.channel],
            name='data_input')

        # These are the lists for each tower
        self.tower_z            = []
        self.tower_G            = []
        self.tower_D_p          = []
        self.tower_D_p_logits   = []
        self.tower_D_p_feature  = []
        self.tower_D_n          = []
        self.tower_D_n_logits   = []
        self.tower_D_n_feature  = []
        self.tower_D_loss       = []
        self.tower_G_loss       = []
        self.tower_Denoise_loss = []
        self.tower_denoiser_p   = []
        self.tower_denoiser_n   = []
        self.sum_list           = []

        self.normal = tf.contrib.distributions.Normal(mu=0., sigma=1.)

        # Define the network for each GPU
        for i in xrange(self.n_gpu):
            with tf.device('/gpu:%d' % i):
                with tf.name_scope('Tower_%d' % (i)) as scope:
                    # Grab this portion of the input
                    x_portion = \
                        self.input_x[i * self.batch_size:(i + 1) * self.batch_size, :]

                    # Construct the model
                    self.build_model_per_gpu(x_portion)

                    # Calculate the loss for this tower
                    D_loss, G_loss, Denoise_loss = \
                        self.get_loss(self.tower_D_p_logits[-1],
                                      self.tower_D_n_logits[-1],
                                      self.tower_D_p_feature[-1],
                                      self.tower_D_n_feature[-1],
                                      self.tower_denoiser_p[-1],
                                      self.tower_denoiser_n[-1])

                    # logging tensorboard
                    self.sum_list.append(
                        tf.summary.scalar('tower_{}/discriminator_loss'.format(i), D_loss))
                    self.sum_list.append(
                        tf.summary.scalar('tower_{}/generator_loss'.format(i), G_loss))

                    # Reuse variables for the next tower
                    # tf.get_variable_scope().reuse_variables()

                    # Keep track of models across all towers
                    self.tower_D_loss.append(D_loss)
                    self.tower_G_loss.append(G_loss)
                    self.tower_Denoise_loss.append(Denoise_loss)

        self.saver = tf.train.Saver()

    def build_model_per_gpu(self, x):
        if self.z_dist == 'uniform':
            z = tf.random_uniform((self.batch_size, 1, 1, self.z_dim), -1., 1.)
        elif self.z_dist == 'normal':
            z = tf.random_normal([self.batch_size, 1, 1, self.z_dim])
        else:
            print ('z_dist error! It must be either uniform or normal.')

        D_p_feature, D_p, D_p_logits = self.discriminator(x, reuse=False)
        currupted_feature = D_p_feature + self.normal.sample(D_p_feature.get_shape().as_list())
        denoiser_p = self.denoiser(currupted_feature, 1024, reuse=False)
        G = self.generator(z)
        D_n_feature, D_n, D_n_logits = self.discriminator(G, reuse=True)
        denoiser_n = self.denoiser(D_n_feature, 1024, reuse=True)

        self.tower_z.append(z)
        self.tower_G.append(G)
        self.tower_D_p.append(D_p)
        self.tower_D_p_logits.append(D_p_logits)
        self.tower_D_p_feature.append(D_p_feature)
        self.tower_D_n.append(D_n)
        self.tower_D_n_logits.append(D_n_logits)
        self.tower_D_n_feature.append(D_n_feature)
        self.tower_denoiser_p.append(denoiser_p)
        self.tower_denoiser_n.append(denoiser_n)

    def get_loss(self, D_p_logits, D_n_logits, D_p_feature, D_n_feature, denoiser_p, denoiser_n):
        D_loss = tf.reduce_mean(tf.nn.softplus(-D_p_logits)) \
            + tf.reduce_mean(tf.nn.softplus(-D_n_logits) + D_n_logits)

        denoiser_n = tf.stop_gradient(denoiser_n)
        G_loss = tf.reduce_mean(
            self.lambda_adv * tf.nn.softplus(-D_n_logits) +
            self.lambda_denoise * tf.reduce_sum(tf.square(D_n_feature - denoiser_n), axis=1))

        Denoise_loss = tf.reduce_mean(
            tf.reduce_sum(tf.square(D_p_feature - denoiser_p), axis=1))

        return D_loss, G_loss, Denoise_loss

    def discriminator(self, x, reuse=False):
        with tf.variable_scope('discriminator', reuse=reuse) as scope:

            batch_norm_params = {
                'is_training': self.is_training, 'updates_collections': None}

            with slim.arg_scope([slim.conv2d],
                                padding='SAME',
                                normalizer_fn=slim.batch_norm,
                                normalizer_params=batch_norm_params,
                                activation_fn=common.leaky_relu):

                if self.data == 'cifar10':
                    net = slim.conv2d(x, 64, 5, 2, normalizer_fn=None, scope='conv1')
                    net = slim.conv2d(net, 128, 5, 2, scope='conv2')
                    net = slim.conv2d(net, 256, 5, 2, scope='conv3')
                    net = slim.conv2d(net, 512, 5, 2, scope='conv4')
                    net = slim.flatten(net)
                    feature = slim.fully_connected(
                        net, 1024,
                        activation_fn=None,
                        normalizer_fn=None,
                        normalizer_params=None, scope='fc1')
                    logits = slim.fully_connected(
                        feature, 1,
                        activation_fn=None, scope='fc2')

                elif self.data == 'mnist':
                    net = slim.conv2d(x, 64, 5, 2, normalizer_fn=None, scope='conv1')
                    net = slim.conv2d(net, 128, 5, 2, scope='conv2')
                    net = slim.flatten(net)
                    feature = slim.fully_connected(
                        net, 1024,
                        activation_fn=None,
                        normalizer_fn=None,
                        normalizer_params=None, scope='fc1')
                    logits = slim.fully_connected(
                        feature, 1,
                        activation_fn=None, scope='fc2')

            return feature, tf.nn.sigmoid(logits), logits

    def generator(self, z):
        with tf.variable_scope('generator') as scope:
            if self.data == 'cifar10':
                dim = 2
            elif self.data == 'mnist':
                dim = 7

            batch_norm_params = {
                'is_training': self.is_training, 'updates_collections': None}
            with slim.arg_scope([slim.conv2d_transpose],
                                padding='SAME',
                                normalizer_fn=slim.batch_norm,
                                normalizer_params=batch_norm_params,
                                activation_fn=tf.nn.relu):
                if self.data == 'cifar10':
                    net = slim.fully_connected(
                        z, dim * dim * 1024,
                        normalizer_fn=slim.batch_norm,
                        normalizer_params=batch_norm_params,
                        activation_fn=tf.nn.relu, scope='proj')
                    net = tf.reshape(net, [-1, dim, dim, 1024])
                    net = slim.conv2d_transpose(net, 512, 5, 2, scope='deconv1')
                    net = slim.conv2d_transpose(net, 256, 5, 2, scope='deconv2')
                    net = slim.conv2d_transpose(net, 128, 5, 2, scope='deconv3')
                    net = slim.conv2d_transpose(net, 3, 5, 2, normalizer_fn=None, activation_fn=tf.sigmoid, scope='deconv4')
                elif self.data == 'mnist':
                    net = slim.fully_connected(
                        z, 1024,
                        normalizer_fn=slim.batch_norm,
                        normalizer_params=batch_norm_params,
                        activation_fn=tf.nn.relu, scope='fc1')
                    net = slim.fully_connected(
                        net, dim * dim * 128,
                        normalizer_fn=slim.batch_norm,
                        normalizer_params=batch_norm_params,
                        activation_fn=tf.nn.relu, scope='fc2')
                    net = tf.reshape(net, [-1, dim, dim, 128])
                    net = slim.conv2d_transpose(net, 128, 5, 2, scope='deconv1')
                    net = slim.conv2d_transpose(net, 1, 5, 2, normalizer_fn=None, activation_fn=tf.sigmoid, scope='deconv2')
            return net

    def denoiser(self, inputs, n_hidden, reuse=False):
        with tf.variable_scope('denoiser', reuse=reuse) as scope:
            net = inputs
            batch_norm_params = {
                'is_training': self.is_training, 'updates_collections': None}
            with slim.arg_scope([slim.fully_connected],
                                normalizer_fn=slim.batch_norm,
                                normalizer_params=batch_norm_params,
                                activation_fn=tf.nn.relu):
                net = slim.repeat(net, 9, slim.fully_connected, n_hidden, scope='fc')
                net = slim.fully_connected(
                    net, n_hidden, normalizer_fn=None, activation_fn=None, scope='fc_last')
            return net

    def train(self, config):

        d_opt = tf.train.AdamOptimizer(config.d_lr, beta1=config.beta1)
        g_opt = tf.train.AdamOptimizer(config.g_lr, beta1=config.beta1)
        denoise_opt = tf.train.AdamOptimizer(config.denoise_lr, beta1=config.beta1)

        G_params = [param for param in tf.trainable_variables()
                    if 'generator' in param.name]
        D_params = [param for param in tf.trainable_variables()
                    if 'discriminator' in param.name]
        Denoise_params = [param for param in tf.trainable_variables()
                          if 'denoiser' in param.name]

        tower_gen_grads     = []
        tower_disc_grads    = []
        tower_denoise_grads = []
        for i in xrange(self.n_gpu):
            # Calculate the gradients for the batch of data on this tower
            disc_grads = d_opt.compute_gradients(self.tower_D_loss[i], var_list=D_params)
            gen_grads = g_opt.compute_gradients(self.tower_G_loss[i], var_list=G_params)
            denoise_grads = denoise_opt.compute_gradients(
                self.tower_Denoise_loss[i], var_list=Denoise_params)

            # Keep track of the gradients across all towers
            tower_disc_grads.append(disc_grads)
            tower_gen_grads.append(gen_grads)
            tower_denoise_grads.append(denoise_grads)

        # Average the gradients
        disc_grads = common.average_gradient(tower_disc_grads)
        gen_grads = common.average_gradient(tower_gen_grads)
        denoise_grads = common.average_gradient(tower_denoise_grads)

        # Apply the gradients with our optimizers
        D_train = d_opt.apply_gradients(disc_grads)
        G_train = g_opt.apply_gradients(gen_grads)
        Denoise_train = denoise_opt.apply_gradients(denoise_grads)

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

        # Merge all the summaries and write them out
        self.merged = tf.summary.merge(self.sum_list)
        self.board_writer = tf.summary.FileWriter(self.summary_dir, self.sess.graph)

        n_train = self.data_set.n_train
        total_batch = int(np.floor(n_train / (self.batch_size * self.n_gpu)))

        for epoch in xrange(config.max_epoch + 1):
            d_total_loss = g_total_loss = denoise_total_loss = 0
            with tqdm(total=total_batch, leave=False) as pbar:
                for batch, label in self.data_set.iter(
                        self.batch_size * self.n_gpu, which='train'):
                    if batch.shape[0] != self.batch_size * self.n_gpu:
                        break

                    self.sess.run(
                        [D_train, G_train, Denoise_train],
                        feed_dict={self.input_x: batch,
                                   self.is_training: True})

                    d_losses, g_losses, denoise_losses = self.sess.run(
                        [self.tower_D_loss, self.tower_G_loss, self.tower_Denoise_loss],
                        feed_dict={self.input_x: batch,
                                   self.is_training: True})

                    # Write Tensorboard log
                    summary = self.sess.run(
                        self.merged, feed_dict={self.input_x: batch,
                                                self.is_training: True})
                    self.board_writer.add_summary(summary, counter)

                    d_total_loss += np.array(d_losses).mean()
                    g_total_loss += np.array(g_losses).mean()
                    denoise_total_loss += np.array(denoise_losses).mean()

                    pbar.set_description('Epoch {}'.format(epoch))
                    pbar.update()

                    # monitor generated data
                    if config.monitor:
                        gen_imgs = self.sess.run(
                            self.tower_G[0], feed_dict={self.is_training: False})
                        gen_tiled_imgs = common.img_tile(
                            gen_imgs[0:100], border_color=1.0, stretch=True)
                        gen_tiled_imgs = gen_tiled_imgs[:, :, ::-1]
                        cv2.imshow('generated data', gen_tiled_imgs)
                        cv2.waitKey(1)

            # monitor training data
            if config.monitor:
                training_tiled_imgs = common.img_tile(
                    batch, border_color=1.0, stretch=True)
                training_tiled_imgs = training_tiled_imgs[:, :, ::-1]
                cv2.imshow('training data', training_tiled_imgs)
                cv2.waitKey(1)

            d_total_loss /= total_batch
            g_total_loss /= total_batch
            denoise_total_loss /= total_batch

            # Print display network output
            print('Counter: {}\tD_loss: {:.4f}\tG_loss: {:.4f}\tDenoise: {:.4f}'.format(
                counter, d_total_loss, g_total_loss, denoise_total_loss))
            self.f.flush()

            # Save model
            if counter % 100 == 0:
                self.save_model(self.model_dir, counter)

            # Save images
            if counter % 10 == 0:
                gen_imgs = self.sess.run(
                    self.tower_G[0], feed_dict={self.is_training: False})
                gen_tiled_imgs = common.img_tile(
                    gen_imgs[0:100], border_color=1.0, stretch=True)
                file_name = ''.join([self.img_dir, '/generated_', str(counter).zfill(4), '.jpg'])
                gen_tiled_imgs = gen_tiled_imgs[:, :, ::-1]
                cv2.imwrite(file_name, gen_tiled_imgs * 255.)

            counter += 1

        cv2.destroyAllWindows()
        self.f.close()

    def save_model(self, model_dir, step):
        model_name = 'DCGAN_DFM.model'
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
