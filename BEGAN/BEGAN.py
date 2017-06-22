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


class BEGAN(object):
    def __init__(
            self, sess, log_dir, data, data_dir, gamma, lambda_,
            batch_size=128, z_dim=100, z_dist='uniform'):
        self.sess = sess

        print('[*] Training start.... {}'.format(
            time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())))
        self.img_dir = os.path.join(log_dir, 'imgs')
        self.summary_dir = os.path.join(log_dir, 'summary')
        self.train_dir = os.path.join(self.summary_dir, 'train')
        self.test_dir = os.path.join(self.summary_dir, 'test')
        self.model_dir = os.path.join(log_dir, 'models')
        self.f = open(os.path.join(log_dir, 'training.log'), 'a')
        self.prepare_directory(self.img_dir)
        self.prepare_directory(self.summary_dir)
        self.prepare_directory(self.train_dir)
        self.prepare_directory(self.test_dir)
        self.prepare_directory(self.model_dir)

        self.is_training = tf.placeholder(tf.bool, [], name='is_training')

        self.data = data
        self.data_dir = data_dir
        self.load_data(data_dir)

        self.batch_size = batch_size

        self.z_dim = z_dim
        self.z_dist = z_dist
        self.gamma = gamma
        self.lambda_ = lambda_
        self.hidden_num = 128
        self.repeat_num = 3

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
        self.k = tf.Variable(0.0, 'k')

        # These are the lists for each tower
        self.tower_z           = []
        self.tower_G           = []
        self.tower_encoded_z_p = []
        self.tower_reconst_x_p = []
        self.tower_encoded_z_n = []
        self.tower_reconst_x_n = []
        self.tower_D_loss      = []
        self.tower_G_loss      = []
        self.tower_measure     = []

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
                    D_loss, G_loss = self.get_loss(x_portion)

                    measure = self.convergence_measuer(x_portion)

                    # logging tensorboard
                    tf.summary.scalar('tower_{}/discriminator_loss'.format(i), D_loss)
                    tf.summary.scalar('tower_{}/generator_loss'.format(i), G_loss)
                    tf.summary.scalar('tower_{}/convergence'.format(i), measure)
                    tf.summary.scalar('lambda_k', self.k)
                    tf.summary.image('generated', self.tower_G[0])

                    # Reuse variables for the next tower
                    # tf.get_variable_scope().reuse_variables()

                    # Keep track of models across all towers
                    self.tower_D_loss.append(D_loss)
                    self.tower_G_loss.append(G_loss)
                    self.tower_measure.append(measure)

        self.saver = tf.train.Saver()

    def build_model_per_gpu(self, x):
        if self.z_dist == 'uniform':
            z = tf.random_uniform((self.batch_size, 1, 1, self.z_dim), -1., 1.)
        elif self.z_dist == 'normal':
            z = tf.random_normal([self.batch_size, 1, 1, self.z_dim])
        else:
            print ('z_dist error! It must be either uniform or normal.')

        encoded_z_p, reconst_x_p = self.discriminator(x)
        G = self.generator(z)
        encoded_z_n, reconst_x_n = self.discriminator(G, reuse=True)

        self.tower_z.append(z)
        self.tower_G.append(G)
        self.tower_encoded_z_p.append(encoded_z_p)
        self.tower_reconst_x_p.append(reconst_x_p)
        self.tower_encoded_z_n.append(encoded_z_n)
        self.tower_reconst_x_n.append(reconst_x_n)

    def get_loss(self, x):
        reconst_x_p = self.tower_reconst_x_p[-1]
        G = self.tower_G[-1]
        reconst_x_n = self.tower_reconst_x_n[-1]

        loss_real = tf.reduce_mean(tf.abs(x - reconst_x_p))
        loss_fake = tf.reduce_mean(tf.abs(G - reconst_x_n))

        D_loss = loss_real - self.k * loss_fake
        G_loss = loss_fake

        self.update_k = tf.assign(
            self.k,
            tf.clip_by_value(self.k + self.lambda_ * (self.gamma * loss_real - loss_fake), 0, 1))

        return D_loss, G_loss

    def convergence_measuer(self, x):
        reconst_x_p = self.tower_reconst_x_p[-1]
        G = self.tower_G[-1]
        reconst_x_n = self.tower_reconst_x_n[-1]

        loss_real = tf.reduce_mean(tf.abs(x - reconst_x_p))
        loss_fake = tf.reduce_mean(tf.abs(G - reconst_x_n))

        measure = loss_real + tf.abs(self.gamma * loss_real - loss_fake)

        return measure

    def discriminator(self, x, reuse=None):
        with tf.variable_scope('discriminator', reuse=reuse) as scope:

            # Encoder
            net = slim.conv2d(x, self.hidden_num, 3, 1, activation_fn=tf.nn.elu)

            channel_num = self.hidden_num
            for idx in range(self.repeat_num):
                net = slim.conv2d(net, channel_num, 3, 1, activation_fn=tf.nn.elu)
                net = slim.conv2d(net, channel_num, 3, 1, activation_fn=tf.nn.elu)
                if idx < self.repeat_num - 1:
                    net = slim.conv2d(net, channel_num, 3, 2, activation_fn=tf.nn.elu)
                channel_num = channel_num * 2
            net = slim.flatten(net)
            encoded_z = slim.fully_connected(net, self.z_dim, activation_fn=None)

            # Decoder
            if self.data == 'mnist':
                dim = 7
            elif self.data == 'cifar10':
                dim = 8

            num_output = int(np.prod([dim, dim, self.hidden_num]))
            net = slim.fully_connected(encoded_z, num_output, activation_fn=None)
            net = tf.reshape(net, [-1, dim, dim, self.hidden_num])

            for idx in range(self.repeat_num):
                net = slim.conv2d(net, self.hidden_num, 3, 1, activation_fn=tf.nn.elu)
                net = slim.conv2d(net, self.hidden_num, 3, 1, activation_fn=tf.nn.elu)
                if idx < self.repeat_num - 1:
                    _, h, w, _ = net.get_shape().as_list()
                    net = tf.image.resize_nearest_neighbor(net, (h * 2, w * 2))

            if self.data == 'mnist':
                reconst_x = slim.conv2d(net, 1, 3, 1, activation_fn=None)
            elif self.data == 'cifar10':
                reconst_x = slim.conv2d(net, 3, 3, 1, activation_fn=None)

        return encoded_z, reconst_x

    def generator(self, z):
        with tf.variable_scope('generator') as scope:
            if self.data == 'mnist':
                dim = 7
            elif self.data == 'cifar10':
                dim = 8

            num_output = int(np.prod([dim, dim, self.hidden_num]))
            net = slim.fully_connected(z, num_output, activation_fn=None)
            net = tf.reshape(net, [-1, dim, dim, self.hidden_num])

            for idx in range(self.repeat_num):
                net = slim.conv2d(net, self.hidden_num, 3, 1, activation_fn=tf.nn.elu)
                net = slim.conv2d(net, self.hidden_num, 3, 1, activation_fn=tf.nn.elu)
                if idx < self.repeat_num - 1:
                    _, h, w, _ = net.get_shape().as_list()
                    net = tf.image.resize_nearest_neighbor(net, (h * 2, w * 2))

            if self.data == 'mnist':
                net = slim.conv2d(net, 1, 3, 1, activation_fn=None)
            elif self.data == 'cifar10':
                net = slim.conv2d(net, 3, 3, 1, activation_fn=None)

        return net

    def train(self, config):
        d_opt = tf.train.AdamOptimizer(config.d_lr, beta1=config.beta1, name='D_opt')
        g_opt = tf.train.AdamOptimizer(config.g_lr, beta1=config.beta1, name='G_opt')

        G_params = [param for param in tf.trainable_variables()
                    if 'generator' in param.name]
        D_params = [param for param in tf.trainable_variables()
                    if 'discriminator' in param.name]

        D_train = d_opt.minimize(self.tower_D_loss[0], var_list=D_params)
        G_train = g_opt.minimize(self.tower_G_loss[0], var_list=G_params)

        # with tf.variable_scope('compute_gradients'):
            # tower_gen_grads = []
            # tower_disc_grads = []
            # for i in xrange(self.n_gpu):
                # # Calculate the gradients for the batch of data on this tower
                # disc_grads = d_opt.compute_gradients(self.tower_D_loss[i], var_list=D_params)
                # gen_grads = g_opt.compute_gradients(self.tower_G_loss[i], var_list=G_params)

                # # Keep track of the gradients across all towers
                # tower_disc_grads.append(disc_grads)
                # tower_gen_grads.append(gen_grads)

            # # Average the gradients
            # disc_grads = common.average_gradient(tower_disc_grads)
            # gen_grads = common.average_gradient(tower_gen_grads)

        # # Apply the gradients with our optimizers
        # D_train = d_opt.apply_gradients(disc_grads, name='D_train')
        # G_train = g_opt.apply_gradients(gen_grads, name='G_train')

        # Train starts
        init = tf.global_variables_initializer()
        self.sess.run(init)

        counter = 0
        could_load, checkpoint_counter = self.load_model(self.model_dir)
        if could_load:
            counter = checkpoint_counter + 1
            print('[*] Load SUCCESS')
        else:
            print('[!] Load failed...')

        # Merge all the summaries and write them out
        self.merged = tf.summary.merge_all()
        self.train_writer = tf.summary.FileWriter(self.train_dir, self.sess.graph)
        self.test_writer = tf.summary.FileWriter(self.test_dir)

        n_train = self.data_set.n_train
        total_batch = int(np.floor(n_train / (self.batch_size * self.n_gpu)))

        step = counter * total_batch

        for epoch in xrange(config.max_epoch + 1):
            d_total_loss = g_total_loss = 0
            with tqdm(total=total_batch, leave=False) as pbar:
                for batch, label in self.data_set.iter(
                        self.batch_size * self.n_gpu, which='train'):
                    if batch.shape[0] != self.batch_size * self.n_gpu:
                        break

                    self.sess.run(
                        [G_train, D_train],
                        feed_dict={self.input_x: batch,
                                   self.is_training: True})

                    self.sess.run(
                        self.update_k,
                        feed_dict={self.input_x: batch,
                                   self.is_training: True})

                    d_losses, g_losses = self.sess.run(
                        [self.tower_D_loss, self.tower_G_loss],
                        feed_dict={self.input_x: batch,
                                   self.is_training: True})

                    # Write Tensorboard log
                    summary = self.sess.run(
                        self.merged,
                        feed_dict={self.input_x: batch,
                                   self.is_training: False})
                    self.train_writer.add_summary(summary, step)

                    if step % 10 == 0:
                        test_data, _ = self.data_set.get_data(config.batch_size, which='test')
                        summary = self.sess.run(
                            self.merged,
                            feed_dict={self.input_x: test_data,
                                       self.is_training: False})
                        self.test_writer.add_summary(summary, step)

                    d_total_loss += np.array(d_losses).mean()
                    g_total_loss += np.array(g_losses).mean()

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
                    step += 1

            # monitor training data
            if config.monitor:
                training_tiled_imgs = common.img_tile(
                    batch, border_color=1.0, stretch=True)
                training_tiled_imgs = training_tiled_imgs[:, :, ::-1]
                cv2.imshow('training data', training_tiled_imgs)
                cv2.waitKey(1)

            conv_measure, kk = self.sess.run(
                [self.tower_measure[0], self.k],
                feed_dict={self.input_x: test_data,
                           self.is_training: False})

            d_total_loss /= total_batch
            g_total_loss /= total_batch

            # Print display network output
            print('Counter: {}\tD_loss: {:.4f}\tG_loss: {:.4f}\tconv: {}\tk: {}'.format(
                counter, d_total_loss, g_total_loss, conv_measure, kk))
            self.f.flush()

            # Save model
            if counter % 50 == 0:
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
        model_name = 'BEGAN.model'
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
