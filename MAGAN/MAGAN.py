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

# import tf_utils.inception_score as inception_score


def leaky_relu(x, leak=0.2, name='leaky_relu'):
    return tf.maximum(x, leak * x)


class MAGAN(object):
    def __init__(
            self, sess, log_dir, data, data_dir,
            batch_size=128, z_dim=100, z_dist='uniform'):
        self.sess = sess

        self.img_dir = os.path.join(log_dir, 'imgs')
        self.summary_dir = os.path.join(log_dir, 'summary')
        self.model_dir = os.path.join(log_dir, 'models')
        self.f = open(os.path.join(log_dir, 'training.log'), 'a')
        self.prepare_directory(self.img_dir)
        self.prepare_directory(self.summary_dir)
        self.prepare_directory(self.model_dir)

        self.data = data
        self.data_dir = data_dir
        self.load_data(data_dir)

        self.batch_size = batch_size

        self.z_dim = z_dim
        self.z_dist = z_dist

        gpus = os.environ['CUDA_VISIBLE_DEVICES']
        self.n_gpu = len([int(s) for s in gpus.split(',') if s.isdigit()])

        self.build_model()

    def load_data(self, data_dir):
        if self.data == 'mnist':
            self.data_set = DataSet(self.data, data_dir, normalise='sigmoid')
        elif self.data == 'cifar10':
            self.data_set = DataSet(self.data, data_dir, normalise='tanh')

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
        self.input_reset = tf.placeholder(tf.float32, name='reset_value')
        self.S_real = tf.Variable(0.0, name='real_statistic')
        self.S_fake = tf.Variable(0.0, name='fake_statistic')
        self.margin = tf.Variable(0.0, name='margin')

        self.is_training = tf.placeholder(tf.bool, name='is_training')

        # These are the lists for each tower
        self.tower_z          = []
        self.tower_G          = []
        self.tower_x_encoded  = []
        self.tower_x_reconst  = []
        self.tower_g_encoded  = []
        self.tower_g_reconst  = []
        self.tower_D_loss     = []
        self.tower_D_pre_loss = []
        self.tower_G_loss     = []
        self.tower_sum_step   = []
        self.tower_sum_epoch  = []

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
                    self.build_op(x_portion)

                    # logging tensorboard
                    self.tower_sum_step.append(
                        tf.summary.scalar('0_discriminator_loss', self.tower_D_loss[-1]))
                    self.tower_sum_step.append(
                        tf.summary.scalar('0_generator_loss', self.tower_D_loss[-1]))
                    self.tower_sum_step.append(
                        tf.summary.image('0_input', x_portion))
                    self.tower_sum_step.append(
                        tf.summary.image('0_reconst', self.tower_x_reconst[-1]))
                    self.tower_sum_step.append(
                        tf.summary.image('0_fake', self.tower_G[-1]))

                    self.tower_sum_epoch.append(
                        tf.summary.scalar('1_real_statistic', self.S_real))
                    self.tower_sum_epoch.append(
                        tf.summary.scalar('1_fake_statistic', self.S_fake))
                    self.tower_sum_epoch.append(
                        tf.summary.scalar('1_measure', self.measure))
                    self.tower_sum_epoch.append(
                        tf.summary.scalar('1_margin', self.margin))

                    # Reuse variables for the next tower
                    # tf.get_variable_scope().reuse_variables()

        self.saver = tf.train.Saver()

    def build_model_per_gpu(self, x):
        if self.z_dist == 'uniform':
            z = tf.random_uniform((self.batch_size, 1, 1, self.z_dim), -1., 1.)
        elif self.z_dist == 'normal':
            z = tf.random_normal([self.batch_size, 1, 1, self.z_dim])
        else:
            print ('z_dist error! It must be either uniform or normal.')

        x_encoded, x_reconst = self.discriminator(x)
        G = self.generator(z)
        g_encoded, g_reconst = self.discriminator(G, reuse=True)

        self.tower_z.append(z)
        self.tower_G.append(G)
        self.tower_x_encoded.append(x_encoded)
        self.tower_x_reconst.append(x_reconst)
        self.tower_g_encoded.append(g_encoded)
        self.tower_g_reconst.append(g_reconst)

    def build_op(self, x):
        x_reconst = self.tower_x_reconst[-1]
        G = self.tower_G[-1]
        g_reconst = self.tower_g_reconst[-1]

        loss_real = tf.reduce_sum(tf.square(x_reconst - x), reduction_indices=[1, 2, 3])
        loss_fake = tf.reduce_sum(tf.square(g_reconst - G), reduction_indices=[1, 2, 3])

        # additional op
        self.S_real_update = tf.assign(
            self.S_real, self.S_real + tf.reduce_sum(loss_real))
        self.S_fake_update = tf.assign(
            self.S_fake, self.S_fake + tf.reduce_sum(loss_fake))
        self.S_real_assign = tf.assign(self.S_real, self.input_reset)
        self.S_fake_assign = tf.assign(self.S_fake, self.input_reset)
        self.margin_assign = tf.assign(self.margin, self.input_reset)

        # Loss
        D_pre_loss = tf.reduce_mean(loss_real)
        D_loss = tf.reduce_mean(loss_real + tf.maximum(self.margin - loss_fake, 0))
        G_loss = tf.reduce_mean(loss_fake)

        # convergence measure
        expected_real = self.S_real / self.data_set.n_train
        expected_fake = self.S_fake / self.data_set.n_train
        self.measure = expected_real + tf.abs(expected_real - expected_fake)

        self.tower_D_pre_loss.append(D_pre_loss)
        self.tower_D_loss.append(D_loss)
        self.tower_G_loss.append(G_loss)

    def discriminator(self, x, reuse=None):
        with tf.variable_scope('discriminator', reuse=reuse) as scope:

            batch_norm_params = {
                'is_training': self.is_training, 'updates_collections': None}

            if self.data == 'mnist':
                with slim.arg_scope([slim.fully_connected],
                                    normalizer_fn=slim.batch_norm,
                                    normalizer_params=batch_norm_params,
                                    activation_fn=tf.nn.relu):
                    net = slim.flatten(x)
                    net = slim.fully_connected(net, 874, scope='fc1')
                    encoded_z = net = slim.fully_connected(net, 256, scope='fc2')
                    net = slim.fully_connected(net, 256, scope='fc3')
                    net = slim.fully_connected(
                        net, 784, normalizer_fn=None, activation_fn=tf.nn.sigmoid, scope='fc4')
                    reconst_x = tf.reshape(net, [-1, 28, 28, 1])

            elif self.data == 'cifar10':
                with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                                    normalizer_fn=slim.batch_norm,
                                    normalizer_params=batch_norm_params,
                                    activation_fn=leaky_relu):
                    # Encoder
                    net = slim.conv2d(x, 128, 4, 2, scope='conv1')
                    net = slim.conv2d(net, 256, 4, 2, scope='conv2')
                    encoded_z = net = slim.conv2d(net, 512, 4, 2, scope='conv3')
                    # Decoder
                    net = slim.conv2d_transpose(net, 256, 4, 2, scope='deconv1')
                    net = slim.conv2d_transpose(net, 128, 4, 2, scope='deconv2')
                    reconst_x = slim.conv2d_transpose(
                        net, 3, 4, 2, normalizer_fn=None, activation_fn=tf.nn.tanh, scope='deconv3')

            return encoded_z, reconst_x

    def generator(self, z):
        with tf.variable_scope('generator') as scope:

            batch_norm_params = {
                'is_training': self.is_training, 'updates_collections': None}

            if self.data == 'mnist':
                with slim.arg_scope([slim.conv2d_transpose, slim.fully_connected],
                                    normalizer_fn=slim.batch_norm,
                                    normalizer_params=batch_norm_params,
                                    activation_fn=tf.nn.relu):
                    net = slim.fully_connected(z, 1024, scope='fc1')
                    net = slim.fully_connected(net, 7 * 7 * 128, scope='proj')
                    net = tf.reshape(net, [-1, 7, 7, 128])
                    # net = slim.conv2d_transpose(net, 128, 7, 1, scope='deconv1')
                    # net = slim.conv2d_transpose(net, 64, 4, 2, scope='deconv2')
                    # fake = slim.conv2d_transpose(
                        # net, 1, 4, 2, activation_fn=tf.nn.sigmoid, scope='deconv3')

                    net = slim.conv2d_transpose(net, 128, 5, 2, scope='deconv1')
                    fake = slim.conv2d_transpose(
                        net, 1, 5, 2, normalizer_fn=None, activation_fn=tf.sigmoid, scope='deconv2')

            elif self.data == 'cifar10':
                with slim.arg_scope([slim.conv2d_transpose, slim.fully_connected],
                                    normalizer_fn=slim.batch_norm,
                                    normalizer_params=batch_norm_params,
                                    activation_fn=tf.nn.relu):
                    net = slim.fully_connected(z, 2 * 2 * 1024, scope='proj')
                    net = tf.reshape(net, [-1, 2, 2, 1024])
                    net = slim.conv2d_transpose(net, 512, 5, 2, scope='deconv1')
                    net = slim.conv2d_transpose(net, 256, 5, 2, scope='deconv2')
                    net = slim.conv2d_transpose(net, 128, 5, 2, scope='deconv3')
                    fake = slim.conv2d_transpose(
                        net, 3, 5, 2, normalizer_fn=None, activation_fn=tf.nn.tanh, scope='deconv4')
            return fake

    def train(self, config):

        d_opt = tf.train.AdamOptimizer(config.d_lr, beta1=config.beta1)
        d_pre_opt = tf.train.AdamOptimizer(config.d_lr, beta1=config.beta1)
        g_opt = tf.train.AdamOptimizer(config.g_lr, beta1=config.beta1)

        G_params = [param for param in tf.trainable_variables()
                    if 'generator' in param.name]
        D_params = [param for param in tf.trainable_variables()
                    if 'discriminator' in param.name]

        tower_gen_grads = []
        tower_disc_grads = []
        tower_disc_pre_grads = []
        for i in xrange(self.n_gpu):
            # Calculate the gradients for the batch of data on this tower
            disc_grads = d_opt.compute_gradients(self.tower_D_loss[i], var_list=D_params)
            disc_pre_grads = d_pre_opt.compute_gradients(self.tower_D_pre_loss[i], var_list=D_params)
            gen_grads = g_opt.compute_gradients(self.tower_G_loss[i], var_list=G_params)

            # Keep track of the gradients across all towers
            tower_disc_grads.append(disc_grads)
            tower_disc_pre_grads.append(disc_pre_grads)
            tower_gen_grads.append(gen_grads)

        # Average the gradients
        disc_grads = common.average_gradient(tower_disc_grads)
        disc_pre_grads = common.average_gradient(tower_disc_pre_grads)
        gen_grads = common.average_gradient(tower_gen_grads)

        # Apply the gradients with our optimizers
        D_train = d_opt.apply_gradients(disc_grads)
        D_pre_train = d_pre_opt.apply_gradients(disc_pre_grads)
        G_train = g_opt.apply_gradients(gen_grads)

        # Train starts
        init = tf.global_variables_initializer()
        self.sess.run(init)

        print('[*] Training start.... {}'.format(
            time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())))

        n_train = self.data_set.n_train
        total_batch = int(np.floor(n_train / (self.batch_size * self.n_gpu)))

        counter = 0
        could_load, checkpoint_counter = self.load_model(self.model_dir)
        if could_load:
            counter = checkpoint_counter + 1
            print('[*] Load SUCCESS')
        else:
            print('[!] Load failed...')

            # pre train the discriminator
            for i in range(2):
                self.sess.run(self.S_real_assign, {self.input_reset: 0})
                for batch, label in self.data_set.iter(
                        self.batch_size * self.n_gpu, which='train'):
                    if batch.shape[0] != self.batch_size * self.n_gpu:
                        break

                    self.sess.run(
                        [D_pre_train, self.S_real_update],
                        feed_dict={self.input_x: batch,
                                   self.is_training: True})

            S_real_sum = self.sess.run(self.S_real)
            self.sess.run(self.margin_assign, {self.input_reset: S_real_sum / n_train})
            S_prev_fake = np.inf
            print('[*] Pre-train discriminator finished')

        # Merge all the summaries and write them out
        self.merged_step = tf.summary.merge(self.tower_sum_step)
        self.merged_epoch = tf.summary.merge(self.tower_sum_epoch)
        self.board_writer = tf.summary.FileWriter(self.summary_dir, self.sess.graph)

        step = counter * total_batch
        for epoch in xrange(config.max_epoch + 1):
            d_total_loss = g_total_loss = 0
            self.sess.run(
                [self.S_real_assign, self.S_fake_assign],
                feed_dict={self.input_reset: 0})

            with tqdm(total=total_batch, leave=False) as pbar:
                for batch, label in self.data_set.iter(
                        self.batch_size * self.n_gpu, which='train'):
                    if batch.shape[0] != self.batch_size * self.n_gpu:
                        break

                    self.sess.run(
                        [D_train, G_train],
                        feed_dict={self.input_x: batch,
                                   self.is_training: True})

                    d_losses, g_losses = self.sess.run(
                        [self.tower_D_loss, self.tower_G_loss],
                        feed_dict={self.input_x: batch,
                                   self.is_training: True})

                    self.sess.run(
                        [self.S_fake_update, self.S_real_update],
                        feed_dict={self.input_x: batch,
                                   self.is_training: True})

                    # _, d_losses = self.sess.run(
                        # [D_train, self.tower_D_loss],
                        # feed_dict={self.input_x: batch})

                    # self.sess.run(self.S_real_update, feed_dict={self.input_x: batch})

                    # _, g_losses = self.sess.run(
                        # [G_train, self.tower_G_loss])

                    # self.sess.run(self.S_fake_update)

                    # Write Tensorboard log
                    summary = self.sess.run(
                        self.merged_step, feed_dict={self.input_x: batch,
                                                     self.is_training: True})
                    self.board_writer.add_summary(summary, step)

                    d_total_loss += np.array(d_losses).mean()
                    g_total_loss += np.array(g_losses).mean()

                    pbar.set_description('Epoch {}'.format(epoch))
                    pbar.update()

                    # monitor generated data
                    if config.monitor:
                        gen_imgs = self.sess.run(self.tower_G[0], {self.is_training: False})
                        self.cv2_imshow(gen_imgs, 'generated data')
                    step += 1

            S_real = self.S_real.eval()
            S_fake = self.S_fake.eval()
            margin = self.margin.eval()

            # Write Tensorboard log
            summary = self.sess.run(self.merged_epoch, {self.is_training: True})
            self.board_writer.add_summary(summary, epoch)

            if (S_real / n_train) < margin and S_real < S_fake and S_prev_fake < S_fake:
                self.sess.run(self.margin_assign, {self.input_reset: S_real / n_train})
                new_margin = self.margin.eval()
                print 'margin {} to {}'.format(margin, new_margin)
                
            S_prev_fake = S_fake

            # monitor training data
            if config.monitor:
                self.cv2_imshow(batch, 'training data')

            d_total_loss /= total_batch
            g_total_loss /= total_batch

            # Print display network output
            print('Counter: {}\tD_loss: {:.4f}\tG_loss: {:.4f}\tS_real: {:.4f}\tS_fake: {:.4f}'.format(
                counter, d_total_loss, g_total_loss, S_real, S_fake))
            self.f.flush()

            # Save model
            if counter % 50 == 0:
                self.save_model(self.model_dir, counter)

            # Save images
            if counter % 10 == 0:
                gen_imgs = self.sess.run(self.tower_G[0], {self.is_training: False})
                self.cv2_imsave(gen_imgs, 'generated', counter)

            counter += 1

        cv2.destroyAllWindows()
        self.f.close()

    def test(self):

        self.load_model(self.model_dir)

        test_data, _ = self.data_set.get_data(50000, which='train')
        test_data = test_data[:, :, :, ::-1]
        test_data = common.img_stretch(test_data)
        test_data *= 255.0

        n = np.ceil(50000. / self.batch_size).astype(np.int)
        gen_data = []
        for i in range(n):
            gen_data.append(self.sess.run(self.tower_G[0], {self.is_training: False}))
        gen_data = np.vstack(gen_data)
        gen_data = gen_data[:50000]
        gen_data = gen_data[:, :, :, ::-1]
        gen_data = common.img_stretch(gen_data)
        gen_data *= 255.0

        # np.save('gen_data', gen_data)

        # print test_data.shape, gen_data.shape, test_data.max(), test_data.min(), gen_data.max(), gen_data.min()

        # test_list = []
        # gen_list = []
        # for i in range(50000):
            # test_list.append(test_data[i])
            # gen_list.append(gen_data[i])

        # print 'real inception: '
        # inception_score.get_inception_score(test_list)
        # print 'generated inception '
        # inception_score.get_inception_score(gen_list)

    def save_model(self, model_dir, step):
        model_name = 'MAGAN.model'
        self.saver.save(self.sess, os.path.join(model_dir, model_name), global_step=step)

    def load_model(self, model_dir):
        import re
        print('[*] Reading checkpoints...')

        ckpt = tf.train.get_checkpoint_state(model_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            print model_dir, ckpt_name
            self.saver.restore(self.sess, os.path.join(model_dir, ckpt_name))
            counter = int(next(re.finditer('(\d+)(?!.*d)', ckpt_name)).group(0))
            print('[*] Success to read {}'.format(ckpt_name))
            return True, counter
        else:
            print('[*] Failed to find a checkpoint')
            return False, 0

    def cv2_imshow(self, imgs, title):
        imgs = common.img_tile(imgs[0:100], border_color=1.0, stretch=True)
        imgs = imgs[:, :, ::-1]
        cv2.imshow(title, imgs)
        cv2.waitKey(1)

    def cv2_imsave(self, imgs, title, counter):
        imgs = common.img_tile(imgs[0:100], border_color=1.0, stretch=True)
        file_name = ''.join([self.img_dir, '/', title, '_', str(counter).zfill(4), '.jpg'])
        imgs = imgs[:, :, ::-1]
        cv2.imwrite(file_name, imgs * 255.)
