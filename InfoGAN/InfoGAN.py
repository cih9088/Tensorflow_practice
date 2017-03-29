import os
import time
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import matplotlib.pyplot as plt

import tf_utils.common as common
from tf_utils.data.dataset import Dataset

from tqdm import tqdm


class InfoGAN(object):
    def __init__(self, sess, log_dir, data, data_dir,
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

        self.data = data
        self.load_data(data_dir)

        self.batch_size = batch_size

        self.z_dim = z_dim
        self.z_dist = z_dist

        gpus = os.environ['CUDA_VISIBLE_DEVICES']
        self.n_gpu = len([int(s) for s in gpus.split(',') if s.isdigit()])

        self.build_model()

    def load_data(self, data_dir):
        self.data_set = Dataset(self.data, data_dir, normalise=True)

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
        self.tower_inputs        = []
        self.tower_c             = []
        self.tower_G             = []
        self.tower_G_categorical = []
        self.tower_D_p           = []
        self.tower_D_n           = []
        self.tower_Q_p           = []
        self.tower_Q_n           = []
        self.tower_D_loss        = []
        self.tower_Q_loss        = []
        self.tower_G_loss        = []
        self.sum_list            = []

        # Define the network for each GPU
        for i in xrange(self.n_gpu):
            with tf.device('/gpu:%d' % i):
                with tf.name_scope('Tower_%d' % (i)) as scope:
                    # Grab this portion of the input
                    inputs_per_gpu = self.inputs[i * self.batch_size:(i + 1) * self.batch_size, :]

                    # Construct the model
                    D_p, D_n, Q_p, Q_n, G, G_categorical, c = \
                        self.build_model_per_gpu(inputs_per_gpu)

                    # Calculate the loss for this tower
                    D_loss = self.get_D_loss(D_p, D_n)
                    G_loss, Q_loss = self.get_G_Q_loss(D_n, Q_n, c)

                    # logging tensorboard
                    self.sum_list.append(tf.summary.scalar('tower_{}/discriminator_loss'.format(i), D_loss))
                    self.sum_list.append(tf.summary.scalar('tower_{}/generator_loss'.format(i), G_loss))
                    self.sum_list.append(tf.summary.scalar('tower_{}/recognitor_loss'.format(i), Q_loss))

                    # Reuse variables for the next tower
                    # tf.get_variable_scope().reuse_variables()

                    # Keep track of models across all towers
                    self.tower_inputs.append(inputs_per_gpu)
                    self.tower_c.append(c)
                    self.tower_G.append(G)
                    self.tower_G_categorical.append(G_categorical)
                    self.tower_D_p.append(D_p)
                    self.tower_D_n.append(D_n)
                    self.tower_Q_p.append(Q_p)
                    self.tower_Q_n.append(Q_n)
                    self.tower_D_loss.append(D_loss)
                    self.tower_G_loss.append(G_loss)
                    self.tower_Q_loss.append(Q_loss)

        self.saver = tf.train.Saver()

    def build_model_per_gpu(self, inputs):
        if self.z_dist == 'uniform':
            z = tf.random_uniform((self.batch_size, 1, 1, self.z_dim), -1., 1.)
        elif self.z_dist == 'normal':
            z = tf.random_normal([self.batch_size, 1, 1, self.z_dim])
        else:
            print ('z_dist error! It must be either uniform or normal.')

        if self.data == 'mnist':
            p = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
            dist = tf.contrib.distributions.Categorical(p=p)
            samples = dist.sample(self.batch_size)
            c1 = tf.expand_dims(tf.expand_dims(tf.one_hot(samples, 10), 1), 1)
            c2 = tf.random_uniform((self.batch_size, 1, 1, 1), -1.0, 1.0)
            c3 = tf.random_uniform((self.batch_size, 1, 1, 1), -1.0, 1.0)
            c = tf.concat([c1, c2, c3], 3)

            positive_shared = self.shared_structure(inputs)
            D_p = self.discriminator(positive_shared)
            Q_p = self.recognitor(positive_shared)

            G = self.generator(z, c)

            negative_shared = self.shared_structure(G, reuse=True)
            D_n = self.discriminator(negative_shared, reuse=True)
            Q_n = self.recognitor(negative_shared, reuse=True)

        G_categorical = []
        for i in range(10):
            categorical_p = np.zeros([10])
            categorical_p[i] = 1.0
            categorical_dist = tf.contrib.distributions.Categorical(p=categorical_p)
            categorical_samples = categorical_dist.sample(self.batch_size)
            categorical_c1 = tf.expand_dims(tf.expand_dims(tf.one_hot(categorical_samples, 10), 1), 1)
            categorical_c = tf.concat([categorical_c1, c2, c3], 3)

            G_categorical.append(self.generator(z, categorical_c))

        return D_p, D_n, Q_p, Q_n, G, G_categorical, c

    def get_D_loss(self, D_p, D_n):
        return tf.reduce_mean(tf.nn.softplus(-D_p)) \
            + tf.reduce_mean(tf.nn.softplus(-D_n) + D_n)

    def get_G_Q_loss(self, D_n, Q_n, c, cont_lamb=1, disc_lamb=1, epsilon=1e-8):
        c = tf.squeeze(c)
        if self.data == 'mnist':
            in_c1 = tf.slice(c, [0, 0], [-1, 10])
            in_c2 = tf.slice(c, [0, 9], [-1, 1])
            in_c3 = tf.slice(c, [0, 10], [-1, 1])
            c1, c2_mean, c2_log_sigma_sq, c3_mean, c3_log_sigma_sq = Q_n

            c1 = tf.clip_by_value(c1, epsilon, 1 - epsilon)
            categorical_loss = -tf.reduce_sum(in_c1 * tf.log(c1))

            c2_err = (in_c2 - c2_mean) / (tf.sqrt(tf.exp(c2_log_sigma_sq)) + epsilon)
            c2_loss = (0.5 * c2_log_sigma_sq + 0.5 * tf.square(c2_err))
            c3_err = (in_c3 - c3_mean) / (tf.sqrt(tf.exp(c3_log_sigma_sq)) + epsilon)
            c3_loss = (0.5 * c3_log_sigma_sq + 0.5 * tf.square(c3_err))
            continuous_loss = tf.reduce_mean(c2_loss + c3_loss)

        Q_loss = disc_lamb * categorical_loss + cont_lamb * continuous_loss
        G_loss = tf.reduce_mean(tf.nn.softplus(-D_n)) + Q_loss

        return G_loss, Q_loss

    def get_loss(self, D_p_logits, D_n_logits):
        D_loss = tf.reduce_mean(tf.nn.softplus(-D_p_logits))\
            + tf.reduce_mean(tf.nn.softplus(-D_n_logits) + D_n_logits)
        G_loss = tf.reduce_mean(tf.nn.softplus(-D_n_logits))

        return D_loss, G_loss

    def shared_structure(self, inputs, reuse=False):
        with tf.variable_scope('shared') as scope:
            if reuse:
                scope.reuse_variables()

            with slim.arg_scope([slim.conv2d],
                                padding='SAME',
                                normalizer_fn=slim.batch_norm,
                                normalizer_params={'scale': True},
                                activation_fn=common.leaky_relu):
                net = slim.conv2d(inputs, 32, [5, 5], 2, scope='conv1')
                net = slim.conv2d(net, 64, [5, 5], 2, scope='conv2')
                net = slim.conv2d(net, 128, [5, 5], 1, padding='VALID', scope='conv3')
                net = slim.dropout(net, 0.9, scope='dropout3')
                net = slim.flatten(net)
        return net

    def discriminator(self, inputs, reuse=False):
        with tf.variable_scope('discriminator') as scope:
            if reuse:
                scope.reuse_variables()

            logits = slim.fully_connected(inputs, 1, activation_fn=None, scope='fc1')

        return logits

    def recognitor(self, inputs, reuse=False):
        with tf.variable_scope('recognitor') as scope:
            if reuse:
                scope.reuse_variables()

            if self.data == 'mnist':
                output = slim.fully_connected(inputs, 128, activation_fn=None, scope='fc1')

                c1 = slim.fully_connected(output, 10, activation_fn=tf.nn.softmax, scope='c1')
                c2_logits = slim.fully_connected(output, 2, activation_fn=None, scope='c2')
                c3_logits = slim.fully_connected(output, 2, activation_fn=None, scope='c3')
                c2_mean, c2_log_sigma_sq = tf.split(c2_logits, 2, 1)
                c3_mean, c3_log_sigma_sq = tf.split(c3_logits, 2, 1)

        return [c1, c2_mean, c2_log_sigma_sq, c3_mean, c3_log_sigma_sq]

    def generator(self, z_input, c_input):
        inputs = tf.concat([z_input, c_input], 3)
        with tf.variable_scope('generator') as scope:
            with slim.arg_scope([slim.conv2d_transpose],
                                padding='SAME',
                                normalizer_fn=slim.batch_norm,
                                normalizer_params={'scale': True},
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

    def train(self, config):
        if config.monitor:
            import cv2

        d_opt = tf.train.AdamOptimizer(config.d_lr, beta1=config.beta1)
        g_opt = tf.train.AdamOptimizer(config.g_lr, beta1=config.beta1)
        q_opt = tf.train.AdamOptimizer(config.q_lr, beta1=config.beta1)

        # Merge all the summaries and write them out
        self.merged = tf.summary.merge(self.sum_list)
        self.board_writer = tf.summary.FileWriter(self.summary_dir, self.sess.graph)

        G_params = [param for param in tf.trainable_variables() if 'generator' in param.name]
        D_params = [param for param in tf.trainable_variables()
                if 'discriminator' in param.name or 'shared' in param.name]
        Q_params = [param for param in tf.trainable_variables()
                if 'recognitor' in param.name or 'shared' in param.name]

        tower_gen_grads  = []
        tower_disc_grads = []
        tower_reco_grads = []
        for i in xrange(self.n_gpu):
            # Calculate the gradients for the batch of data on this tower
            disc_grads = d_opt.compute_gradients(self.tower_D_loss[i], var_list=D_params)
            reco_grads = q_opt.compute_gradients(self.tower_Q_loss[i], var_list=Q_params)
            gen_grads = g_opt.compute_gradients(self.tower_G_loss[i], var_list=G_params)

            # Keep track of the gradients across all towers
            tower_disc_grads.append(disc_grads)
            tower_reco_grads.append(reco_grads)
            tower_gen_grads.append(gen_grads)

        # Average the gradients
        disc_grads = common.average_gradient(tower_disc_grads)
        reco_grads = common.average_gradient(tower_reco_grads)
        gen_grads  = common.average_gradient(tower_gen_grads)

        # Apply the gradients with our optimizers
        D_train = d_opt.apply_gradients(disc_grads)
        Q_train = q_opt.apply_gradients(reco_grads)
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

        for epoch in xrange(config.max_epoch):
            d_total_loss = g_total_loss = q_total_loss = 0
            with tqdm(total=total_batch) as pbar:
                for batch, label in self.data_set._iter(self.batch_size * self.n_gpu, 'train'):
                    if batch.shape[0] != self.batch_size * self.n_gpu:
                        break

                    _, d_losses = self.sess.run([D_train, self.tower_D_loss],
                            feed_dict={self.inputs: batch})
                    _, q_losses = self.sess.run([Q_train, self.tower_Q_loss],
                            feed_dict={self.inputs: batch})
                    _, g_losses = self.sess.run([G_train, self.tower_G_loss],
                            feed_dict={self.inputs: batch})

                    # Write Tensorboard log
                    summary = self.sess.run(self.merged, feed_dict={self.inputs: batch})
                    self.board_writer.add_summary(summary, counter)

                    d_total_loss += np.array(d_losses).mean()
                    q_total_loss += np.array(q_losses).mean()
                    g_total_loss += np.array(g_losses).mean()

                    pbar.set_description('Epoch {}'.format(epoch))
                    pbar.update()

                    # monitor generated data
                    if config.monitor:
                        gen_imgs = self.sess.run(self.tower_G[0])
                        gen_tiled_imgs = common.img_tile(gen_imgs[0:100], border_color=1.0, stretch=True)
                        cv2.imshow('generated data', gen_tiled_imgs)
                        cv2.waitKey(1)

            categorical_imgs = self.sess.run(self.tower_G_categorical[0])
            categorical_tiled_imgs = []
            for i in range(10):
                categorical_imgs[i] = categorical_imgs[i][:, :, :, ::-1]
                categorical_tiled_imgs.append(
                    common.img_tile(categorical_imgs[i], stretch=True))
            row1 = np.concatenate((categorical_tiled_imgs[0], categorical_tiled_imgs[1], categorical_tiled_imgs[2], categorical_tiled_imgs[3], categorical_tiled_imgs[4]), axis=1)
            row2 = np.concatenate((categorical_tiled_imgs[5], categorical_tiled_imgs[6], categorical_tiled_imgs[7], categorical_tiled_imgs[8], categorical_tiled_imgs[9]), axis=1)
            merged_tiled_imgs = np.concatenate((row1, row2), axis=0)

            cv2.imwrite(''.join([self.img_dir, '/categorical_', str(counter).zfill(4), '.jpg']), merged_tiled_imgs * 255.)
            if config.monitor:
                # monitor training data
                training_tiled_imgs = common.img_tile(batch[0:100], border_color=1.0, stretch=True)
                cv2.imshow('training data', training_tiled_imgs)

                # Monitor categorically generated data
                cv2.imshow('categorical generated data', merged_tiled_imgs)

            d_total_loss /= total_batch
            q_total_loss /= total_batch
            g_total_loss /= total_batch

            # Print display network output
            print('Counter: {}\t Epoch: {}\t D_loss: {:.4f}\t G_loss: {:.4f}'.format(counter, epoch, d_total_loss, g_total_loss))
            self.f.flush()

            # Save model
            if counter % 100 == 0:
                self.save_model(self.model_dir, counter)

            # Save images
            if counter % 10 == 0:
                cv2.imwrite(''.join([self.img_dir, '/generated_', str(counter).zfill(4), '.jpg']), gen_tiled_imgs * 255.)

            counter += 1

        cv2.destroyAllWindows()
        self.f.close()

    def save_model(self, model_dir, step):
        model_name = 'GAN.model'

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
