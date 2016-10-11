'''TensorFlow implementation of http://arxiv.org/pdf/1502.04623v2.pdf

DISCLAIMER
Work in progress. This code requires massive refactoring.
'''

from __future__ import absolute_import, division, print_function

import math
import os

import numpy as np
import prettytensor as pt
import tensorflow as tf

import tf_utils.data.mnist_data as mnist_data
import tf_utils.data.manage as manage
import tf_utils.common as common

import tqdm

flags = tf.flags
logging = tf.logging
flags.DEFINE_integer('batch_size', 8, 'size of batches to use(per GPU)')
flags.DEFINE_string('log_dir', 'draw-mnist_logs/', 'saved image directory')
flags.DEFINE_integer('max_epoch', 1000, 'a number of epochs to run')
flags.DEFINE_string('data_dir', '/home/mlg/ihcho/data', 'data directory')
flags.DEFINE_float('learning_rate', 1e-3, 'Learning rate')
flags.DEFINE_integer("rnn_size", 256, "size of RNN unit")
flags.DEFINE_integer("rnn_len", 64, "time T")
flags.DEFINE_integer("hidden_size", 100, "size of the hidden VAE unit")
flags.DEFINE_integer("r_N", 2, "read attention")
flags.DEFINE_integer("w_N", 5, "write attention")
flags.DEFINE_boolean("test", True, "Test it or not")
FLAGS = flags.FLAGS

A = 28 # width
B = 28 # height
C = 1 # channel

def filterbank_matrices(g_x, g_y, sigma_sq, delta, N, A, B, epsilon=1e-8):
    
    # (N) to (1, N)
    i_s = tf.reshape(tf.cast(tf.range(N), tf.float32), [1, -1])

    # Eq 19
    mu_x = g_x + (i_s - N / 2 - 0.5) * delta

    # Eq 20
    mu_y = g_y + (i_s - N / 2 - 0.5) * delta

    a = tf.cast(tf.range(A), tf.float32)
    b = tf.cast(tf.range(B), tf.float32)

    # reshape for broadcasting
    mu_x = tf.reshape(mu_x, [-1, N, 1])
    mu_y = tf.reshape(mu_y, [-1, N, 1])
    a = tf.reshape(a, [1, 1, -1])
    b = tf.reshape(b, [1, 1, -1])
    sigma_sq = tf.reshape(sigma_sq, [-1, 1, 1])

    # Eq 25
    F_x = tf.exp(-tf.square(a - mu_x) / (2 * sigma_sq))
    F_x = F_x / tf.maximum(tf.reduce_sum(F_x, 2, True), epsilon)

    # Eq 26
    F_y = tf.exp(-tf.square(b - mu_y) / (2 * sigma_sq))
    F_y = F_y / tf.maximum(tf.reduce_sum(F_y, 2, True), epsilon)

    return F_x, F_y

def apply_filters(image, F_x, F_y, gamma, N, A, B, read=True):
    '''
    Args:
        N: size of corp
        A: width
        B: height
        image: [batch, hight, width, channel]
        F_x: [batch, N, A]
        F_y: [batch, N, B]
        gamma: [batch, 1]
    '''
    if read == True:
        F_y = tf.reshape(F_y, [-1, N, B, 1, 1])
        F_y = tf.tile(F_y, [1, 1, 1, A, C])
        image = tf.reshape(image, [-1, 1, B, A, C])
        image = tf.tile(image, [1, N, 1, 1, 1])
        image = tf.reduce_sum((F_y * image), 2)

        image = tf.reshape(image, [-1, N, A, C, 1])
        image = tf.tile(image, [1, 1, 1, 1, N])
        F_x = tf.transpose(F_x, [0, 2, 1])
        F_x = tf.reshape(F_x, [-1, 1, A, 1, N])
        F_x = tf.tile(F_x, [1, N, 1, C, 1])
        image = tf.reduce_sum((image * F_x), 2)

        # image: [batch, N(height), N(width), C(channel)]
        image = tf.transpose(image, [0, 1, 3, 2])
        return image * tf.reshape(gamma, [-1, 1, 1, 1])
    else:
        F_y = tf.transpose(F_y, [0, 2, 1])
        F_y = tf.reshape(F_y, [-1, B, N, 1, 1])
        F_y = tf.tile(F_y, [1, 1, 1, N, C])
        image = tf.reshape(image, [-1, 1, N, N, C])
        image = tf.tile(image, [1, B, 1, 1, 1])
        image = tf.reduce_sum((F_y * image), 2)
        
        image = tf.reshape(image, [-1, B, N, C, 1])
        image = tf.tile(image, [1, 1, 1, 1, A])
        F_x = tf.reshape(F_x, [-1, 1, N, 1, A])
        F_x = tf.tile(F_x, [1, B, 1, C, 1])
        image = tf.reduce_sum((image * F_x), 2)

        # image: [batch, B(height), A(width), C(channel)]
        image = tf.transpose(image, [0, 1, 3, 2])
        return image * tf.reshape( 1.0 / gamma, [-1, 1, 1, 1])

def transform_params(input_tensor, N, A, B):
    
    g_x, g_y, log_sigma_sq, log_delta, log_gamma = tf.split(1, 5, input_tensor)

    g_x = (A + 1) / 2 * (g_x + 1)
    g_y = (B + 1) / 2 * (g_y + 1)
    sigma_sq = tf.exp(log_sigma_sq)
    delta = (max(A, B) - 1) / (N - 1) * tf.exp(log_delta)
    gamma = tf.exp(log_gamma)

    return g_x, g_y, sigma_sq, delta, gamma

def get_vae_loss(mean, stddev, epsilon=1e-8):
    return tf.reduce_sum(-0.5 * (1.0 + 2.0 * tf.log(stddev + epsilon) - tf.square(mean) - tf.square(stddev)))

def get_reconst_loss(output, target, epsilon=1e-8):
    return -tf.reduce_sum(target * tf.log(output + epsilon) + (1.0 - target) * tf.log(1.0 - output + epsilon))


if __name__ == '__main__':

    # Create log directory
    common.delete_and_create_directory(FLAGS.log_dir)
    img_dir = os.path.join(FLAGS.log_dir, 'imgs')
    common.delete_and_create_directory(img_dir)
    model_dir = os.path.join(FLAGS.log_dir, 'models')
    common.delete_and_create_directory(model_dir)

    # Create tensorboard summary data dir
    summary_dir = os.path.join(FLAGS.log_dir, 'board')
    common.delete_and_create_directory(summary_dir)

    # Create logfile
    f = open(os.path.join(FLAGS.log_dir, 'training.log'), 'w')

    # print hyperparameters
    f.write('batch_size:%d, learning_rate:%f, rnn_size:%d, rnn_len:%d, hidden_size:%d, read_n:%d, write_n:%d\n' %(FLAGS.batch_size, FLAGS.learning_rate, FLAGS.rnn_size, FLAGS.rnn_len, FLAGS.hidden_size, FLAGS.r_N, FLAGS.w_N))
    print('batch_size:%d, learning_rate:%f, rnn_size:%d, rnn_len:%d, hidden_size:%d, read_n:%d, write_n:%d\n' %(FLAGS.batch_size, FLAGS.learning_rate, FLAGS.rnn_size, FLAGS.rnn_len, FLAGS.hidden_size, FLAGS.r_N, FLAGS.w_N))


    # load mnist
    trainx, trainy = mnist_data.load(FLAGS.data_dir, subset='train')
    testx, testy = mnist_data.load(FLAGS.data_dir, subset='test')

    # state of each rnn
    encoder_state = (pt.wrap(tf.zeros([FLAGS.batch_size, FLAGS.rnn_size], tf.float32)),)
    decoder_state = (pt.wrap(tf.zeros([FLAGS.batch_size, FLAGS.rnn_size], tf.float32)),)
    sampled_state = (pt.wrap(tf.zeros([FLAGS.batch_size, FLAGS.rnn_size], tf.float32)),)

    input_image = tf.placeholder(tf.float32, [FLAGS.batch_size, B, A, C])
    output_image = tf.zeros([FLAGS.batch_size, B, A, C], tf.float32)
    sampled_image = tf.zeros([FLAGS.batch_size, B, A, C], tf.float32)

    with tf.variable_scope('model'):
        # Eq 5
        encoder_template = (pt.template('input').
                            gru_cell(state=pt.UnboundVariable('state'), num_units=FLAGS.rnn_size))

        # Eq 7
        decoder_template = (pt.template('input').
                            gru_cell(state=pt.UnboundVariable('state'), num_units=FLAGS.rnn_size))

        # Eq 1, 2
        encoder_hidden_params_template = (pt.template('input').
                                        fully_connected(FLAGS.hidden_size * 2, activation_fn=None))

        # Eq 21
        decoder_read_params_template = (pt.template('input').
                                        fully_connected(5, activation_fn=None))
        decoder_write_params_template = (pt.template('input').
                                        fully_connected(5, activation_fn=None))
        
        # Eq 28
        decoder_write_template = (pt.template('input').
                                  fully_connected(FLAGS.w_N * FLAGS.w_N * C, activation_fn=None))

        vae_loss_sum = 0
        for _ in range(FLAGS.rnn_len):
            epsilon = tf.random_normal([FLAGS.batch_size, FLAGS.hidden_size])

            read_params = decoder_read_params_template.construct(input=decoder_state[0].tensor)
            g_x, g_y, sigma_sq, delta, gamma = transform_params(read_params, FLAGS.r_N, A, B)
            F_x, F_y = filterbank_matrices(g_x, g_y, sigma_sq, delta, FLAGS.r_N, A, B)
            image_glimpse = apply_filters(input_image, F_x, F_y, gamma, FLAGS.r_N, A, B, True)

            image_hat = input_image - tf.nn.sigmoid(output_image)
            image_hat_glimpse = apply_filters(image_hat, F_x, F_y, gamma, FLAGS.r_N, A, B, True)

            encoder_input_tensor = pt.wrap(tf.concat(1, [tf.reshape(image_glimpse, [FLAGS.batch_size, -1]),
                    tf.reshape(image_hat_glimpse, [FLAGS.batch_size, -1]), decoder_state[0].tensor]))

            encoded_tensor, encoder_state = \
                    encoder_template.construct(input=encoder_input_tensor, state=encoder_state[0].tensor)

            hidden_params = encoder_hidden_params_template.construct(input=encoded_tensor)
            mean = hidden_params[:, :FLAGS.hidden_size]
            stddev = tf.sqrt(tf.exp(hidden_params[:, FLAGS.hidden_size:]))

            decoder_input_tensor = mean + stddev * epsilon

            decoded_tensor, decoder_state = \
                    decoder_template.construct(input=decoder_input_tensor, state=decoder_state[0].tensor)

            write_params = decoder_write_params_template.construct(input=decoder_state[0].tensor)
            g_x, g_y, sigma_sq, delta, gamma = transform_params(write_params, FLAGS.w_N, A, B)
            F_x, F_y = filterbank_matrices(g_x, g_y, sigma_sq, delta, FLAGS.w_N, A, B)

            # Eq 28
            w = decoder_write_template.construct(input=decoded_tensor)
            image_patch = tf.reshape(w, [FLAGS.batch_size, FLAGS.w_N, FLAGS.w_N, C])

            write_glimpse = apply_filters(image_patch, F_x, F_y, gamma, FLAGS.w_N, A, B, False)

            output_image = output_image + write_glimpse

            vae_loss = get_vae_loss(mean, stddev)
            vae_loss_sum +=  vae_loss

            sampled_tensor, sampled_state = \
                    decoder_template.construct(input=epsilon, state=sampled_state[0].tensor)

            write_params = decoder_write_params_template.construct(input=sampled_state[0].tensor)
            g_x, g_y, sigma_sq, delta, gamma = transform_params(write_params, FLAGS.w_N, A, B)
            F_x, F_y = filterbank_matrices(g_x, g_y, sigma_sq, delta, FLAGS.w_N, A, B)

            w = decoder_write_template.construct(input=sampled_tensor)
            image_patch = tf.reshape(w, [FLAGS.batch_size, FLAGS.w_N, FLAGS.w_N, C])

            write_glimpse = apply_filters(image_patch, F_x, F_y, gamma, FLAGS.w_N, A, B, False)

            sampled_image = sampled_image + write_glimpse

        output_image = tf.nn.sigmoid(output_image)
        sampled_image = tf.nn.sigmoid(sampled_image)

    reconst_loss = get_reconst_loss(output_image, input_image)
    loss = vae_loss_sum + reconst_loss

    optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate, beta1=0.5)
    train = pt.apply_optimizer(optimizer, losses=[loss])

    init = tf.initialize_all_variables()
    saver = tf.train.Saver(max_to_keep=0)

    # how many batches are in an epoch
    total_batch = int(np.floor(trainx.shape[0]/(FLAGS.batch_size)))

    with tf.Session() as sess:
        sess.run(init)

        if FLAGS.test:
            saver.restore(sess, model_dir + '/0000.ckpt')

            iter_ = manage.data_iterate(trainx, FLAGS.batch_size)
            imgs = []
            for i in range(13):
                sampled = sess.run(sampled_image)

                if len(imgs) == 0:
                    imgs = sampled
                else:
                    imgs = np.concatenate((imgs, sampled), axis=0)

                img_tile = common.img_tile(imgs[:100], border_color=1.0, stretch=True)
                common.plot_img(img_tile, 'generated MNIST')
                common.plt.savefig(os.path.join(img_dir, 'draw.png'))

        else:
            for epoch in range(FLAGS.max_epoch):
                training_loss = 0
                iter_ = manage.data_iterate(trainx, FLAGS.batch_size)

                for i in tqdm.tqdm(range(total_batch)):

                    next_batch = iter_.next()
                    next_batch = np.reshape(next_batch, [-1, B, A, C])

                    _, loss_value = sess.run([train, loss], feed_dict={input_image: next_batch})
                    training_loss += loss_value

                training_loss = training_loss / total_batch
                print('Epoch: %d|\t Training Loss: %f' %(epoch, training_loss))
                f.write('Epoch: %d|\t Training Loss: %f\n' %(epoch, training_loss))
                f.flush()

                if epoch % 10 == 0:
                    sample_, output_ = \
                            sess.run([sampled_image, output_image], feed_dict={input_image: next_batch})
                    common.plot_generative_output(np.squeeze(sample_), \
                                                    np.squeeze(next_batch), np.squeeze(output_))

                    n = os.path.join(img_dir, str(epoch).zfill(4) + '.png')
                    common.plt.savefig(n, dpi=100)
                    common.plt.clf()

                if (epoch) % 20 == 0:
                    save_path = saver.save(sess, model_dir + '/' + str(epoch).zfill(0) + '.ckpt')

f.close()
