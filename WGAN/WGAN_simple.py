import os
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import matplotlib.pyplot as plt
import seaborn as sns

import tf_utils.common as common
import tf_utils.data.manage as manage

from tqdm import tqdm
import cv2
from fuel.datasets.hdf5 import H5PYDataset
from fuel.schemes import ShuffledScheme
from fuel.transformers import ScaleAndShift
from fuel.streams import DataStream
import h5py

from sklearn.metrics import roc_curve, auc

# logging = tf.logging
flags = tf.flags
flags.DEFINE_integer('batch_size', 100, 'size of batches to use')
flags.DEFINE_integer('n_latent', 10, 'a number of latent variable')
flags.DEFINE_string('log_dir', 'results_WGAN_simple/', 'saved image directory')
flags.DEFINE_integer('max_epoch', 500, 'a number of epoch to run')
flags.DEFINE_string('data_dir', '/home/mlg/ihcho/data', 'data directory')
flags.DEFINE_string('data', 'mnist', 'fuel data')
flags.DEFINE_bool('monitering', True, 'set true if you want to moniter training process')
flags.DEFINE_bool('train', False, 'train or test')
FLAGS = flags.FLAGS


def leaky_relu(x, name='leaky_relu'):
    return tf.where(tf.greater(x, 0), x, 0.1 * x, name=name)


def load_data_with_fuel():
    if FLAGS.data == 'mnist':
        data = 'mnist.hdf5'
    elif FLAGS.data == 'cifar10':
        data = 'cifar10.hdf5'

    data_dir = os.path.join(FLAGS.data_dir, data)
    train_set = H5PYDataset(data_dir, which_sets=('train',))
    test_set = H5PYDataset(data_dir, which_sets=('test',))

    return train_set, test_set


def prepare_log_directory(delete=False):
    if delete:
        # Create log directory
        common.delete_and_create_directory(FLAGS.log_dir)
        img_dir = os.path.join(FLAGS.log_dir, 'imgs')
        common.delete_and_create_directory(img_dir)
        model_dir = os.path.join(FLAGS.log_dir, 'models')
        common.delete_and_create_directory(model_dir)

        # Create tensorboard summary data dir
        summary_dir = os.path.join(FLAGS.log_dir, 'summary')
        common.delete_and_create_directory(summary_dir)

        # Create logfile
        f = open(os.path.join(FLAGS.log_dir, 'training.log'), 'w')
    else:
        # Create log directory
        common.create_directory(FLAGS.log_dir)
        img_dir = os.path.join(FLAGS.log_dir, 'imgs')
        common.create_directory(img_dir)
        model_dir = os.path.join(FLAGS.log_dir, 'models')
        common.create_directory(model_dir)

        # Create tensorboard summary data dir
        summary_dir = os.path.join(FLAGS.log_dir, 'summary')
        common.create_directory(summary_dir)

        # Create logfile
        f = open(os.path.join(FLAGS.log_dir, 'training.log'), 'w')

    return img_dir, model_dir, summary_dir, f


def critic(input_tensor):
    with slim.arg_scope([slim.conv2d],
                        padding='SAME',
                        normalizer_fn=slim.batch_norm,
                        normalizer_params={'scale': True},
                        activation_fn=leaky_relu):
        net = slim.conv2d(input_tensor, 32, [5, 5], 2, scope='conv1')
        net = slim.conv2d(net, 64, [5, 5], 2, scope='conv2')
        net = slim.conv2d(net, 128, [5, 5], 1, padding='VALID', scope='conv3')
        net = slim.dropout(net, 0.9, scope='dropout3')
        net = slim.flatten(net)
        net = slim.fully_connected(net, 1, activation_fn=None, scope='fc1')
    return net


def generator(input_tensor):
    with slim.arg_scope([slim.conv2d_transpose],
                        padding='SAME',
                        normalizer_fn=slim.batch_norm,
                        normalizer_params={'scale': True},
                        activation_fn=leaky_relu):
        if FLAGS.data == 'mnist':
            net = slim.conv2d_transpose(input_tensor, 128, [3, 3], 1, padding='VALID', scope='deconv1')
            net = slim.conv2d_transpose(net, 64, [5, 5], 1, padding='VALID', scope='deconv2')
            net = slim.conv2d_transpose(net, 32, [5, 5], 2, scope='deconv3')
            net = slim.conv2d_transpose(net, 1, [5, 5], 2, normalizer_fn=None, activation_fn=tf.sigmoid, scope='deconv4')

        elif FLAGS.data == 'cifar10':
            net = slim.conv2d_transpose(input_tensor, 128, [4, 4], 1, padding='VALID', scope='deconv1')
            net = slim.conv2d_transpose(net, 64, [5, 5], 1, padding='VALID', scope='deconv2')
            net = slim.conv2d_transpose(net, 32, [5, 5], 2, scope='deconv3')
            net = slim.conv2d_transpose(net, 3, [5, 5], 2, normalizer_fn=None, activation_fn=tf.sigmoid, scope='deconv4')

    return net


def build_model(input_tensor, batch_size, n_latent):
    # z_p = tf.random_normal([batch_size, n_latent])
    z_p = tf.random_uniform((batch_size, 1, 1, n_latent), -1.0, 1.0)

    with tf.variable_scope("critic"):
        cri_positive_out = critic(input_tensor)

    with tf.variable_scope("generator"):
        gen_out = generator(z_p)

    with tf.variable_scope("critic"):
        cri_negative_out = critic(gen_out)

    return cri_positive_out, gen_out, cri_negative_out


def get_loss(input_tensor, cri_positive_out, gen_out, cri_negative_out):
    D_loss = tf.reduce_mean(-cri_positive_out + cri_negative_out)
    C_loss = tf.reduce_mean(-cri_negative_out)

    return D_loss, C_loss


def divide_closed_set(data_set, ys_list, n=5):
    state = data_set.open()
    data, label = data_set.get_data(state, slice(0, data_set.num_examples))
    data_set.close(state)

    known_data = None
    known_label = None
    unknown_data = None
    unknown_label = None
    for i in range(len(ys_list)):
        idx = np.where(label == ys_list[i])[0]
        if i < n:
            if known_data is None:
                known_data = data[idx]
                known_label = label[idx]
            else:
                known_data = np.vstack((known_data, data[idx]))
                known_label = np.concatenate((known_label, label[idx]), axis=0)
        else:
            if unknown_data is None:
                unknown_data = data[idx]
                unknown_label = label[idx]
            else:
                unknown_data = np.vstack((unknown_data, data[idx]))
                unknown_label = np.concatenate((unknown_label, label[idx]), axis=0)

    return known_data, known_label, unknown_data, unknown_label


# Configuration of WGAN
cfg = dict(
    batch_size=FLAGS.batch_size,
    n_latent=FLAGS.n_latent,
    max_epoch=FLAGS.max_epoch,
    gen_lr=5e-5,
    cri_lr=5e-5,
    clip_critic=0.1,
    n_critic=5,
    n_for_test=2500
)


def main():

    if FLAGS.monitering:
        plt.ion()

    # Load data and save dimensions to configuration
    train_set, test_set = load_data_with_fuel()
    train_state = train_set.open()
    train_data, _ = train_set.get_data(train_state, slice(0, 1))
    train_set.close(train_state)

    cfg['channel'] = train_data.shape[1]
    cfg['height']  = train_data.shape[2]
    cfg['width']   = train_data.shape[3]
    cfg['n_train'] = train_set.num_examples()

    # prepare log directory and print configuration
    if FLAGS.train:
        img_dir, model_dir, summary_dir, f = prepare_log_directory(delete=True)
        f.write(str(cfg))
        f.write('\n\n')
    else:
        img_dir, model_dir, summary_dir, f = prepare_log_directory(delete=False)

    # Construct tensorflow graph
    graph = tf.Graph()
    with graph.as_default():
        # Create a variable to count number of train calls
        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
        C_step = tf.get_variable('critic_step', [], initializer=tf.constant_initializer(0), trainable=False)
        G_step = tf.get_variable('generator_step', [], initializer=tf.constant_initializer(0), trainable=False)

        lr = tf.placeholder(tf.float32, shape=[])
        opt = tf.train.RMSPropOptimizer(lr)

        sum_list = []

        input = tf.placeholder(tf.float32, [cfg['batch_size'], cfg['height'], cfg['width'], cfg['channel']])

        # Construct the model
        cri_positive_out, gen_out, cri_negative_out = \
            build_model(input, cfg['batch_size'], cfg['n_latent'])

        # Calculate the loss for this tower
        C_loss, G_loss = \
            get_loss(input, cri_positive_out, gen_out, cri_negative_out)

        sum_list.append(tf.summary.scalar('Discriminator_loss', C_loss))
        sum_list.append(tf.summary.scalar('Generator_loss', G_loss))

        # Specify loss to parameters
        G_params = []
        C_params = []

        for param in tf.trainable_variables():
            if 'critic' in param.name:
                C_params.append(param)
            elif 'generator' in param.name:
                G_params.append(param)

        clip_C = [p.assign(tf.clip_by_value(p, -cfg['clip_critic'], cfg['clip_critic'])) for p in C_params]

        cri_train = opt.minimize(C_loss, global_step=C_step, var_list=C_params)
        gen_train = opt.minimize(G_loss, global_step=G_step, var_list=G_params)

        # Start the Session
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        sess = tf.InteractiveSession(graph=graph, config=config)

        # Merge all the summaries and write them out
        merged = tf.summary.merge(sum_list)
        board_writer = tf.summary.FileWriter(summary_dir, sess.graph)

        sess.run(init)

    if FLAGS.train:
        # prepare log directory and print configuration
        img_dir, model_dir, summary_dir, f = prepare_log_directory(delete=True)
        f.write(str(cfg))
        f.write('\n\n')

        # Set initial training setting

        epoch = 0
        scale = 1.0 / 255.0
        shift = 0
        scheme = ShuffledScheme(examples=cfg['n_train'], batch_size=cfg['batch_size'])
        datastream = ScaleAndShift(
                DataStream(dataset=train_set, iteration_scheme=scheme),
                scale=scale, shift=shift, which_sources='features')

        total_batch  = int(np.floor(cfg['n_train'] / (cfg['batch_size'])))

        # Start training
        print('Training start.......')
        for epoch in range(cfg['max_epoch']):

            with tqdm(total=total_batch) as pbar:
                c_total_loss = g_total_loss = 0

                for batch in datastream.get_epoch_iterator():
                    batch = np.transpose(batch, [0, 2, 3, 1])
                    if batch.shape[0] != cfg['batch_size']:
                        break

                    for i in range(cfg['n_critic']):
                        _, c_loss, = sess.run([cri_train, C_loss], {lr: cfg['cri_lr'], input: batch})
                        _ = sess.run(clip_C)
                    _, g_loss = sess.run([gen_train, G_loss], {lr: cfg['gen_lr'], input: batch})

                    # Write Tensorboard log
                    summary = sess.run(merged, feed_dict={input: batch})
                    board_writer.add_summary(summary, epoch)

                    c_total_loss += c_loss
                    g_total_loss += g_loss

                    # Monitor the generated samples
                    gen_imgs = sess.run(gen_out)
                    gen_imgs = gen_imgs[:, :, :, ::-1]
                    gen_tiled_imgs = common.img_tile(gen_imgs, border_color=1.0, stretch=True)
                    cv2.imshow('generated data', gen_tiled_imgs)
                    cv2.waitKey(1)

                    pbar.set_description('Epoch {} '.format(epoch))
                    pbar.update()

                # Monitor train data
                batch = batch[:, :, :, ::-1]
                tiled_img = common.img_tile(batch, border_color=1.0, stretch=True)
                cv2.imshow('training data', tiled_img)

            c_total_loss /= total_batch
            g_total_loss /= total_batch

            # Show roc and histogram
            if epoch % 10 == 0:
                # Save generated samples per each epoch
                cv2.imwrite(''.join([img_dir, '/generated_', str(epoch).zfill(4), '.jpg']), gen_tiled_imgs * 255.)

            # Save network
            if epoch % 100 == 0:
                saver.save(sess, ''.join([model_dir, '/WGAN.ckpt']), global_step=epoch + 1)

            # Print display network output
            tqdm.write('\tC_loss: {:.4f}\t G_loss: {:.4f}'.format(c_total_loss, g_total_loss))
            f.write('Epoch: {}\t C_loss: {:.4f}\t G_loss: {:.4f}\n'.format(epoch, c_total_loss, g_total_loss))
            f.flush()
    else:
        ckpt = tf.train.get_checkpoint_state(model_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print('Evaluation start.......')
        else:
            print('Error!!! No check point found')

    cv2.destroyAllWindows()
    f.close()


if __name__ == '__main__':
    main()
