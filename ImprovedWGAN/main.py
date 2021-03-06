import os
import sys
from tf_utils.common import Logger

from ImprovedWGAN import ImprovedWGAN

import tensorflow as tf
import pprint

flags = tf.app.flags
flags.DEFINE_integer('batch_size', 100, 'Size of batches to use')
flags.DEFINE_float('g_lr', 1e-4, 'Learning rate for generator')
flags.DEFINE_float('d_lr', 1e-4, 'Learning rate for discriminator')
flags.DEFINE_float('beta1', 0.5, 'beta1 for Adam Optimizer')
flags.DEFINE_float('beta2', 0.9, 'beta2 for Adam Opmimizer')
flags.DEFINE_integer('lambda_', 10, 'hyper-parameter for gradient penalty')
flags.DEFINE_integer('z_dim', 100, 'a number of z dimension layer')
flags.DEFINE_string('z_dist', 'uniform', 'Distribution for z [uniform, normal]')
flags.DEFINE_string('log_dir', 'results_simple/', 'saved image directory')
flags.DEFINE_integer('max_epoch', 500, 'A number of epochs to train')
flags.DEFINE_boolean('is_train', False, 'True for training, False for testing')
flags.DEFINE_string('data_dir', '/home/mlg/ihcho/data', 'data directory')
flags.DEFINE_string('data', 'mnist', 'fuel data')
flags.DEFINE_boolean('monitor', False, 'True for monitoring training process')
flags.DEFINE_integer('n_critic', 5, 'A number of updates for critic')
FLAGS = flags.FLAGS


def main(_):

    if not os.path.exists(FLAGS.log_dir):
        os.makedirs(FLAGS.log_dir)
    f = open(os.path.join(FLAGS.log_dir, 'training.log'), 'a')
    sys.stdout = Logger(f)

    print('\n======================================')
    pprint.PrettyPrinter().pprint(flags.FLAGS.__flags)

    run_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    with tf.Session(config=run_config) as sess:
        wgan = ImprovedWGAN(
            sess=sess,
            log_dir=FLAGS.log_dir,
            data=FLAGS.data,
            data_dir=FLAGS.data_dir,
            batch_size=FLAGS.batch_size,
            z_dim=FLAGS.z_dim,
            z_dist=FLAGS.z_dist,
            lambda_=FLAGS.lambda_
        )

        if FLAGS.is_train:
            wgan.train(FLAGS)
        else:
            if not wgan.load_model(FLAGS.log_dir):
                raise Exception('[!] Train a model first, then run test mode')

    f.close()


if __name__ == '__main__':
    tf.app.run()
