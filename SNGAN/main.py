import os
import sys
import tensorflow as tf
import argparse
import logging
import matplotlib
matplotlib.use('Agg')

import utils as utils
from model import Model

parser = argparse.ArgumentParser()

# Train parameters
parser.add_argument(
    '--batch_size',
    type=int,
    default=64,
    help='Size of batches to train the model')
parser.add_argument(
    '--max_epoch',
    type=int,
    default=100,
    help='The number of epochs to train')
parser.add_argument(
    '--data_dir',
    type=str,
    required=True,
    help='Data directory for train or test')
parser.add_argument(
    '--data',
    type=str,
    default='imagenet64',
    help='data version')
parser.add_argument(
    '--normalise',
    type=str,
    default='tanh',
    help='normalisation method for the data')
parser.add_argument(
    '--log_dir',
    type=str,
    default='log',
    help='Data directory for logging')
parser.add_argument(
    '--is_train',
    action='store_true',
    default=False,
    help='Whether train or test')
parser.add_argument(
    '--loss_type',
    type=str,
    choices=['gan', 'hinge', 'wasserstain'],
    default='hinge',
    help='normalisation method for the data')
parser.add_argument(
    '--g_lr',
    type=float,
    default=2e-4,
    help='Learning rate for model')
parser.add_argument(
    '--d_lr',
    type=float,
    default=2e-4,
    help='Learning rate for discriminator')
parser.add_argument(
    '--beta1',
    type=float,
    default=0.5,
    help='beta1 for Adam Optimizer')
parser.add_argument(
    '--beta2',
    type=float,
    default=0.999,
    help='beta2 for Adam Optimizer')
parser.add_argument(
    '--lr_decay_epoch',
    type=float,
    default=10,
    help='Learning rate decay step')
parser.add_argument(
    '--lr_decay_rate',
    type=float,
    default=0.5,
    help='Learning rate decay rate')
parser.add_argument(
    '--z_dim',
    type=float,
    default=128,
    help='Dimension of z')
parser.add_argument(
    '--z_dist',
    choices=['uniform', 'normal'],
    type=str,
    default='normal',
    help='Distribution of z')
parser.add_argument(
    '--weight_decay',
    type=float,
    default=1e-4,
    help='Weight decay for the network')
parser.add_argument(
    '--ps_strategy',
    choices=['CPU', 'GPU'],
    type=str,
    default='CPU',
    help='Where to locate variable operations (parameter server)')
parser.add_argument(
    '--num_gpus',
    type=int,
    default=1,
    help='The number of gpus used. Uses only CPU if set to 0.')
parser.add_argument(
    '--data_format',
    choices=['CPU', 'GPU', None],
    type=str,
    default=None,
    help='If not set, the data format best for the training device is used.')
parser.add_argument(
    '--gpu_memory_fraction',
    type=float,
    default=100,
    help='Fraction of per gpu memory that tensorflow would occupy.' \
    'It should be range between (0~100)%.')

# Input data parameter

# Occlusion parameters for heatmap
#  parser.add_argument('--occ_size', type=int, default=8, help='occlusion size')
#  parser.add_argument('--occ_stride', type=int, default=8, help='occlusion stride')
#  parser.add_argument('--occ_random', action='store_true', default=False, help='occlusion with random values (defualt: 0)')

FLAGS = parser.parse_args()

# Check whether parameters are appropriate
if FLAGS.num_gpus > 0:
    # Get available gpus
    os.environ['CUDA_VISIBLE_DEVICES'] = str(utils.get_available_gpus(FLAGS.num_gpus, FLAGS.gpu_memory_fraction))
if FLAGS.num_gpus == 0:
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
if FLAGS.num_gpus < 0:
    raise ValueError(
        'Invalid GPU count: \"--num_gpus\" must be 0 or a positive integer.')
if FLAGS.num_gpus == 0 and FLAGS.ps_strategy == 'GPU':
    raise ValueError('num_gpus=0, CPU must be used as parameter server. Set'
                     '--ps_strategy=CPU.')
if FLAGS.num_gpus != 0 and FLAGS.batch_size % FLAGS.num_gpus != 0:
    raise ValueError('--batch_size must be multiple of --num_gpus.')

def main():

    if not os.path.exists(FLAGS.log_dir):
        os.makedirs(FLAGS.log_dir)

    if FLAGS.is_train:
        logging.basicConfig(
            level=logging.DEBUG,
            #  format='%(asctime)s:%(levelname)s:%(name)s:%(message)s',
            format='%(message)s',
            filename=os.path.join(FLAGS.log_dir, 'training.log'),
            filemode='w'
        )
    else:
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(message)s',
            filename=os.path.join(FLAGS.log_dir, 'testing.log'),
            filemode='w'
        )
    stdout_logger = logging.getLogger('STDOUT')
    sys.stdout = utils.Logger(sys.stdout, stdout_logger, logging.INFO)
    stderr_logger = logging.getLogger('STDERR')
    sys.stderr = utils.Logger(sys.stderr, stderr_logger, logging.ERROR)

    utils.pprint_dict(vars(FLAGS), 'Input arguments')

    run_config = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=False,
        gpu_options=tf.GPUOptions(force_gpu_compatible=True,
                                  per_process_gpu_memory_fraction=FLAGS.gpu_memory_fraction / 100.,
                                  allow_growth=False))

    with tf.Session(config=run_config) as sess:
        model = Model(name='SNGAN_resnet', **vars(FLAGS))

        if FLAGS.is_train:
            model.train(FLAGS, sess)
        else:
            model.test(FLAGS, sess)
            #  model.t_sne(FLAGS, sess)


if __name__ == '__main__':
    main()
