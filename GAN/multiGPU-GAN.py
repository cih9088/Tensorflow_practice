import os
import numpy as np
import prettytensor as pt
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('agg')

from tf_utils.pt.deconv import deconv2d
import tf_utils.data.manage as manage
import tf_utils.common as common

from tqdm import tqdm
import cv2
from fuel.datasets.hdf5 import H5PYDataset
from fuel.schemes import ShuffledScheme
from fuel.transformers import ScaleAndShift
from fuel.streams import DataStream

#logging = tf.logging

flags = tf.flags
flags.DEFINE_integer('batch_size', 50, 'size of batches to use(per GPU)')
flags.DEFINE_integer('n_hidden', 10, 'a number of hidden layer')
flags.DEFINE_string('log_dir', 'multiGPU-GAN/', 'saved image directory')
flags.DEFINE_integer('max_epoch', 200, 'a number of epochs to run')
flags.DEFINE_integer('n_gpu', 2, 'the number of gpus to use')
flags.DEFINE_string('data_dir', '/home/mlg/ihcho/data', 'data directory')
flags.DEFINE_string('data', 'mnist', 'fuel data')
FLAGS = flags.FLAGS

def leaky_relu(x, name='leaky_relu'):
    return tf.select(tf.greater(x, 0), x, 0.01 * x, name=name) # VAE?
    #return tf.select(tf.greater(x, 0), x, 0.1 * x, name=name) # LVAE paper

def prepare_for_train():
    # Create log directory
    common.delete_and_create_directory(FLAGS.log_dir)
    img_dir = os.path.join(FLAGS.log_dir, 'imgs')
    common.delete_and_create_directory(img_dir)
    model_dir = os.path.join(FLAGS.log_dir, 'models')
    common.delete_and_create_directory(model_dir)

    # Create tensorboard summary data dir
    summary_dir = os.path.join(FLAGS.log_dir, 'train')
    common.delete_and_create_directory(summary_dir)

    # Create logfile
    f = open(os.path.join(FLAGS.log_dir, 'training.log'), 'w')

    return img_dir, model_dir, summary_dir, f

def load_data_with_fuel():
    if FLAGS.data == 'mnist':
        data = 'mnist.hdf5'
    elif FLAGS.data == 'cifar10':
        data = 'cifar10.hdf5'

    data_dir = os.path.join(FLAGS.data_dir, data)
    train_set = H5PYDataset(data_dir, which_sets=('train',))
    test_set = H5PYDataset(data_dir, which_sets=('test',))

    return train_set, test_set

def average_gradient(tower_grads):
    """ Calculate the average gradient for each shared variable across all towers.
    Note that this function provides a synchronization point across all towers.
    Args:
        tower_grads: List of lists of (gradient, variable) tuples. The outer list
                    is over individual gradients. The inner is over the gradient
                    calculation for each tower.
    Returns:
        List of pairs of (graident, variable) where the gradient has been averaged
        across all towers.
    """

    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        # ((grad0_gpu0, var0_gpu0), ..., (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(0, grads)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # accross towers. So.. we will just return the first tower's point to
        # the Variables
        v = grad_and_vars[0][1]
        grad_and_vars = (grad, v)
        average_grads.append(grad_and_vars)

    return average_grads

def discriminator(input_tensor):
    return (pt.wrap(input_tensor).
            conv2d(5, 32, stride=2).
            conv2d(5, 64, stride=2).
            conv2d(5, 128, edges='VALID').
            dropout(0.9).
            flatten().fully_connected(1, activation_fn=None))

def generator(input_tensor):
    return (pt.wrap(input_tensor).
            deconv2d(3, 128, edges='VALID').
            deconv2d(5, 64, edges='VALID').
            deconv2d(5, 32, stride=2).
            deconv2d(5, 1, stride=2))

def build_model(input_tensor, batch_size, n_hidden):
    z_p = tf.random_uniform((batch_size, 1, 1, n_hidden), -1.0, 1.0) # normal dist for GAN

    with pt.defaults_scope(activation_fn=leaky_relu,
                            batch_normalize=True,
                            learned_moments_update_rate=0.0003,
                            variance_epsilon=0.001,
                            scale_after_normalization=True):
        with tf.variable_scope("discriminator"):
            disc_positive_out = discriminator(input_tensor)

        with tf.variable_scope("generator"):
            gen_out = generator(z_p)

        with tf.variable_scope("discriminator", reuse=True):
            disc_negative_out = discriminator(gen_out)

    return disc_positive_out, gen_out, disc_negative_out

def get_loss(input_tensor, disc_positive_out, gen_out, disc_negative_out, epsilon=1e-6):
#    D_loss = -tf.reduce_mean(tf.log(tf.clip_by_value(tf.sigmoid(disc_positive_out), epsilon, 1.0-epsilon))) \
#            -tf.reduce_mean(1.0 - tf.log(tf.clip_by_value(tf.sigmoid(disc_negative_out), epsilon, 1.0-epsilon)))
#    G_loss = -tf.reduce_mean(tf.log(tf.clip_by_value(tf.sigmoid(disc_negative_out), epsilon, 1.0-epsilon)))

#    D_loss = tf.reduce_mean(tf.nn.relu(disc_positive_out) - disc_positive_out + tf.log(1.0 + tf.exp(-tf.abs(disc_positive_out)))) + tf.reduce_mean(tf.nn.relu(disc_negative_out) + tf.log(1.0 + tf.exp(-tf.abs(disc_negative_out))))
#    G_loss = tf.reduce_mean(tf.nn.relu(disc_negative_out) - disc_negative_out + tf.log(1.0 + tf.exp(-tf.abs(disc_negative_out))))
    D_loss = tf.reduce_mean(tf.nn.softplus(-disc_positive_out))\
            + tf.reduce_mean(tf.nn.softplus(-disc_negative_out) + disc_negative_out)
    G_loss = tf.reduce_mean(tf.nn.softplus(-disc_negative_out))

    return D_loss, G_loss


if __name__ == '__main__':

#    # Control the number of gpus being used
#    gpus = np.arange(0, FLAGS.n_gpu)
#    os.environ["CUDA_VISIBLE_DEVICES"]=','.join([str(i) for i in gpus])

    img_dir, model_dir, summary_dir, f = prepare_for_train()

    train_set, test_set = load_data_with_fuel()
    state = train_set.open()
    data = train_set.get_data(state, slice(0,1))

    # configuration of GAN
    batch_size         = FLAGS.batch_size
    n_hidden           = FLAGS.n_hidden
    n_train            = train_set.num_examples
    channel            = data[0].shape[1]
    height             = data[0].shape[2]
    width              = data[0].shape[3]
    gen_learning_rate  = 0.003
    disc_learning_rate = 0.003

    # Construct tensorflow graph
    graph = tf.Graph()
    with graph.as_default():
        # Create a variable to count number of train calls
        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)

        lr = tf.placeholder(tf.float32, shape=[])
        opt = tf.train.AdamOptimizer(lr, epsilon=1.0)

        # These are the lists for each tower
        tower_disc_grads = []
        tower_gen_grads = []
        tower_data_acc = []
        tower_sample_acc = []

        all_input = tf.placeholder(tf.float32, [batch_size*FLAGS.n_gpu, height, width, channel])

        # Define the network for each GPU
        for i in xrange(FLAGS.n_gpu):
            with tf.device('/gpu:%d' % i):
                with tf.name_scope('Tower_%d' % (i)) as scope:
                    # Grab this portion of the input
                    next_batch = all_input[i*batch_size:(i+1)*batch_size, :]

                    # Construct the model
                    disc_positive_out, gen_out, disc_negative_out = \
                            build_model(next_batch, batch_size, n_hidden)

                    # Calculate the loss for this tower
                    D_loss, G_loss = \
                            get_loss(next_batch, disc_positive_out, gen_out, disc_negative_out)

                    # Additional accuracy information for GAN training
#                    data_acc = tf.reduce_mean(
#                            tf.cast(disc_positive_out >= 0.5, tf.float32), name='accuracy_data')
#                    sample_acc = tf.reduce_mean(
#                            tf.casst(disc_negative_out < 0.5, tf.float32), name='accuracy_sample')
                    
                    # Logging tensorboard
#                    tf.scalar_summary('Tower_%d/data_accuracy' %(i), data_acc)
#                    tf.scalar_summary('Tower_%d/sample_accuracy' %(i), sample_acc)
                    tf.scalar_summary('Tower_%d/Discriminator_loss' % (i), D_loss)
                    tf.scalar_summary('Tower_%d/Generator_loss' % (i), G_loss)

                    # Specify loss to parameters
                    G_params = []
                    D_params = []

                    for param in tf.trainable_variables():
                        if 'discriminator' in param.name:
                            D_params.append(param)
                        elif 'generator' in param.name:
                            G_params.append(param)

                    # Reuse variables for the next tower
                    tf.get_variable_scope().reuse_variables()

                    # Calculate the gradients for the batch of data on this tower
                    disc_grads = opt.compute_gradients(D_loss, var_list=D_params)
                    gen_grads = opt.compute_gradients(G_loss, var_list=G_params)

                    # Keep track of the gradients across all towers
                    tower_disc_grads.append(disc_grads)
                    tower_gen_grads.append(gen_grads)
#                    tower_data_acc.append(data_acc)
#                    tower_sample_acc.append(sample_acc)

        # Average the gradients
        disc_grads = average_gradient(tower_disc_grads)
        gen_grads = average_gradient(tower_gen_grads)

        # Apply the gradients with our optimizers
        disc_train = opt.apply_gradients(disc_grads, global_step=global_step)
        gen_train = opt.apply_gradients(gen_grads, global_step=global_step)

        # Start the Session
        init = tf.initialize_all_variables()
        saver = tf.train.Saver()
        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
        sess = tf.InteractiveSession(graph=graph, config=config)

        # Merge all the summaries and write them out
        merged = tf.merge_all_summaries()
        board_writer = tf.train.SummaryWriter(summary_dir, sess.graph)

        sess.run(init)

    #tf.train.Saver.resotre(saver, sess, 'models/cifar_multiGPU.tfmod')
    epoch = 0
    scale = 1.0 / 255.0
    shift = 0
    scheme = ShuffledScheme(examples=n_train, batch_size=batch_size*FLAGS.n_gpu)
    datastream = ScaleAndShift(
            DataStream(dataset=train_set, iteration_scheme=scheme),
            scale=scale, shift=shift, which_sources='features')

    total_batch = int(np.floor(n_train/(batch_size * FLAGS.n_gpu)))

    # Start training
    for epoch in range(FLAGS.max_epoch):
        print('Training start.......')

        with tqdm(total=total_batch) as pbar:
            pbar.set_description('Epoch %d ' % epoch)
            d_total_loss = g_total_loss = 0

            for batch in datastream.get_epoch_iterator():
                pbar.update()
                batch = np.transpose(batch[0], [0, 2, 3, 1])

                # Train the model and calculate loss for each model
                _, d_loss = sess.run([disc_train, D_loss], {lr: disc_learning_rate, all_input: batch})
                _, g_loss = sess.run([gen_train, G_loss], {lr: gen_learning_rate, all_input: batch})

                # Write Tensorboard log
                summary = sess.run(merged, feed_dict={all_input: batch})
                board_writer.add_summary(summary, epoch)

                d_total_loss += d_loss
                g_total_loss += g_loss

#                # Monitor the generated samples
#                img1, img2 = sess.run([gen_out, gen_out])
#                generated_imgs = np.vstack((img1, img2))
#                tiled_img = common.img_tile(generated_imgs, border_color=1.0, stretch=True)
#                cv2.imshow('generated', tiled_img)
#                cv2.waitKey(1)

            d_total_loss /= total_batch
            g_total_loss /= total_batch

        # Print display network output
        tqdm.write('\tD_loss: %.4f\t G_loss: %.4f' %(d_total_loss, g_total_loss))
        f.write('Epoch: %d\t D_loss: %.4f\t G_loss: %.4f\n' %(epoch, d_total_loss, g_total_loss))
        f.flush()

        # Save generated samples per each epoch
        img1, img2 = sess.run([gen_out, gen_out])
        generated_imgs = np.vstack((img1, img2))
        tiled_img = common.img_tile(generated_imgs, border_color=1.0, stretch=True)
        cv2.imwrite(''.join([img_dir, '/generated_', str(epoch).zfill(4), '.jpg']), tiled_img * 255.)

    # Save network
    saver.save(sess, ''.join([model_dir, '/multigpu_GAN_' + str(epoch).zfill(4), '.tfmod']))
    f.close()


