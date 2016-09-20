import os
import numpy as np
import prettytensor as pt
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('agg')

from tf_utils.pt.deconv import deconv2d
import tf_utils.data.cifar10_data as cifar10_data
import tf_utils.data.manage as manage
import tf_utils.common as common

import IPython.display
import math
import tqdm

#logging = tf.logging

flags = tf.flags
flags.DEFINE_integer('batch_size', 50, 'size of batches to use(per GPU)')
flags.DEFINE_integer('n_hidden', 2048, 'a number of hidden layer')
flags.DEFINE_string('log_dir', 'VAE-CIFAR_logs/', 'saved image directory')
flags.DEFINE_integer('max_epoch', 10000, 'a number of epochs to run')
flags.DEFINE_integer('n_gpu', 2, 'the number of gpus to use')
flags.DEFINE_string('data_dir', '/home/mlg/ihcho/data', 'data directory')
FLAGS = flags.FLAGS

dim1 = 32 # first dimension of input data
dim2 = 32 # second dimension of input data
dim3 = 3 # third dimension of input data (colors)
### we can train our different networks  with different learning rates if we want to
e_learning_rate = 1e-3
d_learning_rate = 1e-3

gpus = np.arange(0, FLAGS.n_gpu)
os.environ["CUDA_VISIBLE_DEVICES"]=','.join([str(i) for i in gpus])

def sigmoid(x, shift, mult):
    """
    Using this sigmoid to discourage one network overpowering the other
    """
    return 1 / (1 + math.exp(-(x+shift)*mult))

def create_image(im):
    return np.reshape(im, (dim1, dim2, dim3))

def encoder(x):
    """ Create encoder network
    Args:
        x: a btch of flattend images [FLAGS.batch_size, 32*32*3]
    Returns:
        A tensor that express the encoder network
            # The transformation is parameterized and can be learned.
            # returns network output, mean, setd
    """
    lay_end = (pt.wrap(x).
                reshape([FLAGS.batch_size, dim1, dim2, dim3]).
                conv2d(5, 64, stride=2).
                conv2d(5, 128, stride=2).
                conv2d(5, 256, stride=2).
                flatten())
    #print lay_end.get_shape()
    z_mean = lay_end.fully_connected(FLAGS.n_hidden, activation_fn=None)
    z_log_sigma_sq = lay_end.fully_connected(FLAGS.n_hidden, activation_fn=None)

    return z_mean, z_log_sigma_sq

def decoder(z):
    """ Create decoder network
        If input tensor is provied then decodes it, otherwise samples from a sampled vector.
    Args:
        z: a batch of vectors of decoder
    Returns:
        A tensor that express the generator network
    """
    return (pt.wrap(z).
            fully_connected(4*4*256).reshape([FLAGS.batch_size, 4, 4, 256]).
            deconv2d(5, 256, stride=2).
            deconv2d(5, 128, stride=2).
            deconv2d(5, 32, stride=2).
            deconv2d(1, dim3, stride=1, activation_fn=tf.sigmoid).
            flatten()
            )

def inference(x):
    """
    Run the models. Called inference because it does the same thing as tensorflow's cifar tutorial
    """

    z_p = tf.random_normal((FLAGS.batch_size, FLAGS.n_hidden), 0, 1) # normal dist for GAN
    eps = tf.random_normal((FLAGS.batch_size, FLAGS.n_hidden), 0, 1) # normal dist for VAE

    with pt.defaults_scope(activation_fn=tf.nn.elu,
                            batch_normalize=True,
                            learned_moments_update_rate=0.0003,
                            variance_epsilon=0.001,
                            scale_after_normalization=True):
        with tf.variable_scope("enc"):
            z_mean, z_log_sigma_sq = encoder(x) # get z from the input

        with tf.variable_scope("dec"):
            z = tf.add(z_mean, tf.mul(tf.sqrt(tf.exp(z_log_sigma_sq)), eps)) # grab our actual z
            recon_x = decoder(z)

        with tf.variable_scope("dec", reuse=True):
            sample_x = decoder(eps)

        return z_mean, z_log_sigma_sq, z, recon_x, sample_x

def loss(x, recon_x, z_log_sigma_sq, z_mean):
    """
    Loss function for SSE, KL divergence, Discrim, Generator, Lth Layer Similarity
    """
    ### We don't actually use SSE(MSE) loss for anything (but maybe prettraining)
    SSE_loss = tf.reduce_mean(tf.square(x - recon_x)) # This is what a normal VAE uses

    # We clip gradients of KL divergence to prevent NANs
    #KL_loss = tf.reduce_sum( -0.5 * tf.reduce_sum(1 + tf.clip_by_value(z_x_log_sigma_sq, -10.0, 10.0) - tf.square(tf.clip_by_value(z_x_mean, -10.0, 10.0)) - tf.exp(tf.clip_by_value(z_x_log_sigma_sq, -10.0, 10)),1))/dim1/dim2/dim3
    regular_loss = -0.5 * tf.reduce_sum(1 + z_log_sigma_sq - tf.square(z_mean) - tf.exp(z_log_sigma_sq))

    # Generator Loss
    #G_loss = tf.reduce_mean(-1. * (tf.log(tf.clip_by_value(d_x_p, 1e-5, 1.0)))) #+ tf.log(tf.clip_by_value(1.0 - d_x, 1e-5, 1.0)))
    reconst_loss = -tf.reduce_sum(x * tf.log(recon_x + 1e-10) + (1.0 - x) * tf.log(1.0 - recon_x + 1e-10))

    return SSE_loss, regular_loss, reconst_loss

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

def plot_network_output(example_data):
    """ Just plots the output of the network, error, reconstructions, etc
    """
    random_x, reconst_z = sess.run((sample_x, z_mean), {all_input: example_data})
    reconst_x = sess.run(recon_x, {z: reconst_z})
    examples = 8
    random_x = np.squeeze(random_x)
    reconst_x = np.squeeze(reconst_x)
    random_x = random_x[0:8]

    fig, ax = plt.subplots(nrows=3, ncols=examples, figsize=(18, 6))
    for i in xrange(examples):
        ax[(0,i)].imshow(create_image(random_x[i]), cmap=plt.cm.gray, interpolation='nearest')
        ax[(1,i)].imshow(create_image(example_data[i + (FLAGS.n_gpu-1)*FLAGS.batch_size]), cmap=plt.cm.gray, interpolation='nearest')
        ax[(2,i)].imshow(create_image(reconst_x[i]), cmap=plt.cm.gray, interpolation='nearest')
        ax[(0,i)].axis('off')
        ax[(1,i)].axis('off')
        ax[(2,i)].axis('off')

    fig.suptitle('Top: random points in z space | Middle: inputs | Bottom: reconstructions')
#    plt.show()
    fig.savefig(''.join([img_dir, '/test_', str(epoch).zfill(4), '.png']), dpi=100)
    plt.clf()
"""
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(20,10), linewidth=4)
    KL_plt, = plt.semilogy((KL_loss_list), linewidth=4, ls='-', color='r', alpha=.5, label='KL')
    D_plt, = plt.semilogy((D_loss_list), linewidth=4, ls='-', color='b', alpha=.5, label='D')
    G_plt, = plt.semilogy((G_loss_list), linewidth=4, ls='-', color='k', alpha=.5, label='G')
    SSE_plt, = plt.semilogy((SSE_loss_list), linewidth=4, ls='-', color='g', alpha=.5, label='SSE')
    LL_plt, = plt.semilogy((LL_loss_list), linewidth=4, ls='-', color='m', alpha=.5, label='LL')

    axes = plt.gca()
    leg = plt.legend(handles=[KL_plt, D_plt, G_plt, SSE_plt, LL_plt], fontsize=20)
    leg.get_frame().set_alpha(0.5)
#    plt.show()
    fig.savefig(''.join(['imgs/test_', str(epoch).zfill(4), '_r.png']), dpi=100)
"""
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

    '''
    # print network architecture
    f.write('batch_size=%d\n' %(FLAGS.batch_size))
    '''

    # load CIFAR-10
    trainx, trainy = cifar10_data.load(FLAGS.data_dir, subset='train')
    testx, testy = cifar10_data.load(FLAGS.data_dir, subset='test')

    # Normalize the dataset between 0 and 1
    trainx = (trainx) / 255.
    testx = (testx) / 255.

    # flatten
    trainx = np.reshape(trainx, (trainx.shape[0], dim1 * dim2 * dim3))
    testx = np.reshape(testx, (testx.shape[0], dim1 * dim2 * dim3))

    graph = tf.Graph()

    # Make lists to save the losses to
    # You should probably just be using tensorboard to do any visualization
    regular_loss_list = []
    reconst_loss_list = []
    SSE_loss_list = []

    """
    # test to calculate encoded shape
    x = tf.placeholder(tf.float32, (FLAGS.batch_size, dim1 * dim2 * dim3))
    encoder(x)
    """
    
    with graph.as_default():
        # Create a variable to count number of train calls
        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)

        # different optimizers are needed for different learning rates
        # (uing the same learning rate seems to work fine though)
        lr_D = tf.placeholder(tf.float32, shape=[])
        lr_E = tf.placeholder(tf.float32, shape=[])
        opt_D = tf.train.AdamOptimizer(lr_D, epsilon=1.0)
        opt_E = tf.train.AdamOptimizer(lr_E, epsilon=1.0)

        # These are the lists of gradients for each tower
        tower_grads_e = []
        tower_grads_d = []

        all_input = tf.placeholder(tf.float32, [FLAGS.batch_size*FLAGS.n_gpu, dim1*dim2*dim3])

        # Define the network for each GPU
        for i in xrange(FLAGS.n_gpu):
            with tf.device('/gpu:%d' % i):
                with tf.name_scope('Tower_%d' % (i)) as scope:
                    # grab this portion of the input
                    next_batch = all_input[i*FLAGS.batch_size:(i+1)*FLAGS.batch_size,:]

                    # Construct the model
                    z_mean, z_log_sigma_sq, z, recon_x, sample_x = inference(next_batch)

                    # Calculate the loss for this tower
                    SSE_loss, regular_loss, reconst_loss = \
                            loss(next_batch, recon_x, z_log_sigma_sq, z_mean)
                    
                    # logging tensorboard
                    tf.scalar_summary('Tower_%d/SSE_loss' % (i), SSE_loss)
                    tf.scalar_summary('Tower_%d/regular_loss' % (i), regular_loss)
                    tf.scalar_summary('Tower_%d/reconst_loss' % (i), reconst_loss)                   

                    # specify loss to parameters
                    params = tf.trainable_variables()
                    E_params = [i for i in params if 'enc' in i.name]
                    D_params = [i for i in params if 'dec' in i.name]

                    # Calculate the losses specific to encoder, generator, decoder
                    Loss = regular_loss + reconst_loss

                    # Reuse variables for the next tower
                    tf.get_variable_scope().reuse_variables()

                    # Calculate the gradients for the batch of data on this CIFAR tower
                    grads_e = opt_E.compute_gradients(Loss, var_list=E_params)
                    grads_d = opt_D.compute_gradients(Loss, var_list=D_params)

                    # Keep track of the gradients across all towers
                    tower_grads_e.append(grads_e)
                    tower_grads_d.append(grads_d)

#        # logging tensorboard
#        img_origin = tf.reshape(next_batch, (FLAGS.batch_size, dim1, dim2, dim3))
#        img_recon = tf.reshape(recon_x, (FLAGS.batch_size, dim1, dim2, dim3))
#        img_gener = tf.reshape(sample_x, (FLAGS.batch_size, dim1, dim2, dim3))
#        tf.image_summary('original', img_origin)
#        tf.image_summary('reconstruction', img_recon)
#        tf.image_summary('generation', img_gener)

        # Average the gradients
        grads_e = average_gradient(tower_grads_e)
        grads_d = average_gradient(tower_grads_d)

        # apply the gradients with our optimizers
        train_E = opt_E.apply_gradients(grads_e, global_step=global_step)
        train_D = opt_D.apply_gradients(grads_d, global_step=global_step)


        # Start the Session
        init = tf.initialize_all_variables()
        saver = tf.train.Saver()
        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
        sess = tf.InteractiveSession(graph=graph, config=config)

        # Merge all the summaries and write them out
        merged = tf.merge_all_summaries()
        board_writer = tf.train.SummaryWriter(summary_dir, sess.graph)

        sess.run(init)

    epoch = 0
    #tf.train.Saver.resotre(saver, sess, 'models/cifar_multiGPU.tfmod')

    # how many batches are in an epoch
    total_batch = int(np.floor(trainx.shape[0]/(FLAGS.batch_size * FLAGS.n_gpu)))

    e_current_lr = e_learning_rate
    d_current_lr = d_learning_rate

    while epoch < FLAGS.max_epoch:
        regular_e = reconst_e = SSE_e = 0
        for i in tqdm.tqdm(range(total_batch)):
            iter_ = manage.data_iterate(trainx, FLAGS.batch_size)

            ## balance dec and descrim
            #e_current_lr = e_learning_rate * sigmoid(np.mean(d_real), -.5, 15)
            #d_current_lr = d_learning_rate * sigmoid(np.mean(d_real), -.5, 15)

            next_batch = np.array([])
            for j in xrange(FLAGS.n_gpu):
                partial_batch = iter_.next()
                if next_batch.size == 0:
                    next_batch = partial_batch
                else:
                    next_batch = np.concatenate((next_batch, partial_batch), axis=0)

            _, _, regular_err, reconst_err, SSE_err = \
                    sess.run([train_E, train_D, regular_loss, reconst_loss, SSE_loss],
                            {lr_E: e_current_lr, lr_D: d_current_lr, all_input: next_batch})

            regular_e += regular_err
            reconst_e += reconst_err
            SSE_e += SSE_err

            # Save our lists
            regular_loss_list.append(regular_err)
            reconst_loss_list.append(reconst_err)
            SSE_loss_list.append(SSE_err)

        regular_e /= total_batch
        reconst_e /= total_batch
        SSE_e /= total_batch

        # print display network output
        print('Epoch: %d\t regular: %.4f\t reconst: %.4f\t SSE: %.6f' %(epoch, regular_e, reconst_e, SSE_e))
        f.write('Epoch: %d\t regular: %.4f\t reconst: %.4f\t SSE: %.6f\n' %(epoch, regular_e, reconst_e, SSE_e))
        f.flush()
        plot_network_output(next_batch)

        # Write Tensorboard log
        summary = sess.run(merged, feed_dict={all_input: next_batch})
        board_writer.add_summary(summary, epoch)

        if epoch % 100 == 0:
            # save network
            saver.save(sess, ''.join([model_dir, '/cifar_multiGPU_' + str(epoch).zfill(4), '.tfmod']))

        epoch += 1
f.close()
