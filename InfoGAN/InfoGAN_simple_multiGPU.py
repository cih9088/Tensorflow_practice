import os
import numpy as np
import prettytensor as pt
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt

from tf_utils.pt.deconv import deconv2d
import tf_utils.data.manage as manage
import tf_utils.common as common

from tqdm import tqdm
import cv2
from fuel.datasets.hdf5 import H5PYDataset
from fuel.schemes import ShuffledScheme
from fuel.transformers import ScaleAndShift
from fuel.streams import DataStream
matplotlib.use('agg')

# logging = tf.logging
flags = tf.flags
flags.DEFINE_integer('batch_size', 50, 'size of batches to use(per GPU)')
flags.DEFINE_integer('n_latent', 62, 'a number of latent variable')
flags.DEFINE_string('log_dir', 'results_simple/', 'saved image directory')
flags.DEFINE_integer('max_epoch', 3000, 'a number of epochs to run')
flags.DEFINE_integer('n_gpu', 2, 'the number of gpus to use')
flags.DEFINE_string('data_dir', '/home/mlg/ihcho/data', 'data directory')
flags.DEFINE_string('data', 'mnist', 'fuel data')
FLAGS = flags.FLAGS


def leaky_relu(x, name='leaky_relu'):
    return tf.select(tf.greater(x, 0), x, 0.1 * x, name=name)


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


def shared_structure(input_tensor):
    return (pt.wrap(input_tensor).
            conv2d(5, 32, stride=2).
            conv2d(5, 64, stride=2).
            conv2d(5, 128, edges='VALID').
            flatten()).tensor


def discriminator(input_tensor):
    return (pt.wrap(input_tensor).
            fully_connected(1, activation_fn=None)).tensor


def recognitor(input_tensor):
    if FLAGS.data == 'mnist':
        output = (pt.wrap(input_tensor).fully_connected(128)).tensor

        c1 = (pt.wrap(output).fully_connected(10, activation_fn=tf.nn.softmax)).tensor
        c2_logit = (pt.wrap(output).fully_connected(2, activation_fn=None)).tensor
        c3_logit = (pt.wrap(output).fully_connected(2, activation_fn=None)).tensor
        c2_mean, c2_log_sigma_sq = tf.split(1, 2, c2_logit)
        c3_mean, c3_log_sigma_sq = tf.split(1, 2, c3_logit)

        return [c1, c2_mean, c2_log_sigma_sq, c3_mean, c3_log_sigma_sq]


def generator(z_input, c_input):
    input_tensor = tf.concat(3, [z_input, c_input])
    if FLAGS.data == 'mnist':
        return (pt.wrap(input_tensor).
                deconv2d(3, 128, edges='VALID').
                deconv2d(5, 64, edges='VALID').
                deconv2d(5, 32, stride=2).
                deconv2d(5, 1, stride=2, batch_normalize=False, activation_fn=tf.sigmoid)).tensor

    elif FLAGS.data == 'cifar10':
        return (pt.wrap(input_tensor).
                deconv2d(4, 128, edges='VALID').
                deconv2d(5, 64, edges='VALID').
                deconv2d(5, 32, stride=2).
                deconv2d(5, 3, stride=2, batch_normalize=False, activation_fn=tf.sigmoid)).tensor


def build_model(input_tensor, batch_size, n_latent):
    z_p = tf.random_uniform((batch_size, 1, 1, n_latent), -1.0, 1.0)  # normal dist for GAN

    if FLAGS.data == 'mnist':
        p = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
        dist = tf.contrib.distributions.Categorical(p=p)
        samples = dist.sample_n(batch_size)
        c1 = tf.expand_dims(tf.expand_dims(tf.one_hot(samples, 10), 1), 1)
        c2 = tf.random_uniform((batch_size, 1, 1, 1), -1.0, 1.0)
        c3 = tf.random_uniform((batch_size, 1, 1, 1), -1.0, 1.0)
        c_p = tf.concat(3, [c1, c2, c3])

    with pt.defaults_scope(
            activation_fn=leaky_relu,
            batch_normalize=True,
            learned_moments_update_rate=0.0003,
            variance_epsilon=0.001,
            scale_after_normalization=True):

        with tf.variable_scope('shared'):
            positive_shared = shared_structure(input_tensor)

        with tf.variable_scope('discriminator'):
            positive_disc = discriminator(positive_shared)

        with tf.variable_scope('recognitor'):
            positive_recog = recognitor(positive_shared)

        with tf.variable_scope('generator'):
            output_gen = generator(z_p, c_p)

        with tf.variable_scope('shared', reuse=True):
            negative_shared = shared_structure(output_gen)

        with tf.variable_scope('recognitor', reuse=True):
            negative_recog = recognitor(negative_shared)

        with tf.variable_scope('discriminator', reuse=True):
            negative_disc = discriminator(negative_shared)

        categorical_gen = []
        for i in range(10):
            categorical_p = np.zeros([10])
            categorical_p[i] = 1.0
            categorical_dist = tf.contrib.distributions.Categorical(p=categorical_p)
            categorical_samples = categorical_dist.sample_n(batch_size)
            categorical_c1 = tf.expand_dims(tf.expand_dims(tf.one_hot(categorical_samples, 10), 1), 1)
            categorical_c_p = tf.concat(3, [categorical_c1, c2, c3])

            with tf.variable_scope('generator', reuse=True):
                categorical_gen.append(generator(z_p, categorical_c_p))

    return positive_disc, negative_disc, positive_recog, negative_recog, output_gen, categorical_gen, c_p


def get_D_loss(positive_disc, negative_disc):
    return tf.reduce_mean(tf.nn.softplus(-positive_disc)) \
        + tf.reduce_mean(tf.nn.softplus(-negative_disc) + negative_disc)


def get_G_Q_loss(negative_disc, negative_recog, c_p, cont_lamb=1, disc_lamb=1, epsilon=1e-8):
    c_p = tf.squeeze(c_p)
    if FLAGS.data == 'mnist':
        in_c1 = tf.slice(c_p, [0, 0], [-1, 10])
        in_c2 = tf.slice(c_p, [0, 9], [-1, 1])
        in_c3 = tf.slice(c_p, [0, 10], [-1, 1])
        c1, c2_mean, c2_log_sigma_sq, c3_mean, c3_log_sigma_sq = negative_recog

        c1 = tf.clip_by_value(c1, epsilon, 1 - epsilon)
        categorical_loss = -tf.reduce_sum(in_c1 * tf.log(c1))

        c2_err = (in_c2 - c2_mean) / (tf.sqrt(tf.exp(c2_log_sigma_sq)) + epsilon)
        c2_loss = (0.5 * c2_log_sigma_sq + 0.5 * tf.square(c2_err))
        c3_err = (in_c3 - c3_mean) / (tf.sqrt(tf.exp(c3_log_sigma_sq)) + epsilon)
        c3_loss = (0.5 * c3_log_sigma_sq + 0.5 * tf.square(c3_err))
        continuous_loss = tf.reduce_mean(c2_loss + c3_loss)

    Q_loss = disc_lamb * categorical_loss + cont_lamb * continuous_loss
    G_loss = tf.reduce_mean(tf.nn.softplus(-negative_disc)) + Q_loss

    return G_loss, Q_loss


def main():
    # Control the number of gpus being used
    gpus = np.arange(0, FLAGS.n_gpu)
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(i) for i in gpus])

    img_dir, model_dir, summary_dir, f = prepare_for_train()

    train_set, _ = load_data_with_fuel()
    state = train_set.open()
    data = train_set.get_data(state, slice(0, 100))

    # configuration of GAN
    batch_size          = FLAGS.batch_size
    n_latent            = FLAGS.n_latent
    n_train             = train_set.num_examples
    channel             = data[0].shape[1]
    height              = data[0].shape[2]
    width               = data[0].shape[3]
    gen_learning_rate   = 0.001
    disc_learning_rate  = 0.001
    recog_learning_rate = 0.001

    # Show training data
    data = np.transpose(data[0], (0, 2, 3, 1))
    data = data[:, :, :, ::-1]
    tiled_img = common.img_tile(data, border_color=1.0, stretch=True)
    cv2.imshow('training data', tiled_img)
    cv2.waitKey(100)

    # Construct tensorflow graph
    graph = tf.Graph()
    with graph.as_default(), tf.device('/cpu:0'):
        # Create a variable to count number of train calls
        global_step = \
            tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)

        lr = tf.placeholder(tf.float32, shape=[])
        opt = tf.train.AdamOptimizer(lr)

        # These are the lists for each tower
        tower_disc_grads      = []
        tower_gen_grads       = []
        tower_recog_grads     = []
        tower_data_acc        = []
        tower_sample_acc      = []
        tower_acc             = []
        tower_gen             = []
        tower_categorical_gen = []
        sum_list              = []

        all_input = tf.placeholder(tf.float32, [batch_size * FLAGS.n_gpu, height, width, channel])

        # Define the network for each GPU
        for i in xrange(FLAGS.n_gpu):
            with tf.device('/gpu:%d' % i):
                with tf.name_scope('Tower_%d' % (i)) as scope:
                    # Grab this portion of the input
                    next_batch = all_input[i * batch_size:(i + 1) * batch_size, :]

                    # Construct the model
                    positive_disc, negative_disc, positive_recog, negative_recog, output_gen, categorical_gen, c_p = \
                        build_model(next_batch, batch_size, n_latent)

                    # Calculate the loss for this tower
                    D_loss = get_D_loss(positive_disc, negative_disc)
                    G_loss, Q_loss = get_G_Q_loss(negative_disc, negative_recog, c_p)

                    # Additional accuracy information for GAN training
                    data_acc = tf.reduce_mean(
                        tf.cast(tf.greater_equal(positive_disc, 0), tf.float32), name='accuracy_data')
                    sample_acc = tf.reduce_mean(
                        tf.cast(tf.less(negative_disc, 0), tf.float32), name='accuracy_sample')
                    acc = (data_acc + sample_acc) / 2.0

                    # Logging tensorboard
                    sum_list.append(tf.summary.scalar('Tower_%d/data_accuracy' % (i), data_acc))
                    sum_list.append(tf.summary.scalar('Tower_%d/sample_accuracy' % (i), sample_acc))
                    sum_list.append(tf.summary.scalar('Tower_%d/accuracy' % (i), acc))
                    sum_list.append(tf.summary.scalar('Tower_%d/Discriminator_loss' % (i), D_loss))
                    sum_list.append(tf.summary.scalar('Tower_%d/Generator_loss' % (i), G_loss))
                    sum_list.append(tf.summary.scalar('Tower_%d/Recognitor_loss' % (i), Q_loss))

                    # Specify loss to parameters
                    G_params = []
                    D_params = []
                    Q_params = []

                    for param in tf.trainable_variables():
                        if 'shared' in param.name:
                            D_params.append(param)
                            Q_params.append(param)
                        elif 'discriminator' in param.name:
                            D_params.append(param)
                        elif 'recognitor' in param.name:
                            Q_params.append(param)
                        elif 'generator' in param.name:
                            G_params.append(param)

                    # Reuse variables for the next tower
                    tf.get_variable_scope().reuse_variables()

                    # Calculate the gradients for the batch of data on this tower
                    disc_grads = opt.compute_gradients(D_loss, var_list=D_params)
                    recog_grads = opt.compute_gradients(Q_loss, var_list=Q_params)
                    gen_grads = opt.compute_gradients(G_loss, var_list=G_params)

                    # Keep track of the gradients across all towers
                    tower_disc_grads.append(disc_grads)
                    tower_gen_grads.append(gen_grads)
                    tower_recog_grads.append(recog_grads)
                    tower_data_acc.append(data_acc)
                    tower_sample_acc.append(sample_acc)
                    tower_gen.append(output_gen)
                    tower_categorical_gen.append(categorical_gen)
                    tower_acc.append(acc)

        # Merge tower information
        acc = None
        output_gen = None
        categorical_gen = None
        for i in range(FLAGS.n_gpu):
            if acc is None:
                acc = tower_acc[i]
            else:
                acc += tower_acc[i]
            if output_gen is None:
                output_gen = tower_gen[i]
            else:
                output_gen = tf.concat(0, [output_gen, tower_gen[i]])
            if categorical_gen is None:
                categorical_gen = tower_categorical_gen[i]
            else:
                for j in range(10):
                    categorical_gen[j] = tf.concat(0, [categorical_gen[j], tower_categorical_gen[i][j]])
        acc /= FLAGS.n_gpu

        # Average the gradients
        recog_grads = average_gradient(tower_recog_grads)
        disc_grads = average_gradient(tower_disc_grads)
        gen_grads = average_gradient(tower_gen_grads)

        # Apply the gradients with our optimizers
        disc_train = opt.apply_gradients(disc_grads)
        gen_train = opt.apply_gradients(gen_grads)
        recog_train = opt.apply_gradients(recog_grads, global_step=global_step)

        # Start the Session
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        sess = tf.InteractiveSession(graph=graph, config=config)

        # Merge all the summaries and write them out
        merged = tf.summary.merge(sum_list)
        board_writer = tf.train.SummaryWriter(summary_dir, sess.graph)

    sess.run(init)
    # tf.train.Saver.resotre(saver, sess, 'models/cifar_multiGPU.tfmod')
    epoch = 0
    scale = 1.0 / 255.0
    shift = 0
    scheme = ShuffledScheme(examples=n_train, batch_size=batch_size * FLAGS.n_gpu)
    datastream = ScaleAndShift(
        DataStream(dataset=train_set, iteration_scheme=scheme),
        scale=scale, shift=shift, which_sources='features')

    total_batch = int(np.floor(n_train / (batch_size * FLAGS.n_gpu)))

    prev_accuracy = 0.0
    accuracy = 0.0
    gen_iter = 1
    # Start training
    print('Training start.......')
    for epoch in range(FLAGS.max_epoch):

        gen_ctr = 0
        disc_ctr = 0
        with tqdm(total=total_batch) as pbar:
            d_total_loss = g_total_loss = q_total_loss = 0

            for batch in datastream.get_epoch_iterator():
                batch = np.transpose(batch[0], [0, 2, 3, 1])

                # if prev_accuracy >= 0.85:
                    # gen_iter += 1
                # elif prev_accuracy < 0.85:
                    # gen_iter = 1

                # if gen_iter >= 5:
                    # gen_iter = 5

                # if prev_accuracy < 0.99:
                    # _, d_loss, accuracy = \
                        # sess.run([disc_train, D_loss, acc], {lr: disc_learning_rate, all_input: batch})
                    # disc_ctr += 1
                # else:
                    # d_loss, accuracy = \
                        # sess.run([D_loss, acc], {lr: disc_learning_rate, all_input: batch})

                # if accuracy > 0.7:
                    # for i in range(gen_iter):
                        # _, g_loss = sess.run([gen_train, G_loss], {lr: gen_learning_rate, all_input: batch})
                        # _, q_loss = sess.run([recog_train, Q_loss], {lr: recog_learning_rate})
                    # gen_ctr += 1
                # else:
                    # g_loss = sess.run(G_loss, {all_input: batch})
                    # q_loss = sess.run(Q_loss, {lr: recog_learning_rate})

                # prev_accuracy = accuracy

                _, d_loss = sess.run([disc_train, D_loss], {lr: disc_learning_rate, all_input: batch})
                _, q_loss = sess.run([recog_train, Q_loss], {lr: recog_learning_rate, all_input: batch})
                _, g_loss = sess.run([gen_train, G_loss], {lr: gen_learning_rate, all_input: batch})

                # Write Tensorboard log
                summary = sess.run(merged, feed_dict={all_input: batch})
                board_writer.add_summary(summary, epoch)

                d_total_loss += d_loss
                g_total_loss += g_loss
                q_total_loss += q_loss

                # Monitor the generated samples
                gen_imgs = sess.run(output_gen)
                gen_imgs = gen_imgs[:, :, :, ::-1]
                gen_tiled_imgs = common.img_tile(gen_imgs, border_color=1.0, stretch=True)
                cv2.imshow('generated data', gen_tiled_imgs)
                cv2.waitKey(1)

                # pbar.set_description('Epoch {}, ({:.3f}, {}), {}, {}'.format(epoch, accuracy, gen_iter, gen_ctr, disc_ctr))
                pbar.set_description('Epoch {} '.format(epoch))
                pbar.update()

            # Monitor the data
            batch = batch[:, :, :, ::-1]
            tiled_img = common.img_tile(batch, border_color=1.0, stretch=True)
            cv2.imshow('training data', tiled_img)

            d_total_loss /= total_batch
            g_total_loss /= total_batch
            q_total_loss /= total_batch

        # Monitor categorically generated data
        categorical_imgs = sess.run(categorical_gen)
        categorical_tiled_imgs = []
        for i in range(10):
            categorical_imgs[i] = categorical_imgs[i][:, :, :, ::-1]
            categorical_tiled_imgs.append(
                common.img_tile(categorical_imgs[i], stretch=True))
        row1 = np.concatenate((categorical_tiled_imgs[0], categorical_tiled_imgs[1], categorical_tiled_imgs[2], categorical_tiled_imgs[3], categorical_tiled_imgs[4]), axis=1)
        row2 = np.concatenate((categorical_tiled_imgs[5], categorical_tiled_imgs[6], categorical_tiled_imgs[7], categorical_tiled_imgs[8], categorical_tiled_imgs[9]), axis=1)
        merged_tiled_imgs = np.concatenate((row1, row2), axis=0)
        cv2.imshow('categorical generated data', merged_tiled_imgs)

        # Print display network output
        tqdm.write('\tD_loss: %.4f\t G_loss: %.4f\t Q_loss: %.4f' % (d_total_loss, g_total_loss, q_total_loss))
        f.write('Epoch: %d\t D_loss: %.4f\t G_loss: %.4f\t Q_loss: %.4f\n' % (epoch, d_total_loss, g_total_loss, q_total_loss))
        f.flush()

        # Save generated samples per each epoch
        cv2.imwrite(''.join([img_dir, '/generated_', str(epoch).zfill(4), '.jpg']), gen_tiled_imgs * 255.)
        cv2.imwrite(''.join([img_dir, '/categorical_', str(epoch).zfill(4), '.jpg']), merged_tiled_imgs * 255.)

    cv2.destroyAllWindows()
    # Save network
    saver.save(sess, ''.join([model_dir, '/InfoGAN_' + str(epoch).zfill(4), '.tfmod']))
    f.close()


if __name__ == '__main__':
    main()
