import os
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

import tf_utils.common as common

from tqdm import tqdm
matplotlib.use('GTKagg')

# logging = tf.logging
flags = tf.flags
flags.DEFINE_integer('batch_size', 250, 'size of batches to use(per GPU)')
flags.DEFINE_integer('n_hidden', 128, 'a number of hidden layer')
flags.DEFINE_integer('n_latent', 256, 'a number of latent variable')
flags.DEFINE_string('log_dir', 'results_synthe/', 'saved image directory')
flags.DEFINE_integer('max_step', 45001, 'a number of steps to run')
flags.DEFINE_integer('n_gpu', 2, 'the number of gpus to use')
flags.DEFINE_string('data_dir', './', 'data directory')
flags.DEFINE_bool('monitering', True, 'set true if you want to moniter training process')
FLAGS = flags.FLAGS


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


def sample_mog(batch_size, n_mixture=8, std=0.02, radius=2.0):
    thetas = np.linspace(0, 2 * np.pi, n_mixture + 1)
    thetas = thetas[:-1]
    xs, ys = radius * np.sin(thetas), radius * np.cos(thetas)
    cat = tf.contrib.distributions.Categorical(tf.ones(n_mixture))
    comps = [tf.contrib.distributions.MultivariateNormalDiag([xi, yi], [std, std]) for xi, yi in zip(xs.ravel(), ys.ravel())]
    data = tf.contrib.distributions.Mixture(cat, comps)
    return data.sample_n(batch_size)


def average_gradient(tower_grads):
    ''' Calculate the average gradient for each shared variable across all towers.
    Note that this function provides a synchronization point across all towers.
    Args:
        tower_grads: List of lists of (gradient, variable) tuples. The outer list
                    is over individual gradients. The inner is over the gradient
                    calculation for each tower.
    Returns:
        List of pairs of (graident, variable) where the gradient has been averaged
        across all towers.
    '''

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
        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # accross towers. So.. we will just return the first tower's point to
        # the Variables
        v = grad_and_vars[0][1]
        grad_and_vars = (grad, v)
        average_grads.append(grad_and_vars)

    return average_grads


def discriminator(input_tensor, n_hidden):
    net = slim.fully_connected(input_tensor, n_hidden, scop='fc1')
    net = slim.fully_connected(net, n_hidden, scope='fc2')
    net = slim.fully_connected(net, 1, activation_fn=None, scope='fc3')
    return net


def generator(input_tensor, n_hidden):
    net = slim.fully_connected(input_tensor, n_hidden, scope='fc1')
    net = slim.fully_connected(net, n_hiddeen, scope='fc2')
    net = slim.fully_connected(net, 2, activation_fn=None, scope='fc3')
    return net


def build_model(input_tensor, batch_size, n_latent, n_hidden, in_place):
    # z_p = tf.random_uniform((batch_size, n_latent), -1.0, 1.0)
    z_p = tf.random_normal([batch_size, n_latent])

    with slim.arg_scope([slim.fully_connected],
                        padding='SAME',
                        activation_fn=tf.nn.relu):
        with tf.variable_scope("discriminator"):
            disc_positive_out = discriminator(input_tensor, n_hidden)

        with tf.variable_scope("generator"):
            gen_out = generator(z_p, n_hidden)

        with tf.variable_scope("discriminator", reuse=True):
            disc_negative_out = discriminator(gen_out, n_hidden)

        with tf.variable_scope("discriminator", reuse=True):
            disc_rand = discriminator(in_place, n_hidden)

    return disc_positive_out, gen_out, disc_negative_out, disc_rand


def get_loss(input_tensor, disc_positive_out, gen_out, disc_negative_out):
    D_loss = tf.reduce_mean(-disc_positive_out + disc_negative_out)
    G_loss = tf.reduce_mean(-disc_negative_out)

    return D_loss, G_loss


def main():
    # Control the number of gpus being used
    gpus = np.arange(0, FLAGS.n_gpu)
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(i) for i in gpus])

    if FLAGS.monitering:
        plt.ion()

    # configuration of GAN
    batch_size         = FLAGS.batch_size
    n_latent           = FLAGS.n_latent
    n_hidden           = FLAGS.n_hidden
    gen_learning_rate  = 5e-5
    disc_learning_rate = 5e-5
    clip_critic        = 0.1
    n_critic           = 5
    xmax               = 4

    img_dir, model_dir, summary_dir, f = prepare_for_train()

    # Setting figure
    fig = plt.figure(num=1, figsize=(18, 6))
    ax1 = plt.subplot2grid((2, 6), (0, 0), colspan=2, rowspan=2)
    ax2 = plt.subplot2grid((2, 6), (0, 2), colspan=2, rowspan=2)
    ax3 = plt.subplot2grid((2, 6), (0, 4), colspan=2, rowspan=2)
    bg_color = sns.color_palette('Blues', n_colors=256)[0]

    # Construct tensorflow graph
    graph = tf.Graph()
    with graph.as_default(), tf.device('/cpu:0'):
        data = sample_mog(batch_size * FLAGS.n_gpu)

        # Create a variable to count number of train calls
        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)

        lr = tf.placeholder(tf.float32, shape=[])
        in_place = tf.placeholder(tf.float32, shape=[None, 2])
        opt = tf.train.RMSPropOptimizer(lr)

        # These are the lists for each tower
        tower_disc_grads = []
        tower_gen_grads = []
        tower_gen = []
        sum_list = []

        all_input = data

        # Define the network for each GPU
        for i in xrange(FLAGS.n_gpu):
            with tf.device('/gpu:%d' % i):
                with tf.name_scope('Tower_%d' % (i)) as scope:
                    # Grab this portion of the input
                    next_batch = all_input[i * batch_size:(i + 1) * batch_size, :]

                    # Construct the model
                    disc_positive_out, gen_out, disc_negative_out, disc_rand = \
                        build_model(next_batch, batch_size, n_latent, n_hidden, in_place)

                    # Calculate the loss for this tower
                    D_loss, G_loss = \
                        get_loss(next_batch, disc_positive_out, gen_out, disc_negative_out)

                    sum_list.append(tf.scalar_summary('Tower_%d/Discriminator_loss' % (i), D_loss))
                    sum_list.append(tf.scalar_summary('Tower_%d/Generator_loss' % (i), G_loss))

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
                    tower_gen.append(gen_out)

        # Merge tower information
        gen_out = None
        for i in range(FLAGS.n_gpu):
            if gen_out is None:
                gen_out = tower_gen[i]
            else:
                gen_out = tf.concat(0, [gen_out, tower_gen[i]])

        clip_D = [p.assign(tf.clip_by_value(p, -clip_critic, clip_critic)) for p in D_params]

        # Average the gradients
        disc_grads = average_gradient(tower_disc_grads)
        gen_grads = average_gradient(tower_gen_grads)

        # Apply the gradients with our optimizers
        disc_train = opt.apply_gradients(disc_grads, global_step=global_step)
        gen_train = opt.apply_gradients(gen_grads, global_step=global_step)

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
    ctr = 0
    np_samples = []
    # Start training
    print('Training start.......')
    with tqdm(total=FLAGS.max_step) as pbar:
        for step in range(FLAGS.max_step):

            if step % 5000 == 0:
                np_samples.append(np.vstack([sess.run(gen_out) for _ in xrange(10)]))

            if step % 100 == 0:
                xx, yy = sess.run([gen_out, data])
                
                ax2.clear()
                ax2.set_xlim(-xmax, xmax)
                ax2.set_ylim(-xmax, xmax)
                sns.kdeplot(xx[:, 0], xx[:, 1], cmap='Blues', shade=True, n_levels=20, clip=[[-xmax, xmax]] * 2, shade_lowest=False, ax=ax2)
                ax2.set_axis_bgcolor(bg_color)
                ax2.set_title('Step {}'.format(step))
                ax2.scatter(xx[:, 0], xx[:, 1], c='r', edgecolor='none', alpha=0.1)
                ax2.scatter(yy[:, 0], yy[:, 1], c='g', edgecolor='none')

                ax3.clear()
                ax3.set_xlim(-xmax, xmax)
                ax3.set_ylim(-xmax, xmax)
                grid = np.mgrid[-xmax:xmax:100j, -xmax:xmax:100j]
                grid_2d = np.transpose(grid.reshape(2, -1))
                x, y = grid
                n_iter = 100000 / FLAGS.batch_size
                out_list = []
                for i in range(n_iter):
                    start = i * FLAGS.batch_size 
                    end = start + FLAGS.batch_size
                    out_list.append(sess.run(disc_rand, feed_dict={in_place: grid_2d[start:end, :]}))
                out = np.transpose(np.vstack(out_list))
                # cmap = sns.diverging_palette(250, 12, n=9, s=85, l=25, as_cmap=True)
                cmap = sns.diverging_palette(250, 12, n=9, s=85, l=25, as_cmap=True)
                contour = ax3.contourf(x, y, out.reshape(100, 100), cmap=cmap)
                ax3.scatter(xx[:, 0], xx[:, 1], c='b', edgecolor='none')
                ax3.scatter(yy[:, 0], yy[:, 1], c='r', edgecolor='none')

                ax1.clear()
                ax1.set_xlim(-xmax, xmax)
                ax1.set_ylim(-xmax, xmax)
                sns.kdeplot(yy[:, 0], yy[:, 1], cmap='Blues', shade=True, n_levels=20, clip=[[-xmax, xmax]] * 2, shade_lowest=False, ax=ax1)
                ax1.set_axis_bgcolor(bg_color)
                ax1.set_title('Train data')

                plt.gcf().tight_layout()
                plt.savefig(os.path.join(img_dir, '{}.png'.format(step)))
                if FLAGS.monitering:
                    plt.pause(0.05)

            for i in range(n_critic):
                _, d_loss, = sess.run([disc_train, D_loss], {lr: disc_learning_rate})
                _ = sess.run(clip_D)
            _, g_loss = sess.run([gen_train, G_loss], {lr: gen_learning_rate})

            # Write Tensorboard log
            summary = sess.run(merged)
            board_writer.add_summary(summary, step)

            pbar.set_description('Step {}, D_loss: {:.4f}, G_loss: {:.4f} '.format(step, d_loss, g_loss))
            pbar.update()

    np_samples.append(np.vstack([sess.run(data) for _ in xrange(10)]))
    np_samples_ = np_samples[::1]
    cols = len(np_samples_)
    bg_color = sns.color_palette('Greens', n_colors=256)[0]
    plt.figure(num=2, figsize=(2 * cols, 2))
    for i, samps in enumerate(np_samples_):
        if i == 0:
            ax = plt.subplot(1, cols, 1)
        else:
            plt.subplot(1, cols, i + 1, sharex=ax, sharey=ax)
        ax2 = sns.kdeplot(samps[:, 0], samps[:, 1], shade=True, cmap='Greens', n_levels=20, clip=[[-xmax, xmax]] * 2)
        ax2.set_axis_bgcolor(bg_color)
        plt.xticks([]); plt.yticks([])
        if i + 1 == cols:
            plt.title('Target')
        else:
            plt.title('step %d'%(i*5000))
    ax.set_ylabel('clip: {}, lr: {}'.format(clip_critic, disc_learning_rate))
    plt.gcf().tight_layout()
    plt.savefig(os.path.join(FLAGS.log_dir, 'Density_{}_{}.png'.format(clip_critic, disc_learning_rate)))
    plt.pause(60)

    # Save network
    saver.save(sess, ''.join([model_dir, '/GAN_' + str(step).zfill(4), '.tfmod']))
    f.close()

if __name__ == '__main__':
    main()
