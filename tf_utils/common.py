import os
import sys
import shutil
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

import seaborn as sns
sns.set_context("paper")
sns.set_style("white")

def get_one_hot(label):
    one_hot = np.zeros((label.shape[0], 5))
    one_hot[np.arange(label.shape[0]), label.astype(int)] = 1
    return one_hot


def image_mix(input, num=2):
    batch_size = input.shape[0]
    idx = np.random.randint(0, batch_size, size=[batch_size, num])

    data = np.zeros(input.shape)
    for i in range(num):
        data += 0.5 * input[idx[:, i]]

    data[data >= 1.0] = 1.0
    # data = common.scale_to_unit_interval(data)
    return data


class Logger(object):
    def __init__(self, f):
        self.terminal = sys.stdout
        self.log = f

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        # this flush method is nedded for python 3 compatibility.
        # this handles the flush command by doing nothing
        # you might wnat to specify some extra behavior here.
        pass


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
        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # accross towers. So.. we will just return the first tower's point to
        # the Variables
        v = grad_and_vars[0][1]
        grad_and_vars = (grad, v)
        average_grads.append(grad_and_vars)

    return average_grads

def show_roc(y_true, y_score, title, f=1):
    from sklearn.metrics import roc_curve, auc
    from scipy.optimize import brentq
    from scipy.interpolate import interp1d

    fpr, tpr, threshold = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    # eer2 = fpr[np.argmin(abs(1-(fpr+tpr)))]
    thresh = interp1d(fpr, threshold)(eer)

    plt.figure(f)
    plt.clf()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='(AUC = {:.4f}, EER = {:.4f})'.format(roc_auc, eer))
    print('{}\tAUC: {}, EER: {}'.format(title, roc_auc, eer))
    # plt.plot(fpr, tpr, color='darkorange', lw=2, label='(AUC = {:.4f})'.format(roc_auc))
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc='lower right')


def show_hist(data, label, title, f=1):
    assert len(data) == len(label)

    fig = plt.figure(num=f)
    fig.clf()
    fig, ax = plt.subplots(num=f)

    for i in range(len(data)):
        sns.kdeplot(data[i], label=label[i], shade=True, shade_lowest=False, ax=ax)

    ax.relim()
    ax.autoscale()
    ax.set_ylim(0, None)
    ax.set_title(title)
    ax.legend()


def delete_and_create_directory(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)


def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)


def binarize(images):
    """
    Stochastically binarize values in [0, 1] by treating them as p-values of
    a Bernoulli distribution.
    """
    return (np.random.uniform(size=images.shape) < images).astype('float32')


# Plot image examples.
def plot_img(img, title):
    plt.figure()
    plt.imshow(img, interpolation='nearest')
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()

    assert len(data) == len(label)

    fig = plt.figure(num=f)
    fig.clf()
    fig, ax = plt.subplots(num=f)

    for i in range(len(data)):
        sns.kdeplot(data[i], label=label[i], shade=True, shade_lowest=False, ax=ax)

    ax.relim()
    ax.autoscale()
    ax.set_ylim(0, None)
    ax.set_title(title)
    ax.legend()

def plot_output(data, label, title='plot', examples=8, f=1):
    """ Just plots the output of the network, error, reconstructions, etc
    """
    assert len(data) == len(label)

    nrows = len(data)

    fig = plt.figure(num=f)
    fig.clf()
    fig, ax = plt.subplots(nrows=nrows, ncols=examples, figsize=(18, 6), num=f)
    for i in xrange(examples):
        for j in xrange(nrows):
            ax[(j, i)].imshow(np.squeeze(data[j][i]), cmap=plt.cm.gray, interpolation='nearest')
            ax[(j, i)].axis('off')

    # fig.suptitle('Top: random points in z space | Middle: inputs | Bottom: reconstructions')

def plot_generative_output(gens, inputs, reconsts, examples=8):
    """ Just plots the output of the network, error, reconstructions, etc
    """
    gens = img_stretch(np.squeeze(gens))
    inputs = img_stretch(np.squeeze(inputs))
    reconsts = img_stretch(np.squeeze(reconsts))
    fig, ax = plt.subplots(nrows=3, ncols=examples, figsize=(18, 6))
    for i in xrange(examples):
        ax[(0, i)].imshow(gens[i], cmap=plt.cm.gray, interpolation='nearest')
        ax[(1, i)].imshow(inputs[i], cmap=plt.cm.gray, interpolation='nearest')
        ax[(2, i)].imshow(reconsts[i], cmap=plt.cm.gray, interpolation='nearest')
        ax[(0, i)].axis('off')
        ax[(1, i)].axis('off')
        ax[(2, i)].axis('off')

    fig.suptitle('Top: random points in z space | Middle: inputs | Bottom: reconstructions')


def img_stretch(img):
    img = img.astype(float)
    img -= np.min(img)
    img /= np.max(img) + 1e-12
    return img


def img_tile(imgs, aspect_ratio=1.0, tile_shape=None, border=1,
             border_color=0, stretch=False):
    ''' Tile images in a grid.
    If tile_shape is provided only as many images as specified in tile_shape
    will be included in the output.
    '''

    # Prepare images
    if stretch:
        imgs = img_stretch(imgs)
    imgs = np.array(imgs)
    if imgs.ndim != 3 and imgs.ndim != 4:
        raise ValueError('imgs has wrong number of dimensions.')
    n_imgs = imgs.shape[0]

    # Grid shape
    img_shape = np.array(imgs.shape[1:3])
    if tile_shape is None:
        img_aspect_ratio = img_shape[1] / float(img_shape[0])
        aspect_ratio *= img_aspect_ratio
        tile_height = int(np.ceil(np.sqrt(n_imgs * aspect_ratio)))
        tile_width = int(np.ceil(np.sqrt(n_imgs / aspect_ratio)))
        grid_shape = np.array((tile_height, tile_width))
    else:
        assert len(tile_shape) == 2
        grid_shape = np.array(tile_shape)

    # Tile image shape
    tile_img_shape = np.array(imgs.shape[1:])
    tile_img_shape[:2] = (img_shape[:2] + border) * grid_shape[:2] - border

    # Assemble tile image
    tile_img = np.empty(tile_img_shape)
    tile_img[:] = border_color
    for i in range(grid_shape[0]):
        for j in range(grid_shape[1]):
            img_idx = j + i * grid_shape[1]
            if img_idx >= n_imgs:
                # No more images - stop filling out the grid.
                break
            img = imgs[img_idx]
            yoff = (img_shape[0] + border) * i
            xoff = (img_shape[1] + border) * j
            tile_img[yoff:yoff + img_shape[0], xoff:xoff + img_shape[1], ...] = img

    return tile_img


def conv_filter_tile(filters):
    n_filters, n_channels, height, width = filters.shape
    tile_shape = None
    if n_channels == 3:
        # Interpret 3 color channels as RGB
        filters = np.transpose(filters, (0, 2, 3, 1))
    else:
        # Organize tile such that each row corresponds to a filter and the
        # columns are the filter channels
        tile_shape = (n_channels, n_filters)
        filters = np.transpose(filters, (1, 0, 2, 3))
        filters = np.resize(filters, (n_filters * n_channels, height, width))
    filters = img_stretch(filters)
    return img_tile(filters, tile_shape=tile_shape)
    

def scale_to_unit_interval(ndar, eps=1e-8):
    """ Scales all values in the ndarray ndar to be between 0 and 1 """
    ndar = ndar.copy()
    ndar -= ndar.min()
    ndar *= 1.0 / (ndar.max() + eps)
    return ndar


def tile_raster_images(X, img_shape, tile_shape, tile_spacing=(0, 0),
                       scale_rows_to_unit_interval=True,
                       output_pixel_vals=True):
    """
    Transform an array with one flattened image per row, into an array in
    which images are reshaped and layed out like tiles on a floor.

    This function is useful for visualizing datasets whose rows are images,
    and also columns of matrices for transforming those rows
    (such as the first layer of a neural net).

    :type X: a 2-D ndarray or a tuple of 4 channels, elements of which can
    be 2-D ndarrays or None;
    :param X: a 2-D array in which every row is a flattened image.

    :type img_shape: tuple; (height, width)
    :param img_shape: the original shape of each image

    :type tile_shape: tuple; (rows, cols)
    :param tile_shape: the number of images to tile (rows, cols)

    :param output_pixel_vals: if output should be pixel values (i.e. int8
    values) or floats

    :param scale_rows_to_unit_interval: if the values need to be scaled before
    being plotted to [0,1] or not


    :returns: array suitable for viewing as an image.
    (See:`PIL.Image.fromarray`.)
    :rtype: a 2-d array with same dtype as X.

    """

    assert len(img_shape) == 2
    assert len(tile_shape) == 2
    assert len(tile_spacing) == 2

    # The expression below can be re-written in a more C style as
    # follows :
    #
    # out_shape = [0,0]
    # out_shape[0] = (img_shape[0] + tile_spacing[0]) * tile_shape[0] -
    #                tile_spacing[0]
    # out_shape[1] = (img_shape[1] + tile_spacing[1]) * tile_shape[1] -
    #                tile_spacing[1]
    out_shape = [(ishp + tsp) * tshp - tsp for ishp, tshp, tsp
                  in zip(img_shape, tile_shape, tile_spacing)]

    if isinstance(X, tuple):
        assert len(X) == 4
        # Create an output numpy ndarray to store the image
        if output_pixel_vals:
            out_array = np.zeros((out_shape[0], out_shape[1], 4), dtype='uint8')
        else:
            out_array = np.zeros((out_shape[0], out_shape[1], 4), dtype=X.dtype)

        #colors default to 0, alpha defaults to 1 (opaque)
        if output_pixel_vals:
            channel_defaults = [0, 0, 0, 255]
        else:
            channel_defaults = [0., 0., 0., 1.]

        for i in range(4):
            if X[i] is None:
                # if channel is None, fill it with zeros of the correct
                # dtype
                out_array[:, :, i] = np.zeros(out_shape,
                      dtype='uint8' if output_pixel_vals else out_array.dtype
                      ) + channel_defaults[i]
            else:
                # use a recurrent call to compute the channel and store it
                # in the output
                out_array[:, :, i] = tile_raster_images(X[i], img_shape, tile_shape, tile_spacing, scale_rows_to_unit_interval, output_pixel_vals)
        return out_array

    else:
        # if we are dealing with only one channel
        H, W = img_shape
        Hs, Ws = tile_spacing

        # generate a matrix to store the output
        out_array = np.zeros(out_shape, dtype='uint8' if output_pixel_vals else X.dtype)


        for tile_row in range(tile_shape[0]):
            for tile_col in range(tile_shape[1]):
                if tile_row * tile_shape[1] + tile_col < X.shape[0]:
                    if scale_rows_to_unit_interval:
                        # if we should scale values to be between 0 and 1
                        # do this by calling the `scale_to_unit_interval`
                        # function
                        this_img = scale_to_unit_interval(X[tile_row * tile_shape[1] + tile_col].reshape(img_shape))
                    else:
                        this_img = X[tile_row * tile_shape[1] + tile_col].reshape(img_shape)
                    # add the slice to the corresponding position in the
                    # output array
                    out_array[
                            tile_row * (H+Hs): tile_row * (H + Hs) + H,
                            tile_col * (W+Ws): tile_col * (W + Ws) + W
                            ] \
                            = this_img * (255 if output_pixel_vals else 1)
        return out_array

