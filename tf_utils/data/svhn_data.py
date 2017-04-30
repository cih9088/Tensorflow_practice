import os
import sys
from six.moves import urllib
import numpy as np
import scipy.io

url = 'http://ufldl.stanford.edu/housenumbers/'
filenames = ['train_32x32.mat', 'test_32x32.mat', 'extra_32x32.mat']

def maybe_download(data_dir):
    for filename in filenames:
        filepath = os.path.join(data_dir, filename)
        if not os.path.isfile(filepath):
            def _progress(count, block_size, total_size):
                sys.stdout.write('\r>> Downloading %s %.1f%%'
                    % (filename, float(count * block_size) / float(total_size) * 100.0))
                sys.stdout.flush()
            filepath, _ = urllib.request.urlretrieve(os.path.join(url, filename), filepath, _progress)
            print()
            statinfo = os.stat(filepath)
            print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')


def load(data_dir, subset='train'):
    maybe_download(data_dir)

    if subset == 'train':
        filepath = os.path.join(data_dir, filenames[0])
        data_set = scipy.io.loadmat(filepath)
        datax = data_set['X']
        datay = data_set['y']

        datax = np.transpose(datax, [3, 0, 1, 2])
        datax = datax.astype(np.float32)

        datay[datay == 10] = 0
        datay.astype(np.uint8)

        return datax, datay

    elif subset == 'test':
        filepath = os.path.join(data_dir, filenames[1])
        data_set = scipy.io.loadmat(filepath)
        datax = data_set['X']
        datay = data_set['y']

        datax = np.transpose(datax, [3, 0, 1, 2])
        datax = datax.astype(np.float32)

        datay[datay == 10] = 0
        datay.astype(np.uint8)

        return datax, datay
    else:
        raise NotImplementedError('subset should be train or test')
