import pickle
import os
import sys
import tarfile
from six.moves import urllib
import numpy as np

def maybe_download_and_extract(data_dir, url='http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'):
    if not os.path.exists(os.path.join(data_dir, url.split('/')[-1])):
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        filename = url.split('/')[-1]
        filepath = os.path.join(data_dir, filename)
        if not os.path.exists(filepath):
            def _progress(count, block_size, total_size):
                sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
                    float(count * block_size) / float(total_size) * 100.0))
                sys.stdout.flush()
            filepath, _ = urllib.request.urlretrieve(url, filepath, _progress)
            print()
            statinfo = os.stat(filepath)
            print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
            tarfile.open(filepath, 'r:gz').extractall(data_dir)

def unpickle(file):
    fo = open(file, 'rb')
    d = pickle.load(fo)
    fo.close()
    return {'x': np.cast[np.uint8]((np.transpose(d['data'].reshape((d['data'].shape[0],3,32,32)), (0, 2, 3, 1)))),
            'y': np.array(d['labels']).astype(np.uint8)}

def load(data_dir, subset='train'):
    maybe_download_and_extract(data_dir, 'http://www.image-net.org/image/downsample/Imagenet32_val.zip')
    maybe_download_and_extract(data_dir, 'http://www.image-net.org/image/downsample/Imagenet32_train.zip')
    if subset=='train':
        train_data = [unpickle(os.path.join(data_dir,'Imagenet32_train/train_data_batch_{}'.format(i))) for i in range(1,11)]
        trainx = np.concatenate([d['x'] for d in train_data], axis=0)
        trainy = np.concatenate([d['y'] for d in train_data], axis=0)
        return trainx, trainy
    elif subset=='test':
        test_data = unpickle(os.path.join(data_dir,'Imagenet32_val/val_data'))
        testx = test_data['x']
        testy = test_data['y']
        return testx, testy
    else:
        raise NotImplementedError('subset should be either train or test')
