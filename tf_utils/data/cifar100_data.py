import cPickle
import os
import sys
import tarfile
from six.moves import urllib
import numpy as np

def maybe_download_and_extract(data_dir, url='http://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz'):
    if not os.path.exists(os.path.join(data_dir, 'cifar-100-python')):
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
    d = cPickle.load(fo)
    # print d.keys()
    fo.close()
    return {'x': np.cast[np.uint8]((np.transpose(d['data'].reshape((-1,3,32,32)), (0, 2, 3, 1)))), 
            'y': np.array(d['fine_labels']).astype(np.uint8)}

def load(data_dir, subset='train'):
    maybe_download_and_extract(data_dir)
    if subset=='train':
        train_data = unpickle(os.path.join(data_dir,'cifar-100-python/train'))
        trainx = train_data['x']
        trainy = train_data['y']
        return trainx, trainy
    elif subset=='test':
        test_data = unpickle(os.path.join(data_dir,'cifar-100-python/test'))
        testx = test_data['x']
        testy = test_data['y']
        return testx, testy
    else:
        raise NotImplementedError('subset should be either train or test')

if __name__ == '__main__':
    trainx, trainy = load('/home/mlg/ihcho/data', 'train')
    print trainx.shape, trainy.shape, trainx.max(), trainx.min(), trainy.max(), trainy.min()
