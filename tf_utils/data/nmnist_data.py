import cPickle
import os
import sys
import tarfile
from six.moves import urllib
import numpy as np
from PIL import Image

url = 'http://yaroslavvb.com/upload/notMNIST/notMNIST_large.tar.gz'
filename = url.split('/')[-1]
foldername = 'notMNIST_large'

label_dict = {
    'A': 0,
    'B': 1,
    'C': 2,
    'D': 3,
    'E': 4,
    'F': 5,
    'G': 6,
    'H': 7,
    'I': 8,
    'J': 9
}


def maybe_download(data_dir):
    filepath = os.path.join(data_dir, filename)
    if not os.path.isfile(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%'
                    % (filename, float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()
        filepath, _ = urllib.request.urlretrieve(url, filepath, _progress)
        print()
        statinfo = os.stat(filepath)
        print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
        tarfile.open(filepath, 'r:gz').extractall(data_dir)


def load(data_dir, subset='train'):
    maybe_download(data_dir)
    folderpath = os.path.join(data_dir, foldername)

    x = []
    y = []
    for label in os.listdir(folderpath):
        ctr = 0
        x_tmp = []
        for img_name in os.listdir(os.path.join(folderpath, label)):
            img_path = os.path.join(folderpath, label, img_name)
            try:
                img = Image.open(img_path)
                imgfile = np.expand_dims(
                    np.expand_dims(np.array(img), axis=0),
                    axis=3
                )
                ctr += 1
                x_tmp.append(imgfile.astype(np.float32))
            except:
                pass
        x.append(np.vstack(x_tmp))
        y.append(np.zeros((ctr)) + label_dict[label])
        y[-1].astype(np.uint8)

    n = []
    for i in range(10):
        n.append(int(len(x[i]) / 10 * 8))

    if subset == 'train':
        trainx = []
        trainy = []
        for i in range(10):
            trainx.append(x[i][:n[i]])
            trainy.append(y[i][:n[i]])
        return np.concatenate(trainx, axis=0), np.concatenate(trainy, axis=0)
    elif subset == 'test':
        testx = []
        testy = []
        for i in range(10):
            testx.append(x[i][n[i]:])
            testy.append(y[i][n[i]:])
        return np.concatenate(testx, axis=0), np.concatenate(testy, axis=0)
    else:
        raise NotImplementedError('subset should be train or test')


if __name__ == '__main__':
    trainx, trainy = load('/home/mlg/ihcho/data')
    testx, testy = load('/home/mlg/ihcho/data', 'test')
    print trainx.shape, trainy.shape
    print testx.shape, testy.shape

