import tf_utils.inception_score as inception_score
from tf_utils.data.dataset import DataSet
import tf_utils.common as common

import numpy as np

data_set = DataSet('cifar10', '/home/mlg/ihcho/data', normalise='tanh')
test_data, _ = data_set.get_data(50000, which='train')
test_data = common.img_stretch(test_data)
test_data *= 255.0

gen_data = np.load('gen_data.npy')

test_list = []
gen_list = []
for i in range(50000):
    test_list.append(test_data[i])
    gen_list.append(gen_data[i])

real = inception_score.get_inception_score(test_list)
print '\n'
fake = inception_score.get_inception_score(gen_list)
print '\n'
print 'real inception: {}'.format(real)
print 'fake inception: {}'.format(fake)
