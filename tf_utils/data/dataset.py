import numpy as np


def scale_to_unit_interval(ndar, eps=1e-8):
    """ Scales all values in the ndarray ndar to be between 0 and 1 """
    ndar = ndar.copy()
    ndar -= ndar.min()
    scale = 1.0 / (ndar.max() + eps)
    ndar = ndar * scale
    return ndar


class DataSet(object):
    def __init__(self, data, data_dir, normalise=True):
        self.data = data
        self.data_dir = data_dir

        self.train_data  = None
        self.train_label = None
        self.test_data   = None
        self.test_label  = None
        self.valid_data  = None
        self.valid_label = None
        self.n_train     = 0
        self.n_test      = 0
        self.n_valid     = 0
        self.lable_class = None
        self.n_label     = None
        self.data_shape  = None

        self._load_data()

        if normalise:
            self._normalise()

    def _load_data(self):
        if self.data == 'mnist':
            import mnist_data
            self.train_data, self.train_label = mnist_data.load(self.data_dir, 'train')
            self.test_data, self.test_label = mnist_data.load(self.data_dir, 'test')
            self.valid_data, self.valid_label = mnist_data.load(self.data_dir, 'valid')
            self.train_data = np.expand_dims(self.train_data, 3)
            self.test_data = np.expand_dims(self.test_data, 3)
            self.valid_data = np.expand_dims(self.valid_data, 3)
        elif self.data == 'nmist':
            import nmnist_data
            self.train_data, self.train_label = nmnist_data.load(self.data_dir, 'train')
            self.test_data, self.test_label = nmnist_data.load(self.data_dir, 'test')
        elif self.data == 'cifar10':
            import cifar10_data
            self.train_data, self.train_label = cifar10_data.load(self.data_dir, 'train')
            self.test_data, self.test_label = cifar10_data.load(self.data_dir, 'test')
        elif self.data == 'mog':
            n_mixture = 8
            std       = 0.01
            radius    = 1.0
            self.train_data, self.train_label = self.sample_mog(
                50000, n_mixture, std, radius)
            self.test_data, self.test_label = self.sample_mog(
                10000, n_mixture, std, radius)
            self.valid_data, self.valid_label = self.sample_mog(
                10000, n_mixture, std, radius)
        else:
            raise ValueError('data argument must be in range [mnist, cifar10, mog]')

        if self.train_data is not None:
            self.n_train = len(self.train_data)
        if self.test_data is not None:
            self.n_test = len(self.test_data)
        if self.valid_data is not None:
            self.n_valid = len(self.valid_data)

        self.label_class = np.unique(self.train_label)
        self.n_label = len(self.label_class)

        self.data_shape = self.train_data.shape[1:]

    def _normalise(self):
        if self.n_train != 0:
            self.train_data = scale_to_unit_interval(self.train_data)
        if self.n_test != 0:
            self.test_data = scale_to_unit_interval(self.test_data)
        if self.n_valid != 0:
            self.valid_data = scale_to_unit_interval(self.valid_data)

    def sample_mog(self, size, n_mixture=8, std=0.01, radius=1.0):
        thetas = np.linspace(0, 2 * np.pi, n_mixture + 1)
        xs, ys = radius * np.sin(thetas[:-1]), radius * np.cos(thetas[:-1])
        cat = np.random.choice(n_mixture, size)

        data  = []
        label = []
        for i in range(size):
            xi, yi = xs[cat[i]], ys[cat[i]]
            compas = np.random.multivariate_normal([xi, yi], np.eye(2) * std)
            data.append(np.reshape(compas, (1, 2)))
            label.append(cat[i])

        return np.concatenate(data), np.array(label)

    def iter(self, batch_size, request_label=None, which='train'):
        """ A simple data iterator """
        data, label = self._data_selection(which)

        if request_label is None:
            request_label = self.label_class

        # error check
        ctr = 0
        for i in request_label:
            for j in self.label_class:
                if i == j:
                    ctr += 1
                    break
        if ctr != len(request_label):
            raise ValueError('Incorrect requested label {}'.format(request_label))

        idx = []
        for i in request_label:
            idx.append(np.where(label == i)[0])
        idx = np.concatenate(idx)
        data = data[idx]
        label = label[idx]

        batch_idx = 0
        idxs = np.arange(0, len(data))
        np.random.shuffle(idxs)
        for batch_idx in range(0, len(data), batch_size):
            cur_idxs = idxs[batch_idx:batch_idx + batch_size]
            data_batch = data[cur_idxs]
            label_batch = label[cur_idxs]
            # print data_batch.shape, label_batch.shape
            yield data_batch, label_batch

    def get_num_data(self, request_label=None, which='train'):
        data, label = self._data_selection(which)

        if request_label is None:
            request_label = self.label_class

        # error check
        ctr = 0
        for i in request_label:
            for j in self.label_class:
                if i == j:
                    ctr += 1
                    break
        if ctr != len(request_label):
            raise ValueError('Incorrect requested label {}'.format(request_label))

        idx = []
        for i in request_label:
            idx.append(np.where(label == i)[0])
        idx = np.concatenate(idx)
        return len(idx)

    def get_data(self, batch_size, request_label=None, which='train'):
        """ A simple data iterator """
        data, label = self._data_selection(which)

        if request_label is None:
            request_label = self.label_class

        # error check
        ctr = 0
        for i in request_label:
            for j in self.label_class:
                if i == j:
                    ctr += 1
                    break
        if ctr != len(request_label):
            raise ValueError('Incorrect requested label {}'.format(request_label))

        idx = []
        for i in request_label:
            idx.append(np.where(label == i)[0])
        idx = np.concatenate(idx)
        data = data[idx]
        label = label[idx]

        idx = np.arange(len(data))
        np.random.shuffle(idx)
        data = data[idx]
        label = label[idx]

        return data[0:batch_size], label[0:batch_size]

    def _divide_data(self, ys_list, n=5, which='train'):
        data, label = self._data_selection(which)

        first_data = None
        first_label = None
        second_data = None
        second_label = None
        for i in range(self.n_label):
            idx = np.where(label == ys_list[i])[0]
            if i < n:
                if first_data is None:
                    first_data = data[idx]
                    first_label = label[idx]
                else:
                    first_data = np.vstack((first_data, data[idx]))
                    first_label = np.concatenate((first_label, label[idx]), axis=0)
            else:
                if second_data is None:
                    second_data = data[idx]
                    second_label = label[idx]
                else:
                    second_data = np.vstack((second_data, data[idx]))
                    second_label = np.concatenate((second_label, label[idx]), axis=0)

        return first_data, first_label, second_data, second_label

    def _data_selection(self, which):
        if which == 'train':
            data = self.train_data
            label = self.train_label
        elif which == 'test':
            data = self.test_data
            label = self.test_label
        elif which == 'valid':
            data = self.valid_data
            label = self.valid_label
        else:
            print('Dataset error: There is no such {} in this data set'.format(which))
            data = None
            label = None
        return data, label

    def divide_data(self, shuffle=False, n=5):
        ys_list = self.label_class.copy()

        if shuffle:
            np.random.shuffle(ys_list)

        known_data, known_label, _, _ = \
            self._divide_data(ys_list, n, 'train')

        test_data, test_label, unknown_data, unknown_label = \
            self._divide_data(ys_list, n, 'test')

        self.train_data  = known_data
        self.train_label = known_label
        self.test_data   = test_data
        self.test_label  = test_label
        self.valid_data  = unknown_data
        self.valid_label = unknown_label

        self.n_train = len(self.train_data)
        self.n_test = len(self.test_data)
        self.n_valid = len(self.valid_data)


if __name__ == '__main__':
    dataset = DataSet('mnist', '/home/mlg/ihcho/data', True)
    # iter = dataset.iter(100, [0, 1, 2, 3, 4], which='train')
    # data, label = iter.next()

    # aa = []
    # for k in range(10):
        # aa.append(np.sum(label == k))
    # print aa

    # data, label = dataset.get_data(100, [0, 1, 2, 3, 4], which='test')
    # aa = []
    # for k in range(10):
        # aa.append(np.sum(label == k))
    # print aa
    
    # dataset = DataSet('mog', '/home/mlg/ihcho/data', False)
    # data, label = dataset.get_data(1000, [3, 4, 7])
    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots(figsize=(10, 6))
    # c = ['r', 'g', 'b', 'k', 'm', 'y', 'coral', 'cyan']
    # for i in range(8):
        # ax.scatter(data[label == i, 0], data[label == i, 1], color=c[i])
    # plt.show()
