import numpy as np

def data_label_iterate(data, label, batch_size):
    """ A simple data iterator """
    batch_idx = 0
    while True:
        idxs = np.arange(0, len(data))
        np.random.shuffle(idxs)
        for batch_idx in range(0, len(data), batch_size):
            cur_idxs = idxs[batch_idx:batch_idx+batch_size]
            data_batch = data[cur_idxs]
            label_batch = label[cur_idxs]
            yield data_batch, label_batch

def data_iterate(data, batch_size):
    """ A simple data iterator """
    batch_idx = 0
    while True:
        idxs = np.arange(0, len(data))
        np.random.shuffle(idxs)
        for batch_idx in range(0, len(data), batch_size):
            cur_idxs = idxs[batch_idx:batch_idx+batch_size]
            data_batch = data[cur_idxs]
            yield data_batch
