import h5py
import matplotlib.pyplot as plt
import numpy as np

np.set_printoptions(threshold=np.inf)

filename = '/Users/wangy/Downloads/exported2.hdf5'


def traverse_datasets(hdf_file):
    def h5py_dataset_iterator(g, prefix=''):
        for key in g.keys():
            item = g[key]
            path = f'{prefix}/{key}'
            # print('  {}, Min: {}, Mean: {}, Max: {}, size: {}'.format(key, np.min(data), np.mean(data), np.max(data), dataset[key].shape))
            if isinstance(item, h5py.Dataset):  # test for dataset
                yield (path, item)
            elif isinstance(item, h5py.Group):  # test for group (go down)
                yield from h5py_dataset_iterator(item, path)

    with h5py.File(hdf_file, 'r') as f:
        for (path, dset) in h5py_dataset_iterator(f):
            print(path, dset)


def normal(X):
    m = (X - X.min()) / (X.max() - X.min())
    return 2 * m - 1


def normal2(X):
    m = (X - X.min()) / (X.max() - X.min())
    return m

def binData(ave, event, po, index, t_s):
    ave[po][0] = (ave[po][0] * (index - 1) + event[2]) / index
    ave[po][1] = (ave[po][1] * (index - 1) + event[1]) / index
    #ave[po][2] = (ave[po][2] * (index - 1) + event[0] - t_s) / index


if __name__ == '__main__':
    d = h5py.File(filename, 'r')
    data = d.get('/event')
    # print(data[2])
    print(data.shape)
    events = np.array(data, dtype=np.int32)
    d.close()

    sphis_ave = np.zeros((2, 3), dtype=np.float16)
    sphis_cm = np.zeros((2, 3, 3), dtype=np.float16)

    p = 1
    ms_i = 0
    indix_n = 0
    indix_p = 0

    ts = events[:, 0]

    time_stand = events[0][0]
    while events[ms_i][0] - time_stand <= 50000:
        if events[ms_i][3] == p:
            indix_p += 1
            binData(sphis_ave, events[ms_i], 1, indix_p, time_stand)
        else:
            indix_n += 1
            binData(sphis_ave, events[ms_i], 0, indix_n, time_stand)
        ms_i += 1

    ts_norm = normal(ts[:ms_i])

    for i in range(2):
        ma_x, ma_y, ma_t  = [], [], []
        for j in range(ms_i):
            if events[j][3] == i:
                ma_x.append(events[j][2])
                ma_y.append(events[j][1])
                ma_t.append(ts_norm[j])
        X = np.stack((ma_x, ma_y, ma_t), axis=0)
        result = np.cov(X)
        sphis_cm[i] = result
        sphis_ave[i][2] = np.average(ma_t)