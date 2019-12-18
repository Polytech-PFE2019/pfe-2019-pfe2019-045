import h5py
import matplotlib.pyplot as plt
import numpy as np
import random

np.set_printoptions(threshold=np.inf)

filename = '/Volumes/SD/exported_9175.h5'


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


def ed(a, b):
    """
    Compute Euclidean Distance
    :param a: vector a
    :param b: vector b
    :return: euclidian distance between a & b
    """
    # np.sqrt(np.sum(np.square(vector1-vector2)))
    return np.linalg.norm(a - b)


def time_context(data, event, r):
    x = event[1]
    y = event[2]
    t = event[0]
    p = event[3]
    tc = np.zeros((data.shape[0], 4), dtype=np.int16)

    for i in range(x - r, x + r):
        for j in range(y - r, y + r):
            print(np.where(data[:, ]))


def model(data, n, k, r, t_c):
    """
    Hierarchical Model from HOTS
    @param data: events or the output from previous layer
    @param n: polarity
    @param k: k cluster
    @param r: radius of neighborhood
    @param t_c: time constant
    @return: clustered events
    """
    R = 2 * r + 1

    temp = np.zeros((260 + 2 * r, 346 + 2 * r), dtype=np.int32)


    feat_c = np.zeros((data.shape), dtype=np.int32)
    size_pre = 0

    for m in range(n):
        data_on = data[np.where(data[:, 3] == m)]
        feat = np.zeros((data_on.shape), dtype=np.int32)
        C = np.zeros((k, R ** 2), dtype=np.double)
        p = np.ones(k)
        print(m)
        print(data_on.shape)
        print(feat.shape)
        for j in range(k):
            rand = random.randint(0, data_on.shape[0])
            temp[data_on[rand][2] + r][data_on[rand][1] + r] = data_on[j][0]
            t = np.array(
                temp[(data_on[j][2]): (data_on[j][2] + 2 * r + 1),
                (data_on[j][1]): (data_on[j][1] + 2 * r + 1)]).flatten()
            C[j] = np.exp((t - data_on[j][0]) / t_c)
            feat[j] = data_on[j]
            feat[j][3] = k * m + j

        for i in range(data_on.shape[0]):
            temp[data_on[i][2] + r][data_on[i][1] + r] = data_on[i][0]
            t = np.array(
                temp[(data_on[i][2]): (data_on[i][2] + 2 * r + 1),
                (data_on[i][1]): (data_on[i][1] + 2 * r + 1)]).flatten()
            S = np.exp((t - data_on[j][0]) / t_c)
            dist_min = ed(S, C[0])
            index = 0
            for j in range(1, k):
                dist = ed(S, C[j])
                if dist < dist_min:
                    dist_min = dist
                    index = j

            feat[i] = data_on[i]
            feat[i][3] = k * m + index

            alpha = 0.01 / (1 + p[index] / 20000)
            beta = np.cos(C[j].reshape(R, R), S.reshape(R, R))
            C[index] = C[index] + alpha * (S - beta.flatten() * C[index])

            p[index] += 1
        feat_c[size_pre:(size_pre + data_on.shape[0])] = feat.copy()
        size_pre = data_on.shape[0]

    fig, axes = plt.subplots(n, k, figsize=(15, 10))

    for j in range(k * n):
        temp[data_on[j][2] + r][data_on[j][1] + r] = data_on[j][0]
        frame = np.zeros((260, 346), dtype=np.int16)
        cluster = feat_c[np.where(feat_c[:, 3] == j)]
        for i in range(cluster.shape[0]):
            frame[cluster[i][2]][cluster[i][1]] = 1
        axes.ravel()[j].imshow(frame, cmap='gray')
        axes.ravel()[j].set_title(str(j))
        print("cluster ", j)
        print(cluster.shape[0])
    fig.tight_layout()
    plt.show()
    return feat_c


if __name__ == '__main__':
    d = h5py.File(filename, 'r')
    data = d.get('/event')
    print(data[2])
    print(data.shape)
    events = np.array(data, dtype=np.int32)
    d.close()

    events = events[np.where(events[:, 0] - events[0, 0] <= 50000)]
    print(events.shape)

    k = 2
    r = 5
    t_c = 50

    out = model(events, 2, k, r, t_c).copy()

    r = 8
    t_c = 500

    out = model(out, 4, k, r, t_c).copy()

    r = 80
    t_c = 5000

    out = model(out, 8, k, r, t_c).copy()