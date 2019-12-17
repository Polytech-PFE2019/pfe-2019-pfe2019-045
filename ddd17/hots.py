import h5py
import matplotlib.pyplot as plt
import numpy as np

np.set_printoptions(threshold=np.inf)

filename ='/Volumes/SD/exported_9175.h5'


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
    #np.sqrt(np.sum(np.square(vector1-vector2)))
    return np.linalg.norm(a - b)


def time_context(data, event, r):
    x = event[1]
    y = event[2]
    t = event[0]
    p = event[3]
    tc = np.zeros((data.shape[0], 4), dtype=np.int16)

    for i in range(x-r, x+r):
        for j in range(y-r, y+r):
            print(np.where(data[:,]))



if __name__ == '__main__':
    d = h5py.File(filename, 'r')
    data = d.get('/event')
    print(data[2])
    print(data.shape)
    events = np.array(data, dtype=np.int32)
    d.close()

    p = 1
    k = 5
    r = 2
    R = 2 * r + 1
    t_c = 50

    temp = np.zeros((260+2*r, 346+2*r), dtype=np.int32)
    C = np.zeros((k, R**2), dtype=np.double)
    p = np.ones(k)

    i = k
    data_on = events[np.where(events[:, 3] == 1)]
    feat = np.zeros((data_on.shape), dtype=np.int32)

    for j in range(k):

        temp[data_on[j][2]+r][data_on[j][1]+r] = data_on[j][0]
        t = np.array(temp[(data_on[j][2]): (data_on[j][2]+2*r+1), (data_on[j][1]): (data_on[j][1]+2*r+1)]).flatten()
        C[j] = np.exp((t-data_on[j][0])/t_c)
        feat[j] = data_on[j]
        feat[j][3] = j

    while data_on[i][0] - data_on[0][0] <= 50000:
        temp[data_on[i][2] + r][data_on[i][1] + r] = data_on[i][0]
        t = np.array(
            temp[(data_on[i][2]): (data_on[i][2] + 2 * r + 1), (data_on[i][1]): (data_on[i][1] + 2 * r + 1)]).flatten()
        S = np.exp((t - data_on[j][0]) / t_c)
        dist_min = ed(S, C[0])
        index = 0
        for j in range(1, k):
            dist = ed(S, C[j])
            if dist < dist_min:
                dist_min = dist
                index = j

        feat[i] = data_on[i]
        feat[i][3] = index

        alpha = 0.01/(1+p[index]/20000)
        beta = np.cos(C[j].reshape(R, R), S.reshape(R, R))
        C[index] = C[index]+alpha*(S-beta.flatten()*C[index])

        p[index] += 1
        i += 1

    print(i)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    for j in range(k):
        temp[data_on[j][2] + r][data_on[j][1] + r] = data_on[j][0]
        frame = np.zeros((260, 346), dtype=np.int16)
        cluster = feat[np.where(feat[:, 3] == j)]
        for i in range(cluster.shape[0]):
            frame[cluster[i][2]][cluster[i][1]] = 1
        axes.ravel()[j].imshow(frame, cmap='gray')
        axes.ravel()[j].set_title(str(j))
    fig.tight_layout()
    plt.show()

    k = 8
    i = k
    r = 4
    R = 2 * r + 1
    t_c = 500
    data_on = feat
    feat = np.zeros((data_on.shape), dtype=np.int32)
    temp = np.zeros((260 + 2 * r, 346 + 2 * r), dtype=np.int32)
    C = np.zeros((k, R ** 2), dtype=np.double)
    p = np.ones(k)

    for j in range(k):
        temp[data_on[j][2] + r][data_on[j][1] + r] = data_on[j][0]
        t = np.array(
            temp[(data_on[j][2]): (data_on[j][2] + 2 * r + 1), (data_on[j][1]): (data_on[j][1] + 2 * r + 1)]).flatten()
        C[j] = np.exp((t - data_on[j][0]) / t_c)
        feat[j] = data_on[j]
        feat[j][3] = j

    print(i)
    while data_on[i][0] - data_on[0][0] <= 50000:
        temp[data_on[i][2] + r][data_on[i][1] + r] = data_on[i][0]
        t = np.array(
            temp[(data_on[i][2]): (data_on[i][2] + 2 * r + 1), (data_on[i][1]): (data_on[i][1] + 2 * r + 1)]).flatten()
        S = np.exp((t - data_on[j][0]) / t_c)
        dist_min = ed(S, C[0])
        index = 0
        for j in range(1, k):
            dist = ed(S, C[j])
            if dist < dist_min:
                dist_min = dist
                index = j

        feat[i] = data_on[i]
        feat[i][3] = index

        alpha = 0.01 / (1 + p[index] / 20000)
        beta = np.cos(C[j].reshape(R, R), S.reshape(R, R))
        C[index] = C[index] + alpha * (S - beta.flatten() * C[index])

        p[index] += 1
        i += 1

    fig, axes = plt.subplots(2, 4, figsize=(15, 10))

    for j in range(k):
        temp[data_on[j][2] + r][data_on[j][1] + r] = data_on[j][0]
        frame = np.zeros((260, 346), dtype=np.int16)
        cluster = feat[np.where(feat[:, 3] == j)]
        for i in range(cluster.shape[0]):
            frame[cluster[i][2]][cluster[i][1]] = 1
        axes.ravel()[j].imshow(frame, cmap='gray')
        axes.ravel()[j].set_title(str(j))
    fig.tight_layout()
    plt.show()

    k = 16
    i = k
    r = 8
    R = 2 * r + 1
    t_c = 5000
    data_on = feat
    feat = np.zeros((data_on.shape), dtype=np.int32)
    temp = np.zeros((260 + 2 * r, 346 + 2 * r), dtype=np.int32)
    C = np.zeros((k, R ** 2), dtype=np.double)
    p = np.ones(k)

    for j in range(k):
        t = np.array(
            temp[(data_on[j][2]): (data_on[j][2] + 2 * r + 1), (data_on[j][1]): (data_on[j][1] + 2 * r + 1)]).flatten()
        C[j] = np.exp((t - data_on[j][0]) / t_c)
        feat[j] = data_on[j]
        feat[j][3] = j

    while data_on[i][0] - data_on[0][0] <= 50000:
        temp[data_on[i][2] + r][data_on[i][1] + r] = data_on[i][0]
        t = np.array(
            temp[(data_on[i][2]): (data_on[i][2] + 2 * r + 1), (data_on[i][1]): (data_on[i][1] + 2 * r + 1)]).flatten()
        S = np.exp((t - data_on[j][0]) / t_c)
        dist_min = ed(S, C[0])
        index = 0
        for j in range(1, k):
            dist = ed(S, C[j])
            if dist < dist_min:
                dist_min = dist
                index = j

        feat[i] = data_on[i]
        feat[i][3] = index

        alpha = 0.01 / (1 + p[index] / 20000)
        beta = np.cos(C[j].reshape(R, R), S.reshape(R, R))
        C[index] = C[index] + alpha * (S - beta.flatten() * C[index])

        p[index] += 1
        i += 1

    fig, axes = plt.subplots(2, 8, figsize=(15, 10))

    for j in range(k):
        frame = np.zeros((260, 346), dtype=np.int16)
        cluster = feat[np.where(feat[:, 3] == j)]
        for i in range(cluster.shape[0]):
            frame[cluster[i][2]][cluster[i][1]] = 1
        axes.ravel()[j].imshow(frame, cmap='gray')
        axes.ravel()[j].set_title(str(j))
    fig.tight_layout()
    plt.show()















