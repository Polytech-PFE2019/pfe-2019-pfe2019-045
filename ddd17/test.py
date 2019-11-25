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


def normal(X):
    m = (X-X.min())/(X.max()-X.min())
    return 2* m - 1

def normal2(X):
    m = (X-X.min())/(X.max()-X.min())
    return m

if __name__ == '__main__':
    d = h5py.File(filename, 'r')
    data = d.get('/event')
    print(data[2])
    print(data.shape)
    events = np.array(data, dtype = np.int32)

    # plt.imshow(d['/frame'].value[0], cmap='gray')
    # plt.show()

    d.close()

    frame = np.zeros((260,346), dtype=np.int16)
    mean = np.zeros((260, 346), dtype=np.float16)
    sd = np.zeros((260,346), dtype=np.float16)

    p = 1
    i = 0

    ts = events[:, 0]

    while events[i][0] - events[0][0] <= 50000:
        if events[i][3] == p:
            frame[events[i][2]][events[i][1]] += 1
        i += 1

    j = 0
    ts_norm = normal(ts[:i])
    while events[j][0] - events[0][0] <= 50000:
        if events[j][3] == p:
            mean[events[j][2]][events[j][1]] += ts_norm[j]
        j += 1

    frame_normalize = normal(frame)

    mean_=normal(mean)

    mean_total = np.array([mean[i][j]/frame[i][j] if frame[i][j] != 0 else 0
            for i in range(len(mean)) for j in range(len(mean[0]))]).reshape((260, 346))
    print(mean_total[0][10])
    mean_normalize = normal(mean_total)

    s = 0
    while events[s][0] - events[0][0] <= 50000:
        if events[s][3] == p:
            sd[events[s][2]][events[s][1]] += ((ts_norm[s] - mean_total[events[s][2]][events[s][1]]) ** 2)
        s += 1

    sd_ = [np.sqrt(sd[i][j]/(frame[i][j]-1)) if frame[i][j]-1 != 0 else 0
            for i in range(len(sd)) for j in range(len(sd[0]))]

    # sd_normalize = normal(np.array(sd_).reshape((260, 346)))
    sd_normalize = normal2(np.array(sd_).reshape((260, 346)))

    plt.imshow(sd_normalize, cmap='gray')
    print(np.array(sd_normalize).reshape((260, 346))[0])
    plt.show()

    plt.imshow(mean_normalize, cmap='gray')
    print(mean_normalize)
    plt.show()

    plt.imshow(frame_normalize, cmap='gray')
    print(frame_normalize)
    plt.show()





