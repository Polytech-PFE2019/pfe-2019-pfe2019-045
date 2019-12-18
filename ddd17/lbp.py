import h5py
import numpy as np
import matplotlib.pyplot as plt

from skimage import feature

# from ddd17.localbinarypatterns import LocalBinaryPatterns

np.set_printoptions(threshold=np.inf)

filename = '/Users/wangy/Downloads/exported2.hdf5'

def change_normal(x):
    x[np.where(x[:,:] == 8)] = 9
    x[np.where(x[:, :] == 0)] = 8
    x[np.where(x[:, :] == 9)] = 0
    return x



def describe(image,numPoints,  radius, eps=1e-7):
    # compute the Local Binary Pattern representation
    # of the image, and then use the LBP representation
    # to build the histogram of patterns
    lbp = feature.local_binary_pattern(image,  numPoints,
                                        radius, method="uniform")

    change_normal(lbp)

    # plt.imshow(lbp)
    # plt.show()

    (hist, _) = np.histogram(lbp.ravel(),
                             bins=np.arange(0, numPoints + 3),
                             range=(0,  numPoints + 2))

    # normalize the histogram
    hist = hist.astype("float")
    hist /= (hist.sum() + eps)

    # return the histogram of Local Binary Patterns
    return hist, lbp

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

if __name__ == '__main__':
    d = h5py.File(filename, 'r')
    data = d.get('/event')
    # print(data[2])
    print(data.shape)
    events = np.array(data, dtype=np.int32)

    d.close()

    ms_i = 0
    fp = np.zeros((260, 346), dtype=np.int16)
    fn = np.zeros((260, 346), dtype=np.int16)
    fram_p = np.zeros((260, 346), dtype=np.int16)
    fram_n = np.zeros((260, 346), dtype=np.int16)
    time_stand = events[0][0]

    while events[ms_i][0] - time_stand <= 50000:
        if events[ms_i][3] == 1:
            fp[events[ms_i][2]][events[ms_i][1]] += 1
            fram_p[events[ms_i][2]][events[ms_i][1]] = 1
        else:
            fn[events[ms_i][2]][events[ms_i][1]] += 1
            fram_n[events[ms_i][2]][events[ms_i][1]] = -1
        ms_i += 1

    hist_p, lbp_p =  describe(fp,8, 2)
    hist_n, lbp_n =  describe(fn,8, 2)

    fig, axes = plt.subplots(2,2)
    axes.ravel()[0].bar(range(len(hist_p)), hist_p)
    axes.ravel()[0].set_title('ON')

    axes.ravel()[1].bar(range(len(hist_n)), hist_n)
    axes.ravel()[1].set_title('Off')


    axes.ravel()[2].imshow(lbp_p)
    axes.ravel()[2].set_title('ON')

    axes.ravel()[3].imshow(lbp_n)
    axes.ravel()[3].set_title('Off')

    fig.tight_layout()
    #plt.bar(range(len(hist_p)), hist_p)
    plt.show()