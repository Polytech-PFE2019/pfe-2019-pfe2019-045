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


if __name__ == '__main__':
    d = h5py.File(filename, 'r')
    data = d.get('/event')
    # print(data[2])
    print(data.shape)
    events = np.array(data, dtype=np.int32)

    # plt.imshow(d['/frame'].value[0], cmap='gray')
    # plt.show()

    d.close()

    frame = np.zeros((260, 346), dtype=np.int16)
    mean = np.zeros((260, 346), dtype=np.float16)
    sd = np.zeros((260, 346), dtype=np.float16)
    sphis = np.zeros((260, 346), dtype=np.float16)
    sphis_ave = np.zeros((21, 3), dtype=np.float16)
    sphis_cm = np.zeros((21, 3, 3), dtype=np.float16)

    p = 1
    ms_i = 0
    indix_n = 0
    indix_p = 0

    ts = events[:, 0]

    while events[ms_i][0] - events[0][0] <= 50000:
        # if events[ms_i][3] == p:
        #     indix_p += 1
        #     sphis_ave[0][0] = (sphis_ave[0][0]*(indix_p-1) + events[ms_i][2] )/indix_p
        #     sphis_ave[0][1] = (sphis_ave[0][1] * (indix_p - 1) + events[ms_i][1]) / indix_p
        #     sphis_ave[0][2] = (sphis_ave[0][2] * (indix_p - 1) + events[ms_i][3]) / indix_p
        # else:
        #     indix_n += 1
        #     sphis_ave[1][0] = (sphis_ave[1][0] * (indix_n - 1) + events[ms_i][2]) / indix_n
        #     sphis_ave[1][1] = (sphis_ave[1][1] * (indix_n - 1) + events[ms_i][1]) / indix_n
        #     sphis_ave[1][2] = (sphis_ave[1][2] * (indix_n - 1) + events[ms_i][3]) / indix_n
        frame[events[ms_i][2]][events[ms_i][1]] += 1
        ms_i += 1

    j = 0
    ts_norm = normal(ts[:ms_i])
    while events[j][0] - events[0][0] <= 50000:
        #if events[j][3] == p:
        mean[events[j][2]][events[j][1]] += ts_norm[j]
        j += 1

    frame_normalize = normal(frame)

    mean_ = normal(mean)
    # print(mean)

    frame_reshapa = np.array(frame_normalize).reshape((260, 346))

    sphis = [np.floor((frame_reshapa[i][j] + 1) / 0.1) for i in range(mean.shape[0]) for j in range(mean.shape[1])]
    sphis = np.array(sphis).reshape((260, 346))

    test_dic = [0] * 21
    #indix = 0
    for i in range(260):
        for j in range(346):
            #indix += 1
            bin_number = int(sphis[i][j])
            test_dic[bin_number] += 1
            sphis_ave[bin_number][0] = (i + sphis_ave[bin_number][0] * (test_dic[bin_number] - 1)) / test_dic[
                bin_number]
            sphis_ave[bin_number][1] = (j + sphis_ave[bin_number][1] * (test_dic[bin_number] - 1)) / test_dic[
                bin_number]
            sphis_ave[bin_number][2] = (mean[i][j] + sphis_ave[bin_number][2] * (test_dic[bin_number] - 1)) / test_dic[
                bin_number]
    #print(indix)

    for n in range(21):
        if test_dic[n] != 0 and test_dic[n] != 1:
            ma_x = []
            ma_y = []
            ma_t = []
            for i in range(260):
                for j in range(346):
                    if sphis[i][j] == n:
                        ma_x.append(i)
                        ma_y.append(j)
                        ma_t.append(mean[i][j])
            X = np.stack((ma_x, ma_y, ma_t), axis=0)
            result = np.cov(X)
            sphis_cm[n] = result

    # test_dic = [0] * 21
    # for es in range(ms_i):
    #     if events[es][3] == p:
    #         x, y = events[es][2], events[es][1]
    #         bin_number = int(sphis[x][y])
    #         test_dic[bin_number] += 1
    #         sphis_ave[bin_number][2] = (events[es][0] + sphis_ave[bin_number][2] * (test_dic[bin_number] - 1)) / test_dic[bin_number]

    # print(sphis)
    # mean_total = np.array([mean[i][j]/frame[i][j] if frame[i][j] != 0 else 0
    #         for i in range(len(mean)) for j in range(len(mean[0]))]).reshape((260, 346))
    # print(mean_total[0][10])
    # mean_normalize = normal(mean_total)
    #
    # s = 0
    # while events[s][0] - events[0][0] <= 50000:
    #     if events[s][3] == p:
    #         sd[events[s][2]][events[s][1]] += ((ts_norm[s] - mean_total[events[s][2]][events[s][1]]) ** 2)
    #     s += 1
    #
    # sd_ = [np.sqrt(sd[i][j]/(frame[i][j]-1)) if frame[i][j]-1 != 0 else 0
    #         for i in range(len(sd)) for j in range(len(sd[0]))]
    #
    # # sd_normalize = normal(np.array(sd_).reshape((260, 346)))
    # sd_normalize = normal2(np.array(sd_).reshape((260, 346)))
    #
    # plt.imshow(sd_normalize, cmap='gray')
    # print(np.array(sd_normalize).reshape((260, 346))[0])
    # plt.show()
    #
    # plt.imshow(mean_normalize, cmap='gray')
    # print(mean_normalize)
    # plt.show()
    #
    # plt.imshow(frame_normalize, cmap='gray')
    # print(frame_normalize)
    # plt.show()
