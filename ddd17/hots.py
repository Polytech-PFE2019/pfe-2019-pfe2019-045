import h5py
import matplotlib.pyplot as plt
import numpy as np
import random
from PIL import Image
from mpl_toolkits.mplot3d import Axes3D


np.set_printoptions(threshold=np.inf)

filename = '/Volumes/SD/our_data/exported_9175.h5'



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


def normal(X):
    m = (X - X.min()) / (X.max() - X.min())
    return 2 * m - 1


def normal2(X):
    m = (X - X.min()) / (X.max() - X.min())
    return m


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

    # fig, axes = plt.subplots(n, k, figsize=(15, 10))
    #
    # for j in range(k * n):
    #     temp[data_on[j][2] + r][data_on[j][1] + r] = data_on[j][0]
    #     frame = np.zeros((260, 346), dtype=np.int16)
    #     cluster = feat_c[np.where(feat_c[:, 3] == j)]
    #     for i in range(cluster.shape[0]):
    #         frame[cluster[i][2]][cluster[i][1]] = 1
    #     axes.ravel()[j].imshow(frame, cmap='gray')
    #     axes.ravel()[j].set_title(str(j))
    #     print("cluster ", j)
    #     print(cluster.shape[0])
    # fig.tight_layout()
    plt.show()
    return feat_c


if __name__ == '__main__':
    # im = Image.open('data/ground_truth/rec1487339175_export_0.png')
    # im_array = np.asarray(im)
    # print(im_array.shape)
    # im_array = im_array * 40
    # plt.imshow(im_array)
    # plt.show()

    b = np.load("/Volumes/SD/our_data/6842_1954.npy")

    # for i in b:
    #     print(i[0])

    events = b[np.where(b[:, 3] == 0)]
    ts = events[:, 0]
    time_surface = np.zeros((260, 346), dtype=np.double)
    total_nb = np.zeros((260, 346), dtype=np.int32)
    hist = np.zeros((260, 346), dtype=np.double)
    mean = np.zeros((260, 346), dtype=np.double)

    t_c = 5000
    r = 3

    for i in events:
        data = events[np.where(events[:, 0] - i[0] <= 0)]
        data = data[np.where(data[:, 0] - i[0] >= - t_c)]
        data = data[np.where(data[:, 1] - i[1] >= -r)]
        data = data[np.where(data[:, 1] - i[1] <= r)]
        data = data[np.where(data[:, 2] - i[2] >= -r)]
        data = data[np.where(data[:, 2] - i[2] <= r)]
        small_time_mean = np.zeros((2 * r + 1, 2 * r + 1), dtype=np.double)
        small_nb = np.zeros((2 * r + 1, 2 * r + 1), dtype=np.int32)
        x = i[1]
        y = i[2]
        t = i[0]
        m = (t - ts.min()) / (ts.max() - ts.min())
        t_norm = 2 * m - 1
        t_s = 0
        for k in range(data.shape[0]):
            j = data[data.shape[0] - k - 1]
            x_j = j[1] - x + r
            y_j = j[2] - y + r
            t_j = j[0]
            if small_nb[y_j][x_j] > 6:
                small_time_mean[y_j][x_j] = (small_time_mean[y_j][x_j] * small_nb[y_j][x_j]) / (small_nb[y_j][x_j] + 1)
            else:
                small_time_mean[y_j][x_j] = (small_time_mean[y_j][x_j] * small_nb[y_j][x_j] + t_j) / (small_nb[y_j][x_j] + 1)
            small_nb[y_j][x_j] += 1
        count = 0
        for j in small_time_mean.flatten():
            if j != 0:
                t_s += np.exp((j - t) / t_c)
                count += 1
        if count - small_nb[r][r] == 0 and (small_nb[r][r] > 6 or t_norm > 0.5): #total_nb[y][x] > 6 and and total_nb[y][x] != 0
            time_surface[y][x] = (time_surface[y][x] * total_nb[y][x]) / (small_nb[r][r] + total_nb[y][x])

        else:

            time_surface[y][x] = (time_surface[y][x] * total_nb[y][x] + t_s * t_norm / count) / (small_nb[r][r] + total_nb[y][x])
        total_nb[y][x] += small_nb[r][r]
        hist[y][x] += 1
        mean[y][x] += t_norm

    mean_total = np.array([mean[i][j] / hist[i][j] if hist[i][j] != 0 else 0
                           for i in range(len(mean)) for j in range(len(mean[0]))]).reshape((260, 346))

    sd = np.zeros((260, 346), dtype=np.float16)
    for i in events:
        m = (i[0] - ts.min()) / (ts.max() - ts.min())
        t_norm = 2 * m - 1
        sd[i[2]][i[1]] += ((t_norm - mean_total[i[2]][i[1]]) ** 2)

    sd_ = [np.sqrt(sd[i][j] / (hist[i][j] - 1)) if hist[i][j] - 1 != 0 else 0for i in range(len(sd)) for j in range(len(sd[0]))]
    # sd_normalize = normal(np.array(sd_).reshape((260, 346)))
    sd_normalize = normal2(np.array(sd_).reshape((260, 346)))
    frame_normalize = normal(hist)
    mean_norm = normal(mean_total)

    plt.imshow(mean_norm, cmap='gray')
    plt.show()
    time_surface_normal = normal(time_surface)
    # mean = np.mean(time_surface_normal)
    # time_surface_normal = time_surface_normal - mean
    # time_surface_normal = time_surface_normal * 2 + mean * 0.8
    plt.imshow(time_surface_normal, cmap='gray')
    plt.show()
    c = np.load('/Volumes/Untitled/dataset_our_codification/events/train/rec1487846842_export_1954.npy')
    d = c[:,:,5]
    mn = c[:,:,4]
    hn = c[:,:,1]
    test = np.array([c[:,:,0], c[:,:,1], c[:,:,2], c[:, :, 3], c[:, :, 4], c[:, :, 5]])
    test = test.transpose(1,2,0)
    print(test.shape)
    print(c.shape)
    print((test == c).all())







    # for i in range(int(50000/5000)):
    #     if i == 0:
    #         floor = 0
    #     else:
    #         temp = np.where(b[:, 0] - b[0, 0] > 5000 * i)
    #         floor = temp[0][0]
    #     temp = np.where(b[:, 0] - b[0, 0] > 5000 * (i + 1))
    #     upper = temp[0][0]
    #     events = b[floor:upper]
    #     events = events[np.where(events[:, 3] == 0)]
    #     events = events[np.where(events[:, 3] == 0)]
    #
    #     frame = np.zeros((260, 346), dtype=np.int32)
    #     for j in events:
    #         frame[j[2]][j[1]] += 1
    #
    #     print("finish")






    # d = h5py.File(filename, 'r')
    # data = d.get('/event')
    # print(data[2])
    # print(data.shape)
    # events = np.array(data, dtype=np.int32)
    # d.close()
    #
    # events = events[np.where((events[:, 1] - 70).any() and (events[:, 2] - 51).any())]
    # print(events.shape)
    #
    # events = events[np.where(events[:, 0] - events[0, 0] <= 50000)]
    # print(events.shape)

    # k = 2
    # r = 5
    # t_c = 50
    #
    # out = model(b, 2, k, r, t_c).copy()

    #
    # r = 8
    # t_c = 500
    #
    # out = model(out, 4, k, r, t_c).copy()
    #
    # r = 80
    # t_c = 5000
    #
    # out = model(out, 8, k, r, t_c).copy()
