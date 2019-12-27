import glob
import numpy as np
from skimage import feature


def normal(X):
    m = (X - X.min()) / (X.max() - X.min())
    return 2 * m - 1


def normal2(X):
    m = (X - X.min()) / (X.max() - X.min())
    return m


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

    # return the histogram of Local Binary Patterns
    return lbp

if __name__ == '__main__':

    for name in glob.glob('/Volumes/SD/9465/*'):
        print(name)
        b = np.load(name)

        for p in range(2):
            events = b[np.where(b[:, 3] == p)]
            ts = events[:, 0]
            time_surface = np.zeros((260, 346), dtype=np.double)
            total_nb = np.zeros((260, 346), dtype=np.int32)
            hist = np.zeros((260, 346), dtype=np.double)
            mean = np.zeros((260, 346), dtype=np.double)
            out_put_1 = np.zeros((260, 346, 3), dtype=np.double)

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
                        small_time_mean[y_j][x_j] = (small_time_mean[y_j][x_j] * small_nb[y_j][x_j]) / (
                        small_nb[y_j][x_j] + 1)
                    else:
                        small_time_mean[y_j][x_j] = (small_time_mean[y_j][x_j] * small_nb[y_j][x_j] + t_j) / (
                        small_nb[y_j][x_j] + 1)
                    small_nb[y_j][x_j] += 1
                count = 0
                for j in small_time_mean.flatten():
                    if j != 0:
                        t_s += np.exp((j - t) / t_c)
                        count += 1
                if count - small_nb[r][r] == 0 and (
                        small_nb[r][r] > 6 or t_norm > 0.5):  # total_nb[y][x] > 6 and and total_nb[y][x] != 0
                    time_surface[y][x] = (time_surface[y][x] * total_nb[y][x]) / (small_nb[r][r] + total_nb[y][x])

                else:

                    time_surface[y][x] = (time_surface[y][x] * total_nb[y][x] + t_s * t_norm / count) / (
                    small_nb[r][r] + total_nb[y][x])
                total_nb[y][x] += small_nb[r][r]
                hist[y][x] += 1
                mean[y][x] += t_norm

            lbp = describe(hist, 8, 2)
            mean_total = np.array([mean[i][j] / hist[i][j] if hist[i][j] != 0 else 0
                                   for i in range(len(mean)) for j in range(len(mean[0]))]).reshape((260, 346))

            sd = np.zeros((260, 346), dtype=np.float16)
            for i in events:
                m = (i[0] - ts.min()) / (ts.max() - ts.min())
                t_norm = 2 * m - 1
                sd[i[2]][i[1]] += ((t_norm - mean_total[i[2]][i[1]]) ** 2)

            sd_ = [np.sqrt(sd[i][j] / (hist[i][j] - 1)) if hist[i][j] - 1 != 0 else 0 for i in range(len(sd)) for j in
                   range(len(sd[0]))]
            sd_normalize = normal2(np.array(sd_).reshape((260, 346)))
            time_surface_normal = normal(time_surface)
            lbp = normal(lbp)

            out_put_1 = np.asarray([lbp, time_surface_normal, sd_normalize])
            out_put_1 = out_put_1.transpose(1, 2, 0)
            if p == 0:
                out = out_put_1.copy()

        export = np.asarray([out[:, :, 0], out[:, :, 1], out[:, :, 2], out_put_1[:, :, 0], out_put_1[:, :, 1], out_put_1[:, :, 2]])# lbp-, ts-, sd-, lbp+, ts+, sd+
        export = export.transpose(1, 2, 0)
        out_prefix = '/Volumes/SD/data_our/events/train/'
        name = name.split('/')[-1]
        out_postfix = '_export_' + name.split('_')[-1]
        name = name.split('_')[0]
        if name == '9175':
            name = 'rec1487339175'
        if name == '2276':
            name = 'rec1487842276'
        if name == '3224':
            name = 'rec1487593224'
        if name == '6842':
            name = 'rec1487846842'
        if name == '9465':
            name = 'rec1487779465'
        if name == '7411':
            name = 'rec1487417411'
        name = out_prefix + name + out_postfix
        print(name)
        # out_name = out_prefix
        # out_name +=
        # out_name += ".npy"
        np.save(name, export)





