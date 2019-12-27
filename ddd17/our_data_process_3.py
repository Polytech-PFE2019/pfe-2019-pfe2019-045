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

    list=['/Volumes/SD/9175/9175_5717.npy',
'/Volumes/SD/9175/9175_1936.npy',
'/Volumes/SD/9175/9175_5490.npy',
'/Volumes/SD/9175/9175_1892.npy',
'/Volumes/SD/9175/9175_5476.npy',
'/Volumes/SD/9175/9175_5716.npy',
'/Volumes/SD/9175/9175_1909.npy',
'/Volumes/SD/9175/9175_5680.npy',
'/Volumes/SD/9175/9175_1953.npy',
'/Volumes/SD/9175/9175_5671.npy',
'/Volumes/SD/9175/9175_5486.npy',
'/Volumes/SD/9175/9175_1916.npy',
'/Volumes/SD/9175/9175_5681.npy',
'/Volumes/SD/9175/9175_1957.npy',
'/Volumes/SD/9175/9175_5479.npy',
'/Volumes/SD/9175/9175_1949.npy',
'/Volumes/SD/9175/9175_1894.npy',
'/Volumes/SD/9175/9175_5688.npy',
'/Volumes/SD/9175/9175_1895.npy',
'/Volumes/SD/9175/9175_5703.npy',
'/Volumes/SD/9175/9175_5475.npy',
'/Volumes/SD/9175/9175_5492.npy',
'/Volumes/SD/9175/9175_1963.npy',
'/Volumes/SD/9175/9175_5672.npy',
'/Volumes/SD/9175/9175_1923.npy',
'/Volumes/SD/9175/9175_1947.npy',
'/Volumes/SD/9175/9175_5720.npy',
'/Volumes/SD/9175/9175_1961.npy',
'/Volumes/SD/9175/9175_5658.npy',
'/Volumes/SD/9175/9175_1919.npy',
'/Volumes/SD/9175/9175_1965.npy',
'/Volumes/SD/9175/9175_5480.npy',
'/Volumes/SD/9175/9175_5710.npy',
'/Volumes/SD/9175/9175_5769.npy',
'/Volumes/SD/9175/9175_5691.npy',
'/Volumes/SD/9175/9175_5638.npy',
'/Volumes/SD/9175/9175_1922.npy',
'/Volumes/SD/9175/9175_1896.npy',
'/Volumes/SD/9175/9175_1959.npy',
'/Volumes/SD/9175/9175_5699.npy',
'/Volumes/SD/9175/9175_5472.npy',
'/Volumes/SD/9175/9175_5668.npy',
'/Volumes/SD/9175/9175_1933.npy',
'/Volumes/SD/9175/9175_1964.npy',
'/Volumes/SD/9175/9175_1929.npy',
'/Volumes/SD/9175/9175_5488.npy',
'/Volumes/SD/9175/9175_5644.npy',
'/Volumes/SD/9175/9175_1956.npy',
'/Volumes/SD/9175/9175_5642.npy',
'/Volumes/SD/9175/9175_5713.npy',
'/Volumes/SD/9175/9175_5643.npy',
'/Volumes/SD/9175/9175_5692.npy',
'/Volumes/SD/9175/9175_5689.npy',
'/Volumes/SD/9175/9175_5610.npy',
'/Volumes/SD/9175/9175_5487.npy',
'/Volumes/SD/9175/9175_5702.npy',
'/Volumes/SD/9175/9175_1937.npy',
'/Volumes/SD/9175/9175_5663.npy',
'/Volumes/SD/9175/9175_1914.npy',
'/Volumes/SD/9175/9175_5711.npy',
'/Volumes/SD/9175/9175_5660.npy',
'/Volumes/SD/9175/9175_1908.npy',
'/Volumes/SD/9175/9175_1945.npy',
'/Volumes/SD/9175/9175_1939.npy',
'/Volumes/SD/9175/9175_5771.npy',
'/Volumes/SD/9175/9175_1913.npy',
'/Volumes/SD/9175/9175_5482.npy',
'/Volumes/SD/9175/9175_5666.npy',
'/Volumes/SD/9175/9175_1934.npy',
'/Volumes/SD/9175/9175_1935.npy',
'/Volumes/SD/9175/9175_5602.npy',
'/Volumes/SD/9175/9175_5489.npy',
'/Volumes/SD/9175/9175_5670.npy',
'/Volumes/SD/9175/9175_1921.npy',
'/Volumes/SD/9175/9175_5767.npy',
'/Volumes/SD/9175/9175_5648.npy',
'/Volumes/SD/9175/9175_5682.npy',
'/Volumes/SD/9175/9175_1944.npy',
'/Volumes/SD/9175/9175_5669.npy',
'/Volumes/SD/9175/9175_1906.npy',
'/Volumes/SD/9175/9175_5607.npy',
'/Volumes/SD/9175/9175_1943.npy',
'/Volumes/SD/9175/9175_1954.npy',
'/Volumes/SD/9175/9175_1901.npy',
'/Volumes/SD/9175/9175_1903.npy']
    for name in list:
        print(name)
        b = np.load(name)


        events = b
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

        sd_ = [np.sqrt(sd[i][j] / (hist[i][j] - 1)) if hist[i][j] - 1 != 0 else 1 for i in range(len(sd)) for j in
               range(len(sd[0]))]
        sd_normalize = normal2(np.array(sd_).reshape((260, 346)))
        time_surface_normal = normal(time_surface)
        lbp = normal(lbp)

        out_put_1 = np.asarray([lbp, time_surface_normal, sd_normalize])
        out_put_1 = out_put_1.transpose(1, 2, 0)


        export = np.asarray([out_put_1[:, :, 0], out_put_1[:, :, 1], out_put_1[:, :, 2]])# lbp-, ts-, sd-, lbp+, ts+, sd+
        export = export.transpose(1, 2, 0)
        export = np.around(export, decimals=4)
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





