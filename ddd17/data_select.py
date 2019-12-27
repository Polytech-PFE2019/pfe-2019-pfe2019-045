import h5py
import numpy as np

np.set_printoptions(threshold=np.inf)
# /Volumes/SD/
filename = '/Volumes/SD/exported_9175.h5'
out_prefix = '/Volumes/SD/9175/9175_'



if __name__ == '__main__':
    d = h5py.File(filename, 'r')
    data = d.get('/event')
    # print(data[2])
    print(data.shape)
    events = np.array(data, dtype=np.int32)

    # plt.imshow(d['/frame'].value[0], cmap='gray')
    # plt.show()

    d.close()



    for i in range(0, 2000):
        temp = np.where(events[:, 0] - events[0, 0] > 50000 * i)
        floor = temp[0][0]
        temp = np.where(events[:, 0] - events[0, 0] > 50000 * (i + 1))
        upper = temp[0][0]
        out_name = out_prefix
        out_name += str(i)
        out_name += ".npy"
        print(out_name)
        np.save(out_name, events[floor:upper])

    for i in range(5200, 5975):
        temp = np.where(events[:, 0] - events[0, 0] > 50000 * i)
        floor = temp[0][0]
        temp = np.where(events[:, 0] - events[0, 0] > 50000 * (i + 1))
        upper = temp[0][0]
        out_name = out_prefix
        out_name += str(i)
        out_name += ".npy"
        print(out_name)
        np.save(out_name, events[floor:upper])

    # for i in range(8400, 8515):
    #     temp = np.where(events[:, 0] - events[0, 0] > 50000 * i)
    #     floor = temp[0][0]
    #     temp = np.where(events[:, 0] - events[0, 0] > 50000 * (i + 1))
    #     upper = temp[0][0]
    #     out_name = out_prefix
    #     out_name += str(i)
    #     out_name += ".npy"
    #     print(out_name)
    #     np.save(out_name, events[floor:upper])
    #
    # for i in range(8800, 8980):
    #     temp = np.where(events[:, 0] - events[0, 0] > 50000 * i)
    #     floor = temp[0][0]
    #     temp = np.where(events[:, 0] - events[0, 0] > 50000 * (i + 1))
    #     upper = temp[0][0]
    #     out_name = out_prefix
    #     out_name += str(i)
    #     out_name += ".npy"
    #     print(out_name)
    #     np.save(out_name, events[floor:upper])
    #
    # for i in range(9920, 10048):
    #     temp = np.where(events[:, 0] - events[0, 0] > 50000 * i)
    #     floor = temp[0][0]
    #     temp = np.where(events[:, 0] - events[0, 0] > 50000 * (i + 1))
    #     upper = temp[0][0]
    #     out_name = out_prefix
    #     out_name += str(i)
    #     out_name += ".npy"
    #     print(out_name)
    #     np.save(out_name, events[floor:upper])
    #
    # for i in range(18500, 20400):
    #     temp = np.where(events[:, 0] - events[0, 0] > 50000 * i)
    #     floor = temp[0][0]
    #     temp = np.where(events[:, 0] - events[0, 0] > 50000 * (i + 1))
    #     upper = temp[0][0]
    #     out_name = out_prefix
    #     out_name += str(i)
    #     out_name += ".npy"
    #     print(out_name)
    #     np.save(out_name, events[floor:upper])


    # b = np.load("result.npy")

    # while events[i][0] - events[0][0] <= 50000:
    #     if events[i][3] == p:
    #         frame[events[i][2]][events[i][1]] += 1
    #     i += 1
