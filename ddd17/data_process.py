import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
from mpl_toolkits.mplot3d import Axes3D

filename = '/Volumes/SD/exported_9175.h5'

if __name__ == '__main__':
    d = h5py.File(filename, 'r')
    data = d.get('/event')
    events = np.array(data, dtype=np.int32)

    # plt.imshow(d['/frame'].value[0], cmap='gray')
    # plt.show()

    d.close()

    frame_positive = np.zeros((260, 346), dtype=np.int16)
    frame_negative = np.zeros((260, 346), dtype=np.int16)

    # for i in range(50):
    #     if events[i][3] == 1:
    #         frame_positive[events[i][2]][events[i][1]] += 1
    #     else:
    #         frame_negative[events[i][2]][events[i][1]] += 1
    #
    # print(frame_positive)
    # print(events[:50].shape)

    x = [k[0] for k in events[:1000]]
    y = [k[1] for k in events[:1000]]
    z = [k[2] for k in events[:1000]]
    # plt.imshow(frame_positive, cmap=plt.cm.gray, interpolation='nearest', origin='upper')
    # plt.show()
    #
    # plt.imshow(frame_negative, cmap=plt.cm.gray, interpolation='nearest', origin='upper')
    # plt.show()



    fig = plt.figure(dpi=120)
    ax = fig.add_subplot(111, projection='3d')
    # 标题
    plt.title('point cloud')
    # 利用xyz的值，生成每个点的相应坐标（x,y,z）
    ax.scatter(x, y, z, c='b', marker='.', s=2, linewidth=0, alpha=1, cmap='spectral')
    ax.axis('scaled')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    # 显示
    plt.show()