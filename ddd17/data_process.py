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

    for i in events:
        if i[3] == 1:
            frame_positive[i[2]][i[1]] += 1
        else:
            frame_negative[i[2]][i[1]] += 1

    print(frame_positive)

    plt.imshow(frame_positive, cmap=plt.cm.gray, interpolation='nearest', origin='upper')
    plt.show()

    plt.imshow(frame_negative, cmap=plt.cm.gray, interpolation='nearest', origin='upper')
    plt.show()
