import glob

import numpy as np

if __name__ == '__main__':
    prefix = '/Volumes/SD/outoutout/test/'
    for name in glob.glob('/Volumes/SD/not reduced 2/test/*'):
        tensor = np.load(name)
        tensor = np.around(tensor, decimals=4)
        name = prefix + name.split('/')[-1]
        print(name)
        np.save(name, tensor)

