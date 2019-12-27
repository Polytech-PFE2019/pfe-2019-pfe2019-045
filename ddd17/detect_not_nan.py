import glob

import numpy as np

if __name__ == '__main__':
    # prefix = '/Volumes/SD/outoutout/train/'
    for name in glob.glob('../data/dataset_our_codification/events/train/*'):
        tensor = np.load(name)
        print(name)
        assert not np.any(np.isnan(tensor))