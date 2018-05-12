import os
import sys
import tarfile

import numpy
import six.moves.cPickle as pickle

from chainer.dataset import download
from chainer.datasets import tuple_dataset
import chainer.functions as F

import sys
import os, sys, tarfile, errno
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt

if sys.version_info >= (3, 0, 0):
    import urllib.request as urllib # ugly but works
else:
    import urllib

# image shape
HEIGHT = 96
WIDTH = 96
DEPTH = 3

# size of a single image in bytes
SIZE = HEIGHT * WIDTH * DEPTH

# path to the directory with the data
DATA_DIR = os.path.expanduser('~/dataset')

# url of the binary data
DATA_URL = 'http://ai.stanford.edu/~acoates/stl10/stl10_binary.tar.gz'

# path to the binary train file with image data
DATA_PATH = DATA_DIR + '/stl10_binary/train_X.bin'

# path to the binary train file with labels
LABEL_PATH = DATA_DIR + '/stl10_binary/train_y.bin'

# path to the binary train file with image data
TEST_DATA_PATH = DATA_DIR + '/stl10_binary/test_X.bin'

# path to the binary train file with labels
TEST_LABEL_PATH = DATA_DIR + '/stl10_binary/test_y.bin'

# path to the binary train file with image data
UNLABELED_DATA_PATH = DATA_DIR + '/stl10_binary/unlabeled_X.bin'

# all
All_DATA_PATH = DATA_DIR + '/stl10_binary/all.npy'

def read_labels(path_to_labels):
    """
    :param path_to_labels: path to the binary file containing labels from the STL-10 dataset
    :return: an array containing the labels
    """
    with open(path_to_labels, 'rb') as f:
        labels = np.fromfile(f, dtype=np.uint8)
        return labels


def read_all_images(path_to_data):
    """
    :param path_to_data: the file containing the binary images from the STL-10 dataset
    :return: an array containing all the images
    """

    with open(path_to_data, 'rb') as f:
        # read whole file in uint8 chunks
        everything = np.fromfile(f, dtype=np.uint8)

        # We force the data into 3x96x96 chunks, since the
        # images are stored in "column-major order", meaning
        # that "the first 96*96 values are the red channel,
        # the next 96*96 are green, and the last are blue."
        # The -1 is since the size of the pictures depends
        # on the input file, and this way numpy determines
        # the size on its own.

        images = np.reshape(everything, (-1, 3, 96, 96))

        # Now transpose the images into a standard image format
        # readable by, for example, matplotlib.imshow
        # You might want to comment this line or reverse the shuffle
        # if you will use a learning algorithm like CNN, since they like
        # their channels separated.
        # images = np.transpose(images, (0, 3, 2, 1))
        return images

def download_and_extract():
    """
    Download and extract the STL-10 dataset
    :return: None
    """
    dest_directory = DATA_DIR
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    filename = DATA_URL.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\rDownloading %s %.2f%%' % (filename,
                float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()
        filepath, _ = urllib.urlretrieve(DATA_URL, filepath, reporthook=_progress)
        print('Downloaded', filename)
        tarfile.open(filepath, 'r:gz').extractall(dest_directory)


def getSTL(withlabel=False, ndim=3, scale=1.):
    # download data if needed
    download_and_extract()
    if not os.path.exists(All_DATA_PATH) or True:
        unlabeled_x = read_all_images(UNLABELED_DATA_PATH)

        alldata = _preprocess_STL(unlabeled_x, 0, False, ndim, scale)

        np.save(All_DATA_PATH, unlabeled)
    else:
        alldata = np.load(All_DATA_PATH)

    return alldata

def _preprocess_STL(images, labels, withlabel, ndim, scale):
    print("preprocess image")
    if ndim == 1:
        images = images.reshape(-1, SIZE)
    elif ndim == 3:
        images = images.reshape(-1, 3, 96, 96)
    else:
        raise ValueError('invalid ndim for dataset')
    print("==========================")
    images = images.astype(numpy.float32)
    print("==========================")
    images *= scale / 255.
    print("==========================")
    images = F.resize_images(images,(48,48)).data
    print("==========================")

    if withlabel:
        labels = labels.astype(numpy.int32)
        return tuple_dataset.TupleDataset(images, labels)
    else:
        return images
