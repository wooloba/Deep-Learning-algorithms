
from __future__ import absolute_import
import numpy as np
import os
import sys
import six
from six.moves import cPickle
from six.moves.urllib.error import HTTPError
from six.moves.urllib.error import URLError
from six.moves.urllib.request import urlopen
import tensorflow as tf


def get_file(fname,
             origin,
             untar=False,
             md5_hash=None,
             file_hash=None,
             cache_subdir='datasets',
             hash_algorithm='auto',
             extract=False,
             archive_format='auto',
             cache_dir=None):
    
    if cache_dir is None:
        cache_dir = os.path.expanduser(os.path.join('~', '.keras'))
    if md5_hash is not None and file_hash is None:
        file_hash = md5_hash
        hash_algorithm = 'md5'
    datadir_base = os.path.expanduser(cache_dir)
    if not os.access(datadir_base, os.W_OK):
        datadir_base = os.path.join('/tmp', '.keras')
    datadir = os.path.join(datadir_base, cache_subdir)
    if not os.path.exists(datadir):
        os.makedirs(datadir)

    if untar:
        untar_fpath = os.path.join(datadir, fname)
        fpath = untar_fpath + '.tar.gz'
    else:
        fpath = os.path.join(datadir, fname)

    download = False
    if os.path.exists(fpath):
        # File found; verify integrity if a hash was provided.
        if file_hash is not None:
            if not validate_file(fpath, file_hash, algorithm=hash_algorithm):
                print('A local file was found, but it seems to be '
                      'incomplete or outdated because the ' + hash_algorithm +
                      ' file hash does not match the original value of ' +
                      file_hash + ' so we will re-download the data.')
                download = True
    else:
        download = True

    if download:
        print('Downloading data from', origin)

        class ProgressTracker(object):
            # Maintain progbar for the lifetime of download.
            # This design was chosen for Python 2.7 compatibility.
            progbar = None

        def dl_progress(count, block_size, total_size):
            if ProgressTracker.progbar is None:
                if total_size is -1:
                    total_size = None
                ProgressTracker.progbar = Progbar(total_size)
            else:
                ProgressTracker.progbar.update(count * block_size)

        error_msg = 'URL fetch failure on {}: {} -- {}'
        try:
            try:
                urlretrieve(origin, fpath, dl_progress)
            except URLError as e:
                raise Exception(error_msg.format(origin, e.errno, e.reason))
            except HTTPError as e:
                raise Exception(error_msg.format(origin, e.code, e.msg))
        except (Exception, KeyboardInterrupt) as e:
            if os.path.exists(fpath):
                os.remove(fpath)
            raise
        ProgressTracker.progbar = None

    if untar:
        if not os.path.exists(untar_fpath):
            _extract_archive(fpath, datadir, archive_format='tar')
        return untar_fpath

    if extract:
        _extract_archive(fpath, datadir, archive_format)

    return fpath


def load_batch(fpath, label_key='labels'):
    """Internal utility for parsing CIFAR data.
    # Arguments
        fpath: path the file to parse.
        label_key: key for label data in the retrieve
            dictionary.
    # Returns
        A tuple `(data, labels)`.
    """
    f = open(fpath, 'rb')
    if sys.version_info < (3,):
        d = cPickle.load(f)
    else:
        d = cPickle.load(f, encoding='bytes')
        # decode utf8
        d_decoded = {}
        for k, v in d.items():
            d_decoded[k.decode('utf8')] = v
        d = d_decoded
    f.close()
    data = d['data']
    labels = d[label_key]

    data = data.reshape(data.shape[0], 3, 32, 32)
    return data, labels


def load_data():
    """Loads CIFAR10 dataset.
    # Returns
        Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
    """
    dirname = 'cifar-10-batches-py'
    origin = 'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
   # path = get_file(dirname, origin=origin, untar=True)

    num_train_samples = 50000

    cifar10 = tf.keras.datasets.cifar10
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    print(x_train.shape, x_test.shape)
    #print(x_train.shape)
    y_train = np.reshape(y_train, (len(y_train), 1))
    y_test = np.reshape(y_test, (len(y_test), 1))

    return (x_train, y_train), (x_test, y_test)


class Cifar10(object):
  def __init__(self, batch_size=64, one_hot=False, test=False, shuffle=True):
    (x_train, y_train), (x_test, y_test) = load_data()
    if test:
      images = x_test
      labels = y_test
    else:
      images = x_train
      labels = y_train
    if one_hot:
      one_hot_labels = np.zeros((len(labels), 10))
      one_hot_labels[np.arange(len(labels)), labels.flatten()] = 1
      labels = one_hot_labels
    self.shuffle = shuffle
    self._images = images
    self.images = self._images
    self._labels = labels
    self.labels = self._labels
    self.batch_size = batch_size
    self.num_samples = len(self.images)
    if self.shuffle:
      self.shuffle_samples()
    self.next_batch_pointer = 0

  def shuffle_samples(self):
    image_indices = np.random.permutation(np.arange(self.num_samples))
    self.images = self._images[image_indices]
    self.labels = self._labels[image_indices]

  def get_next_batch(self):
    num_samples_left = self.num_samples - self.next_batch_pointer
    if num_samples_left >= self.batch_size:
      x_batch = self.images[self.next_batch_pointer:self.next_batch_pointer + self.batch_size]
      y_batch = self.labels[self.next_batch_pointer:self.next_batch_pointer + self.batch_size]
      self.next_batch_pointer += self.batch_size
    else:
      x_partial_batch_1 = self.images[self.next_batch_pointer:self.num_samples]
      y_partial_batch_1 = self.labels[self.next_batch_pointer:self.num_samples]
      if self.shuffle:
        self.shuffle_samples()
      x_partial_batch_2 = self.images[0:self.batch_size - num_samples_left]
      y_partial_batch_2 = self.labels[0:self.batch_size - num_samples_left]
      x_batch = np.vstack((x_partial_batch_1, x_partial_batch_2))
      y_batch = np.vstack((y_partial_batch_1, y_partial_batch_2))
      self.next_batch_pointer = self.batch_size - num_samples_left
    return x_batch, y_batch
