from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

class TextureImages(object):
    def __init__(self, subset='train', batch_size=64, shuffle=True):
        if subset == 'train':
            images = np.load('TextureImagesDataset/train_images.npy')
            masks = np.load('TextureImagesDataset/train_masks.npy')
        elif subset == 'test':
            images = np.load('TextureImagesDataset/test_images.npy')
            masks = np.load('TextureImagesDataset/test_masks.npy')
        else:
            raise NotImplementedError
        self._images = images
        self.images = self._images
        self._masks = masks
        self.masks = self._masks
        self.batch_size = batch_size
        self.num_samples = len(self.images)
        self.shuffle = shuffle
        if self.shuffle:
            self.shuffle_samples()
        self.next_batch_pointer = 0

    def shuffle_samples(self):
        image_indices = np.random.permutation(np.arange(self.num_samples))
        self.images = self._images[image_indices]
        self.masks = self._masks[image_indices]

    def get_next_batch(self):
        num_samples_left = self.num_samples - self.next_batch_pointer
        if num_samples_left >= self.batch_size:
            x_batch = self.images[self.next_batch_pointer:self.next_batch_pointer + self.batch_size]
            y_batch = self.masks[self.next_batch_pointer:self.next_batch_pointer + self.batch_size]
            self.next_batch_pointer += self.batch_size
        else:
            x_partial_batch_1 = self.images[self.next_batch_pointer:self.num_samples]
            y_partial_batch_1 = self.masks[self.next_batch_pointer:self.num_samples]
            if self.shuffle:
                self.shuffle_samples()
            x_partial_batch_2 = self.images[0:self.batch_size - num_samples_left]
            y_partial_batch_2 = self.masks[0:self.batch_size - num_samples_left]
            x_batch = np.vstack((x_partial_batch_1, x_partial_batch_2))
            y_batch = np.vstack((y_partial_batch_1, y_partial_batch_2))
            self.next_batch_pointer = self.batch_size - num_samples_left
        return x_batch, y_batch
