import os
import random

import numpy as np
from scipy import ndimage
from torch.utils.data import Dataset

from config import Settings


def random_rot(img, msk):
    angle = np.random.randint(-30, 30)
    image = ndimage.rotate(img, angle, order=0, reshape=False)
    mask = ndimage.rotate(msk, angle, order=0, reshape=False)
    return image, mask


def random_flip_90(img, msk):
    k = np.random.randint(0, 4)
    image = np.rot90(img, k)
    mask = np.rot90(msk, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    mask = np.flip(mask, axis=axis).copy()
    return image, mask


class RandomGenerator:
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def __call__(self, img, msk):
        if random.random() > 0.5:
            img, msk = random_rot(img, msk)
        elif random.random() > 0.5:
            img, msk = random_flip_90(img, msk)

        x, y, _ = img.shape
        assert x == self.width and y == self.height

        return img, msk


class BraTSDataset(Dataset):
    def __init__(self, path_Data, config, train=True):
        super(BraTSDataset, self)
        if train:
            images_list = sorted(os.listdir(path_Data + "train/images/"))
            masks_list = sorted(os.listdir(path_Data + "train/masks/"))
            self.data = []
            for i in range(len(images_list)):
                img_path = path_Data + "train/images/" + images_list[i]
                mask_path = path_Data + "train/masks/" + masks_list[i]
                self.data.append([img_path, mask_path])
        else:
            images_list = sorted(os.listdir(path_Data + "val/images/"))
            masks_list = sorted(os.listdir(path_Data + "val/masks/"))
            self.data = []
            for i in range(len(images_list)):
                img_path = path_Data + "val/images/" + images_list[i]
                mask_path = path_Data + "val/masks/" + masks_list[i]
                self.data.append([img_path, mask_path])

    def __getitem__(self, indx):
        img_path, msk_path = self.data[indx]
        img = np.load(img_path)
        msk = np.load(msk_path)

        augmenter = RandomGenerator(Settings.input_size_w, Settings.input_size_h)
        img, msk = augmenter(img, msk)

        return img.T, msk

    def __len__(self):
        return len(self.data)
