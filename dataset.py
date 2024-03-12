import os

import numpy as np
from torch.utils.data import Dataset


# TODO: Data Augmentation
class RandomGenerator:
    pass


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
            # self.transformer = config.train_transformer
        else:
            images_list = sorted(os.listdir(path_Data + "val/images/"))
            masks_list = sorted(os.listdir(path_Data + "val/masks/"))
            self.data = []
            for i in range(len(images_list)):
                img_path = path_Data + "val/images/" + images_list[i]
                mask_path = path_Data + "val/masks/" + masks_list[i]
                self.data.append([img_path, mask_path])
            # self.transformer = config.test_transformer

    def __getitem__(self, indx):
        img_path, msk_path = self.data[indx]
        img = np.load(img_path).T
        msk = np.load(msk_path)
        return img, msk

    def __len__(self):
        return len(self.data)
