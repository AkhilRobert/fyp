import glob

import nibabel
import numpy as np
import torch
import torch.nn.functional as F
# from keras.utils import to_categorical
from matplotlib.pyplot import imshow
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

scaler = MinMaxScaler()


def has_data(slice):
    if (
        1 - (slice[0] / slice.sum())
    ) > 0.001:  # At least 0.1% useful volume with labels that are not 0
        return True
    else:
        return False


def normalize(volume):
    scaler = MinMaxScaler()
    return scaler.fit_transform(
        volume.reshape(-1, volume.shape[-1]),
    ).reshape(volume.shape)


def to_categorical(y, num_classes, dtype="float32"):
    y = np.array(y, dtype="int")
    input_shape = y.shape

    # Shrink the last dimension if the shape is (..., 1).
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])

    y = y.reshape(-1)
    assert num_classes is not None
    num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=dtype)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical


t2_list = sorted(glob.glob("./data/unprocessed/BraTS2021_*/*t2.nii.gz"))
t1ce_list = sorted(glob.glob("./data/unprocessed/BraTS2021_*/*t1ce.nii.gz"))
flair_list = sorted(glob.glob("./data/unprocessed/BraTS2021_*/*flair.nii.gz"))
seg_list = sorted(glob.glob("./data/unprocessed/BraTS2021_*/*seg.nii.gz"))

assert len(t2_list) == len(t1ce_list) == len(flair_list) == len(seg_list)

for i in tqdm(range(len(t2_list)), ascii=True):
    t2 = nibabel.load(t2_list[i]).get_fdata()
    t1ce = nibabel.load(t1ce_list[i]).get_fdata()
    flair = nibabel.load(flair_list[i]).get_fdata()
    seg = nibabel.load(seg_list[i]).get_fdata()

    # Normalizing all the inputs
    t2 = normalize(t2)
    t1ce = normalize(t1ce)
    flair = normalize(flair)

    # Reassinging the mask value to 3
    seg[seg == 4] = 3

    # Create a multichannel image from the from the mri scans
    combined = np.stack([t2, t1ce, flair])

    if True:
        # Cropping out unwanted pieces of the image
        combined = combined[56:184, 56:184, 13:141]
        seg = seg[56:184, 56:184, 13:141]

    mask = to_categorical(seg, 4)

    c = 0
    for j in range(len(t2_list[-1])):
        val, count = np.unique(seg[:, :, j], return_counts=True)

        if not has_data(count):
            continue

        np.save(f"./data/processed/images/mri_{i}_{c}.npy", combined)
        np.save(f"./data/processed/masks/mri_{i}_{c}.npy", mask[:, :, j, :])
        c += 1
