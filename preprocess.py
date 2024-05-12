import glob

import nibabel
import numpy as np
import splitfolders
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

from config import Settings

scaler = MinMaxScaler()


def has_data(slice):
    if (
        1 - (slice[0] / slice.sum())
    ) > 0.1:  # At least 1% useful volume with labels that are not 0
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


if __name__ == "__main__":
    t2_list = sorted(glob.glob("./data/unprocessed/BraTS2021_*/*t2.nii.gz"))
    t1ce_list = sorted(glob.glob("./data/unprocessed/BraTS2021_*/*t1ce.nii.gz"))
    flair_list = sorted(glob.glob("./data/unprocessed/BraTS2021_*/*flair.nii.gz"))
    seg_list = sorted(glob.glob("./data/unprocessed/BraTS2021_*/*seg.nii.gz"))

    assert len(t2_list) == len(t1ce_list) == len(flair_list) == len(seg_list)

    c = 1
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
        combined = np.stack([t2, t1ce, flair], axis=3)

        mix = Settings.mix
        max = Settings.max

        combined = combined[
            mix:max,
            mix:max,
        ]
        seg = seg[
            mix:max,
            mix:max,
        ]

        for j in range(t2[0].shape[-1]):
            val, count = np.unique(seg[:, :, j], return_counts=True)

            if not has_data(count):
                continue

            np.save(f"./data/processed/images/image_{c}.npy", combined[:, :, j])
            np.save(f"./data/processed/masks/mask_{c}.npy", seg[:, :, j])
            c += 1

    print(f"Obtained {c} significant slices")

    splitfolders.ratio("./data/processed/", ratio=(0.9, 0.1), move=True)
