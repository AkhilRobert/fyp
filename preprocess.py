import matplotlib.pyplot as plt
import nibabel
import numpy as np
from matplotlib.pyplot import imshow
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()


def _slices_to_image(volume: np.array):
    pass


mri = nibabel.load("./data/00495/BraTS2021_00495_t1.nii.gz")
data = mri.get_fdata()

print(
    data.reshape(
        -1,
    ).shape
)

print(data)

data = scaler.fit_transform(data.reshape(-1, data.shape[-1])).reshape(data.shape)
print(np.max(data))

# for i in range(data.shape[-1]):
#     imshow(data[:, :, i])
#     plt.show()
