import nibabel
import numpy as np
import torch
from PIL import Image
from skimage.transform import resize

from config import Settings
from model import UNet
from preprocess import normalize


def main(config):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_cfg = config.model_config
    model = UNet(
        num_classes=model_cfg["num_classes"],
        input_channels=model_cfg["input_channels"],
        depths=model_cfg["depths"],
        depths_decoder=model_cfg["depths_decoder"],
        drop_path_rate=model_cfg["drop_path_rate"],
        load_ckpt_path=model_cfg["load_ckpt_path"],
    )
    model.load_from()
    model.to(device)

    weights = torch.load("./results/best.pth")
    model.load_state_dict(weights)

    t2 = nibabel.load(
        "./data/unprocessed/BraTS2021_00051/BraTS2021_00051_t2.nii.gz"
    ).get_fdata()
    t1 = nibabel.load(
        "./data/unprocessed/BraTS2021_00051/BraTS2021_00051_t1ce.nii.gz"
    ).get_fdata()
    flair = nibabel.load(
        "./data/unprocessed/BraTS2021_00051/BraTS2021_00051_flair.nii.gz"
    ).get_fdata()

    t2 = normalize(t2).astype(np.float32)
    t1 = normalize(t1).astype(np.float32)
    flair = normalize(flair).astype(np.float32)

    combined = np.stack([t2, t1, flair], axis=3)
    # resized_img = resize(
    #     combined, (config.input_size_w, config.input_size_h, t2.shape[-1])
    # )
    mix = Settings.mix
    max = Settings.max

    resized_img = combined[mix:max, mix:max, :]

    final_image = np.empty((config.input_size_w, config.input_size_h, t2.shape[-1]))
    with torch.no_grad():
        for i in range(155):
            img = torch.Tensor(resized_img[:, :, i, :]).T
            img = (
                img.to(device)
                .float()
                .view(
                    (
                        -1,
                        config.input_channels,
                        config.input_size_w,
                        config.input_size_h,
                    )
                )
            )
            out = model(img)

            out = torch.argmax(torch.softmax(out, dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()
            prediction = prediction.reshape(config.input_size_w, config.input_size_h)
            prediction[prediction == 3] = 4

            final_image[:, :, i] = prediction

        image = nibabel.Nifti1Image(final_image, None)
        nibabel.save(image, "./sss-out/pred.nii.gz")


if __name__ == "__main__":
    config = Settings
    main(config)
