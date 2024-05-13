from typing import Any, Dict

import gradio as gr
import nibabel
import numpy as np
import torch

from config import Settings
from model import UNet
from preprocess import normalize

world: Dict[Any, Any] = {
    "current": 70,
}

config = Settings


def load_data(t1ce, t2, flair, seg):
    global world
    t1ce_path = t1ce.name
    t2_path = t2.name
    flair_path = flair.name
    seg_path = seg.name

    print(t1ce_path, t2_path, flair_path)

    t1ce_data = normalize(nibabel.load(t1ce_path).get_fdata()).astype(np.float32)
    t2_data = normalize(nibabel.load(t2_path).get_fdata()).astype(np.float32)
    flair_data = normalize(nibabel.load(flair_path).get_fdata()).astype(np.float32)
    seg_data = nibabel.load(seg_path).get_fdata()
    # t2_data = nibabel.load(
    #     "./data/unprocessed/BraTS2021_00051/BraTS2021_00051_t2.nii.gz"
    # ).get_fdata()
    # t1ce_data = nibabel.load(
    #     "./data/unprocessed/BraTS2021_00051/BraTS2021_00051_t1ce.nii.gz"
    # ).get_fdata()
    # flair_data = nibabel.load(
    #     "./data/unprocessed/BraTS2021_00051/BraTS2021_00051_flair.nii.gz"
    # ).get_fdata()

    # t2_data = normalize(t2_data)
    # t1ce_data = normalize(t1ce_data)
    # flair_data = normalize(flair_data)

    combined = np.stack([t2_data, t1ce_data, flair_data], axis=3)

    mix = config.mix
    max = config.max

    resized_img = combined[mix:max, mix:max, :]
    world["resized_img"] = resized_img

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

    final_image = np.empty(
        ((config.input_size_w, config.input_size_h, t2_data.shape[-1]))
    )

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

    world["final_image"] = final_image
    world["original"] = seg_data

    return [
        (
            resized_img[:, :, world["current"], 2],
            [
                (final_image[:, :, world["current"]] == 1, "tumor core"),
                (final_image[:, :, world["current"]] == 2, "peritumoral edema"),
                (final_image[:, :, world["current"]] == 3, "enchancing tumor"),
            ],
        ),
        (
            resized_img[:, :, world["current"], 2],
            [
                (seg_data[56:184, 56:184, world["current"]] == 1, "tumor core"),
                (seg_data[56:184, 56:184, world["current"]] == 2, "peritumoral edema"),
                (seg_data[56:184, 56:184, world["current"]] == 3, "enchancing tumor"),
            ],
        ),
    ]


def render(slider, state):
    resized_img = world["resized_img"]
    final_image = world["final_image"]
    seg_data = world["original"]

    if resized_img is None or final_image is None:
        gr.Warning("No predicted data is available")

    return [
        (
            resized_img[:, :, slider, 2],
            [
                ((final_image[:, :, slider] == 1), "tumor core"),
                ((final_image[:, :, slider] == 2), "peritumoral edema"),
                ((final_image[:, :, slider] == 4), "enchancing tumor"),
            ],
        ),
        (
            resized_img[:, :, slider, 2],
            [
                (seg_data[56:184, 56:184, slider] == 1, "tumor core"),
                (seg_data[56:184, 56:184, slider] == 2, "peritumoral edema"),
                (seg_data[56:184, 56:184, slider] == 4, "enchancing tumor"),
            ],
        ),
    ]


with gr.Blocks() as demo:
    with gr.Column():
        gr.Markdown(
            """
                    <center style="font-size: 40px;">
                        Glioma Segmentation using Mamba
                    </center>
            """
        )

    with gr.Row():
        t1ce = gr.File(label="upload t1ce scan")
        t2 = gr.File(label="upload t2 scan")
        flair = gr.File(label="upload flair scan")
        seg = gr.File(label="upload segmentation mask")

    with gr.Column():
        upload = gr.Button(value="predict")
        slider = gr.Slider(
            0, 154, step=1, label="select the slice you want to visualize", value=70
        )
        state = gr.State(0)
        with gr.Row():
            seg_res = gr.AnnotatedImage(height="500px", label="Predicted Segmentation")
            seg_tru = gr.AnnotatedImage(
                height="500px", label="Ground Truth Segmentation"
            )

    upload.click(load_data, inputs=[t1ce, t2, flair, seg], outputs=[seg_res, seg_tru])

    slider.change(
        render, inputs=[slider, state], outputs=[seg_res, seg_tru], api_name="render"
    )


if __name__ == "__main__":
    demo.launch(share=True)
