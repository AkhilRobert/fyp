from dataclasses import dataclass

from loss import CeDiceLoss


@dataclass
class Settings:
    model_config = {
        "num_classes": 4,
        "input_channels": 3,
        "depths": [2, 2, 2, 2],
        "depths_decoder": [2, 2, 2, 1],
        "drop_path_rate": 0.2,
        "load_ckpt_path": "./pre_trained_weights/vmamba_small_e238_ema.pth",
    }

    data_path = "output/"

    criterion = CeDiceLoss(4)

    datasets = "BraTS"

    work_dir = "results/" + "Unet" + "_" + datasets + "_" + "100" + "_192_2" + "/"

    pretrained_path = "./pre_trained/"
    num_classes = 4
    input_size_h = 192
    input_size_w = 192
    input_channels = 3
    distributed = False
    local_rank = -1
    num_workers = 0
    seed = 42
    world_size = None
    rank = None
    amp = False
    gpu_id = "0"
    batch_size = 18
    epochs = 1500

    print_interval = 20
    val_interval = 30
    save_interval = 100
    threshold = 0.5

    # AdamW optimizer
    opt = "AdamW"
    lr = 0.001  # default: 1e-3 – learning rate
    betas = (
        0.9,
        0.999,
    )  # default: (0.9, 0.999) – coefficients used for computing running averages of gradient and its square
    eps = 1e-8
    weight_decay = 1e-2
    amsgrad = False
