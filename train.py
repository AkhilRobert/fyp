import os
import sys
import warnings

# import timm
import torch
# from engine import *
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

from config import Settings
from dataset import BraTSDataset
from engine import train_one_epoch, val_one_epoch
from model import VMUNet
from utils import get_logger, log_config_info

warnings.filterwarnings("ignore")


def main(config):

    print("#----------Creating logger----------#")
    sys.path.append(config.work_dir + "/")
    log_dir = os.path.join(config.work_dir, "log")
    checkpoint_dir = os.path.join(config.work_dir, "checkpoints")
    resume_model = os.path.join(checkpoint_dir, "latest.pth")
    outputs = os.path.join(config.work_dir, "outputs")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists(outputs):
        os.makedirs(outputs)

    global logger
    logger = get_logger("train", log_dir)
    global writer
    writer = SummaryWriter(config.work_dir + "summary")

    log_config_info(config, logger)

    print("#----------GPU init----------#")
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_id
    # set_seed(config.seed)
    torch.cuda.empty_cache()

    print("#----------Preparing dataset----------#")
    train_dataset = BraTSDataset(config.data_path, config, train=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=config.num_workers,
    )
    val_dataset = BraTSDataset(config.data_path, config, train=False)
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        num_workers=config.num_workers,
        drop_last=True,
    )

    print("#----------Prepareing Model----------#")
    model_cfg = config.model_config
    model = VMUNet(
        num_classes=model_cfg["num_classes"],
        input_channels=model_cfg["input_channels"],
        depths=model_cfg["depths"],
        depths_decoder=model_cfg["depths_decoder"],
        drop_path_rate=model_cfg["drop_path_rate"],
        load_ckpt_path=model_cfg["load_ckpt_path"],
    )
    model.load_from()

    model = model.cuda()

    print("#----------Prepareing loss, opt, sch and amp----------#")
    criterion = config.criterion
    optimizer = torch.optim.AdamW(
        model.parameters(),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=50,
        eta_min=0.00001,
    )

    print("#----------Set other params----------#")
    min_loss = 0.4
    start_epoch = 1
    min_epoch = 1

    if os.path.exists(resume_model):
        print("#----------Resume Model and Other params----------#")
        checkpoint = torch.load(resume_model, map_location=torch.device("cpu"))
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        saved_epoch = checkpoint["epoch"]
        start_epoch += saved_epoch
        min_loss, min_epoch, result = (
            checkpoint["min_loss"],
            checkpoint["min_epoch"],
            checkpoint["loss"],
        )

        # log_info = f"resuming model from {resume_model}. resume_epoch: {saved_epoch}, min_loss: {min_loss:.4f}, min_epoch: {min_epoch}, loss: {loss:.4f}"
        # logger.info(log_info)

    max_result = 0.6
    step = 0
    print("#----------Training----------#")
    for epoch in range(start_epoch, config.epochs + 1):

        torch.cuda.empty_cache()

        step = train_one_epoch(
            train_loader,
            model,
            criterion,
            optimizer,
            scheduler,
            epoch,
            step,
            logger,
            config,
            writer,
        )

        result, _ = val_one_epoch(val_loader, model, criterion, epoch, logger, config)

        print(f"The current loss of the segmentation is {result}")

        if result > max_result:
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, "best.pth"))
            max_result = result
            min_epoch = epoch

        torch.save(
            {
                "epoch": epoch,
                "min_loss": min_loss,
                "min_epoch": min_epoch,
                "loss": result,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
            },
            os.path.join(checkpoint_dir, "latest.pth"),
        )

    # if os.path.exists(os.path.join(checkpoint_dir, "best.pth")):
    #     print("#----------Testing----------#")
    #     best_weight = torch.load(
    #         config.work_dir + "checkpoints/best.pth", map_location=torch.device("cpu")
    #     )
    #     model.load_state_dict(best_weight)
    #     loss = test_one_epoch(
    #         val_loader,
    #         model,
    #         criterion,
    #         logger,
    #         config,
    #     )
    #     os.rename(
    #         os.path.join(checkpoint_dir, "best.pth"),
    #         os.path.join(
    #             checkpoint_dir, f"best-epoch{min_epoch}-loss{min_loss:.4f}.pth"
    #         ),
    #     )


if __name__ == "__main__":
    config = Settings
    main(config)
