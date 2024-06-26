
import datetime
import os
import sys
import time
import warnings

# import timm
import torch
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

from config import Settings
from dataset import BraTSDataset
from engine import train_one_epoch, val_one_epoch, val_one_epoch__prec_recall
from model import UNet
from utils import get_logger, log_config_info, set_seed

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
    set_seed(config.seed)
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
    model = UNet(
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
    scheduler = torch.optim.lr_scheduler.PolynomialLR(
        optimizer, total_iters=config.epochs
    )

    print("#----------Set other params----------#")
    min_loss = 0.4
    start_epoch = 1
    min_epoch = 1

    result = 0

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

        log_info = f"resuming model from {resume_model}. resume_epoch: {saved_epoch}, min_loss: {min_loss:.4f}, min_epoch: {min_epoch}, loss: {result:.4f}"
        logger.info(log_info)

    max_accuracy = 0.7
    step = 0
    st = time.time()
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

        et = time.time()
        print(
            f"time taken for training current epoch is {datetime.timedelta(seconds=et - st)}"
        )

        dc_score, hdf = val_one_epoch(
            val_loader, model, criterion, epoch, logger, config
        )

        print(f"The current dice score is {dc_score} and distance is {hdf}")

        if dc_score > max_accuracy:
            print("saving the best model")
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, "best.pth"))
            max_accuracy = dc_score
            min_epoch = epoch

            # save the precision and recall
            val_one_epoch__prec_recall(
                val_loader, model, criterion, epoch, logger, config
            )

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
        st = time.time()


if __name__ == "__main__":
    config = Settings
    main(config)
