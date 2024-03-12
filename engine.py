import numpy as np
import torch
from medpy import metric
from tqdm import tqdm


def train_one_epoch(
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
):
    # switch to train mode
    model.train()

    loss_list = []

    for iter, data in enumerate(train_loader):
        step += iter
        optimizer.zero_grad()
        images, targets = data
        images, targets = (
            images.cuda(non_blocking=True).float(),
            targets.cuda(non_blocking=True).float(),
        )

        out = model(images)
        loss = criterion(out, targets)

        loss.backward()
        optimizer.step()

        loss_list.append(loss.item())

        now_lr = optimizer.state_dict()["param_groups"][0]["lr"]

        writer.add_scalar("loss", loss, global_step=step)

        if iter % config.print_interval == 0:
            log_info = f"train: epoch {epoch}, iter:{iter}, loss: {np.mean(loss_list):.4f}, lr: {now_lr}"
            print(log_info)
            logger.info(log_info)
    scheduler.step()
    return step


def calculate_metrics(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    elif pred.sum() > 0 and gt.sum() == 0:
        return 1, 0
    else:
        return 0, 0


def val_one_epoch(test_loader, model, criterion, epoch, logger, config):
    model.eval()

    metrics_list = 0.0
    with torch.no_grad():
        for data in tqdm(test_loader):
            img, msk = data
            img = img.cuda(non_blocking=True).float()

            out = torch.argmax(
                torch.softmax(model(img), dim=1),
                dim=1,
            ).squeeze(0)
            prediction = out.cpu().detach().numpy()
            prediction = prediction.reshape(
                -1, config.input_size_w, config.input_size_h
            )
            msk = msk.cpu().detach().numpy()
            msk = msk.reshape(-1, config.input_size_w, config.input_size_h)

            metrics = []
            for i in range(1, config.num_classes):
                metrics.append(calculate_metrics(prediction == i, msk == i))

            metrics_list += np.array(metrics)

        metrics_list = metrics_list / len(test_loader.dataset)
        dice = np.mean(metrics_list, axis=0)[0]
        h95 = np.mean(metrics_list, axis=0)[1]
        return dice, h95
