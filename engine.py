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


def val_one_epoch__prec_recall(test_loader, model, criterion, epoch, logger, config):
    # switch to evaluate mode
    model.eval()
    preds = []
    gts = []
    loss_list = []
    with torch.no_grad():
        for data in tqdm(test_loader):
            img, msk = data
            img, msk = (
                img.cuda(non_blocking=True).float(),
                msk.cuda(non_blocking=True).float(),
            )

            out = model(img)
            loss = criterion(out, msk)

            loss_list.append(loss.item())
            gts.append(msk.squeeze(1).cpu().detach().numpy())
            if type(out) is tuple:
                out = out[0]
            out = out.squeeze(1).cpu().detach().numpy()
            preds.append(out)

    if epoch % config.val_interval == 0:
        preds = np.array(preds).reshape(-1)
        gts = np.array(gts).reshape(-1)

        y_pre = np.where(preds >= config.threshold, 1, 0)
        y_true = np.where(gts >= 0.5, 1, 0)

        from sklearn.metrics import confusion_matrix

        confusion = confusion_matrix(y_true, y_pre)
        TN, FP, FN, TP = (
            confusion[0, 0],
            confusion[0, 1],
            confusion[1, 0],
            confusion[1, 1],
        )

        precision = float(TP) / float(TP + TN)
        recall = float(TP) / float(TP + FN)

        log_info = f"val epoch: {epoch}, loss: {np.mean(loss_list)} precision: {precision}, recall: {recall}"
        print(log_info)
        logger.info(log_info, save_log=True, content_to_save=log_info)

    else:
        log_info = f"val epoch: {epoch}, loss: {np.mean(loss_list):.4f}"
        print(log_info)
        logger.info(log_info)

    return np.mean(loss_list)
