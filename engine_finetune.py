# --------------------------------------------------------
# References:
# MAE: https://github.com/facebookresearch/mae
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import math
import sys
from typing import Iterable, Optional

import torch
import wandb
import torch.nn.functional as F

from timm.data import Mixup
from timm.utils import accuracy
import util.misc as misc
import util.lr_sched as lr_sched
from torchvision.ops.boxes import box_iou, box_convert
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import tempfile
import os
import json
from PIL import Image
from tqdm import tqdm
import numpy as np

def train_one_epoch_linear(model, data_loader, optimizer, device, epoch, criterion, log_writer=None, args=None):
    model.train()
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = f"Epoch: [{epoch}]"

    for images, targets in metric_logger.log_every(data_loader, 10, header):
        images = [img.to(device) for img in images]
        labels = torch.tensor([t["labels"][0] for t in targets]).to(device)  # assuming 1 label per image

        # Forward
        outputs = model(images)  # shape: [B, num_classes]
        targets = torch.stack([t["labels"] for t in targets]).to(device)  # shape: [B, num_classes]
        loss = criterion(outputs, targets)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Logging
        acc = (outputs.argmax(dim=1) == labels).float().mean().item()
        metric_logger.update(loss=loss.item(), acc=acc)

    # End-of-epoch stats
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def evaluate_linear(data_loader, model, device, criterion=None, class_names=None):
    model.eval()

    total_loss_classifier = 0.0
    correct = 0
    total = 0

    for images, targets in tqdm(data_loader, desc="Evaluating Linear"):
        images = [img.to(device) for img in images]
        labels = torch.tensor([t["labels"][0] for t in targets]).to(device)

        outputs = model(torch.stack(images))  # B x num_classes
        loss = criterion(outputs, labels)

        total_loss_classifier += loss.item()
        preds = torch.sigmoid(outputs) > 0.5
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss_classifier = total_loss_classifier / len(data_loader)
    accuracy = 100.0 * correct / total

    # 🟡 For compatibility with evaluate_frcnn
    return {
        "loss_classifier": avg_loss_classifier,
        "loss_box_reg": 0.0,
        "loss_objectness": 0.0,
        "loss_rpn_box_reg": 0.0,
        "mAP_50": accuracy,
        "mAP_50_95": accuracy,
        "per_class_ap": {},
    }


def train_one_epoch_frcnn(model, data_loader, optimizer, device, epoch, log_writer=None, args=None):
    model.train()
    total_loss = 0.0

    for images, targets in data_loader:
        images = list(img.to(device) for img in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        optimizer.zero_grad()

        # ✅ Pass resized images and transformed ta0rgets
        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())
        losses.backward()
        optimizer.step()
        total_loss += losses.item()

        #print(f"Epoch [{epoch}] - Loss: {losses:.4f}")

    avg_loss = total_loss / len(data_loader)
    return {"loss": avg_loss}



@torch.no_grad()
def evaluate_frcnn(data_loader, model, device, class_names=None):
    model.eval()
    
    total_loss = {
        "loss_classifier": 0.0,
        "loss_box_reg": 0.0,
        "loss_objectness": 0.0,
        "loss_rpn_box_reg": 0.0,
    }
    num_samples = 0
    coco_predictions = []

    # 🔁 For each batch
    for images, targets in tqdm(data_loader, desc="Evaluating"):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # ⚠️ Temporarily switch to train to get loss (eval doesn't compute it)
        model.train()
        loss_dict = model(images, targets)
        for k in total_loss:
            total_loss[k] += loss_dict[k].item()
        num_samples += 1

        # 🔁 Switch back to eval for predictions
        model.eval()
        predictions = model(images)

        for i, pred in enumerate(predictions):
            image_id = int(targets[i]["image_id"].item())  # ✅ must match GT JSON
            boxes = pred["boxes"].cpu()
            scores = pred["scores"].cpu()
            labels = pred["labels"].cpu()

            for box, score, label in zip(boxes, scores, labels):
                if score < 0.05:
                    continue
                x1, y1, x2, y2 = box.tolist()
                coco_predictions.append({
                    "image_id": image_id,
                    "category_id": int(label),
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "score": float(score)
                })

    # ✅ Generate COCO-style GT annotations
    from torch import distributed as dist

# 👇 Only rank 0 creates the JSON
    if misc.is_main_process():
        coco_gt_path = create_coco_json_from_dataset(data_loader.dataset, save_dir=tempfile.gettempdir())
        dist.barrier()  # Let other processes wait
    else:
        dist.barrier()  # Wait for rank 0
        coco_gt_path = os.path.join(tempfile.gettempdir(), "coco_gt.json")

    coco_gt = COCO(coco_gt_path)
    coco_dt = coco_gt.loadRes(coco_predictions)

    # Run evaluation
    coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # Extract metrics
    mAP_50_95 = coco_eval.stats[0] * 100
    mAP_50 = coco_eval.stats[1] * 100

    # Per-class AP
    per_class_ap = {}
    if class_names:
        precisions = coco_eval.eval['precision']  # [T, R, K, A, M]
        for idx, class_name in enumerate(class_names):
            cls_prec = precisions[:, :, idx, 0, 0]
            cls_prec = cls_prec[cls_prec > -1]
            ap = cls_prec.mean() if cls_prec.size > 0 else float('nan')
            per_class_ap[class_name] = round(ap * 100, 2)

    # Final loss averages
    avg_losses = {k: v / num_samples for k, v in total_loss.items()}

    return {
        **avg_losses,
        "mAP_50": mAP_50,
        "mAP_50_95": mAP_50_95,
        "per_class_ap": per_class_ap
    }

def create_coco_json_from_dataset(dataset, save_dir="/tmp", json_name="coco_gt.json"):
    import json, os
    from PIL import Image
    from tqdm import tqdm

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, json_name)

    images, annotations, categories = [], [], []
    annotation_id = 1

    class_to_id = dataset.CLASS_MAPPING
    id_to_class = {v: k for k, v in class_to_id.items()}

    for class_id, class_name in id_to_class.items():
        categories.append({
            "id": class_id,
            "name": class_name,
            "supercategory": "object"
        })

    for idx in tqdm(range(len(dataset))):
        try:
            _, target = dataset[idx]
        except Exception as e:
            print(f"[!] Skipped {idx}: {e}")
            continue

        images.append({
            "id": idx,
            "file_name": dataset.image_filenames[idx] + ".jpg",
            "width": 224,     # ⚠️ force match resized
            "height": 224
        })

        boxes = target["boxes"]
        labels = target["labels"]

        for box, label in zip(boxes, labels):
            x1, y1, x2, y2 = box.tolist()
            width = x2 - x1
            height = y2 - y1
            area = width * height

            annotations.append({
                "id": annotation_id,
                "image_id": idx,  # ⚠️ must match prediction "image_id"
                "category_id": int(label.item()),
                "bbox": [x1, y1, width, height],
                "area": area,
                "iscrowd": 0
            })
            annotation_id += 1

    import tempfile
    temp_path = save_path + ".tmp"

    with open(temp_path, "w") as f:
        json.dump({
            "images": images,
            "annotations": annotations,
            "categories": categories
        }, f)

    os.replace(temp_path, save_path) 
    print(f"[✓] Saved COCO GT to {save_path}")
    return save_path
def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    mixup_fn: Optional[Mixup] = None, log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast():
            outputs = model(samples)
            loss = criterion(outputs, targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            raise ValueError(f"Loss is {loss_value}, stopping training")

        loss /= accum_iter
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', max_lr, epoch_1000x)

            if args.local_rank == 0 and args.wandb is not None:
                try:
                    wandb.log({'train_loss_step': loss_value_reduce,
                               'train_lr_step': max_lr, 'epoch_1000x': epoch_1000x})
                except ValueError:
                    pass

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def train_one_epoch_temporal(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    mixup_fn: Optional[Mixup] = None, log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (samples, timestamps, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)
        timestamps = timestamps.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast():
            outputs = model(samples, timestamps)
            loss = criterion(outputs, targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', max_lr, epoch_1000x)

            if args.local_rank == 0 and args.wandb is not None:
                try:
                    wandb.log({'train_loss_step': loss_value_reduce,
                               'train_lr_step': max_lr, 'epoch_1000x': epoch_1000x})
                except ValueError:
                    pass

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]
        target = batch[-1]
        # print('images and targets')
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # print("before pass model")
        # compute output
        with torch.cuda.amp.autocast():
            output = model(images)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        # print(acc1, acc5, flush=True)

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate_temporal(data_loader, model, device):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    tta = False

    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]
        timestamps = batch[1]
        target = batch[-1]

        batch_size = images.shape[0]
        # print(images.shape, timestamps.shape, target.shape)
        if tta:
            images = images.reshape(-1, 3, 3, 224, 224)
            timestamps = timestamps.reshape(-1, 3, 3)
            target = target.reshape(-1, 1)
        # images = images.reshape()
        # print('images and targets')
        images = images.to(device, non_blocking=True)
        timestamps = timestamps.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # print("before pass model")
        # compute output
        with torch.cuda.amp.autocast():
            output = model(images, timestamps)

            if tta:
                # output = output.reshape(batch_size, 9, -1).mean(dim=1, keepdims=False)

                output = output.reshape(batch_size, 9, -1)
                sp = output.shape
                maxarg = output.argmax(dim=-1)

                output = F.one_hot(maxarg.reshape(-1), num_classes=1000).float()
                output = output.reshape(sp).mean(dim=1, keepdims=False)
                # print(output.shape)
                
                target = target.reshape(batch_size, 9)[:, 0]
            # print(target.shape)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        # print(acc1, acc5, flush=True)

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
