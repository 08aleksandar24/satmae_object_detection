# --------------------------------------------------------
# References:
# MAE: https://github.com/facebookresearch/mae
# --------------------------------------------------------
import argparse
import datetime
import json
import numpy as np
import os
import time
import wandb
from pathlib import Path


import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from torchvision.ops import FeaturePyramidNetwork as FPN
import timm

#assert timm.__version__ == "0.3.2"  # version check
from timm.models.layers import trunc_normal_
from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

import util.lr_decay as lrd
import util.misc as misc
from util.datasets import DIORDataset
from util.pos_embed import interpolate_pos_embed
from util.misc import NativeScalerWithGradNormCount as NativeScaler

import models_resnet
import models_vit
import models_vit_temporal
import models_vit_group_channels
from DINOv2_features import DINOv2
from engine_finetune import (train_one_epoch, train_one_epoch_temporal,
                             evaluate, evaluate_temporal, train_one_epoch_frcnn, evaluate_frcnn, train_one_epoch_linear, evaluate_linear)
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from models_vit import vit_large_patch16_frcnn  
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn as nn

import torchvision.transforms.functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2
def get_args_parser():
    parser = argparse.ArgumentParser('MAE fine-tuning for image classification', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model_type', default=None, choices=['group_c', 'resnet', 'resnet_pre',
                                                               'temporal', 'vanilla'],
                        help='Use channel model')
    parser.add_argument('--model', default='vit_large_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')
    parser.add_argument('--patch_size', default=16, type=int,
                        help='images input size')

    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    # Optimizer parameters
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--layer_decay', type=float, default=0.75,
                        help='layer-wise lr decay from ELECTRA/BEiT')

    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR')

    # Augmentation parameters
    parser.add_argument('--color_jitter', type=float, default=None, metavar='PCT',
                        help='Color jitter factor (enabled only when not using Auto/RandAug)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0,
                        help='mixup alpha, mixup enabled if > 0.')
    parser.add_argument('--cutmix', type=float, default=0,
                        help='cutmix alpha, cutmix enabled if > 0.')
    parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup_prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup_switch_prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup_mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # * Finetuning params
    parser.add_argument('--finetune', default='',
                        help='finetune from checkpoint')
    parser.add_argument('--global_pool', action='store_true')
    parser.set_defaults(global_pool=True)
    parser.add_argument('--cls_token', action='store_false', dest='global_pool',
                        help='Use class token instead of global pool for classification')

    # Dataset parameters
    # parser.add_argument('--data_path', default='/datasets01/imagenet_full_size/061417/', type=str,
    #                     help='dataset path')
    parser.add_argument('--train_path', default='/home/train_62classes.csv', type=str,
                        help='Train .csv path')
    parser.add_argument('--test_path', default='/home/val_62classes.csv', type=str,
                        help='Test .csv path')
    parser.add_argument('--dataset_type', default='rgb', choices=['rgb', 'temporal', 'sentinel', 'euro_sat', 'naip'],
                        help='Whether to use fmow rgb, sentinel, or other dataset.')
    parser.add_argument('--masked_bands', default=None, nargs='+', type=int,
                        help='Sequence of band indices to mask (with mean val) in sentinel dataset')
    parser.add_argument('--dropped_bands', type=int, nargs='+', default=None,
                        help="Which bands (0 indexed) to drop from sentinel data.")
    parser.add_argument('--grouped_bands', type=int, nargs='+', action='append',
                        default=[], help="Bands to group for GroupC vit")

    parser.add_argument('--nb_classes', default=21, type=int,
                        help='number of the classification types')

    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    parser.add_argument('--save_every', type=int, default=25, help='How frequently (in epochs) to save ckpt')
    parser.add_argument('--wandb', type=str, default=None,
                        help="Wandb project name, eg: sentinel_finetune")

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation (recommended during training for faster monitor')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=os.getenv('LOCAL_RANK', 0), type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    parser.add_argument('--detection_head', type=str, default='faster_rcnn',
                        choices=['faster_rcnn', 'linear_probing'],
                        help="Type of detection head to use: 'faster_rcnn' or 'linear_probing'")

    return parser


def main(args):
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True
    mean= [123.675/800, 116.28/800, 103.53/800]#[0.485, 0.456, 0.406]#
    std= [58.395/800, 57.12/800, 57.375/800]#[0.229, 0.224, 0.225]#
    custom_transform = GeneralizedRCNNTransform(
        min_size=(224,),
        max_size=224,
        #image_mean=[0.485, 0.456, 0.406],
        #image_std=[0.229, 0.224, 0.225]
        image_mean=mean,
        image_std=std
    )

    """cropping_transform = A.Compose([
        A.RandomCrop(width=800, height=800),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=mean, std=std),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))
    """
    dataset_root = "/storage/local/ssd/dior"
    """transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])"""
    

    if args.detection_head == "faster_rcnn":
        train_dataset = DIORDataset(root=dataset_root, split="trainval", transform=None)
        val_dataset = DIORDataset(root=dataset_root, split="test", transform=None)
    else:
        train_dataset = DIORDataset(root=dataset_root, split="trainval", transform=custom_transform)
        val_dataset = DIORDataset(root=dataset_root, split="test", transform=custom_transform)



    if len(train_dataset) == 0:
        raise ValueError("DIORDataset is empty! Check dataset loading logic.")
    DIOR_CLASSES = [
        "airplane", "airport", "baseballfield", "basketballcourt", "bridge",
        "chimney", "dam", "Expressway-Service-area", "Expressway-toll-station", "golffield",
        "groundtrackfield", "harbor", "overpass", "ship", "stadium",
        "storagetank", "tenniscourt", "trainstation", "vehicle", "windmill"
    ]
    
    CLASS_MAPPING = {cls: i for i, cls in enumerate(DIOR_CLASSES)}

    def collate_fn(batch):
        images, targets = zip(*batch)

        def process_target(t):
            return {
                "boxes": torch.tensor(t["boxes"], dtype=torch.float32),
                "labels": torch.tensor(t["labels"], dtype=torch.int64),
                "image_id": t["image_id"]  # ✅ Preserve image_id
            }

        targets = [process_target(t) for t in targets]
        return images, targets





    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)


    if True:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            train_dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
        if args.dist_eval:
            if len(val_dataset) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                val_dataset, num_replicas=num_tasks, rank=global_rank,
                shuffle=True)  # shuffle=True to reduce monitor bias
        else:
            sampler_val = torch.utils.data.SequentialSampler(val_dataset)
    else:
        sampler_train = torch.utils.data.RandomSampler(train_dataset)
        sampler_val = torch.utils.data.SequentialSampler(val_dataset)

    if global_rank == 0 and args.log_dir is not None and not args.eval:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        train_dataset, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=0,
        pin_memory=args.pin_mem,
        drop_last=True,
        collate_fn=collate_fn
    )

    data_loader_val = torch.utils.data.DataLoader(
        val_dataset, sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=0,
        pin_memory=args.pin_mem,
        drop_last=False,
        collate_fn=collate_fn
    )

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        print("Mixup is activated!")
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)

    # Define the model
    # Load SatMAE as backbone
    if args.model == "vit_large_patch16":
        backbone = vit_large_patch16_frcnn(pretrained=True)
        backbone.out_channels = 1024  # This should match your ViT embedding dimension
        
    elif args.model.split("_")[0] == "DINOv2":
        backbone = DINOv2("cuda")

    if args.detection_head == "faster_rcnn":
        # Your current Faster R-CNN code (anchor generator, roi pooler, etc.)
        """anchor_generator = AnchorGenerator(
            sizes=((2, 4, 8, 16, 32, 64, 128, 256),),
            aspect_ratios=((0.25, 0.5, 1.0, 2.0, 4.0),)
        )

        roi_pooler = torchvision.ops.MultiScaleRoIAlign(
            featmap_names=['0'], output_size=7, sampling_ratio=2
        )"""
        """anchor_generator = AnchorGenerator(
            sizes=((2, 4, 8, 16, 32, 64, 128, 256),),
            aspect_ratios=((0.25, 0.5, 1.0, 2.0, 4.0),)
        )"""
        anchor_generator = AnchorGenerator(
            sizes=((16,), (32,), (64,), (128,)),  # 5 feature maps
            aspect_ratios=((0.25, 0.5, 1.0, 2.0, 4.0),) * 4
        )

        roi_pooler = torchvision.ops.MultiScaleRoIAlign(
            featmap_names=['3', '5', '8','11'],
            output_size=7,
            sampling_ratio=2
        )
        """roi_pooler = torchvision.ops.MultiScaleRoIAlign(
            featmap_names=['0', '1', '2', '3'],  # must match FPN outputs
            output_size=7,
            sampling_ratio=2
        )

        """
        model = FasterRCNN(
            backbone,
            num_classes=args.nb_classes,
            rpn_anchor_generator=anchor_generator,
            min_size=224,
            transform=custom_transform,
            image_mean=mean,
            image_std=std,
            box_roi_pool=roi_pooler,
            
        )
        """rpn_pre_nms_top_n_train = 2000,
            rpn_post_nms_top_n_train = 1000,
            rpn_pre_nms_top_n_test = 1000,
            rpn_post_nms_top_n_test = 1000,
            rpn_nms_thresh = 0.7,
            box_score_thresh = 0.05,
            box_nms_thresh = 0.5,
            box_detections_per_img = 100"""
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, args.nb_classes)

    elif args.detection_head == "linear_probing":
        # 🔒 Freeze backbone
        for p in backbone.parameters():
            p.requires_grad = False

        # 🧠 Build simple linear classifier
        model = nn.Sequential(
            backbone,
            nn.Flatten(start_dim=1),  # assumes backbone outputs [B, C, H, W] or [B, C]
            nn.Linear(backbone.out_channels, args.nb_classes)
        )

    else:
        raise ValueError(f"Unknown detection_head: {args.detection_head}")


    if args.finetune and not args.eval:
        checkpoint = torch.load(args.finetune, map_location='cpu')
        print("Load pre-trained checkpoint from: %s" % args.finetune)

        # Get raw MAE encoder weights
        raw_checkpoint_model = checkpoint['model']
        print("Raw checkpoint keys:", raw_checkpoint_model.keys())

        # Add "backbone." prefix to encoder keys
        checkpoint_model = {}
        for k, v in raw_checkpoint_model.items():
            if k.startswith("decoder") or k.startswith("mask_token") or k.startswith("decoder_pred"):
                continue  # skip decoder-related stuff
            checkpoint_model[f"backbone.{k}"] = v

        state_dict = model.state_dict()

        # Remove mismatched keys
        for k in list(checkpoint_model.keys()):
            if k in state_dict and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} due to shape mismatch: {checkpoint_model[k].shape} vs {state_dict[k].shape}")
                del checkpoint_model[k]

        #print("Before interpolation:", checkpoint_model['backbone.pos_embed'].shape)
        interpolate_pos_embed(model, checkpoint_model)
        #print("After interpolation (model):", model.backbone.pos_embed.shape)

        print("State dict keys:", list(state_dict.keys())[:20])
        print("Checkpoint keys:", list(checkpoint_model.keys())[:20])
        # load pre-trained model
        msg = model.load_state_dict(checkpoint_model, strict=False)
        print(msg)
        
        # TODO: change assert msg based on patch_embed
        if args.global_pool:
            print(set(msg.missing_keys))
            # assert set(msg.missing_keys) == {'head.weight', 'head.bias', 'fc_norm.weight', 'fc_norm.bias'}
        else:
            print(set(msg.missing_keys))
            # assert set(msg.missing_keys) == {'head.weight', 'head.bias'}

        # manually initialize fc layer
        #trunc_normal_(model.head.weight, std=2e-5)

    model.to(device)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if args.detection_head == "faster_rcnn":
        criterion = None
        train_fn = train_one_epoch_frcnn
        eval_fn = evaluate_frcnn
    else:
        criterion = nn.BCEWithLogitsLoss()
        train_fn = train_one_epoch_linear
        eval_fn = evaluate_linear


    print("Model = %s" % str(model_without_ddp))
    print('number of params (M): %.2f' % (n_parameters / 1.e6))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    def get_layer_id(name, num_layers):
        """
        Assigns a layer ID based on parameter name for ViT-style models.
        Used to apply layer-wise learning rate decay.
        """
        if name.startswith("backbone.model.patch_embed") or name.startswith("backbone.model.cls_token") or name.startswith("backbone.model.pos_embed"):
            return 0
        elif name.startswith("backbone.model.blocks"):
            block_id = int(name.split('.')[3])  # e.g., 'backbone.model.blocks.11.attn.qkv.weight'
            return block_id + 1
        else:
            return num_layers - 1  # head or other non-transformer layers


    def get_vit_param_groups(model, base_lr, weight_decay, layer_decay=0.9):
        param_groups = []
        num_layers = 24
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            layer_id = get_layer_id(name, num_layers)
            lr = base_lr * (layer_decay ** (num_layers - layer_id - 1))
            param_groups.append({"params": [param], "lr": lr, "weight_decay": weight_decay})
        return param_groups

    # Use in optimizer
    optimizer = torch.optim.AdamW(get_vit_param_groups(model_without_ddp, args.lr, weight_decay=0.05))

    # build optimizer with layer-wise lr decay (lrd)
    param_groups = model_without_ddp.parameters()
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr)
    loss_scaler = NativeScaler()

    criterion = None

    print("criterion = %s" % str(criterion))

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    # Set up wandb
    if global_rank == 0 and args.wandb is not None:
        wandb.init(project=args.wandb, entity="mae-sentinel")
        wandb.config.update(args)
        wandb.watch(model)

    if args.eval:
        test_stats = evaluate_frcnn(data_loader_val, model, device)

        print(f"Evaluation on {len(val_dataset)} loss: {test_stats['loss']:.2f}")
        exit(0)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0
    for epoch in range(args.start_epoch, args.epochs):
        print(epoch)
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        if args.detection_head == "faster_rcnn":
            train_stats = train_fn(
                model, data_loader_train, optimizer, device, epoch,
                log_writer=log_writer, args=args,
            )
        else:
            train_stats = train_fn(
                model, data_loader_train, optimizer, device, epoch,
                criterion, log_writer=log_writer, args=args,
            )

        if args.output_dir and (epoch % args.save_every == 0 or epoch + 1 == args.epochs):
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)
        
            test_stats = eval_fn(data_loader_val, model, device, class_names=DIOR_CLASSES)

            # 💬 Print all losses nicely
            print(f"\n[Evaluation on {len(val_dataset)} images]")
            print(" 🔹 Losses:")
            print(f"   - loss_classifier    : {test_stats['loss_classifier']:.4f}")
            print(f"   - loss_box_reg       : {test_stats['loss_box_reg']:.4f}")
            print(f"   - loss_objectness    : {test_stats['loss_objectness']:.4f}")
            print(f"   - loss_rpn_box_reg   : {test_stats['loss_rpn_box_reg']:.4f}")

            print(f"mAP@[.5:.95]: {test_stats['mAP_50_95']:.2f}% | mAP@0.5: {test_stats['mAP_50']:.2f}%")

            # 🧾 Logging dictionary
            log_stats = {
                **{f"train_{k}": v for k, v in train_stats.items()},
                **{f"test_{k}": v for k, v in test_stats.items() if k != "per_class_ap"},  # Skip nested dict
                'epoch': epoch,
                'n_parameters': n_parameters
            }
            # ✍️ Save log
            if args.output_dir and misc.is_main_process():
                if log_writer is not None:
                    log_writer.flush()

                with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                    f.write(json.dumps(log_stats) + "\n")

                if args.wandb is not None:
                    try:
                        wandb.log(log_stats)
                    except ValueError:
                        print(f"Invalid stats?")


    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
