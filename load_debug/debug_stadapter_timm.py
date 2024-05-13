import argparse
import datetime
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json
import os
from pathlib import Path
from collections import OrderedDict
import torch.nn.functional as F
import sys

sys.path.append("/Users/yangshu/Documents/SurgPETL/")

from mixup import Mixup
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from datasets.transforms.optim_factory import (
    create_optimizer,
    get_parameter_groups,
    LayerDecayValueAssigner,
)

import utils
from model import modeling_phase_finetune_stadapter_timm
# import modeling_phase_finetune_aim_petl


def get_args():
    parser = argparse.ArgumentParser(
        "SurgVideoMAE fine-tuning and evaluation script for video phase recognition",
        add_help=False,
    )

    # Model parameters
    parser.add_argument(
        "--model",
        default="vit_base_224_stadapter12x384",
        type=str,
        metavar="MODEL",
        help="Name of model to train",
    )
    parser.add_argument(
        "--pretrained_data",
        default="wit400m",
        choices=["imagenet21k", "wit400m_in12k", "wit400m", "laion2b_in12k", "laion2b", "laion400m", "surgery"],
        type=str,
        help="dataset",
    )
    parser.add_argument("--input_size", default=224, type=int, help="videos input size")
    parser.add_argument("--patch_size", default=16, type=int, help="videos patch size")
    parser.add_argument("--pre_norm", action="store_true", default=False)
    parser.add_argument("--qkv_bias", action="store_true", default=False)
    parser.add_argument("--surgery", action="store_true", default=False)
    parser.add_argument("--layer_decay", type=float, default=0.75)

    parser.add_argument(
        "--drop_path",
        type=float,
        default=0.2,
        metavar="PCT",
        help="Drop path rate (default: 0.1)",
    )

    parser.add_argument(
        "--opt",
        default="adamw",
        type=str,
        metavar="OPTIMIZER",
        help='Optimizer (default: "adamw"',
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.05, help="weight decay (default: 0.05)"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        metavar="LR",
        help="learning rate (default: 1e-3)",
    )

    return parser.parse_args()


def main(args):
    if args.pretrained_data == "imagenet21k":
        if "base" in args.model and args.patch_size == 16:
            pretrained_path = "/Users/yangshu/Documents/SurgPETL/pretrain_params/vit_base_patch16_224_augreg_in21k.bin"
            args.qkv_divided = False
            args.qkv_bias = True
        elif "large" in args.model and args.patch_size == 16:
            pretrained_path = "/Users/yangshu/Documents/SurgPETL/pretrain_params/vit_large_patch16_224_augreg_in21k.bin"
            args.qkv_divided = False
            args.qkv_bias = True
    elif args.pretrained_data == "wit400m_in12k":
        if "base" in args.model and args.patch_size == 16:
            pretrained_path = "/Users/yangshu/Documents/SurgPETL/pretrain_params/vit_base_patch16_224_wit400m_ftin12k.bin"
            args.qkv_divided = False
            args.pre_norm = True
            args.qkv_bias = True
        elif "large" in args.model and args.patch_size == 14:
            pretrained_path = "/Users/yangshu/Documents/SurgPETL/pretrain_params/vit_large_patch14_224_wit400m_ftin12k.bin"
            args.qkv_divided = False
            args.pre_norm = True
            args.qkv_bias = True
    elif args.pretrained_data == "wit400m":
        if "base" in args.model and args.patch_size == 16:
            pretrained_path = "/Users/yangshu/Documents/SurgPETL/pretrain_params/vit_base_patch16_224_wit400m.pth"
            args.qkv_divided = False
            args.pre_norm = True
            args.qkv_bias = True
        elif "large" in args.model and args.patch_size == 14:
            pretrained_path = "/Users/yangshu/Documents/SurgPETL/pretrain_params/vit_large_patch14_224_wit400m.pth"
            args.qkv_divided = False
            args.pre_norm = True
            args.qkv_bias = True
    elif args.pretrained_data == "laion2b_in12k":
        if "base" in args.model and args.patch_size == 16:
            pretrained_path = "/Users/yangshu/Documents/SurgPETL/pretrain_params/vit_base_patch16_224_laion2b_ftin12k.bin"
            args.qkv_divided = False
            args.pre_norm = True
            args.qkv_bias = True
        elif "large" in args.model and args.patch_size == 14:
            pretrained_path = "/Users/yangshu/Documents/SurgPETL/pretrain_params/vit_large_patch14_224_laion2b_ftin12k.bin"
            args.qkv_divided = False
            args.pre_norm = True
            args.qkv_bias = True
    elif args.pretrained_data == "laion400m":
        if "base" in args.model and args.patch_size == 16:
            pretrained_path = "/Users/yangshu/Documents/SurgPETL/pretrain_params/vit_base_patch16_224_laion400m.pth"
            args.qkv_divided = False
            args.pre_norm = True
            args.qkv_bias = True
        elif "large" in args.model and args.patch_size == 14:
            pretrained_path = "/Users/yangshu/Documents/SurgPETL/pretrain_params/vit_large_patch14_224_laion400m.pth"
            args.qkv_divided = False
            args.pre_norm = True
            args.qkv_bias = True
    elif args.pretrained_data == "laion2b":
        if "base" in args.model and args.patch_size == 16:
            pretrained_path = "/Users/yangshu/Documents/SurgPETL/pretrain_params/vit_base_patch16_224_laion2b.pth"
            args.qkv_divided = False
            args.pre_norm = True
            args.qkv_bias = True
        elif "large" in args.model and args.patch_size == 14:
            pretrained_path = "/Users/yangshu/Documents/SurgPETL/pretrain_params/vit_large_patch14_224_laion2b.pth"
            args.qkv_divided = False
            args.pre_norm = True
            args.qkv_bias = True
    # 手术数据预训练模型
    elif args.pretrained_data == "surgery":
        if "base" in args.model and args.patch_size == 16:
            pretrained_path = "/Users/yangshu/Documents/SurgPETL/pretrain_params/vit_base_patch16_224_surgery.pth"
            args.qkv_divided = True
            args.surgery = True
        elif "large" in args.model and args.patch_size == 16:
            pretrained_path = "/Users/yangshu/Documents/SurgPETL/pretrain_params/vit_large_patch16_224_surgery.pth"
            args.qkv_divided = True
            args.surgery = True

    model = create_model(
        args.model,
        pretrained=True,
        pretrain_path=pretrained_path,
        patch_size=args.patch_size,
        qkv_divided=args.qkv_divided,
        pre_norm=args.pre_norm,
        surgery=args.surgery,
        qkv_bias=args.qkv_bias
    )
    
    # num_layers = model.get_num_layers()

    # if args.layer_decay == 0.1:
    #     assigner = LayerDecayValueAssigner(
    #         [args.layer_decay] * (num_layers + 1) + [1.0]
    #     )
    # elif args.layer_decay < 1.0:  # 沿层以几何方式降低学习率
    #     assigner = LayerDecayValueAssigner(
    #         list(
    #             args.layer_decay ** (num_layers + 1 - i) for i in range(num_layers + 2)
    #         )
    #     )
    # else:
    #     assigner = None

    # if assigner is not None:
    #     print("Assigned values = %s" % str(assigner.values))

    # skip_weight_decay_list = model.no_weight_decay()
    # print("Skip weight decay list: ", skip_weight_decay_list)

    # optimizer = create_optimizer(
    #     args,
    #     model,
    #     skip_list=skip_weight_decay_list,
    #     get_num_layer=assigner.get_layer_id if assigner is not None else None,
    #     get_layer_scale=assigner.get_scale if assigner is not None else None,
    # )


if __name__ == "__main__":
    opts = get_args()

    main(opts)
