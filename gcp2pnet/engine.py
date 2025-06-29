# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Modified from https://github.com/TencentYoutuResearch/CrowdCounting-P2PNet/blob/main/engine.py
Train and eval functions used in main.py
Mostly copy-paste from DETR (https://github.com/facebookresearch/detr).
"""
import os
import sys
import math
from typing import Iterable

import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as standard_transforms
import networkx as nx

from scipy import spatial

from . import misc, inference

"""
def vis(samples, targets, pred, vis_dir, epoch, predict_cnt, gt_cnt):
    '''
    samples -> tensor: [batch, 3, H, W]
    targets -> list of dict: [{'points':[], 'image_id': str}]
    pred -> list: [num_preds, 2]
    '''
    gts = [t['point'].tolist() for t in targets]

    pil_to_tensor = standard_transforms.ToTensor()

    restore_transform = standard_transforms.Compose([
        DeNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        standard_transforms.ToPILImage()
    ])
    # draw one by one
    for idx in range(samples.shape[0]):
        sample = restore_transform(samples[idx])
        sample = pil_to_tensor(sample.convert('RGB')).numpy() * 255
        sample_gt = sample.transpose([1, 2, 0])[:, :, ::-1].astype(np.uint8).copy()
        sample_pred = sample.transpose([1, 2, 0])[:, :, ::-1].astype(np.uint8).copy()

        max_len = np.max(sample_gt.shape)

        size = 5
        # draw gt
        for t in gts[idx]:
            sample_gt = cv2.circle(sample_gt, (int(t[0]), int(t[1])), size, (0, 255, 0), -1)
        # draw predictions
        for p in pred[idx]:
            sample_pred = cv2.circle(sample_pred, (int(p[0]), int(p[1])), size, (0, 0, 255), -1)

        name_1 = targets[idx]['image_id_1']
        name_2 = targets[idx]['image_id_2']
        #################
        fig = plt.figure()
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.imshow(sample_gt)
        ax1.get_xaxis().set_visible(False)
        ax1.get_yaxis().set_visible(False)
        ax2 = fig.add_subplot(1, 2, 2)
        ax2.imshow(sample_pred)
        ax2.get_xaxis().set_visible(False)
        ax2.get_yaxis().set_visible(False)
        fig.suptitle('manual count=%4.2f, inferred count=%4.2f'%(gt_cnt, predict_cnt), fontsize=10)
        plt.tight_layout(rect=[0, 0, 0.95, 0.95]) # maize tassels counting
        plt.savefig(os.path.join(vis_dir, '{}_{}_id_{}_ind_{}.jpg'.format(epoch, idx, int(name_1), int(name_2))), bbox_inches='tight', dpi = 300)
        plt.close()
"""

# the training routine
def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0):
    model.train()
    criterion.train()
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))

    # iterate all training samples
    for samples, targets in data_loader:
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items() if k in ['point', 'labels']} for t in targets]

        # forward
        outputs = model(samples)
        # calc the losses
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce all losses (get the mean values)
        loss_dict_reduced = misc.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        # backward
        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        # update logger
        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


# the inference routine
@torch.no_grad()
def evaluate_crowd_no_overlap(model, data_loader, device, num_classes, threshold=0.5):
    model.eval()

    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', misc.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    
    # run inference on all images to calc MAE
    maes, mses  = [], []

    for samples, targets in data_loader:

        ####################################################
        # modify and add functions for multi-class counting
        # --------------------- start ----------------------

        # Count ground truth points per class
        gt_counts = [0] * num_classes

        for label in targets[0]['labels']:
            if 1 <= label <= num_classes:
                gt_counts[label-1] += 1

        samples = samples.to(device)

        # inference the result
        raw_results = inference.apply_model(model, samples, threshold)
        clustered_df = inference.postprocess_point_clusters(raw_results)
        merged_df = inference.postprocess_merge_by_distance(clustered_df, prox_distance=25)

        pred_counts = [0] * num_classes
        for label in range(1, num_classes+1):
            pred_counts[label-1] = len(merged_df[merged_df.cls == label])

        # Calculate metrics
        mae = sum(abs(pred - gt) for pred, gt in zip(pred_counts, gt_counts))
        mse = sum((pred - gt)**2 for pred, gt in zip(pred_counts, gt_counts))

        # ---------------------- end -----------------------
        # modify and add functions for multi-class counting
        ####################################################

        maes.append(float(mae))
        mses.append(float(mse))

    # calc MAE, MSE
    mae = np.mean(maes)
    mse = np.sqrt(np.mean(mses))

    return mae, mse