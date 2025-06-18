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

from . import misc

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
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

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
def evaluate_crowd_no_overlap(model, data_loader, device, class_n, threshold=0.5):
    model.eval()

    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', misc.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    
    # run inference on all images to calc MAE
    maes = []
    mses = []
    for samples, targets in data_loader:

        ####################################################
        # modify and add functions for multi-class counting
        # --------------------- start ----------------------

        # gt_cnt = targets[0]['point'].shape[0]
        gt_cnt_0 = 0
        gt_cnt_1 = 0

        for label in targets[0]['labels']:
            if label == 1:
                gt_cnt_0 += 1
            elif label == 2:
                gt_cnt_1 += 1

        samples = samples.to(device)

        # run inference
        outputs = model(samples)

        outputs_points = outputs['pred_points'][0]

        outputs_scores = []
        for i in range(class_n):
            outputs_scores.append(torch.nn.functional.softmax(outputs['pred_logits'], -1)[:, :, i + 1][0])


        # filter the predictions
        points_n_for_all_class = []
        scores_n_for_all_class = []

        for i in range(class_n):

            points = outputs_points   [outputs_scores[i] > threshold].detach().cpu().numpy()#.tolist()
            scores = outputs_scores[i][outputs_scores[i] > threshold].detach().cpu().numpy()#.tolist()

            if points.shape[0]<10000 and points.shape[0] != 0:
                cutoff = 500/points.shape[0]
                if cutoff<20:
                    cutoff = 20
                components = nx.connected_components(
                    nx.from_edgelist(
                        (i, j) for i, js in enumerate(
                            spatial.KDTree(points).query_ball_point(points, cutoff)
                        )
                        for j in js
                    )
                )


                clusters = {j: i for i, js in enumerate(components) for j in js}


                # reorganize the points to the order of clusters
                points_reo = np.zeros(points.shape)
                scores_reo = np.zeros(scores.shape)

                i = 0
                for key in clusters.keys():
                    points_reo[i,:] = points[key,:]
                    scores_reo[i] = scores[key]
                    i+=1


                # points_n has the same order as clusters
                res = [clusters[key] for key in clusters.keys()]
                res_n = np.array(res).reshape(-1,1)


                points_n = []
                scores_n = []
                for i in np.unique(res_n):

                    tmp_points = points_reo[np.where(res_n[:,0] == i)]
                    tmp_scores = scores_reo[np.where(res_n[:,0] == i)]

                    points_n.append( [np.mean(tmp_points[:,0]), np.mean(tmp_points[:,1])])
                    scores_n.append( [np.mean(tmp_scores[:])])
    #                    scores_n.append( [np.amax(tmp_scores[:])])

            else:
                points_n = points.tolist()
                scores_n = scores.tolist()

            points_n_for_all_class.append(points_n)
            scores_n_for_all_class.append(scores_n)


        #推論結果の点群を走査して、場所が重複するものがあれば、第三のカテゴリに変更する。(縦も横も7ピクセル以内かどうか)

        point_box = []
        score_box = []

        for i, point_set in enumerate(points_n_for_all_class):
            for p in point_set:
                if i == 0:
                    point_box.append([p,0])
                if i == 1:
                    point_box.append([p,1])

        for i, score_set in enumerate(scores_n_for_all_class):
            for s in score_set:
                if i == 0:
                    score_box.append([s,0])
                if i == 1:
                    score_box.append([s,1])


        prox_distance = 25  #これより近い距離に別の点があれば、同じ点であるとみなして統合します。

        count_0 = 0
        count_1 = 0

        #distance以内に、自分よりscoreが高い点が無ければ、valid_flagが１のまま残り、その点をカウントし、画像にプロットする。
        for i, test_point in enumerate(point_box):

            valid_flag = 1

            x_test_point = test_point[0][0]
            y_test_point = test_point[0][1]
            score_test   = score_box[i][0][0]
            class_test   = score_box[i][1]

            for j, compared_point in enumerate(point_box):
                x_compared_point = compared_point[0][0]
                y_compared_point = compared_point[0][1]
                score_compared   = score_box[j][0][0]
                dist = math.sqrt((x_test_point - x_compared_point) * (x_test_point - x_compared_point) + (y_test_point - y_compared_point) * (y_test_point - y_compared_point))

                if ((dist < prox_distance) and (score_test < score_compared)):
                    valid_flag = 0

            if valid_flag == 1:
                if class_test == 0:
                    count_0 = count_0 + 1
                else:
                    count_1 = count_1 + 1

        # ---------------------- end -----------------------
        # modify and add functions for multi-class counting
        ####################################################

        # accumulate MAE, MSE
        mae = abs(count_0 - gt_cnt_0) + abs(count_1 - gt_cnt_1)
        mse = (count_0 - gt_cnt_0) * (count_0 - gt_cnt_0) + (count_1 - gt_cnt_1) * (count_1 - gt_cnt_1)

        maes.append(float(mae))
        mses.append(float(mse))

    # calc MAE, MSE
    mae = np.mean(maes)
    mse = np.sqrt(np.mean(mses))

    return mae, mse