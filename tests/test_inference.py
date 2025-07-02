import os
import re
import shutil
from pathlib import Path

import torch
import pytest

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import patheffects
from adjustText import adjust_text
from PIL import Image

from gcp2pnet.inference import (
    get_inf_arguments, load_model, 
    load_image_to_tensor, apply_model,
    postprocess_point_clusters_one_class, postprocess_point_clusters,
    postprocess_merge_by_distance,
    draw_result_patch_figure
)

from gcp2pnet import datasets

def test_get_inf_arguments():
    args = get_inf_arguments()

    assert args.seed == 42
    assert args.weight_path == Path('demo_best_mae.pth')
    assert args.img_path == Path('.')

def test_load_model():
    args = get_inf_arguments()

    with pytest.raises(FileNotFoundError, match=re.escape("Could not load model weight")):
        args.weight_path = ''
        model = load_model(args)

    with pytest.raises(FileNotFoundError, match=re.escape("Could not load model weight")):
        args.weight_path = "./demo_not_exist.pth"
        model = load_model(args)

    args.weight_path = "./demo_best_mae.pth"
    model = load_model(args)

    assert model.num_classes == 2

def test_load_image_to_tensor():
    args = get_inf_arguments()

    img_path = "./data/20220207_17_Y_a_v03_h02.JPG"
    img_raw = Image.open(img_path).convert('RGB')

    img_np, img_tensor, resize_ratio = load_image_to_tensor(img_raw, args.device)

    assert img_tensor.shape == torch.Size([1, 3, 256, 256])
    assert img_np.shape == (256, 256, 3)
    assert resize_ratio == (1.0, 1.0)

@pytest.fixture
def setup_inference():
    args = get_inf_arguments()

    args.weight_path = "./demo_best_mae.pth"
    args.img_path = "./data/20220207_17_Y_a_v03_h02.JPG"

    # start inferencing
    model, checkpoints = load_model(args)

    img_raw = Image.open(args.img_path).convert('RGB')
    img_numpy, img_tensor, resize_ratio = load_image_to_tensor(img_raw, args.device)

    return args, model, img_numpy, img_tensor

def test_inference_raw_output(setup_inference):
    args, model, img_numpy, img_tensor = setup_inference

    ################
    #  source code
    ################

    # run inference
    outputs = model(img_tensor)

    # question here: why 16384=128*128 not 256*256?
    assert outputs['pred_points'].shape == torch.Size([1, 16384, 2])
    outputs_points = outputs['pred_points'][0]
    assert outputs_points.shape == torch.Size([16384, 2])

    # model.num_classes = label_class + 1 (0 as background I guess)
    assert outputs['pred_logits'].shape == torch.Size([1, 16384, 3]) 

    # iter each class
    test_class = i = 1
    ### for i in range(label_type_count -1)
    outputs_scores = torch.nn.functional.softmax(outputs['pred_logits'], -1)[:, :, i][0]

    points = outputs_points[outputs_scores > args.threshold].detach().cpu().numpy()#.tolist()
    scores = outputs_scores[outputs_scores > args.threshold].detach().cpu().numpy()#.tolist()

    assert points.shape == (75,2)
    assert scores.shape == (75,)

    ################
    # packed func
    ################

    raw_results = apply_model(model, img_tensor, args.threshold)

    assert raw_results[1]['points'].shape == (75, 2)
    assert raw_results[1]['scores'].shape == (75,)

    fig, ax = plt.subplots(1,1)
    ax.imshow(img_numpy)
    ax.scatter(*raw_results[1]['points'].T, c='r', s=1)
    ax.scatter(*raw_results[2]['points'].T, c='b', s=1)
    plt.savefig("tests/outputs/test_inference_raw_output.png")

def test_inference_post_processing(setup_inference):

    args, model, img_numpy, img_tensor = setup_inference

    raw_results = apply_model(model, img_tensor, args.threshold)

    # sub function 

    points_n, scores_n = postprocess_point_clusters_one_class(
        raw_results[1]['points'],
        raw_results[1]['scores']
    )

    assert points_n.shape == (6,2)
    assert scores_n.shape == (6,)

    points_n, scores_n = postprocess_point_clusters_one_class(
        raw_results[2]['points'],
        raw_results[2]['scores']
    )

    assert points_n.shape == (7,2)
    assert scores_n.shape == (7,)

    # merged function
    results_pd = postprocess_point_clusters(raw_results)

    assert len(results_pd) == 13

    ###############
    # draw figures
    ###############
    fig, ax = plt.subplots(1,1)
    ax.imshow(img_numpy)

    c = {1: 'r', 2: 'b'}
    class_color = [c[i] for i in results_pd.cls]
    ax.scatter(results_pd.x, results_pd.y, c=class_color, s=15, marker='o', edgecolors='w')

    texts = []
    for x, y, score, cls in zip(results_pd.x, results_pd.y, results_pd.score, results_pd.cls):
        texts.append(
            ax.text(x, y, f"{score:.2f}", ha='center', va='bottom', 
                    fontsize=10, color=c[cls], alpha=0.7,
                    path_effects=[patheffects.withStroke(linewidth=2, foreground='white')])
        )

    adjust_text(texts, force_text=0.1, arrowprops=dict(arrowstyle="-|>",
                                                    color='w', alpha=0.8))

    plt.savefig("tests/outputs/test_inference_postprocess_point_clusters.png")

def test_inference_postprocess_merge_by_distance(setup_inference):
    args, model, img_numpy, img_tensor = setup_inference
    raw_results = apply_model(model, img_tensor, args.threshold)
    results_df = postprocess_point_clusters(raw_results)
    filtered_df = postprocess_merge_by_distance(results_df, prox_distance=25)

    ###############
    # draw figures
    ###############

    fig, ax = plt.subplots(1,1)
    ax.imshow(img_numpy)

    c = {1: 'r', 2: 'b'}
    class_color = [c[i] for i in filtered_df.cls]
    ax.scatter(filtered_df.x, filtered_df.y, c=class_color, s=15, marker='o', edgecolors='w')

    for x, y, score, cls in zip(filtered_df.x, filtered_df.y, filtered_df.score, filtered_df.cls):
        ax.text(x, y-5, f"{score:.2f}", ha='center', va='bottom', 
                fontsize=10, color=c[cls], alpha=0.7,
                path_effects=[patheffects.withStroke(linewidth=2, foreground='white')])

    plt.savefig("tests/outputs/test_inference_postprocess_merge_by_distance.png")

    # draw_all_figures
    draw_result_patch_figure(img_numpy, raw_results, results_df, filtered_df, show=False,
                        save_path="tests/outputs/test_inference_all.png")

    # draw_gt
    gt_path = "./data/20220207_17_Y_a_v03_h02.txt"

    # parse gt file
    gt_data = []
    with open(gt_path) as f:
        for line in f:
            line = line.strip()[1:-1]  # remove brackets
            x, y, cls, _ = map(int, line.split(','))
            gt_data.append({'x': x, 'y': y, 'cls': cls})

    gt_df = pd.DataFrame(gt_data)

    # draw gt points
    fig, ax = plt.subplots(1,1)
    ax.imshow(img_numpy)

    class_color = [c[i] for i in gt_df.cls]
    ax.scatter(gt_df.x, gt_df.y, c=class_color, s=30, marker='o', edgecolors='w')
    plt.tight_layout()
    plt.savefig("tests/outputs/test_inference_gt_points.png")
    

def test_inference_by_sliding_window():
    # partly same as main()
    args = get_inf_arguments()

    os.environ["CUDA_VISIBLE_DEVICES"] = '{}'.format(args.gpu_id)

    args.weight_path = "./demo_best_mae.pth"
    args.img_path = "./data/20220207_18_G_a.JPG"
    args.sliding_window = True
    args.window_size = 256 * 3
    args.threshold = 0.5
    args.overlap_ratio = 0.2
    args.merge_distance = 25 * 3

    model, checkpoints = load_model(args)

    # load the images
    img_raw = Image.open(args.img_path).convert('RGB')

    # judge if need sliding window to produce results
    width, height = img_raw.size

    if args.sliding_window:
        patches= datasets.generate_patches((width, height), args.window_size, args.overlap_ratio)
        img_list = []
        for p in patches:
            # p -> (start_w, start_h, end_w, end_h)
            # img_raw_crop -> left, top, right, bottom)
            img_list.append( img_raw.crop( p ) )
    else:
        img_list = [img_raw]
        patches = [(0,0, width, height)]

    clustered_df_final = pd.DataFrame(columns=['x', 'y', 'score', 'cls'])
    for img, patch in zip(img_list, patches):
        img_numpy, img_tensor, resize_ratio = load_image_to_tensor(img, args.device, checkpoints['imgsz'] )
        raw_results = apply_model(model, img_tensor, args.threshold)

        clustered_df = postprocess_point_clusters(raw_results)

        if len(clustered_df) > 0:
            clustered_df['x'] = clustered_df['x'] / resize_ratio[0] + patch[0]
            clustered_df['y'] = clustered_df['y'] / resize_ratio[1] + patch[1]

            clustered_df_final = pd.concat([clustered_df_final, clustered_df]).reset_index(drop=True)
    
    # prox_distance need to resize according to window_size?
    merged_df = postprocess_merge_by_distance(clustered_df_final, prox_distance=args.merge_distance)

    print(merged_df)

    fig, ax = plt.subplots(1, 1, figsize=(width // args.window_size, height // args.window_size), dpi=600)

    ax.imshow(img_raw)

    c = {1: 'r', 2: 'b'}
    class_color = [c[i] for i in merged_df.cls]
    ax.scatter(merged_df.x, merged_df.y, c=class_color, s=15, marker='o', edgecolors='w')

    for x, y, score, cls in zip(merged_df.x, merged_df.y, merged_df.score, merged_df.cls):
        ax.text(x, y-5, f"{score:.2f}", ha='center', va='bottom', 
            fontsize=5, color=c[cls], alpha=0.7,
            path_effects=[patheffects.withStroke(linewidth=2, foreground='white')])
        
    ax.set_title("Merged by distance")

    plt.tight_layout()
    plt.savefig("tests/outputs/test_inference_sliding_window.png")