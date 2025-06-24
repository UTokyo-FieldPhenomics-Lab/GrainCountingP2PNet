import os
import re
import shutil

import torch
import pytest

import matplotlib.pyplot as plt
from matplotlib import patheffects
from adjustText import adjust_text

from gcp2pnet.inference import (
    get_inf_arguments, load_model, 
    load_image_to_tensor, apply_model,
    postprocess_point_clusters_one_class, postprocess_point_clusters,
    postprocess_merge_by_distance
)


def test_get_inf_arguments():
    args = get_inf_arguments()

    assert args.seed == 42
    assert args.weight_path == ''
    assert args.img_path == ''

def test_load_model():
    args = get_inf_arguments()

    with pytest.raises(FileNotFoundError, match=re.escape("Could not load model weight")):
        model = load_model(args)

    with pytest.raises(FileNotFoundError, match=re.escape("Could not load model weight")):
        args.weight_path = "./demo_not_exist.pth"
        model = load_model(args)

    args.weight_path = "./demo_best_mae.pth"
    model = load_model(args)

    assert model.num_classes == 3

def test_load_image_to_tensor():
    args = get_inf_arguments()

    with pytest.raises(FileNotFoundError, match=re.escape("Could not load image from")):
        img_path = "./img_path_not_exist.jpg"
        img_tensor = load_image_to_tensor(img_path, args.device)

    img_path = "./data/inference/20220207_17_Y_a_v03_h02.JPG"
    img_tensor = load_image_to_tensor(img_path, args.device)

    assert img_tensor.shape == torch.Size([1, 3, 256, 256])

@pytest.fixture
def setup_inference():
    args = get_inf_arguments()

    args.weight_path = "./demo_best_mae.pth"
    args.img_path = "./data/inference/20220207_17_Y_a_v03_h02.JPG"

    # start inferencing
    model = load_model(args)

    img_numpy, img_tensor = load_image_to_tensor(args.img_path, args.device)

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

    assert points.shape == (78,2)
    assert scores.shape == (78,)

    ################
    # packed func
    ################

    raw_results = apply_model(model, img_tensor, args.threshold)

    assert raw_results[1]['points'].shape == (78, 2)
    assert raw_results[1]['scores'].shape == (78,)

    fig, ax = plt.subplots(1,1)
    ax.imshow(img_numpy)
    ax.scatter(*raw_results[1]['points'].T, c='r', s=1)
    ax.scatter(*raw_results[2]['points'].T, c='b', s=1)
    plt.savefig("tests/outputs/test_inference_raw_output.png")

def test_inference_post_processing(setup_inference):

    args, model, img_numpy, img_tensor = setup_inference

    raw_results = apply_model(model, img_tensor, args.threshold)

    # sub function 
    for key in raw_results.keys():
        points_n, scores_n = postprocess_point_clusters_one_class(
            raw_results[key]['points'],
            raw_results[key]['scores']
        )

        assert points_n.shape == (6,2)
        assert scores_n.shape == (6,)

    # merged function
    results_pd = postprocess_point_clusters(raw_results)

    assert len(results_pd) == 12

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