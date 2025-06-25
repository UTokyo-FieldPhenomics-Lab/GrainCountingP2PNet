import random

import cv2
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches


from gcp2pnet.datasets import (
    SHHADataset, loading_dataset, loading_label_dict,
    parse_v7labs_json_file, generate_patches,
    generate_patches_with_labels
)
from gcp2pnet.utils import fix_random_seed

dataset_dir = "data/demo_dataset"

gt_label_dict = {'Fill': 1, '平べったいけど沈む': 1, '平べったくて浮く': 2, '詰まっているけど浮く': 2, 'Unfill': 2}

def test_demo_dataset_loading():
    train_set, valid_set = loading_dataset( dataset_dir )

    assert len(train_set) == 3029
    assert len(valid_set) == 1333

    assert train_set[0][0].shape == torch.Size([1, 3, 256, 256])
    assert train_set[0][1][0]['image_path'].stem == train_set[0][1][0]['label_path'].stem

def test_parse_dataset_classes_json():

    label_dict, class_n = loading_label_dict( dataset_dir ) 

    

    assert label_dict == gt_label_dict

    assert class_n == 2


v7lab_test_json = "./data/demo_raw/training_labels/20220207_16_B_a.json"
v7lab_test_img  = "./data/demo_raw/training_images/20220207_16_B_a.JPG"
color_dict = {1: 'r', 2: 'b'}

def test_parse_v7labs_json():

    output = parse_v7labs_json_file(v7lab_test_json, gt_label_dict)

    assert len(output) == 134
    assert output.cls[0] == 2
    assert output.x[0] == 4240
    assert output.y[0] == 6439

def test_generate_patches():

    label_df = parse_v7labs_json_file(v7lab_test_json, gt_label_dict)

    img_np = cv2.imread(v7lab_test_img)
    
    patch_list = generate_patches(img_np.shape, patch_size=256*4, overlap_ratio=0.1)

    fig, ax = plt.subplots(1, figsize=(10,10))
    ax.imshow(img_np)

    for p in patch_list:
        rnd_color = plt.cm.get_cmap('Dark2')(random.random())
        rect = patches.Rectangle((p[0], p[1]), p[2]-p[0], p[3]-p[1], linewidth=1, edgecolor=rnd_color, facecolor=rnd_color, alpha=0.3)
        ax.add_patch(rect)

    ax.scatter( label_df.x, label_df.y, c=[color_dict[cls] for cls in label_df.cls], marker='o', s=15)

    plt.tight_layout()
    plt.savefig("tests/outputs/test_dataset_generate_patch_overlap_0.1_preview.png")
    plt.close(fig)

    patch_list = generate_patches(img_np.shape, patch_size=256*3, overlap_ratio=0.0)

    fig, ax = plt.subplots(1, figsize=(10,10))
    ax.imshow(img_np)

    for p in patch_list:
        rnd_color = plt.cm.get_cmap('Dark2')(random.random())
        rect = patches.Rectangle((p[0], p[1]), p[2]-p[0], p[3]-p[1], linewidth=1, edgecolor=rnd_color, facecolor=rnd_color, alpha=0.3)
        ax.add_patch(rect)

    ax.scatter( label_df.x, label_df.y, c=[color_dict[cls] for cls in label_df.cls], marker='o', s=15)

    plt.tight_layout()
    plt.savefig("tests/outputs/test_dataset_generate_patch_no_overlap_preview.png")
    plt.close(fig)

def test_generate_patches_with_labels():

    label_df = parse_v7labs_json_file(v7lab_test_json, gt_label_dict)
    img_np = cv2.imread(v7lab_test_img)
    patch_list = generate_patches(img_np.shape, patch_size=256*3, overlap_ratio=0.0)

    output_patch_list = generate_patches_with_labels(img_np, patch_list, label_df, trimming_size=256)

    assert len(output_patch_list) == 24 # the demo_dataset/labels/train number

    demo_test = output_patch_list[10]

    assert demo_test['imarray'].shape == (256, 256, 3)

    #########################
    # drawing check figures
    #########################
    fig, ax = plt.subplots(4,6, figsize=(12,8))

    for n, patch in enumerate(output_patch_list):
        i, j = n // 6, n % 6

        ax[i,j].imshow(patch['imarray'])
        ax[i,j].scatter(patch['label'].x, patch['label'].y, c=[color_dict[cls] for cls in patch['label'].cls], marker='o', s=15)
        # ax[i,j].axis('off')

    plt.tight_layout()
    plt.savefig("tests/outputs/test_dataset_generate_patches_with_labels_preview.png")
    plt.close(fig)