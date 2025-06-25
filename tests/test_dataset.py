from gcp2pnet.datasets import (
    SHHADataset, loading_dataset, loading_label_dict,
    parse_v7labs_json_file
)
from gcp2pnet.utils import fix_random_seed
import torch

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

def test_parse_v7labs_json():

    v7lab_test_json = "./data/demo_raw/training_labels/20220207_16_B_a.json"

    output = parse_v7labs_json_file(v7lab_test_json, gt_label_dict)

    assert len(output) == 134
    assert output.cls[0] == 2
    assert output.x[0] == 4240
    assert output.y[0] == 6439