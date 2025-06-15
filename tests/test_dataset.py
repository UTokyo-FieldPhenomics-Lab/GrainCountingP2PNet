from gcp2pnet.datasets import SHHADataset, loading_dataset, loading_label_dict
import torch

dataset_dir = "data/dataset"

def test_demo_dataset_loading():
    train_set, valid_set = loading_dataset( dataset_dir )

    assert len(train_set) == 3029
    assert len(valid_set) == 1333

    assert train_set[0][0].shape == torch.Size([1, 3, 256, 256])


def test_parse_dataset_classes_json():

    label_dict, class_n = loading_label_dict( dataset_dir ) 

    gt_label_dict = {'Fill': 1, '平べったいけど沈む': 1, '平べったくて浮く': 2, '詰まっているけど浮く': 2, 'Unfill': 2}

    assert label_dict == gt_label_dict

    assert class_n == 2