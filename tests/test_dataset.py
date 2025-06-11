from gcp2pnet.datasets import SHHADataset, loading_dataset
import torch

dataset_dir = "data/dataset"

def test_demo_dataset_loading():
    train_set, valid_set = loading_dataset( dataset_dir )

    assert len(train_set) == 3029
    assert len(valid_set) == 1333

    assert train_set[0][0].shape == torch.Size([1, 3, 256, 256])