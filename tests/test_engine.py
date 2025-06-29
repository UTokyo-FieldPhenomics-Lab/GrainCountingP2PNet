from pathlib import Path

import pytest

import torch
from torch.utils.data import DataLoader

from gcp2pnet import datasets, models, engine, utils, train, misc

@pytest.fixture
def setup_init_model():

    args = train.get_train_arguments()

    args.dataset_folder = Path("data/demo_dataset")
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    utils.fix_random_seed(args.seed)

    # create the training and valiation set
    train_set, val_set = datasets.loading_dataset( args.dataset_folder )
    label_dict, num_classes = datasets.loading_label_dict( args.dataset_folder / "classes.json") 

    # create the sampler used during training
    sampler_train = torch.utils.data.RandomSampler(train_set)
    sampler_val = torch.utils.data.SequentialSampler(val_set)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)

    # get the P2PNet model
    model= models.p2pnet.build_model(args, num_classes)
    criterion = models.p2pnet.build_criterion(args, num_classes)

    # move to GPU
    model.to(args.device)
    criterion.to(args.device)

    model_without_ddp = model

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)
    # use different optimation params for different parts of the model
    param_dicts = [
        {"params": [p for n, p in model_without_ddp.named_parameters() if "fpn" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if "fpn" in n and p.requires_grad],
            "lr": args.lr_fpn,
        },
    ]
    # Adam is used by default
    optimizer = torch.optim.Adam(param_dicts, lr=args.lr)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    
    # the dataloader for training
    data_loader_train = DataLoader(train_set, batch_sampler=batch_sampler_train,
                                    collate_fn=misc.collate_fn_crowd, 
                                    num_workers=args.num_workers)

    data_loader_val = DataLoader(val_set, 1, sampler=sampler_val,
                                 drop_last=False, collate_fn=misc.collate_fn_crowd, 
                                 num_workers=args.num_workers)
    
    return model, data_loader_val, args.device, num_classes, args.threshold


def test_evaluate_crowd_no_overlap(setup_init_model):

    model, data_loader_val, device, num_classes, threshold = setup_init_model

    result = engine.evaluate_crowd_no_overlap(model, data_loader_val, device, num_classes, threshold)

    assert len(result) == 2
    assert result[0] == 5.373692077727952
    assert result[1] == 5.604290476847085