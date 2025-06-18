import argparse
import datetime
import os
import time
import yaml
from pathlib import Path

import warnings
warnings.filterwarnings('ignore')

import numpy as np

import torch
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from . import models, utils, datasets, misc, engine

def get_train_arguments():
    """Parse all the arguments provided from the CLI.
    Returns:
      A list of parsed arguments.
    """

    #Change into rice data !!! -> chage default value
    parser = argparse.ArgumentParser(description="P2PNet training script")

    # constant
    parser.add_argument('--lr', default=1e-3, type=float, # originall set 1e-3 and can be reduced at later training stage
                        help="learning rate for background model") 
    parser.add_argument('--lr_fpn', default=1e-4, type=float, help="learning rate for detection head")
    parser.add_argument('--batch_size', default=1, type=int)

    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--lr_drop', default=2000000, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float, help='gradient clipping max norm')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None, 
                        help="Path to the pretrained model. If set, only the mask head will be trained")

    # * Matcher is a Hungarian strategy, minimizing the cost of proposed points and gt points
    parser.add_argument('--set_cost_class', default=0.5, type=float, #クラスマッチングの失敗による重みづけ係数
                        help="Class coefficient in the matching cost") 

    parser.add_argument('--set_cost_point', default=0.5, type=float,
                        help="L1 point coefficient in the matching cost")

    # * Loss coefficients
    parser.add_argument('--point_loss_coef', default=0.02, type=float) # default = 0.0002 #Change into rice data !!! -> chage default value
    parser.add_argument('--eos_coef', default=0.01, type=float, # default 0.05, should be set low at the beginning !
                        help="Relative classification weight of the no-object class") # default = 0.5#Change into rice data !!! -> chage default value

    # a threshold during evaluation for counting and visualization
    parser.add_argument('--threshold', default=0.5, type=float,
                        help="threshold in evalluation: evaluate_crowd_no_overlap")#Change into rice data !!! -> chage default value
    parser.add_argument('--row', default=2, type=int,
                        help="row number of anchor points")#Change into rice data !!! -> chage default value
    parser.add_argument('--line', default=2, type=int,
                        help="line number of anchor points")#Change into rice data !!! -> chage default value

    # dataset parameters
    parser.add_argument('--dataset_folder', default='./data/dataset',
                        help='path where the dataset is')#Change into rice data !!! -> chage default value

    parser.add_argument('--output_dir', default="./runs",
                        help='the output folder contains checkpoints and logs')#Change into rice datawhere to save, empty for no saving')
    parser.add_argument('--run_name', default='p2pnet')
    # parser.add_argument('--checkpoints_dir', default = program_files_root_dir + 'CrowdCounting-P2PNet/MultiLevelPyramidFeature_01',
    #                     help='path where to save checkpoints, empty for no saving') #ckpt_5n was not bad, default 2 X 2#Change into rice data !!! -> chage default value
    # parser.add_argument('--tensorboard_dir', default=program_files_root_dir + 'CrowdCounting-P2PNet/Grain_runs',
    #                     help='path where to save, empty for no saving')#Change into rice data !!! -> chage default value

    parser.add_argument('--seed', default=42, type=int)#Change into rice data !!! -> chage default value
    parser.add_argument('--resume', default='', help='resume from checkpoint')#Change into rice data !!! -> chage default value
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')#Change into rice data !!! -> chage default value
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=1, type=int)
    parser.add_argument('--eval_freq', default=5, type=int,
                        help='frequency of evaluation, default setting is evaluating in every 5 epoch')#Change into rice data !!! -> chage default value
    parser.add_argument('--gpu_id', default=0, type=int, help='the gpu used for training')
    
    return parser.parse_known_args()[0] #if known else parser.parse_args()


def main(args):

    os.environ["CUDA_VISIBLE_DEVICES"] = '{}'.format(args.gpu_id)

    run_output_dir = Path(args.output_dir) / args.run_name
    run_output_dir.mkdir(parents=True, exist_ok=True)

    weights_dir = run_output_dir / 'weights'
    weights_dir.mkdir(parents=True, exist_ok=True)
    best_model_file = weights_dir / "best_mae.pth"

    args.checkpoints_dir = weights_dir
    args.vis_dir = run_output_dir

    args.tensorboard_dir = run_output_dir / "tensorboard_logs"
    args.tensorboard_dir.mkdir(parents=True, exist_ok=True)

    latest_ckpt = weights_dir / 'latest.pth'

    if latest_ckpt.exists() and args.resume == '':
        user_input = input(f"Found previous checkpoints at [{latest_ckpt}], resume training [Y/Yes] or retrain [N/No]? ").lower()
        if user_input in ['y', 'yes']:
            args.resume = str(latest_ckpt)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.device = device

    # create the logging file
    utils.print_args(args)
    run_log_name =  run_output_dir / 'args.yaml'
    with open(run_log_name, 'w', encoding="utf-8") as f:
        yaml.dump({k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, f)

    utils.fix_seed(args.seed)

    # get the P2PNet model
    model, criterion = models.p2pnet.build(args, training=True)

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

    # create the training and valiation set
    # train_set, val_set = loading_data(args.data_root)  # args_dataroot = training_data_root_dir
    train_set, val_set = datasets.loading_dataset( args.dataset_folder )
    label_dict, class_n = datasets.loading_label_dict( args.dataset_folder ) 

    # create the sampler used during training
    sampler_train = torch.utils.data.RandomSampler(train_set)
    sampler_val = torch.utils.data.SequentialSampler(val_set)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)
    
    # the dataloader for training
    data_loader_train = DataLoader(train_set, batch_sampler=batch_sampler_train,
                                    collate_fn=misc.collate_fn_crowd, 
                                    num_workers=args.num_workers)

    data_loader_val = DataLoader(val_set, 1, sampler=sampler_val,
                                 drop_last=False, collate_fn=misc.collate_fn_crowd, 
                                 num_workers=args.num_workers)

    if args.frozen_weights is not None:
        # checkpoint = torch.load(args.frozen_weights, map_location='cpu')
        checkpoint = torch.load(args.frozen_weights, weights_only=False, map_location=device)
        model_without_ddp.detr.load_state_dict(checkpoint['model'])

    # resume the weights and training state if exists
    if args.resume:
        print("Resume previous checkpoints, continue training")
        # checkpoint = torch.load(args.resume, map_location='cpu')
        checkpoint = torch.load(args.resume, weights_only=False, map_location=device)
        model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
        if not args.eval and \
            'optimizer' in checkpoint and \
            'lr_scheduler' in checkpoint and \
            'epoch' in checkpoint:

            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1
            print(checkpoint['epoch'] + 1)

    #######################
    print("Start training")
    #######################

    start_time = time.time()

    # the logger writer
    writer = SummaryWriter(args.tensorboard_dir)

    # save the performance during the training
    mae = []
    mse = []
    step = 0

    # training starts here
    for epoch in range(args.start_epoch, args.epochs):

        # run evaluation
        if epoch % args.eval_freq == 0:
            t1 = time.time()
            result = engine.evaluate_crowd_no_overlap(model, data_loader_val, device, class_n, args.threshold)
            t2 = time.time()


            # save the best model since begining
            if epoch > 0:
                if np.min(mse) > result[1]:
                # if abs(np.min(mae) - result[0]) < 0.01:
                    checkpoint_best_path = best_model_file
                    torch.save({
                        'model': model_without_ddp.state_dict(),
                    }, checkpoint_best_path)
                    print("===updated best model===")

            mae.append(result[0])
            mse.append(result[1])

            print("mae list:", np.asarray(mae))
            print("mse list:", np.asarray(mse))

            # print the evaluation results
            print('=======================================test=======================================')
            print("mae:", result[0], "mse:", result[1], "time:", t2 - t1, "best mse:", np.min(mse), )
            # with open(run_log_name, "a") as log_file:
                # log_file.write("mae:{}, mse:{}, time:{}, best mae:{}".format(result[0],result[1], t2 - t1, np.min(mae)))
                # log_file.write("mae:{}, mse:{}, time:{}, best mse:{}".format(result[0],result[1], t2 - t1, np.min(mse)))
            # recored the evaluation results
            if writer is not None:
                writer.add_scalar('metric/mae', result[0], step)
                writer.add_scalar('metric/mse', result[1], step)
                step += 1

        t1 = time.time()
        stat = engine.train_one_epoch(model, criterion, data_loader_train, optimizer, device, epoch, args.clip_max_norm)

        # record the training states after every epoch
        if writer is not None:
            # with open(run_log_name, "a") as log_file:
            #     log_file.write("loss/loss@{}: {}".format(epoch, stat['loss']))
            #     log_file.write("loss/loss_ce@{}: {}".format(epoch, stat['loss_ce']))
            print(f"epoch: {epoch} with loss= {stat['loss']} loss_ce={stat['loss_ce']}")
            writer.add_scalar('loss/loss', stat['loss'], epoch)
            writer.add_scalar('loss/loss_ce', stat['loss_ce'], epoch)

        t2 = time.time()
        print('[ep %d][lr %.7f][%.2fs]' % (epoch, optimizer.param_groups[0]['lr'], t2 - t1))
        # with open(run_log_name, "a") as log_file:
        #     log_file.write('[ep %d][lr %.7f][%.2fs]' % (epoch, optimizer.param_groups[0]['lr'], t2 - t1))
        
        # change lr according to the scheduler
        lr_scheduler.step()
        
        # save latest weights every epoch
        if not os.path.exists(args.checkpoints_dir):
            os.makedirs(args.checkpoints_dir)
        checkpoint_latest_path = os.path.join(args.checkpoints_dir, 'latest.pth')

        torch.save({
            'model': model_without_ddp.state_dict(),
        }, checkpoint_latest_path)

    # total time for training
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

if __name__ == '__main__':
    import sys
    sys.path.insert(0, '../')

    from gcp2pnet import models, utils, datasets, misc, engine

    args = get_train_arguments()
    main(args)
