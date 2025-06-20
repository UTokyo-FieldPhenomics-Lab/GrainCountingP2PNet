# Modified from https://github.com/TencentYoutuResearch/CrowdCounting-P2PNet/blob/main/run_test.py
import os
import argparse
import datetime
import random
import time
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path

import cv2
import torch
import torchvision.transforms as standard_transforms
import numpy as np

from PIL import Image

from . import models, utils, engine


def get_inf_arguments():
    """
    Parse all the arguments provided from the CLI.

    Returns:
        A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Object Counting Framework")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # a threshold during evaluation for counting and visualization
    parser.add_argument('--threshold', default=0.3, type=float,
                        help="threshold in evalluation: evaluate_crowd_no_overlap")
    parser.add_argument('--row', default=2, type=int,
                        help="row number of anchor points")
    parser.add_argument('--line', default=2, type=int,
                        help="line number of anchor points")
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--weight_path', default="", help='resume from checkpoint')
    parser.add_argument('--img_path', default="", help="The path to image")
    parser.add_argument('--num_workers', default=1, type=int)
    parser.add_argument('--gpu_id', default=0, type=int, help='the gpu used for training')
    parser.add_argument('--device', default=device, type=str, 
                        help="the torch running device, 'cpu' or 'cuda'")

    return parser.parse_known_args()[0] #if known else parser.parse_args()

def load_model(args):
    # ensure model file exists
    if not ( args.weight_path and os.path.exists(args.weight_path) ):
        raise FileNotFoundError(f"Could not load model weight from [{args.weight_path}]")

    utils.fix_random_seed(args.seed)

    # get the P2PNet
    model = models.P2PNet(args.row, args.line)
    model.to(args.device) # move to GPU

    # load trained model
    checkpoint = torch.load(args.weight_path, map_location=args.device)
    model.load_state_dict(checkpoint['model'])

    # convert to eval mode
    model.eval()

    return model

def load_image_to_tensor(img_path, device, trimming_size=256):
    # ensure image file exists
    if not ( img_path and os.path.exists(img_path) ):
        raise FileNotFoundError(f"Could not load image from [{img_path}]")

    # create the pre-processing transform
    transform = standard_transforms.Compose([
        standard_transforms.ToTensor(), 
        standard_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # load the images
    img_raw = Image.open(img_path).convert('RGB')

    # round the size
    width, height = img_raw.size
    new_width = width // trimming_size * trimming_size
    new_height = height // trimming_size * trimming_size

    img_resize = img_raw.resize((new_width, new_height), Image.ANTIALIAS)

    # pre-proccessing
    img_trans = transform(img_resize)

    img_tensor = torch.Tensor(img_trans).unsqueeze(0)
    img_tensor = img_tensor.to(device)

    return img_tensor


def main(args, debug=False):
    os.environ["CUDA_VISIBLE_DEVICES"] = '{}'.format(args.gpu_id)

    utils.print_args(args)

    model = load_model(args)

    img_tensor = load_image_to_tensor(args.img_path, args.device)

    # run inference
    outputs = model(img_tensor)
    outputs_scores = torch.nn.functional.softmax(outputs['pred_logits'], -1)[:, :, 1][0]

    outputs_points = outputs['pred_points'][0]

    # work to here

    threshold = 0.5
    # filter the predictions
    points = outputs_points[outputs_scores > threshold].detach().cpu().numpy().tolist()
    predict_cnt = int((outputs_scores > threshold).sum())

    outputs_scores = torch.nn.functional.softmax(outputs['pred_logits'], -1)[:, :, 1][0]

    outputs_points = outputs['pred_points'][0]
    # draw the predictions
    size = 2
    img_to_draw = cv2.cvtColor(np.array(img_raw), cv2.COLOR_RGB2BGR)
    for p in points:
        img_to_draw = cv2.circle(img_to_draw, (int(p[0]), int(p[1])), size, (0, 0, 255), -1)
    # save the visualized image
    cv2.imwrite(os.path.join(args.output_dir, 'pred{}.jpg'.format(predict_cnt)), img_to_draw)

if __name__ == '__main__':
    import sys
    sys.path.insert(0, '../')

    from gcp2pnet import models, utils, datasets, misc, engine

    args = get_inf_arguments()
    main(args)