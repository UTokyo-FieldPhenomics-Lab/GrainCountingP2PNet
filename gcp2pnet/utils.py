import random

import torch
import numpy as np

from .misc import get_rank

# DeNormalize used to get original images
class DeNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor
    
def gen_random_scale_n(img, rnd=3):
    np.random.seed(rnd)
    scale = torch.tensor(np.random.uniform(0.1, 1.91, (1,1, img.size()[2],img.size()[3]))).float()
    return torch.mul(img, scale)

def print_args(args):
    print("="*50 + " Training Arguments " + "="*50)
    for arg in vars(args):
        print(f"{arg:>20}: {getattr(args, arg)}")
    print("="*120)

def fix_seed(args_seed):
    # fix the seed for reproducibility
    seed = args_seed + get_rank()
    seed = args_seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)