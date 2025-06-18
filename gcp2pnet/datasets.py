# this code is partly modified from 
# https://github.com/TencentYoutuResearch/CrowdCounting-P2PNet/blob/main/crowd_datasets/

import json
import random
from pathlib import Path

import cv2
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as standard_transforms


class SHHADataset(Dataset):

    def __init__(self, image_file_list, label_file_list, transform=None, train=False, patch=False, flip=False, trimming_size=256):

        self.transform = transform
        self.train = train
        self.patch = patch
        self.flip = flip
        self.trimming_size = trimming_size

        self.image_file_list = image_file_list
        self.label_file_list = label_file_list

        # number of samples
        self.nSamples = len(self.image_file_list)

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'

        image_path = self.image_file_list[index]
        label_path = self.label_file_list[index]

        # load image and ground truth
        img = self.load_image_data(image_path)
        point, labels = self.load_label_data(label_path)

        # applu augumentation
        if self.transform is not None:
            img = self.transform(img)

        if self.train:
            # data augmentation -> random scale
            scale_range = [0.5, 1.3]
            min_size = min(img.shape[1:])
            scale = random.uniform(*scale_range)
            # scale the image and points
            if scale * min_size > self.trimming_size:
                img = torch.nn.functional.upsample_bilinear(img.unsqueeze(0), scale_factor=scale).squeeze(0)
                point *= scale

        # random crop augumentaiton
        if self.train and self.patch:
            img, point, labels = self.random_crop_augment(img, point, labels)

            for i, _ in enumerate(point):  #03/21 debug
                point[i] = torch.Tensor(point[i])
                labels[i] = torch.Tensor(labels[i])

        # random flipping
        if random.random() > 0.5 and self.train and self.flip:
            # random flip
            img = torch.Tensor(img[:, :, :, ::-1].copy())

            for i, _ in enumerate(point):
                point[i][:, 0] = self.trimming_size - point[i][:, 0]

        if not self.train:
            point = [point]
            labels = [labels]

        img = torch.Tensor(img)
        # pack up related infos
        target = [{} for i in range(len(point))]

        for i, _ in enumerate(point):  #03/21 debug
            target[i]['point'] = torch.Tensor(point[i])

            if len(labels[0]) > 1:
                target[i]['labels'] = torch.Tensor(labels[i].flatten()).long()
            else:
                target[i]['labels'] = torch.Tensor(labels[i]).long()

            # image_id_1 = int(image_path.split('/')[-1].split('.')[0][5:7])
            # image_id_1 = int(image_path.name[5:7])
            # image_id_1 = torch.Tensor([image_id_1]).long()
            
            # image_id_2 = int(image_path.split('/')[-1].split('.')[0][5:7])
            # image_id_2 = int(image_path.name[5:7])
            # image_id_2 = torch.Tensor([image_id_2]).long()

            # target[i]['image_id_1'] = image_id_1
            # target[i]['image_id_2'] = image_id_2

            target[i]['image_path'] = image_path
            target[i]['label_path'] = label_path

        return img, target
    
    @staticmethod
    def load_image_data(img_path):

        img = cv2.imread(img_path)

        if not img is None:
            img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        
        return img

    @staticmethod
    def load_label_data(anno_json_path):

        points = []
        labels = []
        with open(anno_json_path, 'r') as f:
            pts = f.read().splitlines()
            for pt_0 in pts:
                pt = eval(pt_0)
                x = float(pt[0])#/2
                y = float(pt[1])#/2
                label = float(pt[2])
                points.append([x, y])
                labels.append([label])

        return np.array(points), np.array(labels)
    
    @staticmethod
    def random_crop_augment(img, den, labels, num_patch=1, trimming_size=256):
        half_h = trimming_size
        half_w = trimming_size
        result_img = np.zeros([num_patch, img.shape[0], half_h, half_w])
        result_den = []
        result_label = []

        # crop num_patch for each image
        for i in range(num_patch):
            start_h = random.randint(0, img.size(1) - half_h)
            start_w = random.randint(0, img.size(2) - half_w)
            end_h = start_h + half_h
            end_w = start_w + half_w
            # copy the cropped rect
            result_img[i] = img[:, start_h:end_h, start_w:end_w]

            # copy the cropped points
            idx = (den[:, 0] >= start_w) & (den[:, 0] <= end_w) & (den[:, 1] >= start_h) & (den[:, 1] <= end_h)
            # shift the corrdinates
            record_den = den[idx]
            record_label = labels[idx]
            record_den[:, 0] -= start_w
            record_den[:, 1] -= start_h

            result_den.append(record_den)
            result_label.append(record_label)

        return result_img, result_den, result_label
    
def _listdir_all_images(pathlib_folder):
    return list(pathlib_folder.glob("*.[jJ][pP][gG]")) + \
            list(pathlib_folder.glob("*.[jJ][pP][eE][gG]")) + \
            list(pathlib_folder.glob("*.[pP][nN][gG]")) + \
            list(pathlib_folder.glob("*.[bB][mM][pP]")) + \
            list(pathlib_folder.glob("*.[tT][iI][fF][fF]"))

def _match_image_and_label(img_list, lbl_list):
    img_list_ordered = []
    lbl_list_ordered = []
    img_stems = {img.stem: img for img in img_list}
    lbl_stems = {lbl.stem: lbl for lbl in lbl_list}

    for stem in img_stems:
        if stem in lbl_stems:
            img_list_ordered.append(img_stems[stem])
            lbl_list_ordered.append(lbl_stems[stem])

    return img_list_ordered, lbl_list_ordered
    
def loading_dataset(dataset_root):
    dataset_root = Path(dataset_root)
    # load train and valid labels
    train_image_folder = dataset_root / "images" / "train"
    valid_image_folder = dataset_root / "images" / "valid"

    train_label_folder = dataset_root / "labels" / "train"
    valid_label_folder = dataset_root / "labels" / "valid"

    train_image_list = _listdir_all_images(train_image_folder)
    valid_image_list = _listdir_all_images(valid_image_folder)

    train_label_list = list(train_label_folder.glob("*.txt"))
    valid_label_list = list(valid_label_folder.glob("*.txt"))

    train_image_list, train_label_list = _match_image_and_label(train_image_list, train_label_list)
    valid_image_list, valid_label_list = _match_image_and_label(valid_image_list, valid_label_list)

    # the pre-proccssing transform
    transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]),
    ])

    
    train_set = SHHADataset(train_image_list, train_label_list, train=True, 
                            transform=transform, patch=True, flip=True)
    
    valid_set = SHHADataset(valid_image_list, valid_label_list, train=False, 
                            transform=transform)

    return train_set, valid_set


def loading_label_dict(dataset_root):

    dataset_root = Path(dataset_root)

    label_json_file = dataset_root / "classes.json"

    with open(label_json_file, 'r', encoding='utf-8') as f:
        label_dict = json.load(f)

    class_n = len ( np.unique( np.asarray( list(label_dict.values() ) ) ) )

    return label_dict, class_n

# self defined functions to process v7labs annotation data
# todo