# this code is partly modified from 
# https://github.com/TencentYoutuResearch/CrowdCounting-P2PNet/blob/main/crowd_datasets/

import os
import json
import random
from pathlib import Path

import cv2
import torch
import numpy as np
import pandas as pd
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

############################################################
# self defined functions to process v7labs annotation data
############################################################
def parse_v7labs_json_file(json_path, label_dict):
    output = pd.DataFrame(columns=['cls', 'x', 'y'])
    with open(json_path) as f:
        jsonfile = json.load(f)
        keypoints = jsonfile["annotations"]

        for i, keypoint in enumerate(keypoints):
            if "keypoint" in keypoint.keys():
                label_id = label_dict[str(keypoint["name"])]
                x = int(keypoint["keypoint"]["x"])
                y = int(keypoint["keypoint"]["y"])

                if x < 0 or y < 0:
                    print(f"[Warning] drop annotation [{i}] with x={x} and y={y} for class [{keypoint['name']}]")
                    continue

                output.loc[len(output)] = {"x": x, "y": y, "cls": label_id}

    return output

def generate_patches(img_size, patch_size, overlap_ratio=0):
    """The function to crop images to patch for training data

    <------ Patch size ------>

                   <-overlap->
    +--------------+---------+
    |              |         |
    | <- stride -> |         |
    |     size     |         |
    |              |         |
    +--------------o---------+-------o
    |              |         |       |
    |              |         |       |
    +--------------+---------+       |
                   |                 |
                   |                 |
                   o-----------------o
                   <--  next patch -->

    Parameters
    ----------
    img_path : str
        The path string to image 
    label_pd : pd.DataFrame
        The dataframe after parse json file, columns = [cls, x, y] in raw images
    patch_size : int
        The width/height of each patch on raw images pixels
    overlap_ratio : int, optional
        the buffer area width/patch width inside the patch, by default 0
    trimming_size : int, optional
        the final output size, by default 256
        if patch_size larger then trimming size, resize patch to trimming size then outputs
        e.g. patch_size = 256*3, overlap_ratio = 0, trimming_size=256, 
             will shrink 1/3 smaller as final outputs

    Returns
    -------
    _type_
        _description_
    """
    img_h, img_w = img_size[:2]

    overlap_size = int(patch_size * overlap_ratio)

    patches = []
    stride = patch_size - overlap_size

    for y in range(0, img_h, stride):
        for x in range(0, img_w, stride):
            x1 = x
            y1 = y
            x2 = min(x + patch_size, img_w)
            y2 = min(y + patch_size, img_h)

            # Adjust if patch would exceed image boundaries
            if x2 - x1 < patch_size:
                x1 = max(0, x2 - patch_size)
            if y2 - y1 < patch_size:
                y1 = max(0, y2 - patch_size)

            patches.append((x1, y1, x2, y2))

    return patches
    
def generate_patches_with_labels(img_np, patches, label_df, trimming_size=256):

    output_patches = []

    for p in patches:

        (start_w, start_h, end_w, end_h) = p

        recorded_points = label_df[
            (label_df.x >= start_w) & (label_df.x <= end_w) & \
            (label_df.y >= start_h) & (label_df.y <= end_h)]
        
        if len(recorded_points) == 0:
            continue

        # for patches with annotations
        img_cropped = img_np[start_h:end_h, start_w:end_w]

        recorded_points.x -= start_w
        recorded_points.y -= start_h

        patch_size = end_w - start_w

        ratio = trimming_size / patch_size

        if ratio != 1:
            img_cropped = cv2.resize(img_cropped, (trimming_size, trimming_size))
            recorded_points.x *= ratio
            recorded_points.y *= ratio

        output_patches.append( {'imarray': img_cropped, 'label': recorded_points, 'patch_on_raw': p} )

    return output_patches