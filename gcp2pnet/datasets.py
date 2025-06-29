# this code is partly modified from 
# https://github.com/TencentYoutuResearch/CrowdCounting-P2PNet/blob/main/crowd_datasets/

import os
import json
import random
import argparse
import shutil
from pathlib import Path

import cv2
import torch
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from tqdm import tqdm
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
            target[i]['labels'] = torch.Tensor(labels[i].flatten()).long()

            target[i]['image_path'] = image_path
            target[i]['label_path'] = label_path

        return img, target
    
    @staticmethod
    def load_image_data(img_path):

        # img = cv2.imread(img_path)
        # if not img is None:
        #     img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        
        img = ImageOps.exif_transpose(Image.open(img_path))
        
        return img

    @staticmethod
    def load_label_data(anno_json_path):

        points = []
        labels = []
        with open(anno_json_path, 'r') as f:
            lines = f.read().splitlines()
            for line in lines:
                cls, x, y = line.split(' ')
                x = float(x)
                y = float(y)
                cls = float(cls)
                points.append([x, y])
                labels.append([cls])

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
    if not isinstance(pathlib_folder, Path):
        pathlib_folder = Path(pathlib_folder)

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

    not_in_label = []
    for stem in img_stems:
        if stem in lbl_stems:
            img_list_ordered.append(img_stems[stem])
            lbl_list_ordered.append(lbl_stems[stem])
        else:
            not_in_label.append(stem)

    not_in_img = []
    for stem in lbl_stems:
        if stem not in img_stems:
            not_in_img.append(stem)

    if len(not_in_label) + len(not_in_img) > 0:
        print("Not in Image: ", not_in_img)
        print("Not in Label: ", not_in_label)

    return img_list_ordered, lbl_list_ordered
    
def loading_dataset(dataset_root):
    if not isinstance(dataset_root, Path):
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


def loading_label_dict(label_json_file):

    with open(label_json_file, 'r', encoding='utf-8') as f:
        label_dict = json.load(f)

    class_n = len ( np.unique( np.asarray( list(label_dict.values() ) ) ) )

    return label_dict, class_n

############################################################
# self defined functions to process v7labs annotation data
############################################################
def _parse_v7labs_json_file(json_path, label_dict):
    output = pd.DataFrame(columns=['cls', 'x', 'y'], dtype=float)
    with open(json_path) as f:
        jsonfile = json.load(f)
        keypoints = jsonfile["annotations"]

        for i, keypoint in enumerate(keypoints):
            if "keypoint" in keypoint.keys():
                label_id = label_dict[str(keypoint["name"])]
                x = float(keypoint["keypoint"]["x"])
                y = float(keypoint["keypoint"]["y"])

                if x < 0 or y < 0:
                    print(f"[Warning] drop annotation [{i}] with x={x} and y={y} for class [{keypoint['name']}]")
                    continue

                output.loc[len(output)] = {"x": x, "y": y, "cls": label_id}

    return output

def _parse_labelme_json_file(json_path, label_dict):
    output = pd.DataFrame(columns=['cls', 'x', 'y'], dtype=float)

    with open(json_path) as f:
        jsonfile = json.load(f)

        shapes = jsonfile["shapes"]

        for shape in shapes:
            label = shape["label"]
            points = shape["points"][0]
            x = float(points[0])
            y = float(points[1])

            if label in label_dict:
                label_id = label_dict[label]
                output.loc[len(output)] = {"x": x, "y": y, "cls": label_id}
            else:
                print(f"[Warning] label {label} not found in label_dict")

    return output

def parse_label_json_file(json_path, label_dict, tool):
    if tool == 'v7labs':
        return _parse_v7labs_json_file(json_path, label_dict)
    elif tool == 'labelme':
        return _parse_labelme_json_file(json_path, label_dict)
    else:
        raise KeyError(f"Only v7labs and labelme produced json are supported, not [{tool}]")

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
    overlap_ratio : float, optional
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

        recorded_points.loc[:, 'x'] = recorded_points.loc[:, 'x'] - start_w
        recorded_points.loc[:, 'y'] = recorded_points.loc[:, 'y'] - start_h

        patch_size = end_w - start_w

        ratio = trimming_size / patch_size

        if ratio != 1:
            img_cropped = cv2.resize(img_cropped, (trimming_size, trimming_size))
            recorded_points.loc[:, 'x'] = (recorded_points.loc[:, 'x'] * ratio).astype(float)
            recorded_points.loc[:, 'y'] = (recorded_points.loc[:, 'y'] * ratio).astype(float)

        output_patches.append( {'imarray': img_cropped, 'label': recorded_points, 'patch_on_raw': p} )

    return output_patches

def save_one_output_patch(output_patch_dict, image_stem, image_suffix, image_save_folder, label_save_folder):
    imarray = output_patch_dict['imarray']
    label_df = output_patch_dict['label']
    patch_on_raw = output_patch_dict['patch_on_raw']

    w_pixel = patch_on_raw[0]
    h_pixel = patch_on_raw[1]

    size = patch_on_raw[2] - patch_on_raw[0]

    image_name = f"{image_stem}_x{int(w_pixel)}_y{int(h_pixel)}_s{size}{image_suffix}"
    label_name = f"{image_stem}_x{int(w_pixel)}_y{int(h_pixel)}_s{size}.txt"

    image_path = os.path.join(image_save_folder, image_name)
    label_path = os.path.join(label_save_folder, label_name)

    Image.fromarray(imarray).save(image_path)

    with open(label_path, "w") as f:
        for row, label in label_df.iterrows():
            f.write(f"{int(label.cls)} {float(label.x)} {float(label.y)}\n")


def convert_folder_to_dataset(
    img_folder, label_folder, kind, label_dict, 
    anno_tool, dataset_out_folder,
    patch_size, overlap_ratio, trimming_size
):
    assert kind in ['train', 'valid', 'test'], "kind must be one of 'train', 'valid', 'test'"

    img_folder = Path(img_folder)
    label_folder = Path(label_folder)

    img_files = _listdir_all_images(img_folder)
    label_files = list(label_folder.glob("*.json"))

    img_files, label_files = _match_image_and_label(img_files, label_files)

    dataset_out_folder = Path(dataset_out_folder)
    dataset_out_folder.mkdir(parents=True, exist_ok=True)

    image_save_folder = dataset_out_folder / 'images' / kind
    label_save_folder = dataset_out_folder / 'labels' / kind

    image_save_folder.mkdir(parents=True, exist_ok=True)
    label_save_folder.mkdir(parents=True, exist_ok=True)

    print(f"\n-------- {kind} ----------")

    for img_file, json_file in tqdm(zip(img_files, label_files), total=len(img_files), desc="Processing images"):

        img_path = Path(img_file)

        label_df = parse_label_json_file(json_file, label_dict, tool=anno_tool)
        # img_np = cv2.cvtColor(cv2.imread(img_file), cv2.COLOR_BGR2RGB)
        img_np = np.array(ImageOps.exif_transpose(Image.open(img_file)))
        patch_list = generate_patches(img_np.shape, patch_size=patch_size, overlap_ratio=overlap_ratio)

        output_patch_list = generate_patches_with_labels(img_np, patch_list, label_df, trimming_size=trimming_size)

        for out_patch in output_patch_list:
            save_one_output_patch(
                out_patch, image_stem=img_path.stem, image_suffix=img_path.suffix,
                image_save_folder=image_save_folder,
                label_save_folder=label_save_folder,
            )

def get_dataset_convert_arguments():
    """
    Parse all the arguments provided from the CLI.

    Returns:
        A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Object Counting Framework")

    # a threshold during evaluation for counting and visualization
    parser.add_argument('--anno_tool', default="v7labs", choices=["v7labs", "labelme"], help="the tool used to annotation points and json files")
    parser.add_argument('--dataset_folder', required=True, help="The dataset root folder for recieving converted outputs")
    parser.add_argument('--train_image_folder', required=True, help="the folder contains images for generating training data")
    parser.add_argument('--train_label_folder', required=True, help="the folder contains labels for generating training data")
    parser.add_argument('--valid_image_folder', required=True, help="the folder contains images for generating validation data")
    parser.add_argument('--valid_label_folder', required=True, help="the folder contains labels for generating validation data")
    parser.add_argument('--test_image_folder', required=False, default=None, help="the folder contains images for generating testing data")
    parser.add_argument('--test_label_folder', required=False, default=None, help="the folder contains labels for generating testing data")
    parser.add_argument('--classes_json', required=True, type=str,  help='The classes.json file record id-class pairs')
    parser.add_argument('--patch_size', default=256*3, type=int, help="The patch size in pixels on raw images")
    parser.add_argument('--overlap_ratio', default=0.0, type=float, help="the buffer area width/patch width inside the patch, by default 0.0 no overlap")
    parser.add_argument('--patch_save_size', default=None, type=int, help="The saved patch image size, by default the same as patch size")

    args = parser.parse_known_args()[0]

    args.dataset_folder = Path(args.dataset_folder)
    args.train_image_folder = Path(args.train_image_folder)
    args.train_label_folder = Path(args.train_label_folder)
    args.valid_image_folder = Path(args.valid_image_folder)
    args.valid_label_folder = Path(args.valid_label_folder)

    if args.test_image_folder is not None:
        args.test_image_folder = Path(args.test_image_folder)
    if args.test_label_folder is not None:
        args.test_label_folder = Path(args.test_label_folder)

    args.classes_json = Path(args.classes_json)

    return args

    
if __name__ == '__main__':
    import os
    import sys
    sys.path.insert(0, os.getcwd())
    from gcp2pnet import utils

    args = get_dataset_convert_arguments()
    utils.print_args(args, title="Dataset Convertor")

    label_dict, class_n = loading_label_dict( args.classes_json) 

    dataset_folder = Path(args.dataset_folder)
    dataset_folder.mkdir(parents=True, exist_ok=True)
    if not (dataset_folder / 'classes.json').exists():
        shutil.copy(args.classes_json, dataset_folder / 'classes.json')

    if args.patch_save_size is None:
        trimming_size = args.patch_size
    else:
        trimming_size = args.patch_save_size

    utils.save_args_to_yaml(args, dataset_folder / "generate_args.yaml")

    convert_folder_to_dataset(
        img_folder=args.train_image_folder, 
        label_folder=args.train_label_folder,
        kind='train', 
        label_dict=label_dict,
        anno_tool=args.anno_tool, 
        dataset_out_folder=args.dataset_folder,
        patch_size=args.patch_size,
        overlap_ratio=args.overlap_ratio,
        trimming_size=trimming_size
    )

    convert_folder_to_dataset(
        img_folder=args.valid_image_folder, 
        label_folder=args.valid_label_folder,
        kind='valid', 
        label_dict=label_dict,
        anno_tool=args.anno_tool, 
        dataset_out_folder=args.dataset_folder,
        patch_size=args.patch_size,
        overlap_ratio=args.overlap_ratio,
        trimming_size=trimming_size
    )

    if args.test_image_folder is not None:
        convert_folder_to_dataset(
            img_folder=args.test_image_folder, 
            label_folder=args.test_label_folder,
            kind='test', 
            label_dict=label_dict,
            anno_tool=args.anno_tool, 
            dataset_out_folder=args.dataset_folder,
            patch_size=args.patch_size,
            overlap_ratio=args.overlap_ratio,
            trimming_size=trimming_size
        )