import shutil
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageOps

# the modified function of `datasets.convert_folder_to_dataset()`
def convert_folder_to_dataset_with_variable_patch_size(
    img_folder, label_folder, kind, label_dict, 
    anno_tool, dataset_out_folder,
    patch_size_var, overlap_ratio, trimming_size
):
    assert kind in ['train', 'valid', 'test'], "kind must be one of 'train', 'valid', 'test'"

    img_folder = Path(img_folder)
    label_folder = Path(label_folder)

    img_files = datasets._listdir_all_images(img_folder)
    label_files = list(label_folder.glob("*.json"))

    img_files, label_files = datasets._match_image_and_label(img_files, label_files)

    dataset_out_folder = Path(dataset_out_folder)
    dataset_out_folder.mkdir(parents=True, exist_ok=True)

    image_save_folder = dataset_out_folder / 'images' / kind
    label_save_folder = dataset_out_folder / 'labels' / kind

    image_save_folder.mkdir(parents=True, exist_ok=True)
    label_save_folder.mkdir(parents=True, exist_ok=True)

    print(f"\n-------- {kind} ----------")

    for img_file, json_file in tqdm(zip(img_files, label_files), total=len(img_files), desc="Processing images"):

        img_path = Path(img_file)

        label_df = datasets.parse_label_json_file(json_file, label_dict, tool=anno_tool)
        # img_np = cv2.cvtColor(cv2.imread(img_file), cv2.COLOR_BGR2RGB)
        img_np = np.array(ImageOps.exif_transpose(Image.open(img_file)))

        # {2576: 256, 7728: 256*3}
        if 7728 in img_np.shape:
            patch_size = 256*3
        elif 2576 in img_np.shape:
            patch_size = 256
        else:
            raise ValueError(f"can not find 7728 or 2576 in {img_np.shape}")

        patch_list = datasets.generate_patches(img_np.shape, patch_size=patch_size, overlap_ratio=overlap_ratio)

        output_patch_list = datasets.generate_patches_with_labels(img_np, patch_list, label_df, trimming_size=trimming_size)

        for out_patch in output_patch_list:
            datasets.save_one_output_patch(
                out_patch, image_stem=img_path.stem, image_suffix=img_path.suffix,
                image_save_folder=image_save_folder,
                label_save_folder=label_save_folder,
            )

if __name__ == '__main__':
    # execute this python from from git repo root
    # [GrainCountingP2PNet]$ uv run data/demo_raw/convert2dataset.py
    import os
    import sys
    sys.path.insert(0, os.getcwd())

    print(sys.path)

    from gcp2pnet import utils, datasets

    # here replace 
    # >>> args = datasets.get_dataset_convert_arguments() 
    # by 
    # args.xxx --> args_xxx for easier understanding

    args_anno_tool = "v7labs"
    args_dataset_folder = Path('data/demo_dataset')

    args_train_image_folder = "data/demo_raw/training_images"
    args_train_label_folder = "data/demo_raw/training_labels"
    args_valid_image_folder = "data/demo_raw/evaluation_images"
    args_valid_label_folder = "data/demo_raw/evaluation_labels"

    args_classes_json = Path('data/demo_raw/classes.json')

    # args.patch_size is special for this case
    # => variable to image size, 256 or 256*3=768
    args_patch_size_var = {2576: 256, 7728: 256*3}

    args_overlap_ratio = 0.0
    args_patch_save_size = 256

    #--------------------------------------------------
    # others are just like datasets.py __main__ part
    label_dict, class_n = datasets.loading_label_dict( args_classes_json) 

    args_dataset_folder.mkdir(parents=True, exist_ok=True)
    if not (args_dataset_folder / 'classes.json').exists():
        shutil.copy(args_classes_json, args_dataset_folder / 'classes.json')

    convert_folder_to_dataset_with_variable_patch_size(
        img_folder=args_train_image_folder, 
        label_folder=args_train_label_folder,
        kind='train', 
        label_dict=label_dict,
        anno_tool=args_anno_tool, 
        dataset_out_folder=args_dataset_folder,
        patch_size_var=args_patch_size_var,
        overlap_ratio=args_overlap_ratio,
        trimming_size=args_patch_save_size
    )

    convert_folder_to_dataset_with_variable_patch_size(
        img_folder=args_valid_image_folder, 
        label_folder=args_valid_label_folder,
        kind='valid', 
        label_dict=label_dict,
        anno_tool=args_anno_tool, 
        dataset_out_folder=args_dataset_folder,
        patch_size_var=args_patch_size_var,
        overlap_ratio=args_overlap_ratio,
        trimming_size=args_patch_save_size
    )