# Modified from https://github.com/TencentYoutuResearch/CrowdCounting-P2PNet/blob/main/run_test.py
import os
import argparse
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path

import cv2
import torch
import torchvision.transforms as standard_transforms
import numpy as np
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import patheffects
from matplotlib.colors import ListedColormap
from matplotlib.cm import ScalarMappable
from adjustText import adjust_text
from PIL import Image
from scipy import spatial
from tqdm import tqdm

from . import models, utils


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
    parser.add_argument('--weight_path', default="demo_best_mae.pth", help='resume from checkpoint')
    parser.add_argument('--img_path', default="", help="The path to image")
    parser.add_argument('--result_folder', default=None, help="The path to save result image")
    parser.add_argument('--patch_result_folder', default=None, help="The folder to save patch result images")
    parser.add_argument('--merge_distance', default=25, type=int, help="the pixel distance to merge points of postprocessing")
    parser.add_argument('--sliding_window', default=False, type=bool, help="Whether crop high-resolution images to small patches for inference")
    parser.add_argument('--window_size', default=256, type=int, help="The size of sliding window, default 256x256")
    parser.add_argument('--overlap_ratio', default=0.2, type=float, help="Overlap ratio between patches")
    parser.add_argument('--num_workers', default=1, type=int)
    parser.add_argument('--gpu_id', default=0, type=int, help='the gpu used for training')
    parser.add_argument('--device', default=device, type=str, 
                        help="the torch running device, 'cpu' or 'cuda'")

    args = parser.parse_known_args()[0]

    args.weight_path = Path(args.weight_path)
    args.img_path = Path(args.img_path)

    if args.result_folder is not None:
        args.result_folder = Path(args.result_folder)

        if not args.result_folder.exists():
            args.result_folder.mkdir(parents=True, exist_ok=True)

    if args.patch_result_folder is not None:
        args.patch_result_folder = Path(args.patch_result_folder)

        if not args.patch_result_folder.exists():
            args.patch_result_folder.mkdir(parents=True, exist_ok=True)

    return args

def load_model(args):
    # ensure model file exists
    if not ( args.weight_path and os.path.exists(args.weight_path) ):
        raise FileNotFoundError(f"Could not load model weight from [{args.weight_path}]")

    utils.fix_random_seed(args.seed)

    checkpoint = torch.load(args.weight_path, map_location=args.device)

    # get the P2PNet
    model = models.p2pnet.build_model(args, checkpoint['num_classes'])
    model.to(args.device) # move to GPU

    # load trained model
    model.load_state_dict(checkpoint['model'])

    # convert to eval mode
    model.eval()

    return model, checkpoint

def load_image_to_tensor(img_raw, device, model_train_img_size=256):
    # create the pre-processing transform
    transform = standard_transforms.Compose([
        standard_transforms.ToTensor(), 
        standard_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # round the size
    width, height = img_raw.size
    new_width = new_height = model_train_img_size

    img_resize = img_raw.resize((new_width, new_height), Image.ANTIALIAS)

    resize_ratio = ( new_width / width, new_height / height )

    # pre-proccessing
    img_trans = transform(img_resize)

    img_tensor = torch.Tensor(img_trans).unsqueeze(0)
    img_tensor = img_tensor.to(device)

    return np.asarray(img_resize), img_tensor, resize_ratio

def apply_model(model, img_tensor, threshold):
    # run inference
    outputs = model(img_tensor)

    outputs_points = outputs['pred_points'][0]
    outputs_scores = torch.nn.functional.softmax(outputs['pred_logits'], -1)

    raw_results = {}
    # in our case, model.num_classes = 2, this aims to iter [1, 2]
    for class_i in range(1, model.num_classes + 1):
        outputs_score = outputs_scores[:, :, class_i][0]

        points = outputs_points[outputs_score > threshold].detach().cpu().numpy()#.tolist()
        scores = outputs_score [outputs_score > threshold].detach().cpu().numpy()#.tolist()

        raw_results[class_i] = {'points': points, 'scores': scores}

    return raw_results

def postprocess_point_clusters_one_class(points, scores):
    if points.shape[0] > 10000:
        warnings.warn('Too many points, skip post processing')
        return points, scores
    
    if points.shape[0] == 0:
        warnings.warn('No points, skip post processing')
        return points, scores
    
    cutoff = 500 / points.shape[0]
    if cutoff < 20:
        cutoff = 20

    components = nx.connected_components(
        nx.from_edgelist(
            (i, j) for i, js in enumerate(
                spatial.KDTree(points).query_ball_point(points, cutoff)
            )
            for j in js
        )
    )

    clusters = {j: i for i, js in enumerate(components) for j in js}

    # reorganize the points to the order of clusters
    points_reo = np.zeros(points.shape)
    scores_reo = np.zeros(scores.shape)

    for i, key in enumerate( clusters.keys() ):
        points_reo[i,:] = points[key,:]
        scores_reo[i] = scores[key]

    # points_n has the same order as clusters
    res = [clusters[key] for key in clusters.keys()]
    res_n = np.array(res).reshape(-1,1)

    points_n = []
    scores_n = []
    for i in np.unique(res_n):

        tmp_points = points_reo[np.where(res_n[:,0] == i)]
        tmp_scores = scores_reo[np.where(res_n[:,0] == i)]

        points_n.append( [np.mean(tmp_points[:,0]), np.mean(tmp_points[:,1])])
        scores_n.append( np.mean(tmp_scores[:]) )

    return np.asarray(points_n), np.asarray(scores_n)

def postprocess_point_clusters(raw_dict):
    results = pd.DataFrame(columns=['x', 'y', 'score', 'cls'])
    for class_i in raw_dict.keys():
        points_n, scores_n = postprocess_point_clusters_one_class(
            raw_dict[class_i]['points'],
            raw_dict[class_i]['scores']
        )

        tmp_df = pd.DataFrame({
            'x': points_n[:, 0],
            'y': points_n[:, 1],
            'score': scores_n,
            'cls': class_i
        })
        results = pd.concat([results, tmp_df], ignore_index=True)

    return results

def postprocess_merge_by_distance(results_df, prox_distance=25):

    # Build a KDTree for efficient spatial queries
    points = results_df[['x', 'y']].values
    tree = spatial.KDTree(points)

    # Find all points within `distance` of each point
    to_keep = []
    removed_indices = set()
    for idx, row in results_df.iterrows():
        if idx in removed_indices:
            continue
        neighbors = tree.query_ball_point([row['x'], row['y']], prox_distance)
        neighbors = [n for n in neighbors if n not in removed_indices]
        neighbor_scores = results_df.iloc[neighbors]['score']
        highest_score_idx = neighbor_scores.idxmax()

        to_keep.append(highest_score_idx)
        removed_indices.update(set(neighbors) - {highest_score_idx})

    # Deduplicate and keep only the highest-scoring points in each neighborhood
    filtered_df = results_df.loc[list(set(to_keep))].sort_index()

    return filtered_df

def draw_result_patch_figure(num_classes, img_numpy, raw_results, cluster_df, merged_df, 
                             show=True, save_path=None,):
    fig, ax = plt.subplots(1,3, figsize=(10,4), dpi=300)

    if num_classes <= 9:
        c = {i: plt.cm.Set1(i-1) for i in range(1, num_classes+1)}
    else:
        c = {plt.cm.hsv((i-1)/(num_classes-1)) for i in range(1, num_classes+1)}

    # raw outputs
    ax[0].imshow(img_numpy)
    for i in raw_results.keys():
        ax[0].scatter(*raw_results[i]['points'].T, c=c[i], s=1, alpha=0.5)
    ax[0].set_title("Raw detections")

    # clustered outputs
    ax[1].imshow(img_numpy)
    class_color = [c[i] for i in cluster_df.cls]
    ax[1].scatter(cluster_df.x, cluster_df.y, 
                  c=class_color, s=15, marker='o', edgecolors='w')

    texts = []
    for x, y, score, cls in zip(cluster_df.x, cluster_df.y, cluster_df.score, cluster_df.cls):
        texts.append(
            ax[1].text(x, y, f"{score:.2f}", ha='center', va='bottom', 
                    fontsize=10, color=c[cls], alpha=0.7,
                    path_effects=[patheffects.withStroke(linewidth=2, foreground='white')])
        )

    adjust_text(texts, force_text=0.1, arrowprops=dict(arrowstyle="-|>",
                                                    color='w', alpha=0.8), ax=ax[1])
    ax[1].set_title("Clustered")

    # distance merged outputs
    ax[2].imshow(img_numpy)

    class_color = [c[i] for i in merged_df.cls]
    ax[2].scatter(merged_df.x, merged_df.y, c=class_color, s=15, marker='o', edgecolors='w')

    for x, y, score, cls in zip(merged_df.x, merged_df.y, merged_df.score, merged_df.cls):
        ax[2].text(x, y-5, f"{score:.2f}", ha='center', va='bottom', 
                fontsize=10, color=c[cls], alpha=0.7,
                path_effects=[patheffects.withStroke(linewidth=2, foreground='white')])
        
    ax[2].set_title("Merged by distance")

    # add colorbar at bottom
    cbar_ax = fig.add_axes([0.1, 0.05, 0.8, 0.04])  # [left, bottom, width, height]
    cmap = ListedColormap( [plt.cm.Set1(i-1) for i in raw_results.keys()] )
    cb = plt.colorbar(mappable=ScalarMappable(cmap=cmap), cax=cbar_ax, orientation='horizontal')
    ticks = [( j + 0.5 ) / num_classes for j in range(0, num_classes)]
    labels = [j+1 for j in range(0, num_classes )]
    cb.ax.get_xaxis().set_ticks(ticks, labels)
    cb.ax.get_yaxis().set_ticks([])
    cb.ax.tick_params(bottom=False, labelsize=8)

    plt.tight_layout()

    if show:
        plt.show()
    
    if save_path is not None:
        plt.savefig(save_path)
    
    plt.close(fig) 

def draw_results_raw_image(num_classes, figsize, img_numpy, merged_df, 
                           show=True, save_path=None):
    
    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=300)
    ax.imshow(img_numpy)

    if num_classes <= 9:
        c = {i: plt.cm.Set1(i-1) for i in range(1, num_classes+1)}
    else:
        c = {plt.cm.hsv((i-1)/(num_classes-1)) for i in range(1, num_classes+1)}

    class_color = [c[i] for i in merged_df.cls]
    ax.scatter(merged_df.x, merged_df.y, c=class_color, s=15, marker='o', edgecolors='w')

    for x, y, score, cls in zip(merged_df.x, merged_df.y, merged_df.score, merged_df.cls):
        ax.text(x, y-5, f"{score:.2f}", ha='center', va='bottom', 
            fontsize=5, color=c[cls], alpha=0.7,
            path_effects=[patheffects.withStroke(linewidth=2, foreground='white')])
        
    ax.invert_xaxis()
    ax.invert_yaxis()
    plt.axis('off')
    plt.tight_layout()

    if show:
        # plt.show() -> cv2.show()
        fig.canvas.draw()
        img_argb = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
        img_argb = img_argb.reshape(fig.canvas.get_width_height()[::-1] + (4,))  # (H, W, 4)
        img_rgb = img_argb[..., 1:]  # 去掉 Alpha 通道，保留 RGB (H, W, 3)
        plt.close(fig)  # 关闭 Matplotlib 图形

        print("Press any key to close the window")

        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        cv2.imshow("Results Window", img_bgr)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    if save_path is not None:
        plt.savefig(
            save_path, 
            bbox_inches='tight',
            pad_inches=0)

    plt.close(fig) 

def main(args, return_result=False):
    os.environ["CUDA_VISIBLE_DEVICES"] = '{}'.format(args.gpu_id)

    model, checkpoints = load_model(args)

    # ensure image file exists
    if not ( args.img_path and os.path.exists(args.img_path) ):
        raise FileNotFoundError(f"Could not load image from [{args.img_path}]")
    # load the images
    img_raw = Image.open(args.img_path).convert('RGB')

    # judge if need sliding window to produce results
    width, height = img_raw.size
    img_oversize = max(width, height) > checkpoints['imgsz'] * 2
    # ask user if using sliding window when image oversize than model trained img size
    if img_oversize and not args.sliding_window:
        answer = input(f"[Warning] The input image size ({width}, {height}) exceeds model training size ({checkpoints['imgsz']}, {checkpoints['imgsz']}).\n"
                       f"          This may cause misdetection for small objects.\n"
                       f"          Continue inferencing by default resizing [Y] or using slidling window [N]? (Y/N)")
        
        if answer in ['N', 'n']:
            args.sliding_window = True

    utils.print_args(args, title="Inference Arguements")

    if args.sliding_window:
        patches= datasets.generate_patches((width, height), args.window_size, args.overlap_ratio)
        img_list = []
        for p in patches:
            # p -> (start_w, start_h, end_w, end_h)
            # img_raw_crop -> left, top, right, bottom)
            img_list.append( img_raw.crop( p ) )
    else:
        img_list = [img_raw]
        patches = [(0,0, width, height)]

    raw_results_final = {}
    merged_df_concat = pd.DataFrame(columns=['x', 'y', 'score', 'cls'])
    for img, patch in tqdm(zip(img_list, patches), total=len(patches), desc=f"Processing patches"):

        patch_ndarray, patch_tensor, resize_ratio = load_image_to_tensor(img, args.device, checkpoints['imgsz'] )
        patch_raw_results = apply_model(model, patch_tensor, args.threshold)

        patch_clustered_df = postprocess_point_clusters(patch_raw_results)

        # draw patch results
        if args.patch_result_folder is not None and len(patch_clustered_df) > 0:


            patch_merged_df = postprocess_merge_by_distance(patch_clustered_df, prox_distance=args.merge_distance)

            save_name = Path(args.patch_result_folder) / f"{Path(args.img_path).stem}_x{patch[0]}_y{patch[1]}_s{args.window_size}.png"

            draw_result_patch_figure(
                checkpoints['num_classes'], patch_ndarray, 
                patch_raw_results, patch_clustered_df, patch_merged_df,
                show=False, save_path=save_name)

        if len(patch_clustered_df) > 0:
            patch_clustered_df['x'] = patch_clustered_df['x'] / resize_ratio[0] + patch[0]
            patch_clustered_df['y'] = patch_clustered_df['y'] / resize_ratio[1] + patch[1]

            merged_df_concat = pd.concat([merged_df_concat, patch_clustered_df]).reset_index(drop=True)

    # prox_distance need to resize according to window_size?
    merged_df = postprocess_merge_by_distance(merged_df_concat, prox_distance=args.merge_distance)
    print(merged_df)

    figsize = (width // args.window_size, height // args.window_size)
    if args.result_folder is None:
        # not saving to path, show images instead
        draw_results_raw_image(
            checkpoints['num_classes'], figsize, 
            img_raw, merged_df, show=True)

    else:
        draw_results_raw_image(
            checkpoints['num_classes'], figsize, 
            img_raw, merged_df, show=False, save_path=args.result_folder / f"{args.img_path.stem}_result.png")

    utils.save_args_to_yaml(args, yaml_path=args.result_folder / f"{args.img_path.stem}_result.yaml")

    if return_result:
        return merged_df
        

if __name__ == '__main__':
    import os
    import sys
    sys.path.insert(0, os.getcwd())
    from gcp2pnet import models, utils, datasets, misc, engine

    args = get_inf_arguments()
    main(args)