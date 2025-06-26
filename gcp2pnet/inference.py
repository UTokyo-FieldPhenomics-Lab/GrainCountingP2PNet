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
from adjustText import adjust_text

from PIL import Image
from scipy import spatial

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
    parser.add_argument('--result_path', default=None, help="The path to save result image")
    parser.add_argument('--num_workers', default=1, type=int)
    parser.add_argument('--gpu_id', default=0, type=int, help='the gpu used for training')
    parser.add_argument('--device', default=device, type=str, 
                        help="the torch running device, 'cpu' or 'cuda'")

    args = parser.parse_known_args()[0]

    args.weight_path = Path(args.weight_path)
    args.img_path = Path(args.img_path)

    if args.result_path is not None:
        args.result_path = Path(args.result_path)

    return args

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

    return img_resize, img_tensor

def apply_model(model, img_tensor, threshold):
    class_n = model.num_classes - 1  # num_class = [0, 1, 2] -> [1, 2] are labels -> class_n = 2

    # run inference
    outputs = model(img_tensor)

    outputs_points = outputs['pred_points'][0]
    outputs_scores = torch.nn.functional.softmax(outputs['pred_logits'], -1)

    raw_results = {}
    for class_i in range(1, class_n + 1):
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
    for key in raw_dict.keys():
        points_n, scores_n = postprocess_point_clusters_one_class(
            raw_dict[key]['points'],
            raw_dict[key]['scores']
        )

        class_n = [key] * len(points_n)

        tmp_df = pd.DataFrame({
            'x': points_n[:, 0],
            'y': points_n[:, 1],
            'score': scores_n,
            'cls': class_n
        })
        results = pd.concat([results, tmp_df], ignore_index=True)

    return results

def postprocess_merge_by_distance(results_df, prox_distance=25):

    # Build a KDTree for efficient spatial queries
    points = results_df[['x', 'y']].values
    tree = spatial.KDTree(points)

    # Find all points within `distance` of each point
    to_keep = []
    for idx, row in results_df.iterrows():
        neighbors = tree.query_ball_point([row['x'], row['y']], prox_distance)
        neighbor_scores = results_df.iloc[neighbors]['score']
        highest_score_idx = neighbor_scores.idxmax()
        to_keep.append(highest_score_idx)

    # Deduplicate and keep only the highest-scoring points in each neighborhood
    filtered_df = results_df.loc[list(set(to_keep))].sort_index()

    return filtered_df

def draw_result_figures(img_numpy, raw_results, after_point_clusters, after_merge_by_distance, 
                        show=True, save_path=None,):
    fig, ax = plt.subplots(1,3, figsize=(10,4))

    c = {1: 'r', 2: 'b'}

    # raw outputs
    ax[0].imshow(img_numpy)
    ax[0].scatter(*raw_results[1]['points'].T, c='r', s=1)
    ax[0].scatter(*raw_results[2]['points'].T, c='b', s=1)
    ax[0].set_title("Raw detections")

    # clustered outputs
    ax[1].imshow(img_numpy)
    class_color = [c[i] for i in after_point_clusters.cls]
    ax[1].scatter(after_point_clusters.x, after_point_clusters.y, 
                  c=class_color, s=15, marker='o', edgecolors='w')
    

    texts = []
    for x, y, score, cls in zip(after_point_clusters.x, after_point_clusters.y, after_point_clusters.score, after_point_clusters.cls):
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

    c = {1: 'r', 2: 'b'}
    class_color = [c[i] for i in after_merge_by_distance.cls]
    ax[2].scatter(after_merge_by_distance.x, after_merge_by_distance.y, c=class_color, s=15, marker='o', edgecolors='w')

    for x, y, score, cls in zip(after_merge_by_distance.x, after_merge_by_distance.y, after_merge_by_distance.score, after_merge_by_distance.cls):
        ax[2].text(x, y-5, f"{score:.2f}", ha='center', va='bottom', 
                fontsize=10, color=c[cls], alpha=0.7,
                path_effects=[patheffects.withStroke(linewidth=2, foreground='white')])
        
    ax[2].set_title("Merged by distance")

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
        plt.savefig(save_path)
        plt.close(fig) 


def main(args, debug=False):
    os.environ["CUDA_VISIBLE_DEVICES"] = '{}'.format(args.gpu_id)

    utils.print_args(args, title="Inference Arguements")

    model = load_model(args)

    img_numpy, img_tensor = load_image_to_tensor(args.img_path, args.device)
    raw_results = apply_model(model, img_tensor, args.threshold)

    clustered_df = postprocess_point_clusters(raw_results)
    merged_df = postprocess_merge_by_distance(clustered_df, prox_distance=25)

    print(merged_df)

    if args.result_path is None:
        # not saving to path, show images instead
        draw_result_figures(img_numpy, raw_results, clustered_df, merged_df, show=True)

    else:
        draw_result_figures(img_numpy, raw_results, clustered_df, merged_df, show=False, 
                            save_path=args.result_path)

if __name__ == '__main__':
    import sys
    sys.path.insert(0, '../')

    from gcp2pnet import models, utils, datasets, misc, engine

    args = get_inf_arguments()
    main(args)