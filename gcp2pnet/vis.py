import random

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mplpatches

def draw_patch_split_on_raw(img_np, patch_list, label_df, color_dict, save_path):
    fig, ax = plt.subplots(1, figsize=(10,10))
    ax.imshow(img_np)

    for p in patch_list:
        rnd_color = plt.cm.get_cmap('Dark2')(random.random())
        rect = mplpatches.Rectangle((p[0], p[1]), p[2]-p[0], p[3]-p[1], linewidth=1, edgecolor=rnd_color, facecolor=rnd_color, alpha=0.3)
        ax.add_patch(rect)

    ax.scatter( label_df.x, label_df.y, c=[color_dict[cls] for cls in label_df.cls], marker='o', s=15)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)



def draw_patch_individual(img_np, output_patch_list, color_dict, save_path):
    num_patches = len(output_patch_list)
    cols = int( np.ceil( np.sqrt(num_patches) ) )
    rows = (num_patches + cols - 1) // cols
    fig, ax = plt.subplots(rows, cols, figsize=(cols*2, rows*2))

    for n, patch in enumerate(output_patch_list):
        i, j = n // cols, n % cols

        ax[i,j].imshow(patch['imarray'])
        ax[i,j].scatter(patch['label'].x, patch['label'].y, c=[color_dict[cls] for cls in patch['label'].cls], marker='o', s=15)

    # hide blank axis
    for n in range(num_patches, rows*cols):
        i, j = n // cols, n % cols
        ax[i,j].axis('off')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)