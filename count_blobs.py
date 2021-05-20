"""
Author: Norman Juchler
License: GNU GPLv3
Date: May 2021

Setup:
    python3 -m pip install -r requirements.txt
How to run:
    python3 count_blobs.py -h
    python3 count_blobs.py -i ../samples -t 200 --plot
"""
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import scipy.spatial as spatial
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.collections as mpc
from collections import defaultdict
from functools import reduce
from operator import or_

CENTER_SUFFIX="_Centers.csv"
ORIGINAL_SUFFIX="_OrigImage.tif"
SEGMENTATION_SUFFIX="_Objects.tiff"
PROBABILITIES_SUFFIX="_Probabilities.tif"

OVERLAY_SUFFIX=SEGMENTATION_SUFFIX

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])


def save_figure(path, fig=None, **kwargs):
    dpi = kwargs.pop("dpi", None)
    bbox_inches = kwargs.pop("bbox_inches", "tight")
    transparent = kwargs.pop("transparent", False)
    if fig is not None:
        plt.figure(fig.number)
    if not path.parent.is_dir():
        path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path,
                transparent=transparent,
                bbox_inches="tight",
                dpi=dpi,
                **kwargs)
    return path


def read_centers(filepath):
    centers = pd.read_csv(filepath)
    centers = centers.rename({"Location_Center_X": "x",
                              "Location_Center_Y": "y"}, axis=1)
    centers = centers.loc[:, ["x", "y"]].copy()
    return centers


def form_clusters_step(neighbors):
    """
    Let's be given a dictionary {i: [j]} that maps to each (point) index i
    a list of nearest neighbor (indices, [j]).
    Form clusters by looking up the nearest neighbors for the nearest
    neighbors of the current index i: union(d[j] for j in d[i])

    Example:
        neighbors = {1:[1,2], 2:[2,1], 3:[3,4,5], 4:[4,3], 5:[5,3]}
        ret = form_clusters(neighbors)
        ret = {1:{1,2}, 2:{1,2}, 3:{3,4,5}, 4:{3,4,5}, 5:{3,4,5}}
    """
    def lookup_list(dct, lst):
        return [set(dct[v]) for v in lst if v in dct]
    d = neighbors
    return {k: list(reduce(or_, lookup_list(d, v))) for k,v in d.items()}


def form_clusters(neighbors):
    """
    Expand the clusters until they stabilize.
    """
    max_loops = 10
    ret = neighbors
    for i in range(max_loops):
        new = form_clusters_step(ret)
        if ret==new:
            break
        else:
            ret = new
    return ret


def search_neighbors(centers, threshold):
    point_tree = spatial.cKDTree(centers)
    neighbors = point_tree.query_ball_point(centers, r=threshold)
    neighbors = {i:n for i,n in enumerate(neighbors) if len(n)>1}
    neighbors = form_clusters(neighbors)
    return neighbors


def count_events(centers, neighbors):
    clusters = set(map(tuple, neighbors.values()))
    cluster_sizes = pd.Series([len(n) for n in clusters], dtype=int)
    counts = cluster_sizes.value_counts(sort=False)
    sizes = counts.index
    rel_counts = counts * sizes / len(centers)
    counts = pd.concat([counts, rel_counts], axis=1,
                       keys=["count", "rel_count"])
    print("Total number of blobs: %4d" % len(centers))
    if counts.empty:
        return
    for n, row in counts.iterrows():
        msg = "Number of %d-clusters: %5d (freq: %.3f)"
        print(msg % (n, row["count"], row["rel_count"]))
    print()
    print("Formula for freq: count*cluster_size/total_blobs")
    return counts

def visualize_neighbors(filepath, centers, neighbors, radius=0.005):
    """
    Radius: relative to image diagonal
    """
    # filepath: path to centers file
    dataset_id = filepath.name.replace(CENTER_SUFFIX, "")
    dataset_dir = filepath.parent
    image_path = dataset_dir/(dataset_id+OVERLAY_SUFFIX)
    if not image_path.is_file():
        print("Error: cannot find image: %s" % image_path)
        print("Error: skipping...")
        return
    img = mpimg.imread(image_path)
    img = rgb2gray(img)
    colors = {2:        "#FFFD38",  # yellow
              3:        "#1EB1ED",  # blue
              4:        "#19AF54",  # green
              "default":"#D85326",  # red
             }
    fig, ax = plt.subplots()
    ax.imshow(img, cmap=plt.get_cmap("gray"), vmin=0, vmax=255)
    r = radius*np.linalg.norm(img.shape)
    clusters = set(map(tuple,neighbors.values()))
    for ids in clusters:
        points = centers.iloc[list(ids)]
        color = colors.get(len(ids), colors["default"])
        if True:
            # Connect neighbors by a line
            if len(points)>2:
                hull = spatial.ConvexHull(points.values)
                ax.plot(points.values[hull.vertices,0],
                        points.values[hull.vertices,1],
                        linestyle="-", color=color, marker=None)
            else:
                ax.plot(points["x"], points["y"],
                        color=color, marker=None, linestyle="-")
        circles = [plt.Circle((x,y), radius=r, linewidth=0,
                              facecolor=color, alpha=0.5)
                   for x,y in points.values]
        c = mpc.PatchCollection(circles, match_original=True)
        ax.add_collection(c)

def run(args):
    data_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    threshold = args.threshold
    show = args.show
    plot = args.plot

    centers_files = sorted(data_dir.glob("*"+CENTER_SUFFIX))
    for filepath in centers_files:
        centers = read_centers(filepath=filepath)
        neighbors = search_neighbors(centers=centers, threshold=threshold)
        counts = count_events(centers=centers, neighbors=neighbors)
        if counts is None:
            print("No clusters found (thr=%.1f)!" % threshold)
        if show or plot:
            visualize_neighbors(filepath=filepath,
                                centers=centers,
                                neighbors=neighbors)
            if show:
                plt.show()
            if plot:
                dataset_id = filepath.stem.replace(CENTER_SUFFIX, "")
                plt.axis(False)
                save_figure(path=out_dir/(dataset_id+".pdf"), dpi=600)


def parse_args():
    description = ("Count touching blobs based on blob centers.")
    formatter = argparse.RawDescriptionHelpFormatter
    parser = argparse.ArgumentParser(add_help=False,
                                     formatter_class=formatter,
                                     description=description)
    parser.add_argument("-h", "--help", action="help",
                        help="Show this help text")
    parser.add_argument("-i", "--in-dir", required=True,
                        help="Input directory")
    parser.add_argument("-o", "--out-dir", default="./results/",
                        help="Output directory")
    parser.add_argument("-t", "--threshold", default=5, type=float,
                        help="Distance threshold in pixels.")
    parser.add_argument("--show", action="store_true",
                        help="Visualize and show the figure.")
    parser.add_argument("--plot", action="store_true",
                        help="Visualize the result and save to file.")
    parser.set_defaults(func=run)
    return parser.parse_args()


def main():

    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
