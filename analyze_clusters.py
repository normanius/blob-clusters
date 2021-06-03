"""
Author: Norman Juchler
License: GNU GPLv3
Date: May 2021
"""
import argparse
import cv2 as cv
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

BACKGROUND_SUFFIX=SEGMENTATION_SUFFIX   # or ORIGINAL_SUFFIX

REQUIRED_SUFFIXES={CENTER_SUFFIX,
                   BACKGROUND_SUFFIX,
                   SEGMENTATION_SUFFIX}

CLUSTER_COLORS = {1:        "#FFFFFF",  # white
                  2:        "#FFFD38",  # yellow
                  3:        "#1EB1ED",  # blue
                  4:        "#19AF54",  # green
                  "default":"#D85326",  # red
                 }

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])


def hex2rgb(hx, opencv=False):
    # Usage:
    #   hex2rgb("FFFD38")
    #   hex2rgb("#FFFD38")
    hx = hx.strip("#")
    if opencv:
        return (int(hx[4:6],16),int(hx[2:4],16),int(hx[0:2],16))
    else:
        return (int(hx[0:2],16),int(hx[2:4],16),int(hx[4:6],16))


def ensure_dir(path):
    if not path.is_dir():
        path.mkdir(parents=True, exist_ok=True)
    return path.is_dir()


def save_figure(path, fig=None, **kwargs):
    dpi = kwargs.pop("dpi", None)
    bbox_inches = kwargs.pop("bbox_inches", "tight")
    transparent = kwargs.pop("transparent", False)
    if fig is not None:
        plt.figure(fig.number)
    ensure_dir(path.parent)
    plt.savefig(path,
                transparent=transparent,
                bbox_inches="tight",
                pad_inches=0.0,
                dpi=dpi,
                **kwargs)
    return path


def search_datasets(data_dir):
    datasets = defaultdict(list)
    for suffix in REQUIRED_SUFFIXES:
        files = list(sorted(data_dir.glob("*"+suffix)))
        for f in files:
            dataset_id = f.name.replace(suffix, "")
            datasets[dataset_id].append(suffix)
    dataset_ids = []
    for did, suffixes in datasets.items():
        missing = set(REQUIRED_SUFFIXES) - set(suffixes)
        if missing:
            print("Warning: Dataset '%s' is incomplete. Skipping!" % did)
            print("Missing suffixes: %s" % missing)
        else:
            dataset_ids.append(did)
    return dataset_ids


def ensure_data_path(data_dir, dataset_id, suffix):
    path = data_dir / (dataset_id+suffix)
    if not path.is_file():
        msg = "Error: cannot find file: %s" % path
        raise FileNotFoundError(msg)
    return path


def read_centers(data_dir, dataset_id):
    filepath = ensure_data_path(data_dir=data_dir,
                                dataset_id=dataset_id,
                                suffix=CENTER_SUFFIX)
    centers = pd.read_csv(filepath)
    centers = centers.rename({"Location_Center_X": "x",
                              "Location_Center_Y": "y"}, axis=1)
    centers = centers.loc[:, ["x", "y"]].copy()
    return centers


def unique_colors(img):
    img2d = img.reshape(-1, img.shape[-1])
    col_range = (256, 256, 256) # generically : img2d.max(0)+1
    img1d = np.ravel_multi_index(img2d.T, col_range)
    return np.unravel_index(np.bincount(img1d), col_range)


def form_blob_clusters(blobs, centers, threshold):
    """
    Arguments:
        blobs:          RGB image, with different blobs in different colors.
        centers:        A sequence of center points as extracted from the blob
                        segmentation.
        threshold:      Distance threshold in pixels
    Returns:
        neighbors:      Dictionary mapping a center index (0-based) to a
                        list of indices of neighboring centers:
                            { i: [j0, j1, ... ] }
                        To extract the point coordinates:
                            centers[i], centers[j]
        cluster_ids:    An image with shape==blobs.shape containing the
                        cluster ids (1-based). The image results from
                        thresholding the distance transformed blobs image.
        center_cids:    Maps to every blob center a cluster id.
                            center_cids[i] => cluster id of center[i]
                        Note that the cluster ids are 1-based.
    """
    _COLOR_TO_IND = [1000*1000, 1000, 1]

    def _color_to_ind(img):
        weights = _COLOR_TO_IND
        return img.dot(weights)

    def _pick_a_pixel_per_blob(blobs):
        """
        Pick the first pixel of each blob.
        Can be used as an alternative to blob "centers".
        Warning: the order of pixels doesn't match centers.
        """
        blobs_ids = _color_to_ind(img=blobs_ids)
        colors = np.unique(blobs_ids)
        pixels = np.zeros([len(colors),2])
        for i, color in enumerate(colors):
            # Get first pixel with that value.
            # https://stackoverflow.com/questions/16243955
            idx = np.argmax(blobs_ids==color)
            pixels[i] = np.unravel_index(idx, blobs_ids.shape)
        return pixels

    gray = cv.cvtColor(blobs, cv.COLOR_RGB2GRAY)
    # Threshold anything > 0 and assign value 255.
    mask = cv.threshold(gray, 0, 255, cv.THRESH_BINARY)[1]
    # Options for maskSize: 3, 5, or cv.DIST_MASK_PRECISE
    # Difference is barely noticeable. Runtime increases moderately.
    dist = cv.distanceTransform(src=~mask,
                                distanceType=cv.DIST_L2,
                                maskSize=5)
    # Form clusters: Merge blobs within distance thr.
    _, merged = cv.threshold(src=dist,
                             thresh=threshold,
                             maxval=255,
                             type=cv.THRESH_BINARY_INV)
    merged = merged.astype(np.uint8)
    # Create labels for connected components.
    # cluster_ids is a 2D array with dtype int32.
    n_clusters, cluster_ids = cv.connectedComponents(merged)
    assert(cluster_ids.shape[0] == blobs.shape[0])
    assert(cluster_ids.shape[1] == blobs.shape[1])

    # For the list of centroids, check if the points lie within a blob.
    # In case they don't, throw a warning and pick the closest blob (after
    # merging). This works well for decently shaped blobs.
    # Alternatively, one could try to pick a pixel from the blob:
    # See _pick_a_pixel_per_blob() to pick the first pixel from the blob.
    # Side note: To pick a point well inside an arbitrary contour (with
    # potential holes) is not trivial. See for instance:
    # https://stackoverflow.com/questions/1203135
    # https://stackoverflow.com/questions/35530634/
    centers = centers.round().astype(int)
    colors = blobs[centers["y"], centers["x"], :]
    center_is_inside_blob = (colors != (0,0,0)).any(axis=1)
    all_centers_inside_blob = all(center_is_inside_blob)
    if all_centers_inside_blob:
        # Index of cluster per center.
        center_cids = cluster_ids[centers["y"], centers["x"]]
        # Each point's neighbors are those points with the same cluster id.
        neighbors = { i: np.where(center_cids==cid)[0]
                      for i,cid in enumerate(center_cids) }
    else:
        assert(False)
        # TODO: implement a fallback-option. For instance:
        #       Pick closest blob (after distance-merge)
    # Test: The number of neighbors must be the same for points belonging
    #       to the same cluster.
    for i, ns in neighbors.items():
        c_len = len(ns)
        c_lens = [len(neighbors[j]) for j in ns]
        assert(all(n==c_len for n in c_lens))
    return neighbors, cluster_ids, center_cids


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


def form_point_clusters(centers, threshold):
    point_tree = spatial.cKDTree(centers)
    neighbors = point_tree.query_ball_point(centers, r=threshold)
    neighbors = {i:n for i,n in enumerate(neighbors)}
    neighbors = form_clusters(neighbors)
    return neighbors


def count_events(centers, neighbors):
    clusters = set(map(tuple, neighbors.values()))
    cluster_sizes = pd.Series([len(n) for n in clusters], dtype=int)
    counts = cluster_sizes.value_counts(sort=False)
    sizes = counts.index
    counts.index.name = "cluster_size"
    counts_blobs_rel = counts * sizes / len(centers)
    counts_clusters_rel = counts / sum(counts)
    counts = pd.concat([counts, counts_clusters_rel, counts_blobs_rel], axis=1,
                       keys=["count", "count_clusters_rel", "count_blobs_rel"])
    return counts


def print_counts(counts, centers, indent=4):
    indent = " "*indent
    def _print(msg=""):
        print(indent+msg)
    _print("Formula for freq: count/total_clusters")
    _print("Total number of blobs: %4d" % len(centers))
    if counts.empty:
        return
    for n, row in counts.iterrows():
        msg = "Number of %d-clusters: %5d (freq: %.3f)"
        _print(msg % (n, row["count"], row["count_clusters_rel"]))
        #print(msg % (n, row["count"], row["count_blobs_rel"]))
    _print()
    #print("Formula for freq: count*cluster_size/total_blobs")


def format_count_data(cluster_counts, blob_counts):
    def _merge(df):
        ret = df.loc[:2].copy()
        ret.loc[">=3"] = df.loc[3:].sum(axis=0)
        return ret
    cluster_counts = {did: _merge(df) for did, df in cluster_counts.items()}
    data = pd.concat(cluster_counts.values(), keys=cluster_counts.keys())
    data.index.names = ["dataset_id", "cluster_size"]
    data = data.reset_index()
    data = data.pivot(index="dataset_id", columns="cluster_size",
                      values=["count", "count_clusters_rel", "count_blobs_rel"])
    data.columns.names = ["metric", "cluster_size"]
    data["blobs_counts"] = blob_counts.values()
    data = data.rename({"count": "cluster_counts",
                        "count_clusters_rel": "cluster_counts_rel"},
                       level=0, axis=1)
    try:
        data["cluster_counts"] = data["cluster_counts"].astype(int)
    except ValueError:
        pass
    return data


def colorize_clusters(blobs, clusters, neighbors, center_cids):
    """
    Colorize blobs to indicate if it belongs to a cluster with 1, 2, 3
    or more blobs. (Same as in visualize_neighbors()).
    Strategy: clusters is an image with int32 ids. Look up for each blob
    the number of identified neighbors (lookup 1). In a second step, map
    the cluster sizes with the colors assigned in CLUSTER_COLORS (lookup 2).
    Note that the neighbors container uses the indices of the blob centers
    as read from the blob centers file. To convert those to cluster ids,
    we need to apply yet another lookup (lookup 3).
    """
    # Extract outlines of the blobs.
    edges = cv.Canny(blobs, 0, 0)
    edges = cv.threshold(edges, 0, 255, cv.THRESH_BINARY)[1]
    kernel = np.ones((3,3), np.uint8)
    edges = cv.dilate(edges, kernel=kernel, iterations=2)
    # Extract binary mask of blobs.
    blobs = cv.cvtColor(blobs, cv.COLOR_RGB2GRAY)
    mask = cv.threshold(blobs, 0, 255, cv.THRESH_BINARY)[1]
    # Colorize clusters: clusters is an image with int32 ids.
    # See: https://stackoverflow.com/questions/67761919/
    colors = {k: hex2rgb(v, opencv=True) for k,v in CLUSTER_COLORS.items()}
    # Lookups 1 and 3.
    cluster_sizes = {center_cids[i]: len(n) for i, n in neighbors.items()}
    cluster_sizes[0] = 0
    cluster_sizes = dict(sorted(cluster_sizes.items()))  # Sort by keys.
    # Lookup 2.
    cluster_colmap = [colors.get(n, colors["default"])
                      for n in cluster_sizes.values()]
    cluster_colmap = np.array(cluster_colmap, dtype=np.uint8)
    if clusters.max()>=len(cluster_colmap):
        # This may occur if blobs are present in the blob image for which
        # no corresponding center point exists. Let's mark that blob with
        # a special color.
        print("Warning: Extending the colormap by additional row(s).")
        alert_color = [255, 0, 255]     # magenta (BGR)
        n_rows_to_add = clusters.max()+1-len(cluster_colmap)
        new_rows = np.repeat([alert_color], n_rows_to_add, axis=0)
        cluster_colmap = np.vstack([cluster_colmap, new_rows])
    clusters_color = tuple(cmap[clusters] for cmap in cluster_colmap.T)
    clusters_color = cv.merge(clusters_color)
    # Apply mask of segmented blobs.
    blobs_color = cv.bitwise_and(clusters_color, clusters_color,
                                 mask=blobs)
    edges_color = cv.bitwise_and(clusters_color, clusters_color,
                                 mask=edges)
    # Alpha blend colorized blobs and colorized edges.
    # This is just an operation that improves aesthetics.
    alpha = 0.5
    overlay = cv.addWeighted(src1=blobs_color, alpha=alpha,
                             src2=edges_color, beta=1-alpha,
                             gamma=0.0)
    return overlay


def visualize_neighbors(image_path, centers, neighbors, radius):
    img = cv.imread(str(image_path), cv.IMREAD_GRAYSCALE)
    colors = CLUSTER_COLORS
    fig, ax = plt.subplots()
    ax.imshow(img, cmap=plt.get_cmap("gray"), vmin=0, vmax=255)
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    clusters = set(map(tuple,neighbors.values()))
    for ids in clusters:
        points = centers.iloc[list(ids)]
        color = colors.get(len(ids), colors["default"])
        if len(points)<=1:
            continue
        if True:
            # Connect neighbors by a line.
            if len(points)>2:
                hull = spatial.ConvexHull(points.values)
                ax.plot(points.values[hull.vertices,0],
                        points.values[hull.vertices,1],
                        linestyle="-", color=color,
                        marker=None, alpha=0.8)
            else:
                ax.plot(points["x"], points["y"],
                        color=color, marker=None, linestyle="-")
        circles = [plt.Circle((x,y), radius=radius, linewidth=0,
                              facecolor=color, alpha=0.7)
                   for x,y in points.values]
        c = mpc.PatchCollection(circles, match_original=True)
        ax.add_collection(c)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.axis(False)
    return fig


def run(args):
    data_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    threshold = args.threshold
    out_level = args.out_level
    show = args.show
    mode = args.mode
    dataset_ids = search_datasets(data_dir=data_dir)
    if not dataset_ids:
        print("Error: No datasets found to process in: %s" % data_dir)
        exit(1)

    cluster_counts = {}
    blob_counts = {}
    for i, dataset_id in enumerate(dataset_ids):
        print("Processing %d/%d (%s)" % (i+1, len(dataset_ids), dataset_id))

        ##### Analyze clusters.
        centers = read_centers(data_dir=data_dir,
                               dataset_id=dataset_id)
        if mode=="centers":
            neighbors = form_point_clusters(centers=centers,
                                            threshold=threshold)
        elif mode=="contours":
            segment_file = ensure_data_path(data_dir=data_dir,
                                            dataset_id=dataset_id,
                                            suffix=SEGMENTATION_SUFFIX)
            blobs = cv.imread(str(segment_file), cv.IMREAD_COLOR)
            ret = form_blob_clusters(blobs=blobs,
                                     centers=centers,
                                     threshold=threshold)
            neighbors, clusters, center_cids = ret

        counts = count_events(centers=centers, neighbors=neighbors)
        if counts is None:
            print("No clusters found (thr=%.1f)!" % threshold)
        cluster_counts[dataset_id] = counts
        blob_counts[dataset_id] = len(centers)

        ##### Visualize data.
        clusters_colored = None
        if mode=="contours" and out_level>=2:
            clusters_colored = colorize_clusters(blobs=blobs,
                                                 clusters=clusters,
                                                 neighbors=neighbors,
                                                 center_cids=center_cids)
        fig = None
        if mode=="centers" and out_level>=2:
            background_path = ensure_data_path(data_dir=data_dir,
                                               dataset_id=dataset_id,
                                               suffix=BACKGROUND_SUFFIX)
            fig = visualize_neighbors(image_path=background_path,
                                      centers=centers,
                                      neighbors=neighbors,
                                      radius=threshold/2)

        ##### Write data.
        if out_level >= 0:
            print_counts(counts=counts, centers=centers)
        if out_level >= 1:
            ensure_dir(out_dir)
            counts.to_csv(out_dir/(dataset_id+"_counts.csv"),
                          float_format="%.4f")
        if out_level >= 2:
            if clusters_colored is not None:
                cv.imwrite(filename=str(out_dir/(dataset_id+"_clusters.png")),
                           img=clusters_colored)
            if fig is not None:
                save_figure(path=out_dir/(dataset_id+".pdf"), fig=fig, dpi=600)
                #save_figure(path=out_dir/(dataset_id+".png"), fig=fig, dpi=1200)

        if show:
            plt.show()
        plt.close(fig)

    data = format_count_data(cluster_counts=cluster_counts,
                             blob_counts=blob_counts)
    if out_level >= 1:
        data.to_csv(out_dir/"_summary.csv", float_format="%.4f")


def parse_args():
    description = ("Count touching blobs based on blob centers.")
    formatter = argparse.RawTextHelpFormatter
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
                        help="Distance threshold in pixels")
    parser.add_argument("--show", action="store_true",
                        help="Visualize and show the figure.")
    parser.add_argument("--out-level", type=int, default=2,
                        help=("Controls the amount of data written: \n"
                              "0: No data is written\n"
                              "1: Only text data is written\n"
                              "2: Text and image data is written\n"
                              "Default: 2"))
    parser.add_argument("--mode", choices=("centers", "contours"),
                        default="centers",
                        help="Processing mode. Default: 'centers'")
    parser.set_defaults(func=run)
    return parser.parse_args()


def main():
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
