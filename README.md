# blob-clusters

Utility to analyze blobs in a 2D image. 

**Problem statement**: Given the blobs of a 2D segmentation and a distance threshold $d$. Form clusters of blobs that are within a distance $d$ to each other, and calculate the cluster sizes and the frequency with which clusters of a certain size occur.

Two modes of operation are available:

- **centers**: Given is a list of blob centers $(x,y)$. Two neighboring center points count towards the same cluster if they lie within threshold distance $d$. This approach uses a KD-tree algorithm to identify the neighbors that form a cluster.
- **contours**: Given is (A) a segmentation of the blobs in the form of color, grayscale or binary image; and (B) a list of blob centers $(x,y)$. Two blobs are counted towards the same cluster if their external boundaries (their contours) lie within distance $d$. This approach makes use of a [distance transform](https://docs.opencv.org/4.5.2/d2/dbd/tutorial_distance_transform.html) of the (binarized) segmentation image. The list of blob centers is used to identify a pixel *within* every blob which is used to determine to which cluster a blob belongs. This works under the assumption that the blob centers actually lie within the contour, which holds true for roundish shaped blobs. (This assumption could be alleviated in future by writing a new blob-pixel-pick routine, which also eliminates the dependency on the list of blob centers)

## Installation

The utility requires Python 3.8+.

```bash
git clone https://github.com/normanius/blob-clusters.git
cd blob-clusters
python -m pip install -r requirements.txt
```

## Usage

The bash variable `DATA_DIR` is assumed to point to a directory with one or multiple datasets. The current version of the utility is fully functional if two files are available per dataset:

- The file of blob centers, suffixed by `_Centers.csv`
- The file of segmented blobs, suffixed by `_Objects.tiff`

The suffixes can be adjusted right below the header of [analyze_clusters.py](https://github.com/normanius/blob-clusters/blob/main/analyze_clusters.py).

```bash
# Points to data directory
DATA_DIR="./data"

# Run the script in center mode
python analyze_clusters.py \
       --in-dir "$DATA_DIR" \
       --out-dir "./output" \
       --mode="centers" \
       --threshold=80
       
# Run the script in contour mode
python analyze_clusters.py \
       --in-dir "$DATA_DIR" \
       --out-dir "./output" \
       --mode="contours" \
       --threshold=5
       
# Show a full list of options
python analyze_clusters.py --help
```


## Samples

Input image showing blobs colorized blobs. A corresponding [.csv file](https://github.com/normanius/blob-clusters/blob/main/data/sample_Centers.csv) with columns "Location\_Center\_X" and "Location\_Center\_Y" indicates the blob centers. 

![Input image](https://github.com/normanius/blob-clusters/blob/main/data/sample_Objects.tiff?raw=true)


Visualization of the result using the contours mode:

![Input image](https://github.com/normanius/blob-clusters/blob/main/data/results/contours/sample_clusters_resized.png?raw=true)

Visualization of the result using the centers mode:

![Input image](https://github.com/normanius/blob-clusters/blob/main/data/results/centers/sample.png?raw=true)