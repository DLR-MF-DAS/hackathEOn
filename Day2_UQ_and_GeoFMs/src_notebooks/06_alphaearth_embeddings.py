# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.6
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# **Part 2 · Geo Foundation Models · notebook 3 of 3**
#
# # Notebook 6: AlphaEarth Foundations, ready-made embeddings for every 10 m pixel
#
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DLR-MF-DAS/hackathEOn/blob/main/Day2_UQ_and_GeoFMs/06_alphaearth_embeddings.ipynb)
#
# Needs internet access the first time: the index file and a few internal
# blocks of the data files, up to a few hundred MB depending on the connection. Afterwards everything is cached in `data/alphaearth/`.
#
# In notebooks 4 and 5 *we* computed embeddings with TerraMind. Google DeepMind's
# [AlphaEarth Foundations](https://arxiv.org/abs/2507.22291) (AEF) goes one step further: the embeddings are
# **pre-computed for the whole land surface, every year since 2017, at 10 m resolution**.
#
# * Each pixel gets a vector of **64 numbers** (unit length). It summarises a whole year of
#   Sentinel-2, Sentinel-1, Landsat and other data at that place.
# * Similar places get similar vectors, so the **dot product** of two vectors measures similarity.
# * Nothing to run: you only read the numbers. (You also cannot run the model on your own images.)
#
# The data is free (CC-BY 4.0). It is in Google Earth Engine and, without any account, as Cloud-Optimized GeoTIFFs
# on [Source Cooperative](https://source.coop/tge-labs/aef). We read only a small window over HTTP.
# This notebook follows the official Earth Engine tutorial series on the
# [Satellite Embedding dataset](https://developers.google.com/earth-engine/tutorials/community/satellite-embedding-01-introduction)
# (introduction, unsupervised classification, supervised classification, similarity search, change),
# but in plain Python without Earth Engine:
#
# 1. read the embeddings of a small region for two years
# 2. look at them, cluster them without labels (k-means)
# 3. similarity search: "find all places like this one"
# 4. change detection between two years
# 5. few-label classification with ESA WorldCover labels, plus uncertainty from an ensemble
#
# *"The AlphaEarth Foundations Satellite Embedding dataset is produced by Google and Google DeepMind."*
#
# **Based on** (existing tutorials this one builds on):
# - Earth Engine tutorial series on the Satellite Embedding dataset (CC-BY 4.0 text, Apache-2.0 code):
#   [introduction](https://developers.google.com/earth-engine/tutorials/community/satellite-embedding-01-introduction), [unsupervised classification](https://developers.google.com/earth-engine/tutorials/community/satellite-embedding-02-unsupervised-classification),
#   [supervised classification](https://developers.google.com/earth-engine/tutorials/community/satellite-embedding-03-supervised-classification), [similarity search](https://developers.google.com/earth-engine/tutorials/community/satellite-embedding-05-similarity-search).
#   Same steps, translated from Earth Engine to plain Python (rasterio, scikit-learn).
# - [Data on GCS / Source Cooperative](https://developers.google.com/earth-engine/guides/aef_on_gcs_readme): file layout,
#   index file and the dequantisation formula; [aef-loader](https://github.com/jakenotjay/aef-loader) for the Source Cooperative paths.
# %%
# Google Colab only (skipped on your own computer): a Colab link opens only this notebook, so we fetch
# the rest of the course folder (course_utils.py, scripts) from GitHub and install the missing packages.
import os, sys
REPO = "https://github.com/DLR-MF-DAS/hackathEOn"  # repository that contains the course folder
FOLDER = "Day2_UQ_and_GeoFMs"                        # path of the course folder in the repository
if "google.colab" in sys.modules and not os.path.exists("course_utils.py"):
    shell = get_ipython().system  # like "!" in a notebook cell: the output shows up here
    shell(f"git clone -q --depth 1 --filter=blob:none --sparse {REPO} /content/repo")
    shell(f"git -C /content/repo sparse-checkout set {FOLDER}")  # only the course folder
    os.chdir(f"/content/repo/{FOLDER}")
    shell("pip install -q rasterio")

# %%
import urllib.request

from tqdm import tqdm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import rasterio
from rasterio.windows import Window
from rasterio.warp import transform_bounds, reproject, Resampling
from rasterio.transform import Affine
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from course_utils import set_seed

set_seed(0)

# The region: lon_min, lat_min, lon_max, lat_max (WGS84). About 5 x 4 km around DLR Oberpfaffenhofen.
# Choose any place on land; keep it small (a few km), it is read over the internet.
REGION_NAME = "oberpfaffenhofen"
BBOX = (11.250, 48.070, 11.315, 48.105)
YEARS = (2018, 2024)                 # two years for the change detection

CACHE_DIR = "data/alphaearth"        # downloaded windows are stored here, so a second run works offline
INDEX_URL = "https://data.source.coop/tge-labs/aef/v1/annual/aef_index.parquet"
WORLDCOVER_URL = "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/ESA_WorldCover_10m_2021_v200_{tile}_Map.tif"
os.makedirs(CACHE_DIR, exist_ok=True)

# %% [markdown]
# ## 1. Read the embeddings
#
# *File layout, index and dequantisation from the AEF data README (Google) and aef-loader (Source Cooperative paths).*
#
# The dataset consists of many large GeoTIFF files (8192 × 8192 pixels, 64 bands, one UTM zone each).
# An index file lists which file covers which area and year. We
#
# 1. download the index once,
# 2. find the file that contains our region,
# 3. read only the pixels of our region (a "window"), not the whole file.
#
# The values are stored as small integers (int8) to save space. They are turned back into the original
# numbers with the formula from the dataset documentation, $v = \mathrm{sign}(q)\,(q / 127.5)^2$.
# The value −128 marks "no data".

# %%
def to_gdal_path(path):
    """Index paths (s3:// or gs://) -> a URL that rasterio/GDAL can read over HTTP. Local paths stay."""
    key = None
    if path.startswith("s3://us-west-2.opendata.source.coop/"):
        key = path[len("s3://us-west-2.opendata.source.coop/"):]
    elif path.startswith("gs://alphaearth_foundations/satellite_embedding/v1/annual/"):
        key = "tge-labs/aef/v1/annual/" + path[len("gs://alphaearth_foundations/satellite_embedding/v1/annual/"):]
    return f"/vsicurl/https://data.source.coop/{key}" if key else path


def load_index():
    local = os.path.join(CACHE_DIR, "aef_index.parquet")
    if not INDEX_URL.startswith("http"):  # a local copy of the index
        local = INDEX_URL
    elif not os.path.exists(local):
        with tqdm(desc="downloading the index (once)", unit="B", unit_scale=True, unit_divisor=1024,
                  file=sys.stdout, ncols=100, mininterval=0.5, leave=False) as bar:
            def hook(blocks, block_size, total):
                bar.total = total if total > 0 else None
                bar.update(blocks * block_size - bar.n)
            urllib.request.urlretrieve(INDEX_URL, local + ".part", reporthook=hook)
        os.replace(local + ".part", local)
    cols = ["path", "year", "crs", "wgs84_west", "wgs84_south", "wgs84_east", "wgs84_north"]
    return pd.read_parquet(local, columns=cols)


def find_file(index, bbox, year):
    w, s, e, n = bbox
    rows = index[(index.year == year) & (index.wgs84_west <= w) & (index.wgs84_east >= e)
                 & (index.wgs84_south <= s) & (index.wgs84_north >= n)]
    if len(rows) == 0:
        raise ValueError(f"no single file covers {bbox} in {year}: choose a smaller region or move it a bit")
    return rows.iloc[0]


def read_window(path, bbox):
    """Read the pixels inside bbox (WGS84). Returns int8 array (64, H, W), north-up transform, CRS."""
    with rasterio.open(path) as src:
        left, bottom, right, top = transform_bounds("EPSG:4326", src.crs, *bbox)
        inv = ~src.transform
        cols, rows = zip(*[inv * (x, y) for x in (left, right) for y in (bottom, top)])
        c0, c1 = max(int(np.floor(min(cols))), 0), min(int(np.ceil(max(cols))), src.width)
        r0, r1 = max(int(np.floor(min(rows))), 0), min(int(np.ceil(max(rows))), src.height)
        win = Window(c0, r0, c1 - c0, r1 - r0)
        arr = src.read(window=win)
        t = src.window_transform(win)
        if t.e > 0:  # some files are stored bottom-up: flip so that row 0 is north
            arr = arr[:, ::-1, :]
            t = Affine(t.a, t.b, t.c, t.d, -t.e, t.f + t.e * arr.shape[1])
        return arr, t, src.crs.to_string()


def get_embeddings(year):
    """Embeddings of the region for one year, as float array (H, W, 64), cached on disk."""
    cache = os.path.join(CACHE_DIR, f"{REGION_NAME}_{year}.npz")
    if not os.path.exists(cache):
        row = find_file(load_index(), BBOX, year)
        print(f"{year}: reading a window from {row.path} (a few seconds to a minute) ...", flush=True)
        arr, t, crs = read_window(to_gdal_path(row.path), BBOX)
        np.savez_compressed(cache, q=arr, transform=np.array(t)[:6], crs=crs)
    d = np.load(cache)
    q = d["q"].astype(np.float32)
    emb = np.sign(q) * (q / 127.5) ** 2          # dequantize
    emb[d["q"] == -128] = np.nan                  # no data
    return np.moveaxis(emb, 0, -1), Affine(*d["transform"]), str(d["crs"])


emb = {}
for year in YEARS:
    emb[year], transform, crs = get_embeddings(year)
H, W, D = emb[YEARS[1]].shape
print(f"{H} x {W} pixels (10 m) x {D} dimensions, CRS {crs}")
print("vector length (should be 1):", np.round(np.nanmean(np.linalg.norm(emb[YEARS[1]], axis=-1)), 3))

# %% [markdown]
# ## 2. What do 64 numbers per pixel look like?
#
# *As in the Earth Engine tutorial 'Introduction to the Satellite Embedding Dataset' (three bands as RGB); PCA added.*
#
# Left: three of the 64 dimensions shown as red, green and blue (as in the official tutorial).
# Right: the three main directions of variation (PCA) as colours. Places with similar colours have similar embeddings.
# No labels were used for either picture.

# %%
E = emb[YEARS[1]]
valid = ~np.isnan(E).any(-1)
X = E[valid]                                          # (n_pixels, 64)


def to_rgb(values):
    img = np.zeros((H, W, 3))
    lo, hi = np.percentile(values, [2, 98], axis=0)
    img[valid] = np.clip((values - lo) / (hi - lo), 0, 1)
    return img


pca = PCA(n_components=3, random_state=0).fit(X)
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
axes[0].imshow(to_rgb(X[:, [0, 16, 37]]))
axes[0].set_title(f"{YEARS[1]}: dimensions A00, A16, A37 as RGB")
axes[1].imshow(to_rgb(pca.transform(X)))
axes[1].set_title(f"first 3 principal components ({pca.explained_variance_ratio_.sum():.0%} of the variance)")
for ax in axes:
    ax.axis("off")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 3. Clusters without any labels (k-means)
#
# *As in the Earth Engine tutorial 'Unsupervised Classification'.*
#
# k-means groups the pixels into $k$ clusters of similar embeddings, without any labels.
# Compare the clusters with land cover types you can see (forest, fields, settlements, water).

# %%
N_CLUSTERS = 6
rng = np.random.default_rng(0)
sample = X[rng.choice(len(X), min(20000, len(X)), replace=False)]
kmeans = KMeans(n_clusters=N_CLUSTERS, n_init=4, random_state=0).fit(sample)
clusters = np.full((H, W), np.nan)
clusters[valid] = kmeans.predict(X)

plt.figure(figsize=(6.5, 5.5))
plt.imshow(clusters, cmap="tab10", vmin=0, vmax=9, interpolation="nearest")
plt.title(f"k-means with {N_CLUSTERS} clusters, no labels")
plt.axis("off")
plt.show()

# %% [markdown]
# ## 4. Labels: ESA WorldCover
#
# For the next steps we need labels. We use the [ESA WorldCover](https://esa-worldcover.org) 2021 land cover map
# (10 m, 11 classes), also free and readable over HTTP, and resample it to the pixel grid of the embeddings.
# WorldCover is itself a model product, so its labels contain errors, especially at boundaries (mixed pixels).

# %%
WC_CLASSES = {10: ("Tree cover", "#006400"), 20: ("Shrubland", "#ffbb22"), 30: ("Grassland", "#ffff4c"),
              40: ("Cropland", "#f096ff"), 50: ("Built-up", "#fa0000"), 60: ("Bare / sparse", "#b4b4b4"),
              70: ("Snow and ice", "#f0f0f0"), 80: ("Water", "#0064c8"), 90: ("Herbaceous wetland", "#0096a0"),
              95: ("Mangroves", "#00cf75"), 100: ("Moss and lichen", "#fae6a0")}


def worldcover_tile(lon, lat):
    """WorldCover tiles are 3 x 3 degrees, named by their south-west corner, e.g. N48E009."""
    la, lo = int(np.floor(lat / 3) * 3), int(np.floor(lon / 3) * 3)
    return f"{'N' if la >= 0 else 'S'}{abs(la):02d}{'E' if lo >= 0 else 'W'}{abs(lo):03d}"


def get_worldcover():
    cache = os.path.join(CACHE_DIR, f"{REGION_NAME}_worldcover.npy")
    if not os.path.exists(cache):
        cx, cy = (BBOX[0] + BBOX[2]) / 2, (BBOX[1] + BBOX[3]) / 2
        url = WORLDCOVER_URL.format(tile=worldcover_tile(cx, cy))
        url = "/vsicurl/" + url if url.startswith("http") else url
        print("reading WorldCover from", url, "(a few seconds to a minute) ...", flush=True)
        labels = np.zeros((H, W), dtype=np.uint8)
        with rasterio.open(url) as src:
            reproject(rasterio.band(src, 1), labels, dst_transform=transform, dst_crs=crs,
                      resampling=Resampling.nearest)
        np.save(cache, labels)
    return np.load(cache)


wc = get_worldcover()
present = [c for c in WC_CLASSES if (wc[valid] == c).sum() >= 200]  # classes with enough pixels
cmap = ListedColormap([WC_CLASSES[c][1] for c in present])
wc_idx = np.full((H, W), np.nan)
for i, c in enumerate(present):
    wc_idx[wc == c] = i

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
axes[0].imshow(wc_idx, cmap=cmap, vmin=-0.5, vmax=len(present) - 0.5, interpolation="nearest")
axes[0].set_title("ESA WorldCover 2021")
axes[1].imshow(clusters, cmap="tab10", vmin=0, vmax=9, interpolation="nearest")
axes[1].set_title("k-means clusters of the embeddings")
for ax in axes:
    ax.axis("off")
handles = [plt.Rectangle((0, 0), 1, 1, color=WC_CLASSES[c][1]) for c in present]
axes[0].legend(handles, [WC_CLASSES[c][0] for c in present], loc="lower left", fontsize=8)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 5. Similarity search: "find all places like this one"
#
# *As in the Earth Engine tutorial 'Similarity Search' (dot product with reference embeddings).*
#
# The embeddings have length 1, so the **dot product** of two embeddings is their cosine similarity
# (1: identical, around 0: unrelated). We take the average embedding of a few built-up pixels as a query
# and compute its similarity to every pixel, as in the official similarity search tutorial.

# %%
QUERY_CLASS = 50 if 50 in present else present[0]
q_pixels = np.argwhere((wc == QUERY_CLASS) & valid)
q_pixels = q_pixels[rng.choice(len(q_pixels), min(5, len(q_pixels)), replace=False)]
query = E[q_pixels[:, 0], q_pixels[:, 1]].mean(0)
query /= np.linalg.norm(query)
similarity = E @ query                                 # (H, W): dot product with every pixel

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
im = axes[0].imshow(similarity, cmap="magma", vmin=0, vmax=1)
axes[0].plot(q_pixels[:, 1], q_pixels[:, 0], "c+", ms=12, mew=2, label="query pixels")
axes[0].legend(loc="lower left")
axes[0].set_title(f"similarity to '{WC_CLASSES[QUERY_CLASS][0]}' pixels")
plt.colorbar(im, ax=axes[0], fraction=0.046)
axes[1].imshow(similarity > np.nanpercentile(similarity, 90), cmap="gray")
axes[1].set_title("the 10% most similar pixels")
for ax in axes:
    ax.axis("off")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 6. What changed between two years?
#
# *Dot product between years, as suggested in the dataset documentation.*
#
# For the same pixel in two years, the dot product is close to 1 if nothing changed.
# $1 - e_{2018} \cdot e_{2024}$ is therefore a simple **change score**: new buildings, cleared forest,
# a changed crop rotation. (Small differences also come from different weather and sensor coverage.)

# %%
change = 1 - np.sum(emb[YEARS[0]] * emb[YEARS[1]], axis=-1)
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
im = axes[0].imshow(change, cmap="inferno", vmin=0, vmax=np.nanpercentile(change, 99.5))
axes[0].set_title(f"change score 1 − e{YEARS[0]} · e{YEARS[1]}")
plt.colorbar(im, ax=axes[0], fraction=0.046)
axes[0].axis("off")
axes[1].hist(change[valid], bins=100, color="gray")
axes[1].axvline(np.nanpercentile(change, 99), color="r", label="top 1%")
axes[1].set_yscale("log")
axes[1].set_xlabel("change score")
axes[1].legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 7. Few labels: a linear classifier on the embeddings
#
# *Idea from the Earth Engine tutorial 'Supervised Classification'.*
#
# As in notebook 5: a simple classifier with only a few labelled pixels per class.
#
# **Careful with neighbouring pixels.** Pixels next to each other are almost identical. If training and test pixels
# are mixed randomly, the test is far too easy. We therefore train on the **left half** of the region and test on
# the **right half** (a spatial split).

# %%
y_map = wc_idx
left = np.zeros((H, W), bool)
left[:, : W // 2] = True
train_mask = left & valid & ~np.isnan(y_map)
test_mask = ~left & valid & ~np.isnan(y_map)
classes_ok = [i for i in range(len(present)) if ((y_map == i) & train_mask).sum() >= 50]


def pick_pixels(k, seed):
    r = np.random.default_rng(seed)
    idx = [r.choice(np.argwhere((y_map == i) & train_mask), k, replace=False) for i in classes_ok]
    return np.concatenate(idx)


def fit_probe(pix):
    clf = LogisticRegression(max_iter=2000, C=10.0)
    return clf.fit(E[pix[:, 0], pix[:, 1]], y_map[pix[:, 0], pix[:, 1]].astype(int))


test_pix = np.argwhere(test_mask & np.isin(y_map, classes_ok))
X_test, y_test = E[test_pix[:, 0], test_pix[:, 1]], y_map[test_pix[:, 0], test_pix[:, 1]].astype(int)

ks = [1, 5, 10, 50]
accs = [np.mean([(fit_probe(pick_pixels(k, s)).predict(X_test) == y_test).mean() for s in range(3)]) for k in ks]
for k, a in zip(ks, accs):
    print(f"{k:3d} labelled pixels per class: accuracy {a:.1%}")

plt.figure(figsize=(6, 4))
plt.plot(ks, accs, "o-")
plt.xscale("log")
plt.xticks(ks, [str(k) for k in ks])
plt.ylim(0, 1)
plt.xlabel("labelled pixels per class")
plt.ylabel("accuracy (right half)")
plt.title("AlphaEarth embeddings + logistic regression")
plt.grid(alpha=0.3)
plt.show()

# %% [markdown]
# ## 8. How sure is the classifier? An ensemble of linear heads
#
# Back to Part 1 (uncertainty quantification). A cheap way to get uncertainty on top of a foundation model: train several small heads,
# each on a **different random draw** of 10 labelled pixels per class, and look at how much they disagree.
#
# * **total uncertainty**: entropy of the averaged probabilities
# * **disagreement** (epistemic): total minus the average entropy of the single heads (mutual information)

# %%
N_HEADS = 10
probs = np.stack([fit_probe(pick_pixels(10, seed=100 + m)).predict_proba(X) for m in range(N_HEADS)])  # (M, n, C)


def entropy(p):
    return -(p * np.log(p + 1e-12)).sum(-1)


p_mean = probs.mean(0)
total = entropy(p_mean)
disagreement = total - entropy(probs).mean(0)
pred_map = np.full((H, W), np.nan)
pred_map[valid] = np.array(classes_ok)[p_mean.argmax(1)]
maps = {"prediction (ensemble mean)": pred_map, "total uncertainty (entropy)": total,
        "disagreement between heads": disagreement}

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
for ax, (title, m) in zip(axes, maps.items()):
    if title.startswith("prediction"):
        ax.imshow(m, cmap=cmap, vmin=-0.5, vmax=len(present) - 0.5, interpolation="nearest")
    else:
        img = np.full((H, W), np.nan)
        img[valid] = m
        im = ax.imshow(img, cmap="viridis")
        plt.colorbar(im, ax=ax, fraction=0.046)
    ax.set_title(title)
    ax.axis("off")
plt.tight_layout()
plt.show()

# %% [markdown]
# Does high uncertainty point to wrong predictions? We compare with WorldCover on the right half
# (AUROC: 0.5 = no relation, 1 = uncertainty perfectly separates wrong from right).

# %%
flat_valid = np.argwhere(valid)
is_test = test_mask[flat_valid[:, 0], flat_valid[:, 1]] & np.isin(y_map[flat_valid[:, 0], flat_valid[:, 1]], classes_ok)
truth = y_map[flat_valid[:, 0], flat_valid[:, 1]][is_test]
wrong = np.array(classes_ok)[p_mean[is_test].argmax(1)] != truth
print(f"accuracy of the ensemble on the right half: {1 - wrong.mean():.1%}")
for name, score in [("total uncertainty", total[is_test]), ("disagreement", disagreement[is_test])]:
    print(f"AUROC '{name}' for finding wrong pixels: {roc_auc_score(wrong, score):.2f}")

# %% [markdown]
# Where is the uncertainty high? Look at boundaries between land cover types and mixed pixels (e.g. gardens in
# settlements), where the data itself is ambiguous (aleatoric), and at rare classes (disagreement, epistemic).
# Note: WorldCover is a model product too, so some "errors" are errors of the labels.
#
# ## Exercises
#
# 1. Change `BBOX` to a region you know and compare the k-means map with what you know about it.
# 2. Use a water pixel (WorldCover class 80) as query in the similarity search.
# 3. Change `YEARS` to two years with a known event (a new building area, a forest fire). Do you find it?
# 4. Train the heads with 50 instead of 10 pixels per class. How do accuracy and disagreement change?
#
# **Further reading:** Brown et al. (2025), *AlphaEarth Foundations: An embedding field model for accurate and
# efficient global mapping from sparse label data*, arXiv:2507.22291; the Earth Engine tutorial series on the
# Satellite Embedding dataset.
