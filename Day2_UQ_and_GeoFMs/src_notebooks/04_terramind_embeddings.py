# %% [markdown]
# # Notebook 4: Embeddings from a geo foundation model (TerraMind)
#
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DLR-MF-DAS/hackathEOn/blob/main/Day2_UQ_and_GeoFMs/04_terramind_embeddings.ipynb)
#
# **Session 2, part 1**
#
# [TerraMind](https://huggingface.co/ibm-esa-geospatial/TerraMind-1.0-tiny) is a multimodal
# foundation model for Earth observation by IBM and ESA. It was pre-trained on millions of
# Sentinel-1, Sentinel-2, DEM and land cover samples. We use the smallest version
# (**TerraMind-tiny**, about 6 million parameters) as a frozen feature extractor.
#
# What we do:
#
# 1. Load TerraMind through [TerraTorch](https://github.com/terrastackai/terratorch)
# 2. Turn every EuroSAT image into one vector, the **embedding**
# 3. Look at the embedding space (t-SNE) and use it for similarity search
# 4. Save the embeddings for notebooks 5 and 6 (in `outputs/04_terramind_embeddings/`)
#
# Runs on a CPU; a GPU is faster.
#
# **Based on** (existing notebooks this one builds on):
# - TerraTorch, [`embeddings/embedding_generation_manual_backbone_registry.ipynb`](https://github.com/IBM/terratorch/blob/main/examples/embeddings/embedding_generation_manual_backbone_registry.ipynb)
#   (Apache-2.0): loading TerraMind with the backbone registry, standardising with the pre-training statistics,
#   taking the last layer. We do this for a whole dataset and average the tokens.
# - [TerraMind repository](https://github.com/IBM/terramind) for model names and input modalities.
# %%
# Google Colab only (skipped on your own computer): a Colab link opens only this notebook, so we fetch
# the rest of the course folder (course_utils.py, scripts) from GitHub and install the missing packages.
# The data is downloaded automatically when a notebook needs it.
import os, sys
REPO = "https://github.com/DLR-MF-DAS/hackathEOn"  # repository that contains the course folder
FOLDER = "Day2_UQ_and_GeoFMs"                        # path of the course folder in the repository
if "google.colab" in sys.modules and not os.path.exists("course_utils.py"):
    shell = get_ipython().system  # like "!" in a notebook cell: the output shows up here
    shell(f"git clone -q --depth 1 --filter=blob:none --sparse {REPO} /content/repo")
    shell(f"git -C /content/repo sparse-checkout set {FOLDER}")  # only the course folder
    os.chdir(f"/content/repo/{FOLDER}")
    shell("pip install -q terratorch satellite_cloud_generator torchgeo")  # torchgeo: to build the EuroSAT subset

# %%
import time

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from course_utils import (
    CLASSES, DEVICE, CLOUD, set_seed, load_eurosat, show_images, add_clouds, EMBEDDINGS_FILE,
    TERRAMIND_INPUT, load_terramind, terramind_stats, terramind_embed,
)

set_seed(0)
DATA_VARIANT = "ms"      # "ms": all 13 Sentinel-2 bands (default) | "rgb": red, green, blue only
OUTPUT_DIR = os.path.dirname(EMBEDDINGS_FILE)  # outputs/04_terramind_embeddings
print("device:", DEVICE)

# %% [markdown]
# ## 1. Data: all 13 bands (or RGB)
#
# TerraMind was pre-trained with several modalities. Two of them fit EuroSAT:
#
# | `DATA_VARIANT` | File | TerraMind input | Values |
# |---|---|---|---|
# | `"ms"` (default, ~2 GB download) | `data/eurosat_ms.npz` | `"S2L1C"` (13 bands, Level-1C) | reflectance × 10,000 |
# | `"rgb"` (~90 MB download) | `data/eurosat_rgb.npz` | `"RGB"` (Sentinel-2 RGB) | 0-255 |
#
# The data is downloaded the first time. Notebooks 5 and 6 use the variant of the saved embeddings.

# %%
data = load_eurosat(DATA_VARIANT)   # downloads the data the first time
x = {s: data[f"x_{s}"] for s in ["train", "val", "test"]}   # (N, C, 64, 64) in the units of the data
y = {s: data[f"y_{s}"] for s in ["train", "val", "test"]}
rgb = {s: data[f"rgb_{s}"] for s in ["train", "val", "test"]}  # true-colour view, only for plotting
modality = TERRAMIND_INPUT[DATA_VARIANT][0]
print("TerraMind input:", modality)

show_images(rgb["train"] / 255, [CLASSES[i] for i in y["train"][:8]], n=8)

# %% [markdown]
# ## 2. Load TerraMind-tiny
#
# *Follows the TerraTorch example `embeddings/embedding_generation_manual_backbone_registry.ipynb` (backbone registry).*
#
# TerraTorch downloads the pre-trained weights from Hugging Face (about 25 MB).
# The model takes a dictionary `{modality name: tensor}`, here `{"S2L1C": images}` (or `{"RGB": images}`).

# %%
model = load_terramind(DATA_VARIANT, "terramind_v1_tiny", pretrained=True)
print(f"parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f} M")

# %% [markdown]
# What happens inside `terramind_embed` (see `course_utils.py`):
#
# 1. **Standardise** each channel with the mean and std of the pre-training data
# 2. **Resize** 64x64 -> 224x224, the input size used in pre-training
# 3. The image is cut into 16x16 patches -> 14 x 14 = **196 tokens**, each a 192-d vector
# 4. We take the output of the last layer and **average over all tokens** -> one 192-d vector per image

# %%
with torch.no_grad():
    mean, std = terramind_stats(DATA_VARIANT)
    xb = (x["train"][:2] - mean) / std
    xb = F.interpolate(xb, size=(224, 224), mode="bilinear")
    tokens = model({modality: xb.to(DEVICE)})
print("number of layer outputs:", len(tokens))
print("last layer output:", tuple(tokens[-1].shape), " = (images, tokens, features)")

# %% [markdown]
# ## 3. Compute the embeddings
#
# *Normalisation and last-layer embedding as in the TerraTorch example above.*
#
# We also compute embeddings of **cloudy** test images, which we need in notebook 6.

# %%
t0 = time.time()
emb = {s: terramind_embed(model, x[s], DATA_VARIANT, desc=f"embedding {s}") for s in ["train", "val", "test"]}
emb_cloudy = {}
for c in [0.3, 0.6]:
    x_cloudy = add_clouds(x["test"], coverage=c, cloud_value=CLOUD[DATA_VARIANT], seed=1)  # raw units
    emb_cloudy[c] = terramind_embed(model, x_cloudy, DATA_VARIANT, desc=f"embedding test, {c:.0%} clouds")
print(f"done in {time.time() - t0:.0f} s")

# saved for notebooks 5 and 6
os.makedirs(OUTPUT_DIR, exist_ok=True)
np.savez(
    EMBEDDINGS_FILE,
    **{f"emb_{s}": emb[s].numpy() for s in emb},
    **{f"y_{s}": y[s].numpy() for s in y},
    emb_test_cloud30=emb_cloudy[0.3].numpy(),
    emb_test_cloud60=emb_cloudy[0.6].numpy(),
    variant=DATA_VARIANT,
)
print(f"saved {EMBEDDINGS_FILE}")

print("embedding of one image:", tuple(emb["train"][0].shape))

# %% [markdown]
# ## 4. Look at the embedding space
#
# t-SNE squeezes the 192 dimensions into 2, trying to keep neighbours close.
# Nobody told the model about our 10 classes. If the classes still form groups, the
# embeddings carry useful information.

# %%
z = TSNE(n_components=2, perplexity=30, random_state=0).fit_transform(emb["train"].numpy())

plt.figure(figsize=(7, 6))
for k, name in enumerate(CLASSES):
    m = y["train"].numpy() == k
    plt.scatter(z[m, 0], z[m, 1], s=8, label=name)
plt.legend(fontsize=7, markerscale=2, bbox_to_anchor=(1, 1))
plt.title("t-SNE of TerraMind-tiny embeddings (no labels used)")
plt.xticks([])
plt.yticks([])
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 5. Similarity search
#
# With embeddings, "find images that look like this one" is just a nearest-neighbour search.
# We use the cosine similarity:
#
# $$\text{sim}(a, b) = \frac{a \cdot b}{\|a\|\,\|b\|}$$

# %%
def normalise(e):
    return e / e.norm(dim=1, keepdim=True)


db = normalise(emb["train"])


def search(query_index, k=6):
    q = normalise(emb["test"][query_index:query_index + 1])
    sims = (db @ q.T).squeeze()
    best = sims.argsort(descending=True)[:k]
    images = np.concatenate([rgb["test"][query_index:query_index + 1], rgb["train"][best.numpy()]]) / 255
    titles = [f"query\n{CLASSES[y['test'][query_index]]}"] + [
        f"{CLASSES[y['train'][i]]}\nsim {sims[i]:.2f}" for i in best
    ]
    show_images(images, titles, n=k + 1, ncols=k + 1)


for qi in [0, 10, 20]:
    search(qi)

# %% [markdown]
# ## Exercises
#
# 1. Use the **max** over the tokens instead of the mean (change `terramind_embed`). Does the t-SNE plot change?
# 2. Use an earlier layer, e.g. `tokens[5]` instead of `tokens[-1]`. Are the classes still separated?
# 3. Search with a cloudy query image. What does the model return?
# 4. Run the notebook with `DATA_VARIANT = "rgb"` and compare the t-SNE plots. How much do the other ten bands add?
