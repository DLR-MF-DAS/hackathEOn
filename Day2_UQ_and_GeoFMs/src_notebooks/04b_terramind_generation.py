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
# **Part 2 · Geo Foundation Models · optional notebook (4b)**
#
# # Notebook 4b (optional): Cross-modal generation with TerraMind
#
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DLR-MF-DAS/hackathEOn/blob/main/Day2_UQ_and_GeoFMs/04b_terramind_generation.ipynb)
#
# *A GPU is recommended.*
#
# TerraMind is not only an encoder: it can **generate one modality from another** ("any-to-any").
# From a Sentinel-2 image it produces, for example, a Sentinel-1 radar image, a digital elevation model (DEM),
# a land cover map (LULC) and a vegetation index (NDVI).
#
# What we do:
#
# 1. Load the TerraMind generation model through [TerraTorch](https://github.com/terrastackai/terratorch)
# 2. Generate Sentinel-1, DEM, LULC and NDVI for an example tile of the TerraMind repository and compare with the real data
# 3. Generate the same modalities for our EuroSAT images (13 bands, as in notebook 4)
# 4. Generate several times: how much do the samples differ? A first look at the **uncertainty of generated data**
#
# On a GPU the notebook uses TerraMind-base; on a CPU it uses TerraMind-tiny and fewer diffusion steps.
#
# **Based on** (existing notebooks this one builds on):
# - TerraMind repository (Apache-2.0), [`notebooks/terramind_generation.ipynb`](https://github.com/IBM/terramind/blob/main/notebooks/terramind_generation.ipynb)
#   and [`notebooks/terramind_any_to_any_generation.ipynb`](https://github.com/IBM/terramind/blob/main/notebooks/terramind_any_to_any_generation.ipynb):
#   building the model with `FULL_MODEL_REGISTRY`, running it with diffusion steps, plotting the outputs.
#   Here the input is Sentinel-2 L1C (13 bands) instead of L2A, so that it matches our EuroSAT data.
# - The example tiles (`examples/`) and the plotting helpers (`terramind_plotting.py`, a copy of
#   `notebooks/plotting_utils.py`) come from the same repository.
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
    shell("pip install -q terratorch rasterio satellite_cloud_generator torchgeo")  # torchgeo: to build the EuroSAT subset

# %%
import urllib.request

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import rasterio
from terratorch.registry import FULL_MODEL_REGISTRY

from course_utils import CLASSES, DEVICE, CLOUD, set_seed, load_eurosat, add_clouds, progress
from terramind_plotting import plot_s2, plot_modality

set_seed(0)
print("device:", DEVICE)

# On a GPU the base model with more diffusion steps, on a CPU the tiny model with few steps
MODEL_NAME = "terramind_v1_base_generate" if DEVICE == "cuda" else "terramind_v1_tiny_generate"
TIMESTEPS = 10 if DEVICE == "cuda" else 4      # diffusion steps of the image decoders: more = sharper, slower
OUTPUTS = ["S1RTC", "DEM", "LULC", "NDVI"]     # what TerraMind should generate
EXAMPLE_DIR = "data/terramind_examples"        # the example tiles are downloaded here (a few MB)

# %% [markdown]
# ## 1. The generation model
#
# *As in the TerraMind notebook `terramind_generation.ipynb`; input S2L1C instead of S2L2A.*
#
# `modalities` is the input, `output_modalities` what the model generates. Every output modality needs its own
# **tokenizer** with a diffusion decoder that turns the generated tokens back into an image (see the slides).
# TerraTorch downloads the weights from Hugging Face on first use. With `standardize=True` the model standardises
# the input with the pre-training statistics itself, so we pass raw Sentinel-2 values.
#
# Generation is **chained**: the first output is generated from the input, the second from the input and the
# first output, and so on. So the order of `OUTPUTS` matters a little.

# %%
model = FULL_MODEL_REGISTRY.build(
    MODEL_NAME,
    modalities=["S2L1C"],       # input: Sentinel-2 L1C, 13 bands, raw values
    output_modalities=OUTPUTS,
    pretrained=True,
    standardize=True,
)
model = model.to(DEVICE).eval()
print(MODEL_NAME, "with", TIMESTEPS, "diffusion steps")


def generate(x, seed=0):
    """x: (N, 13, 224, 224) raw S2L1C values -> dict {modality: (N, C, 224, 224)} on the CPU."""
    torch.manual_seed(seed)  # generation samples tokens: a fixed seed gives the same result again
    with torch.no_grad():
        out = model(x.to(DEVICE), timesteps=TIMESTEPS)
    return {k: v.cpu() for k, v in out.items()}

# %% [markdown]
# ## 2. An example tile from the TerraMind repository
#
# The [TerraMind repository](https://github.com/IBM/terramind/tree/main/examples) contains a few 224 × 224 tiles
# of the TerraMesh validation data in all modalities. We take the Sentinel-2 L1C image as input and compare the
# generated modalities with the real ones.

# %%
TILE = "38D_378R_2_3"   # other tiles: 282D_485L_3_3, 433D_629L_3_1, 637U_59R_1_3, 609U_541L_3_0
URL = "https://raw.githubusercontent.com/IBM/terramind/refs/heads/main/examples/{mod}/{tile}.tif"


def read_example(mod, tile=TILE):
    """Download one example tile (once) and return it as a (1, C, 224, 224) tensor."""
    path = os.path.join(EXAMPLE_DIR, mod, f"{tile}.tif")
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        urllib.request.urlretrieve(URL.format(mod=mod, tile=tile), path)
    with rasterio.open(path) as src:
        return torch.from_numpy(src.read().astype(np.float32)).unsqueeze(0)


s2 = read_example("S2L1C")
real = {mod: read_example(mod) for mod in OUTPUTS}
print("input:", tuple(s2.shape), " real:", {k: tuple(v.shape) for k, v in real.items()})

generated = generate(s2)

fig, axes = plt.subplots(2, len(OUTPUTS) + 1, figsize=(4 * (len(OUTPUTS) + 1), 8))
plot_s2(s2, ax=axes[0, 0])
axes[0, 0].set_title("input: Sentinel-2 L1C")
axes[1, 0].axis("off")
for j, mod in enumerate(OUTPUTS, start=1):
    plot_modality(mod, generated[mod], ax=axes[0, j])
    axes[0, j].set_title(f"generated {mod}")
    plot_modality(mod, real[mod], ax=axes[1, j])
    axes[1, j].set_title(f"real {mod}")
plt.tight_layout()
plt.show()

# %% [markdown]
# Compare the generated with the real modalities: large structures (water, fields, built-up areas, relief) and
# details. The model has never seen the radar image or the elevation of this place; it generates what is
# *typical* for such a Sentinel-2 image.

# %% [markdown]
# ## 3. Our EuroSAT images as input
#
# EuroSAT images have the same 13 bands (S2L1C) but only 64 × 64 pixels. As in notebook 4 we resize them to
# 224 × 224. One image per class:

# %%
data = load_eurosat("ms")   # always the 13 bands here (TerraMind input S2L1C); downloads the data the first time
x_test, y_test = data["x_test"], data["y_test"]
idx = [int(torch.nonzero(y_test == c)[0]) for c in range(10)]     # first test image of every class
x_in = F.interpolate(x_test[idx], size=(224, 224), mode="bilinear", align_corners=False)

gen = {mod: [] for mod in OUTPUTS}
for i in progress(range(len(idx)), desc="generating"):
    out = generate(x_in[i:i + 1])
    for mod in OUTPUTS:
        gen[mod].append(out[mod])
gen = {mod: torch.cat(v) for mod, v in gen.items()}

fig, axes = plt.subplots(len(idx), len(OUTPUTS) + 1, figsize=(2.6 * (len(OUTPUTS) + 1), 2.6 * len(idx)))
for r, i in enumerate(idx):
    axes[r, 0].imshow(data["rgb_test"][i])
    axes[r, 0].set_title(CLASSES[int(y_test[i])], fontsize=9)
    axes[r, 0].axis("off")
    for j, mod in enumerate(OUTPUTS, start=1):
        plot_modality(mod, gen[mod][r:r + 1], ax=axes[r, j])
        if r == 0:
            axes[r, j].set_title(f"generated {mod}", fontsize=9)
plt.tight_layout()
plt.show()

# %% [markdown]
# Does the generated land cover (LULC, classes of the ESRI land cover map: water, trees, crops, built area, ...)
# fit the EuroSAT label? Note that some EuroSAT classes (Pasture vs. HerbaceousVegetation, AnnualCrop vs.
# PermanentCrop) are finer than the classes of the land cover map.

# %% [markdown]
# ## 4. How sure is the generator? Sample several times
#
# TerraMind **samples** the tokens of the output, so every run gives a slightly different result. The spread over
# several samples shows where the model is unsure. We compare one image without and with synthetic clouds
# (the clouds of notebook 2, in raw units).

# %%
N_SAMPLES = 4
i = idx[CLASSES.index("Residential")]
x_clear = F.interpolate(x_test[i:i + 1], size=(224, 224), mode="bilinear", align_corners=False)
x_cloudy = F.interpolate(add_clouds(x_test[i:i + 1], coverage=0.6, cloud_value=CLOUD["ms"], seed=1),
                         size=(224, 224), mode="bilinear", align_corners=False)

samples = {}
for name, x in [("clear", x_clear), ("cloudy", x_cloudy)]:
    runs = [generate(x, seed=k) for k in progress(range(N_SAMPLES), desc=f"sampling ({name})")]
    samples[name] = {
        "S1RTC": torch.cat([r["S1RTC"] for r in runs]),                # (N_SAMPLES, 2, 224, 224): VV and VH
        "LULC": torch.cat([r["LULC"].argmax(1) for r in runs]),         # (N_SAMPLES, 224, 224): class per pixel
    }


def lulc_disagreement(classes):
    """Share of samples whose class differs from the most frequent class, per pixel (0 = all agree)."""
    majority = classes.mode(dim=0).values
    return (classes != majority).float().mean(dim=0)


fig, axes = plt.subplots(2, 5, figsize=(20, 8))
s1_max = float(samples["cloudy"]["S1RTC"][:, 0].std(dim=0).max()) or 1.0
for r, (name, x) in enumerate([("clear", x_clear), ("cloudy", x_cloudy)]):
    rgb = x[0, [3, 2, 1]].permute(1, 2, 0).numpy() / 3000
    axes[r, 0].imshow(np.clip(rgb, 0, 1))
    axes[r, 0].set_title(f"input ({name})")
    plot_modality("S1RTC", samples[name]["S1RTC"][0:1], ax=axes[r, 1])
    axes[r, 1].set_title("generated S1RTC, sample 1")
    im = axes[r, 2].imshow(samples[name]["S1RTC"][:, 0].std(dim=0), cmap="magma", vmin=0, vmax=s1_max)
    axes[r, 2].set_title(f"S1RTC: spread over {N_SAMPLES} samples (std, VV)")
    plt.colorbar(im, ax=axes[r, 2], fraction=0.046)
    plot_modality("LULC", samples[name]["LULC"][0], ax=axes[r, 3])
    axes[r, 3].set_title("generated LULC, sample 1")
    im = axes[r, 4].imshow(lulc_disagreement(samples[name]["LULC"]), cmap="magma", vmin=0, vmax=0.75)
    axes[r, 4].set_title("LULC: share of samples that disagree")
    plt.colorbar(im, ax=axes[r, 4], fraction=0.046)
    for ax in axes[r]:
        ax.axis("off")
plt.tight_layout()
plt.show()

for name in samples:
    print(f"{name:7s} S1RTC spread (mean std): {samples[name]['S1RTC'][:, 0].std(dim=0).mean():.3f}   "
          f"LULC pixels where the samples disagree: {(lulc_disagreement(samples[name]['LULC']) > 0).float().mean():.1%}")

# %% [markdown]
# * Where the samples differ, the input does not determine the output: this is the uncertainty of the generated data.
# * Under clouds the optical input carries less information. Compare the spread with and without clouds.
#   The model does not report this by itself; we have to sample several times and measure it.
# * Generated data is a guess, not a measurement. Before using it (e.g. to fill gaps under clouds),
#   check it against real data, and keep its uncertainty in mind.
#
# ## Exercises
#
# 1. Use another example tile (`TILE`). Where does the generated DEM differ most from the real one?
# 2. Change the order of `OUTPUTS` (e.g. LULC first). Does the chained generation change the results?
# 3. Increase `TIMESTEPS` (on a GPU: 50). What changes in the generated images?
# 4. Generate Sentinel-1 several times for a SeaLake and a Forest image. Where do the samples differ more?
