# Day 2: Uncertainty Quantification and Geo Foundation Models for Remote Sensing

Hands-on notebooks for two sessions: estimating the **uncertainty** of deep learning models in remote sensing, and
using pre-trained **geo foundation models** and their embeddings. All examples use Sentinel-2 data and run on a
laptop or in Google Colab.

**Contents**

1. [Setup](#1-setup): local installation with or without a GPU, or Google Colab
2. [Part 1: Uncertainty quantification](#2-part-1-uncertainty-quantification) (notebooks 01-03)
3. [Part 2: Geo foundation models](#3-part-2-geo-foundation-models) (notebooks 04-07)
4. [Data](#4-data)
5. [Sources and further reading](#5-sources-and-further-reading)

## 1. Setup

**Google Colab:** click the Colab badge of a notebook in the tables below, then *Runtime > Run all*.
The first cell installs the missing packages. A GPU can be chosen under *Runtime > Change runtime type*.

**Local installation.** We recommend [uv](https://docs.astral.sh/uv/), a fast Python package manager that also
installs the right Python version (3.12). The dependencies are listed in `pyproject.toml`.

1. **Install uv** (once):

   ```bash
   # macOS / Linux
   curl -LsSf https://astral.sh/uv/install.sh | sh

   # Windows (PowerShell)
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```

2. **Install the course environment** inside this folder. Choose one:

   ```bash
   uv sync --extra cpu      # without GPU: CPU-only PyTorch, much smaller download
   uv sync --extra cuda     # with an NVIDIA GPU (needs a working NVIDIA driver)
   uv sync                  # default PyTorch build of your system
   ```

   Without a GPU, use `--extra cpu`: the CPU-only PyTorch is a download of roughly 200 MB, while the CUDA build
   brings the NVIDIA CUDA libraries along and needs several GB. On Linux, plain `uv sync` installs the CUDA build;
   on Windows and macOS it is CPU-only anyway.

3. **Start Jupyter:**

   ```bash
   uv run --no-sync jupyter lab      # after --extra cpu / --extra cuda
   uv run jupyter lab                # after plain `uv sync`
   ```

   Use `--no-sync` after `--extra cpu` / `--extra cuda`: a plain `uv run` would switch back to the default PyTorch
   and download it again. `uv run --no-sync python scripts/check_gpu.py` shows whether PyTorch sees a GPU.

Everything runs on a laptop CPU. A GPU (your own or the T4 in Colab) makes training and the foundation models
much faster.

## 2. Part 1: Uncertainty quantification

| Notebook | Content | Based on | Colab |
|---|---|---|---|
| [01_uq_basics_regression](01_uq_basics_regression.ipynb) | UQ methods on a toy 1D regression problem; aleatoric and epistemic uncertainty | HAICON 2026 UQ workshop: [`01_regression_uq`](https://github.com/ChrisKo94/HAICON_2026_Trustworthy_UQ_Workshop/blob/main/notebooks/01_regression_uq.ipynb) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DLR-MF-DAS/hackathEOn/blob/main/Day2_UQ_and_GeoFMs/01_uq_basics_regression.ipynb) |
| [02_eurosat_uq](02_eurosat_uq.ipynb) | Calibration and UQ methods for EuroSAT land cover classification; synthetic clouds | HAICON 2026 UQ workshop: [`03_eo_eurosat_uq`](https://github.com/ChrisKo94/HAICON_2026_Trustworthy_UQ_Workshop/blob/main/notebooks/03_eo_eurosat_uq.ipynb); [SatelliteCloudGenerator](https://github.com/strath-ai/SatelliteCloudGenerator) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DLR-MF-DAS/hackathEOn/blob/main/Day2_UQ_and_GeoFMs/02_eurosat_uq.ipynb) |
| [02b_mnist_classification_uq](02b_mnist_classification_uq.ipynb) (optional) | The same for MNIST, with rotated digits | HAICON 2026 UQ workshop: [`02_mnist_classification_uq`](https://github.com/ChrisKo94/HAICON_2026_Trustworthy_UQ_Workshop/blob/main/notebooks/02_mnist_classification_uq.ipynb) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DLR-MF-DAS/hackathEOn/blob/main/Day2_UQ_and_GeoFMs/02b_mnist_classification_uq.ipynb) |
| [03_lightning_uq_box](03_lightning_uq_box.ipynb) | UQ methods from the [Lightning UQ Box](https://github.com/lightning-uq-box/lightning-uq-box) library, on toy data and EuroSAT | Lightning UQ Box tutorials: [regression](https://github.com/lightning-uq-box/lightning-uq-box/tree/main/docs/tutorials/regression), [classification](https://github.com/lightning-uq-box/lightning-uq-box/tree/main/docs/tutorials/classification), [`earth_observation/raps`](https://github.com/lightning-uq-box/lightning-uq-box/blob/main/docs/tutorials/earth_observation/raps.ipynb) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DLR-MF-DAS/hackathEOn/blob/main/Day2_UQ_and_GeoFMs/03_lightning_uq_box.ipynb) |

## 3. Part 2: Geo foundation models

| Notebook | Content | Based on | Colab |
|---|---|---|---|
| [04_terramind_embeddings](04_terramind_embeddings.ipynb) | Embeddings of EuroSAT images with [TerraMind](https://huggingface.co/ibm-esa-geospatial/TerraMind-1.0-tiny) | TerraTorch: [`embedding_generation_manual_backbone_registry`](https://github.com/IBM/terratorch/blob/main/examples/embeddings/embedding_generation_manual_backbone_registry.ipynb) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DLR-MF-DAS/hackathEOn/blob/main/Day2_UQ_and_GeoFMs/04_terramind_embeddings.ipynb) |
| [04b_terramind_generation](04b_terramind_generation.ipynb) (optional) | Generating other modalities (Sentinel-1, DEM, land cover, NDVI) from Sentinel-2 with TerraMind | TerraMind: [`terramind_generation`](https://github.com/IBM/terramind/blob/main/notebooks/terramind_generation.ipynb), [`terramind_any_to_any_generation`](https://github.com/IBM/terramind/blob/main/notebooks/terramind_any_to_any_generation.ipynb) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DLR-MF-DAS/hackathEOn/blob/main/Day2_UQ_and_GeoFMs/04b_terramind_generation.ipynb) |
| [05_few_shot_linear_probe](05_few_shot_linear_probe.ipynb) | Classification with few labels on the embeddings | TerraTorch: [`classification_eurosat`](https://github.com/IBM/terratorch/blob/main/examples/classification/classification_eurosat.ipynb) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DLR-MF-DAS/hackathEOn/blob/main/Day2_UQ_and_GeoFMs/05_few_shot_linear_probe.ipynb) |
| [06_uq_on_embeddings](06_uq_on_embeddings.ipynb) | Uncertainty of classifiers on the embeddings | new | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DLR-MF-DAS/hackathEOn/blob/main/Day2_UQ_and_GeoFMs/06_uq_on_embeddings.ipynb) |
| [07_alphaearth_embeddings](07_alphaearth_embeddings.ipynb) | Pre-computed [AlphaEarth Foundations](https://arxiv.org/abs/2507.22291) embeddings of a small region (needs internet) | Earth Engine [Satellite Embedding tutorials](https://developers.google.com/earth-engine/tutorials/community/satellite-embedding-01-introduction); [aef-loader](https://github.com/jakenotjay/aef-loader) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DLR-MF-DAS/hackathEOn/blob/main/Day2_UQ_and_GeoFMs/07_alphaearth_embeddings.ipynb) |

Notebooks 05 and 06 use the embeddings written by notebook 04 (`outputs/04_terramind_embeddings/`); if they are
missing, they compute them first.

All sources are openly licensed (MIT, Apache-2.0, CC-BY 4.0). Each notebook names them at the top ("Based on") and
in the cells taken over.

## 4. Data

The notebooks use a small, balanced subset of [EuroSAT](https://github.com/phelber/EuroSAT): Sentinel-2 patches of
64 × 64 pixels, 10 land cover classes. Choose the bands at the top of notebooks 02, 03 and 04 (05 and 06 follow
notebook 04; 04b always uses 13 bands):

```python
DATA_VARIANT = "ms"    # default: all 13 Sentinel-2 bands   |   "rgb": red, green, blue only
```

| `DATA_VARIANT` | Download | File | Images (train / val / test) |
|---|---|---|---|
| `"ms"` (default) | ~2 GB (full EuroSAT MS), or ~250 MB if a prepared file is linked | `data/eurosat_ms.npz` | 3000 (1500 / 500 / 1000) |
| `"rgb"` | ~90 MB | `data/eurosat_rgb.npz` | 5000 (3000 / 1000 / 1000) |

The data is downloaded automatically the first time a notebook needs it (locally and in Colab).
To download it beforehand, run in this folder:

```bash
uv run python scripts/download_data.py                  # 13 bands (default)
uv run python scripts/download_data.py --variant rgb    # RGB only
```

Notebook 01 needs no data, 02b downloads MNIST (~12 MB) and 07 reads the AlphaEarth embeddings of a small region
over the internet.

## 5. Sources and further reading

The examples are not built from scratch: they combine and adapt openly available tutorials, libraries, data sets
and papers. All of them are good starting points to go further.

### Tutorials and notebooks the examples build on

| Resource | What we took from it | Notebooks |
|---|---|---|
| [HAICON 2026 workshop "Trustworthy AI: Uncertainty Quantification"](https://github.com/ChrisKo94/HAICON_2026_Trustworthy_UQ_Workshop) | regression and classification UQ examples, EuroSAT UQ, MNIST with rotated digits | 01, 02, 02b |
| [Lightning UQ Box tutorials](https://github.com/lightning-uq-box/lightning-uq-box/tree/main/docs/tutorials) | Bayes by Backprop, Deep Ensembles, Laplace, SWAG, conformal prediction (RAPS) | 03 |
| [TerraTorch examples](https://github.com/IBM/terratorch/tree/main/examples) | building TerraMind, embedding extraction, EuroSAT classification and fine-tuning | 04, 05 |
| [TerraMind notebooks](https://github.com/IBM/terramind/tree/main/notebooks) | cross-modal generation, example tiles, plotting functions | 04b |
| [Earth Engine Satellite Embedding tutorials](https://developers.google.com/earth-engine/tutorials/community/satellite-embedding-01-introduction) | clustering, similarity search, classification with AlphaEarth embeddings | 07 |
| [AlphaEarth data on GCS / Source Cooperative](https://developers.google.com/earth-engine/guides/aef_on_gcs_readme) and [aef-loader](https://github.com/jakenotjay/aef-loader) | file layout, index file, reading small windows without Earth Engine | 07 |
| [SatelliteCloudGenerator](https://github.com/strath-ai/SatelliteCloudGenerator) | realistic synthetic clouds | 02, 03, 04 |

### Libraries

[PyTorch](https://pytorch.org) ·
[Lightning](https://lightning.ai) ·
[Lightning UQ Box](https://github.com/lightning-uq-box/lightning-uq-box) ·
[TerraTorch](https://github.com/IBM/terratorch) ·
[TorchGeo](https://github.com/microsoft/torchgeo) ·
[scikit-learn](https://scikit-learn.org) ·
[rasterio](https://rasterio.readthedocs.io)

### Data

* [EuroSAT](https://github.com/phelber/EuroSAT) (Helber et al., 2019), Sentinel-2 image patches, MIT license.
* [MNIST](https://yann.lecun.com/exdb/mnist/) (LeCun et al., 1998), via torchvision.
* [AlphaEarth Foundations Satellite Embedding dataset](https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_SATELLITE_EMBEDDING_V1_ANNUAL), CC-BY 4.0:
  "The AlphaEarth Foundations Satellite Embedding dataset is produced by Google and Google DeepMind."
* TerraMind example tiles from the [TerraMind repository](https://github.com/IBM/terramind) (Apache-2.0).

### Papers

**Uncertainty quantification**

* Gawlikowski et al. (2023): A survey of uncertainty in deep neural networks. *Artificial Intelligence Review*.
* Gal & Ghahramani (2016): Dropout as a Bayesian approximation. *ICML*.
* Lakshminarayanan et al. (2017): Simple and scalable predictive uncertainty estimation using deep ensembles. *NeurIPS*.
* Blundell et al. (2015): Weight uncertainty in neural networks. *ICML*.
* Maddox et al. (2019): A simple baseline for Bayesian uncertainty in deep learning (SWAG). *NeurIPS*.
* Daxberger et al. (2021): Laplace Redux: effortless Bayesian deep learning. *NeurIPS*.
* Guo et al. (2017): On calibration of modern neural networks. *ICML*.
* Liu et al. (2020): Simple and principled uncertainty estimation with deterministic deep learning via distance awareness (SNGP). *NeurIPS*.
* Mukhoti et al. (2023): Deep deterministic uncertainty: a new simple baseline (DDU). *CVPR*.
* Angelopoulos & Bates (2021): A gentle introduction to conformal prediction and distribution-free uncertainty quantification. arXiv:2107.07511.
* Angelopoulos et al. (2021): Uncertainty sets for image classifiers using conformal prediction (RAPS). *ICLR*.
* Lehmann et al. (2025): Lightning UQ Box: Uncertainty Quantification for Neural Networks. *JMLR*.
* Czerkawski et al. (2023): SatelliteCloudGenerator: controllable cloud and shadow synthesis for multi-spectral optical satellite images. *Remote Sensing*.

**Geo foundation models**

* Jakubik et al. (2025): TerraMind: Large-Scale Generative Multimodality for Earth Observation. *ICCV*, arXiv:2504.11171.
* Brown et al. (2025): AlphaEarth Foundations: an embedding field model for accurate and efficient global mapping from sparse label data. arXiv:2507.22291.
* Helber et al. (2019): EuroSAT: A novel dataset and deep learning benchmark for land use and land cover classification. *IEEE JSTARS*.
