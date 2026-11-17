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
# **Part 2 · Geo Foundation Models · notebook 2 of 3**
#
# # Notebook 5: Few labels? Linear probing on foundation model embeddings
#
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DLR-MF-DAS/hackathEOn/blob/main/Day2_UQ_and_GeoFMs/05_few_shot_linear_probe.ipynb)
#
# Foundation models are often used when only few labels are available. Here we try this on EuroSAT
# with only $k$ labelled images per class ($k$ = 1, 5, 10, 20, 50).
#
# We compare:
#
# | Approach | Input | What is trained |
# |---|---|---|
# | Band statistics + logistic regression | mean and std of each band (26 numbers for 13 bands, 6 for RGB) | a linear classifier |
# | CNN from scratch | image (13 bands, or RGB) | the whole CNN |
# | **TerraMind + logistic regression** ("linear probe") | 192-d embedding from notebook 4 | a linear classifier |
# | **TerraMind + kNN** | 192-d embedding | nothing, just nearest neighbours |
#
# Uses the embeddings from notebook 4 (`outputs/04_terramind_embeddings/`); if they are missing, they are computed first.
#
# **Based on** (existing notebooks this one builds on):
# - Section 7 follows TerraTorch, [`classification/classification_eurosat.ipynb`](https://github.com/IBM/terratorch/blob/main/examples/classification/classification_eurosat.ipynb)
#   (Apache-2.0): the same `ClassificationTask` with TerraMind-tiny and the same model arguments, but on our EuroSAT
#   subset and with a frozen and a trainable encoder.
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
    shell("pip install -q terratorch satellite_cloud_generator torchgeo")  # clouds and torchgeo: if notebook 4's embeddings are computed here

# %%
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from course_utils import (
    CLASSES, EMBEDDINGS_FILE, set_seed, load_npz, load_embeddings, load_eurosat, scale_to_unit,
    SmallCNN, train_classifier, train_model, predict_logits, quiet_lightning, lightning_bar, progress,
)

set_seed(0)

# %% [markdown]
# ## 1. Load images and embeddings

# %%
emb = load_embeddings()  # written by notebook 4 (computed here if missing, e.g. in Colab)
variant = str(emb["variant"])  # the data variant notebook 4 used: "ms" (13 bands) or "rgb"
data = load_eurosat(variant)

y_train, y_test = data["y_train"].numpy(), data["y_test"].numpy()
assert (y_train == emb["y_train"]).all()

# simple hand-crafted features: mean and std of every band
def band_stats(x):
    return torch.cat([x.mean(dim=(2, 3)), x.std(dim=(2, 3))], dim=1).numpy()  # x: (N, C, 64, 64)


feat = {
    "band statistics": (band_stats(data["x_train"]), band_stats(data["x_test"])),
    "TerraMind": (emb["emb_train"], emb["emb_test"]),
}
print({k: v[0].shape for k, v in feat.items()})

# %% [markdown]
# ## 2. A helper that picks k labelled images per class

# %%
def pick_k_per_class(y, k, seed):
    rng = np.random.default_rng(seed)
    return np.concatenate([rng.choice(np.where(y == c)[0], k, replace=False) for c in range(10)])


def linear_probe(f_train, y_tr, f_test):
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
    return clf.fit(f_train, y_tr).predict(f_test)


def knn(f_train, y_tr, f_test, n_neighbors=5):
    # with only 1 label per class, more than 1 neighbour gives ties
    n_neighbors = min(n_neighbors, len(y_tr) // 10)
    clf = KNeighborsClassifier(n_neighbors=max(n_neighbors, 1), metric="cosine")
    return clf.fit(f_train, y_tr).predict(f_test)

# %% [markdown]
# ## 3. Linear probe and kNN for different numbers of labels
#
# We repeat every experiment with 3 random label selections and average.

# %%
ks = [1, 5, 10, 20, 50]
seeds = [0, 1, 2]
results = {}

for name, (f_tr, f_te) in feat.items():
    results[f"{name} + linear"] = [
        np.mean([(linear_probe(f_tr[idx], y_train[idx], f_te) == y_test).mean()
                 for idx in [pick_k_per_class(y_train, k, s) for s in seeds]])
        for k in ks
    ]

f_tr, f_te = feat["TerraMind"]
results["TerraMind + kNN"] = [
    np.mean([(knn(f_tr[idx], y_train[idx], f_te) == y_test).mean()
             for idx in [pick_k_per_class(y_train, k, s) for s in seeds]])
    for k in ks
]

for name, accs in results.items():
    print(f"{name:26s}", "  ".join(f"k={k}: {a:.1%}" for k, a in zip(ks, accs)))

# %% [markdown]
# ## 4. Baseline: a CNN trained from scratch on the same few labels
#
# This is what you would do without a foundation model. Set `RUN_CNN = False` to skip it.

# %%
RUN_CNN = True

if RUN_CNN:
    x_train = scale_to_unit(data["x_train"], variant)
    x_test = scale_to_unit(data["x_test"], variant)
    accs = []
    for k in progress(ks, desc="CNN from scratch, k labels per class"):
        idx = pick_k_per_class(y_train, k, seed=0)
        epochs = max(15, 3000 // (10 * k))  # more epochs when there are fewer images

        def make(c=x_train.shape[1]):
            set_seed(0)
            return SmallCNN(in_channels=c)

        cnn, _ = train_model(
            f"cnn_{variant}_k{k}", make,
            lambda m: train_classifier(m, x_train[idx], torch.from_numpy(y_train[idx]),
                                       epochs=epochs, batch_size=32, seed=0, verbose=False),
            verbose=False,
        )
        pred = predict_logits(cnn, x_test).argmax(-1).numpy()
        accs.append((pred == y_test).mean())
    print(f"{'CNN from scratch':26s}", "  ".join(f"k={k}: {a:.1%}" for k, a in zip(ks, accs)))
    results["CNN from scratch"] = accs

# %% [markdown]
# ## 5. The label efficiency curve

# %%
plt.figure(figsize=(7, 4.5))
for name, accs in results.items():
    style = "o-" if "TerraMind" in name else "s--"
    plt.plot(ks, accs, style, label=name)
plt.xscale("log")
plt.xticks(ks, [str(k) for k in ks])
plt.xlabel("labelled images per class")
plt.ylabel("test accuracy")
plt.ylim(0, 1)
plt.grid(alpha=0.3)
plt.legend()
plt.title("How many labels do we need?")
plt.show()

# %% [markdown]
# ## 6. Where does the linear probe make mistakes? (k = 10)

# %%
idx = pick_k_per_class(y_train, 10, seed=0)
pred = linear_probe(feat["TerraMind"][0][idx], y_train[idx], feat["TerraMind"][1])
cm = confusion_matrix(y_test, pred, normalize="true")
fig, ax = plt.subplots(figsize=(7, 6))
ConfusionMatrixDisplay(cm, display_labels=[c[:12] for c in CLASSES]).plot(
    ax=ax, xticks_rotation=60, values_format=".1f", colorbar=False)
plt.title("TerraMind linear probe, 10 labels per class")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 7. Optional: fine-tuning with TerraTorch (frozen vs. trainable encoder)
#
# So far the encoder was **frozen**: we computed the embeddings once and only trained a linear classifier.
# The other option is **fine-tuning**: the encoder weights are trained further on our labels.
# This follows the official [TerraTorch EuroSAT example](https://github.com/IBM/terratorch/tree/main/examples/classification):
# a `ClassificationTask` with TerraMind-tiny as backbone and a small classification head.
#
# | | frozen encoder | full fine-tuning |
# |---|---|---|
# | trained | head only | encoder + head |
# | cost | low (the encoder only runs forward) | high (gradients through the encoder) |
#
# Both runs use the same 50 labelled images per class. **This needs a GPU** (e.g. Colab T4);
# on a laptop CPU it takes long. Set `RUN_FINETUNE = True` to run it.

# %%
RUN_FINETUNE = False
FT_LABELS_PER_CLASS = 50
FT_EPOCHS = 5

if RUN_FINETUNE:
    import torch.nn.functional as F
    import lightning.pytorch as pl
    from torch.utils.data import DataLoader, Dataset
    from terratorch.tasks import ClassificationTask
    from course_utils import TERRAMIND_INPUT, terramind_stats

    modality, _ = TERRAMIND_INPUT[variant]  # "S2L1C" or "RGB", as in notebook 4
    mean, std = terramind_stats(variant)

    class TerraMindDataset(Dataset):
        """Standardise with the pre-training statistics and resize to 224 x 224, as in notebook 4."""

        def __init__(self, x, y):
            self.x, self.y = x, torch.as_tensor(y)

        def __len__(self):
            return len(self.x)

        def __getitem__(self, i):
            xb = (self.x[i:i + 1].float() - mean) / std
            xb = F.interpolate(xb, size=(224, 224), mode="bilinear", align_corners=False)[0]
            return {"image": xb, "label": self.y[i]}

    idx = pick_k_per_class(y_train, FT_LABELS_PER_CLASS, seed=0)
    train_loader = DataLoader(TerraMindDataset(data["x_train"][idx], y_train[idx]), batch_size=16, shuffle=True,
                              generator=torch.Generator().manual_seed(0))
    test_loader = DataLoader(TerraMindDataset(data["x_test"], y_test), batch_size=64)

    def finetune(freeze_backbone):
        pl.seed_everything(0, verbose=False)
        quiet_lightning()
        task = ClassificationTask(
            model_factory="EncoderDecoderFactory",
            model_args={
                "backbone": "terramind_v1_tiny",
                "backbone_pretrained": True,
                "backbone_modalities": [modality],
                "necks": [{"name": "ReshapeTokensToImage", "remove_cls_token": False}],
                "decoder": "IdentityDecoder",
                "head_dropout": 0.1,
                "head_dim_list": [128],
                "num_classes": 10,
            },
            loss="ce",
            optimizer="AdamW",
            lr=1e-4 if not freeze_backbone else 1e-3,
            freeze_backbone=freeze_backbone,
        )
        name = "frozen encoder" if freeze_backbone else "fine-tuning"
        pl.Trainer(max_epochs=FT_EPOCHS, accelerator="auto", logger=False, enable_checkpointing=False,
                   enable_progress_bar=False, enable_model_summary=False,
                   callbacks=[lightning_bar(name)]).fit(task, train_dataloaders=train_loader)
        task.eval()
        with torch.no_grad():
            pred = torch.cat([task(b["image"].to(task.device)).output.argmax(1).cpu() for b in test_loader])
        return (pred.numpy() == y_test).mean()

    acc_frozen = finetune(freeze_backbone=True)
    acc_full = finetune(freeze_backbone=False)
    k_i = ks.index(FT_LABELS_PER_CLASS) if FT_LABELS_PER_CLASS in ks else None
    print(f"{FT_LABELS_PER_CLASS} labels per class")
    if k_i is not None:
        print(f"  linear probe on embeddings (section 3): {results['TerraMind + linear'][k_i]:.1%}")
    print(f"  frozen encoder + trained head:          {acc_frozen:.1%}")
    print(f"  full fine-tuning:                       {acc_full:.1%}")

# %% [markdown]
# ## Notes
#
# * The linear probe keeps the foundation model **frozen**. Training takes seconds on a CPU.
# * The other option is **fine-tuning** the whole model or parts of it, e.g. with LoRA (section 7).
#   TerraTorch also runs the same from a config file:
#   `terratorch fit -c config.yaml` (see the [TerraMind examples](https://github.com/IBM/terramind)).
# * Compare against a simple baseline such as the band statistics; with 13 bands it uses more spectral
#   information than with RGB only.
#
# ## Exercises
#
# 1. Concatenate band statistics and TerraMind embeddings. Does this help?
# 2. Change the number of neighbours in `knn`. What changes for k = 1 label per class?
# 3. Compare the confusion matrix of the linear probe with the one of the band statistics.
