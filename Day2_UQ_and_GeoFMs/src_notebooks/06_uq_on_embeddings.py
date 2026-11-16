# %% [markdown]
# # Notebook 6: Uncertainty on top of a foundation model
#
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DLR-MF-DAS/hackathEOn/blob/main/Day2_UQ_and_GeoFMs/06_uq_on_embeddings.ipynb)
#
# **Session 2, part 3**
#
# This notebook connects the two sessions: how uncertain is a classifier on top of foundation model embeddings,
# and how do clouds change the embeddings?
#
# Because the embeddings are small (192 numbers per image), UQ methods that are expensive
# for a full CNN become cheap here. Training an ensemble of 10 heads takes seconds.
#
# We compare:
#
# 1. a single small network (MLP head) on the embeddings
# 2. a **Deep Ensemble** of MLP heads
# 3. the **distance to the training data in embedding space** as an uncertainty score
#
# Uses the embeddings from notebook 4 (`outputs/04_terramind_embeddings/`); if they are missing, they are computed first.
#
# **Based on:** the embeddings of notebook 4, the clouds of notebook 2
# (SatelliteCloudGenerator, HAICON 2026 settings) and the methods of Session 1.
# %%
# Google Colab only (skipped on your own computer): a Colab link opens only this notebook, so we fetch
# the rest of the course folder (course_utils.py, scripts) from GitHub.
import os, sys
REPO = "https://github.com/DLR-MF-DAS/hackathEOn"  # repository that contains the course folder
FOLDER = "Day2_UQ_and_GeoFMs"                        # path of the course folder in the repository
if "google.colab" in sys.modules and not os.path.exists("course_utils.py"):
    shell = get_ipython().system  # like "!" in a notebook cell: the output shows up here
    shell(f"git clone -q --depth 1 --filter=blob:none --sparse {REPO} /content/repo")
    shell(f"git -C /content/repo sparse-checkout set {FOLDER}")  # only the course folder
    os.chdir(f"/content/repo/{FOLDER}")
    if not os.path.exists("outputs/04_terramind_embeddings"):  # needed to compute the embeddings of notebook 4 here
        shell("pip install -q terratorch satellite_cloud_generator torchgeo")

# %%
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from course_utils import EMBEDDINGS_FILE, load_embeddings, set_seed, progress, load_npz, decompose_uncertainty, accuracy, nll, ece, reliability_diagram, auroc

set_seed(0)

# %% [markdown]
# ## 1. Load the embeddings

# %%
d = load_embeddings()  # written by notebook 4 (computed here if missing, e.g. in Colab)

# standardise with the statistics of the training embeddings
mu, sd = d["emb_train"].mean(0), d["emb_train"].std(0) + 1e-6
E = {k: torch.from_numpy((d[k] - mu) / sd).float() for k in d if k.startswith("emb_")}
print("embeddings computed from data variant:", d["variant"])
y_train, y_val, y_test = (torch.from_numpy(d[f"y_{s}"]) for s in ["train", "val", "test"])

test_sets = {"clean": E["emb_test"], "30% clouds": E["emb_test_cloud30"], "60% clouds": E["emb_test_cloud60"]}
print({k: tuple(v.shape) for k, v in E.items()})

# %% [markdown]
# ## 2. One MLP head

# %%
def make_head(p_drop=0.2):
    return nn.Sequential(nn.Linear(E["emb_train"].shape[1], 128), nn.ReLU(), nn.Dropout(p_drop), nn.Linear(128, 10))


def train_head(seed, epochs=300):
    torch.manual_seed(seed)
    head = make_head()
    opt = torch.optim.Adam(head.parameters(), lr=1e-3, weight_decay=1e-4)
    for _ in range(epochs):  # full batch, the data is tiny
        loss = F.cross_entropy(head(E["emb_train"]), y_train)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return head.eval()


head = train_head(seed=0)
with torch.no_grad():
    p_single = {name: head(e).softmax(-1) for name, e in test_sets.items()}

for name, p in p_single.items():
    print(f"{name:12s} accuracy {accuracy(p, y_test):.1%}   ECE {ece(p, y_test):.3f}   "
          f"mean confidence {p.max(-1).values.mean():.2f}")

# %% [markdown]
# ## 3. Deep Ensemble of 10 heads

# %%
heads = [train_head(seed) for seed in progress(range(10), desc="training 10 heads", unit="head")]

with torch.no_grad():
    s_ens = {name: torch.stack([h(e).softmax(-1) for h in heads]) for name, e in test_sets.items()}  # (M, N, K)

for name, s in s_ens.items():
    p = s.mean(0)
    _, _, epi = decompose_uncertainty(s)
    print(f"{name:12s} accuracy {accuracy(p, y_test):.1%}   ECE {ece(p, y_test):.3f}   "
          f"NLL {nll(p, y_test):.3f}   epistemic {epi.mean():.3f}")

# %%
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
reliability_diagram(p_single["clean"], y_test, ax=axes[0], title="single head, clean")
reliability_diagram(s_ens["clean"].mean(0), y_test, ax=axes[1], title="ensemble, clean")
reliability_diagram(s_ens["60% clouds"].mean(0), y_test, ax=axes[2], title="ensemble, 60% clouds")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 4. Distance in embedding space
#
# A different idea: if a test embedding is **far away from all training embeddings**,
# the model has not seen anything similar. We use the mean cosine distance to the 10 nearest
# training embeddings as a score. This needs no extra training at all.

# %%
def knn_distance(e, k=10):
    a = F.normalize(e, dim=1)
    b = F.normalize(E["emb_train"], dim=1)
    dist = 1 - a @ b.T  # cosine distance to every training embedding
    return dist.topk(k, largest=False).values.mean(1)


plt.figure(figsize=(7, 3.5))
for name, e in test_sets.items():
    plt.hist(knn_distance(e).numpy(), bins=40, alpha=0.5, label=name)
plt.xlabel("distance to nearest training embeddings")
plt.legend()
plt.show()

# %% [markdown]
# ## 5. Which score separates clean from cloudy images?
#
# Clean test images vs. images with 60% clouds. AUROC: 0.5 = random, 1.0 = perfect.

# %%
clean, cloudy = "clean", "60% clouds"
scores = {
    "1 - max softmax (single head)": (1 - p_single[clean].max(-1).values, 1 - p_single[cloudy].max(-1).values),
    "ensemble total entropy": (decompose_uncertainty(s_ens[clean])[0], decompose_uncertainty(s_ens[cloudy])[0]),
    "ensemble epistemic (MI)": (decompose_uncertainty(s_ens[clean])[2], decompose_uncertainty(s_ens[cloudy])[2]),
    "kNN distance in embedding space": (knn_distance(test_sets[clean]), knn_distance(test_sets[cloudy])),
}
for name, (a, b) in scores.items():
    print(f"{name:33s} AUROC = {auroc(a, b):.3f}")

# %% [markdown]
# ## Points to discuss
#
# * Compare the calibration (ECE) of the heads with the CNNs of notebook 2.
# * With frozen embeddings, an ensemble only means training small heads.
# * The distance in embedding space needs no training. Compare its AUROC with the ensemble.
#
# ## Exercises
#
# 1. Fit a temperature for the ensemble on the validation embeddings (`E["emb_val"]`). How does the ECE change?
# 2. Train the heads on only 10 labels per class (see notebook 5). How do accuracy and epistemic uncertainty change?
# 3. Use the kNN distance to **reject** the 20% most uncertain test images. What is the accuracy on the rest?
