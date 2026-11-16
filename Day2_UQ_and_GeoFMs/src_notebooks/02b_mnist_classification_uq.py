# %% [markdown]
# # Notebook 2b (optional): Uncertainty quantification for MNIST classification
#
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DLR-MF-DAS/hackathEOn/blob/main/Day2_UQ_and_GeoFMs/02b_mnist_classification_uq.ipynb)
#
# **Optional.** A classic, fully controlled example before (or after) EuroSAT in notebook 2.
# This notebook follows the MNIST notebook of the HAICON 2026 workshop
# ["Uncertainty Quantification for Neural Networks"](https://github.com/ChrisKo94/HAICON_2026_Trustworthy_UQ_Workshop)
# (J. Gawlikowski, C. Schweden, DLR).
#
# **Learning goals**
# - Understand **aleatoric** and **epistemic** uncertainty in classification
# - Evaluate the calibration of a classifier
# - Compare methods when the test data drifts away from the training data
#
# | Method | Aleatoric | Epistemic |
# |--------|:---------:|:---------:|
# | Plain softmax CNN | yes | no |
# | MC Dropout | yes | yes |
# | Deep Ensemble | yes | yes |
#
# **Part 1** (sections 1–5): standard MNIST. **Part 2** (sections 6–8): rotated digits as out-of-distribution data.
#
# The MNIST data (about 12 MB) is downloaded automatically into `data/` on the first run.
#
# **Based on** (existing notebooks this one builds on):
# - HAICON 2026 workshop, [`02_mnist_classification_uq.ipynb`](https://github.com/ChrisKo94/HAICON_2026_Trustworthy_UQ_Workshop/blob/main/notebooks/02_mnist_classification_uq.ipynb) (MIT license):
#   taken over almost 1:1 (network, reliability diagram, MC Dropout, ensemble, rotation sweep).
#   Changes: fixed seeds, 5 instead of 15 epochs, progress bars, corrected text.
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

# %%
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as T
import torchvision.transforms.functional as TF

from course_utils import set_seed, progress

SEED = 42
set_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BLUE, ORANGE, GREEN, RED, PURPLE, GRAY, TEAL = "#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2", "#BBBBBB", "#00A9A5"

EPOCHS = 5                                       # the HAICON notebook uses 15; 5 is enough for MNIST on a CPU

# %% [markdown]
# # Part 1: standard MNIST
#
# ## 1. Dataset
#
# *From the HAICON notebook `02_mnist_classification_uq.ipynb`, section 1.*
#
# MNIST contains 70,000 greyscale images (28 × 28 pixels) of handwritten digits 0–9.
# To keep it fast we use a random subset of **8,000 training** and **2,000 test** images.
# The images are normalised with the standard MNIST statistics (mean 0.1307, std 0.3081).

# %%
transform = T.Compose([T.ToTensor(), T.Normalize((0.1307,), (0.3081,))])
train_full = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_full = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=transform)

N_TRAIN, N_TEST = 8_000, 2_000
rng = np.random.default_rng(SEED)
train_set = Subset(train_full, rng.choice(len(train_full), N_TRAIN, replace=False))
test_set = Subset(test_full, rng.choice(len(test_full), N_TEST, replace=False))

train_loader = DataLoader(train_set, batch_size=128, shuffle=True, generator=torch.Generator().manual_seed(SEED))
test_loader = DataLoader(test_set, batch_size=256, shuffle=False)

# all test images as one tensor, used throughout the notebook
X_test = torch.cat([xb for xb, _ in test_loader])  # (N, 1, 28, 28)
y_test = torch.cat([yb for _, yb in test_loader])
y_np = y_test.numpy()
print(f"train: {N_TRAIN:,}   test: {N_TEST:,}")

# %%
fig, axes = plt.subplots(10, 6, figsize=(7, 12))
for cls in range(10):
    idxs = (y_test == cls).nonzero(as_tuple=True)[0][:6]
    for col, idx in enumerate(idxs):
        axes[cls, col].imshow(X_test[idx].squeeze(), cmap="gray")
        axes[cls, col].axis("off")
plt.suptitle("MNIST test samples, 6 per class (rows: digits 0 to 9)", fontsize=12, fontweight="bold", y=1.0)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 2. Network and training
#
# *From the HAICON notebook `02_mnist_classification_uq.ipynb`, section 2.*
#
# All methods use the **same small CNN**, so differences in the uncertainty come from the method,
# not from the architecture:
#
# ```
# Conv(1→16, 3×3) → ReLU
# Conv(16→32, 3×3) → ReLU → MaxPool(2×2)     # output: 32 × 14 × 14
# Flatten → Linear(6272→128) → ReLU
# Dropout(p)                                  # only active for MC Dropout
# Linear(128→10)                              # logits
# ```
#
# - **Baseline**: dropout 0.0
# - **MC Dropout**: dropout 0.2
# - **Deep Ensemble**: 5 networks with different seeds, dropout 0.0

# %%
class MnistCNN(nn.Module):
    def __init__(self, dropout=0.0):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),                # 28×28 → 14×14
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 14 * 14, 128), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


print(f"parameters: {sum(p.numel() for p in MnistCNN().parameters()):,}")


@torch.no_grad()
def get_probs(model, X, bs=256):
    """Softmax probabilities (N, 10), model in eval mode."""
    model.eval()
    return np.concatenate([F.softmax(model(X[i:i + bs].to(device)), dim=-1).cpu().numpy()
                           for i in range(0, len(X), bs)])


def entropy(p):
    """Predictive entropy of probabilities (..., C)."""
    return -(p * np.log(p + 1e-9)).sum(-1)


def train_model(name, dropout=0.0, seed=SEED, n_epochs=EPOCHS, lr=0.005, verbose=True):
    """Train a MnistCNN."""
    set_seed(seed)
    model = MnistCNN(dropout=dropout).to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    loader = DataLoader(train_set, batch_size=128, shuffle=True, generator=torch.Generator().manual_seed(seed))
    bar = progress(total=n_epochs * len(loader), desc=f"training {name}", unit="batch")
    for ep in range(n_epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = F.cross_entropy(model(xb), yb)
            loss.backward()
            opt.step()
            bar.update()
            bar.set_postfix_str(f"epoch {ep + 1}/{n_epochs}  loss {loss.item():.3f}", refresh=False)
    bar.close()
    if verbose:
        print(f"{name}: test accuracy {(get_probs(model, X_test).argmax(-1) == y_np).mean():.1%}")
    return model


clf = train_model("baseline", dropout=0.0)

# %% [markdown]
# ## 3. The overconfidence problem
#
# *From the HAICON notebook `02_mnist_classification_uq.ipynb`, section 3.*
#
# Accuracy tells us *how often* the model is right, but not how confident it is when it is wrong.
# A well-calibrated classifier satisfies
#
# $$P(\hat{y} = y \mid \max_k p_k = c) = c :$$
#
# **when the model says "90% confident", it should be correct about 90% of the time.**
#
# 1. **Reliability diagram**: accuracy vs. predicted confidence in confidence bins. Perfect calibration lies on the diagonal.
# 2. **Expected calibration error (ECE)**: the weighted average gap between confidence and accuracy over the bins:
#
# $$\mathrm{ECE} = \sum_{b=1}^{B} \frac{|\mathcal{B}_b|}{N} \left|\mathrm{acc}(\mathcal{B}_b) - \overline{\mathrm{conf}}(\mathcal{B}_b)\right|$$
#
# **Careful:** when the accuracy is very high (almost all samples correct), calibration plots can be misleading:
# most bins contain only a handful of images.

# %%
num_bins = 20


def reliability_diagram(ax, probs, labels, title="", n_bins=num_bins, color=BLUE):
    """Plots a reliability diagram and returns the ECE."""
    conf = probs.max(-1)
    correct = (probs.argmax(-1) == labels).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    bin_c, acc_b, cnt_b = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf >= lo) & (conf < hi)
        if mask.sum() > 5:
            bin_c.append((lo + hi) / 2)
            acc_b.append(correct[mask].mean())
            cnt_b.append(mask.sum())
    bin_c, acc_b, cnt_b = map(np.array, [bin_c, acc_b, cnt_b])
    ece = (cnt_b * np.abs(acc_b - bin_c)).sum() / cnt_b.sum()

    ax.bar(bin_c, acc_b, width=1.0 / n_bins, alpha=0.8, color=color, edgecolor="white", label="model")
    ax.bar(bin_c, np.maximum(bin_c - acc_b, 0), width=1.0 / n_bins, bottom=acc_b, alpha=0.35, color=RED,
           label="overconfidence gap")
    ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="perfect calibration")
    ax.text(0.05, 0.87, f"ECE = {ece:.4f}", transform=ax.transAxes, fontsize=11, fontweight="bold")
    ax.set(xlabel="predicted confidence", ylabel="fraction correct", title=title, xlim=(0, 1), ylim=(0, 1.05))
    ax.legend(fontsize=8)
    return ece


def confidence_histogram(ax, probs, labels):
    """Confidence of correct and wrong predictions."""
    conf = probs.max(-1)
    correct = probs.argmax(-1) == labels
    bins = np.linspace(0, 1, num_bins + 1)
    ax.hist(conf[correct], bins=bins, color=GREEN, alpha=0.65, label=f"correct ({correct.sum()})", density=True)
    ax.hist(conf[~correct], bins=bins, color=RED, alpha=0.65, label=f"wrong ({(~correct).sum()})", density=True)
    ax.axvline(0.9, color="black", ls="--", lw=1)
    ax.set(xlabel="max softmax confidence", ylabel="density", title="Confidence: correct vs. wrong predictions")
    ax.legend(fontsize=9)
    ax.text(0.45, 0.6, f"{(conf[~correct] > 0.9).mean():.0%} of the errors have conf > 0.9\n"
                       f"{(conf[correct] < 0.9).mean():.0%} of the correct have conf < 0.9",
            transform=ax.transAxes, color=RED, fontsize=10, ha="center")


probs_base = get_probs(clf, X_test)
correct_mask = probs_base.argmax(-1) == y_np
print(f"test accuracy:       {correct_mask.mean():.1%}")
print(f"mean max-confidence: {probs_base.max(-1).mean():.3f}")
print(f"calibration gap:     {probs_base.max(-1).mean() - correct_mask.mean():+.3f}  (positive = overconfident)")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
reliability_diagram(axes[0], probs_base, y_np, title="Baseline softmax: reliability diagram")
confidence_histogram(axes[1], probs_base, y_np)
plt.tight_layout()
plt.show()

# %% [markdown]
# The most dangerous errors: wrong, but very sure.

# %%
preds_base = probs_base.argmax(-1)
wrong_hc = np.where((preds_base != y_np) & (probs_base.max(-1) > 0.85))[0]
n_show = min(10, len(wrong_hc))
print(f"high-confidence errors (conf > 85%): {len(wrong_hc)}, showing {n_show}")

if n_show > 0:
    fig, axes = plt.subplots(2, n_show, figsize=(n_show * 1.7, 4.5), squeeze=False)
    for col, i in enumerate(wrong_hc[:n_show]):
        axes[0, col].imshow(X_test[i].squeeze(), cmap="gray")
        axes[0, col].axis("off")
        axes[0, col].set_title(f"true: {y_np[i]}\npred: {preds_base[i]}", fontsize=8, color=RED)
        axes[1, col].bar(range(10), probs_base[i],
                         color=[GREEN if c == y_np[i] else RED if c == preds_base[i] else GRAY for c in range(10)])
        axes[1, col].set_xticks(range(10))
        axes[1, col].set_xticklabels(range(10), fontsize=6)
        axes[1, col].set_ylim(0, 1)
        axes[1, col].set_title(f"conf={probs_base[i].max():.2f}", fontsize=8)
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ## 4. MC Dropout
#
# *From the HAICON notebook `02_mnist_classification_uq.ipynb`, section 4.*
#
# Keeping dropout active at test time turns every forward pass into a sample from an approximate
# posterior over the weights. With $T$ passes we average the probabilities,
#
# $$\bar{p} = \frac{1}{T}\sum_{t=1}^T p^{(t)}(y|x),$$
#
# and split the uncertainty:
#
# | Quantity | Formula | Meaning |
# |----------|---------|---------|
# | **Total** (predictive entropy) | $H[\bar p]$ | the model is unsure about the answer |
# | **Epistemic** (mutual information) | $I = H[\bar{p}] - \frac{1}{T}\sum_t H[p^{(t)}]$ | the passes disagree: reducible, e.g. unknown inputs |
# | **Aleatoric** | $H - I$ | each pass is unsure: ambiguous digits |
#
# MC Dropout is cheap: one model, $T$ forward passes at test time.

# %%
mc_clf = train_model("mc_dropout", dropout=0.2)


def mc_predict(model, X, T=50, bs=256, seed=SEED):
    """T forward passes with dropout active. Returns mean probabilities, entropy, mutual information."""
    model.train()  # keeps dropout on (this CNN has no BatchNorm)
    torch.manual_seed(seed)
    with torch.no_grad():
        P = np.stack([np.concatenate([F.softmax(model(X[i:i + bs].to(device)), dim=-1).cpu().numpy()
                                      for i in range(0, len(X), bs)]) for _ in range(T)])  # (T, N, 10)
    model.eval()
    mean_p = P.mean(0)
    pred_ent = entropy(mean_p)
    mi = pred_ent - entropy(P).mean(0)
    return mean_p, pred_ent, mi


MC_T = 50
p_mc, ent_mc, mi_mc = mc_predict(mc_clf, X_test, T=MC_T)
ok = p_mc.argmax(-1) == y_np
print(f"MC Dropout accuracy:           {ok.mean():.1%}")
print(f"mean entropy, correct preds:   {ent_mc[ok].mean():.4f}")
if (~ok).any():
    print(f"mean entropy, wrong preds:     {ent_mc[~ok].mean():.4f}")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
reliability_diagram(axes[0], p_mc, y_np, title="MC Dropout: reliability diagram")
confidence_histogram(axes[1], p_mc, y_np)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 5. Deep Ensemble
#
# *From the HAICON notebook `02_mnist_classification_uq.ipynb`, sections 5-6.*
#
# We train **M = 5 networks** from different random seeds. Their *disagreement* on an input measures the
# epistemic uncertainty, with the same mutual-information decomposition as for MC Dropout:
#
# $$\bar{p}^\mathrm{ens} = \frac{1}{M}\sum_{m=1}^M p_m(y|x)$$
#
# The members end up in different minima of the loss, so they can disagree on inputs unlike the training data.
# The cost: $M$ times the training and inference time.

# %%
N_ENS = 5
ens_models = [train_model(f"ensemble member {i + 1}/{N_ENS}", dropout=0.0, seed=SEED + i * 137, verbose=False)
              for i in range(N_ENS)]


def ens_predict(X, bs=256):
    P = np.stack([get_probs(m, X, bs) for m in ens_models])  # (M, N, 10)
    mean_p = P.mean(0)
    pred_ent = entropy(mean_p)
    return mean_p, pred_ent, pred_ent - entropy(P).mean(0)


p_ens, ent_ens, mi_ens = ens_predict(X_test)
print(f"ensemble accuracy: {(p_ens.argmax(-1) == y_np).mean():.1%}")

# %% [markdown]
# Calibration of all three methods on clean MNIST:

# %%
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
reliability_diagram(axes[0], probs_base, y_np, "Baseline", color=RED)
reliability_diagram(axes[1], p_mc, y_np, f"MC Dropout (T={MC_T})", color=PURPLE)
reliability_diagram(axes[2], p_ens, y_np, f"Ensemble (M={N_ENS})", color=GREEN)
plt.tight_layout()
plt.show()

# %% [markdown]
# # Part 2: rotated MNIST as out-of-distribution data
#
# What happens when the test data **drifts away** from the training data?
# We rotate the test digits. The models only saw upright digits, so increasing the rotation angle
# gives a controlled shift: from familiar (0°) to clearly out-of-distribution (90°–180°).
# This is the MNIST version of the cloud experiment on EuroSAT.
#
# ## 6. Rotating without artefacts
#
# *From the HAICON notebook `02_mnist_classification_uq.ipynb`, section 7.*
#
# After normalisation the MNIST background is about −0.42, not 0. `rotate` fills the new corner pixels with 0,
# which is brighter than the background and would create a bright border. Fix: shift the image so that the
# background is 0, rotate, and shift back.

# %%
ROTATION_ANGLES = [0, 15, 30, 45, 60, 90, 120, 150, 180]


def rotate_batch(x, angle):
    """Rotate images (N, C, H, W) by `angle` degrees, keeping the background value."""
    if angle == 0:
        return x
    x_min = x.flatten(1).min(dim=1).values.view(-1, 1, 1, 1)
    return TF.rotate(x - x_min, angle, fill=0) + x_min


show_a = [0, 30, 60, 90, 150, 180, 210, 270, 330]
fig, axes = plt.subplots(3, len(show_a), figsize=(len(show_a) * 1.6, 5.2))
for i, sample in enumerate(X_test[:3]):
    for col, angle in enumerate(show_a):
        axes[i, col].imshow(rotate_batch(sample[None], angle).squeeze(), cmap="gray")
        axes[i, col].axis("off")
        axes[i, col].set_title(f"{angle}°", fontsize=10)
plt.suptitle("Rotated MNIST examples", fontsize=11, fontweight="bold")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 7. The baseline under rotation
#
# *From the HAICON notebook `02_mnist_classification_uq.ipynb`, section 8.*
#
# For each angle we rotate the whole test set and track:
#
# - **accuracy**: does the model still predict correctly?
# - **mean max confidence**: does the model *think* it is still correct?
# - **predictive entropy**: does the uncertainty rise?
#
# The key diagnostic is the **overconfidence gap** (confidence − accuracy). A model that stays 90% confident
# while it is only 50% accurate is dangerously miscalibrated.

# %%
def rotation_sweep(predict_fn, X, y, angles=ROTATION_ANGLES):
    """predict_fn(X_rot) → (probs, entropy, mi). Returns arrays of angle, acc, conf, entropy, mi."""
    out = {k: [] for k in ["angle", "acc", "conf", "entropy", "mi"]}
    for angle in angles:
        probs, ent, mi = predict_fn(rotate_batch(X, angle))
        out["angle"].append(angle)
        out["acc"].append((probs.argmax(-1) == y).mean())
        out["conf"].append(probs.max(-1).mean())
        out["entropy"].append(ent.mean())
        out["mi"].append(mi.mean())
    return {k: np.array(v) for k, v in out.items()}


def baseline_predict(X):
    p = get_probs(clf, X)
    return p, entropy(p), np.zeros(len(p))


base_rot = rotation_sweep(baseline_predict, X_test, y_np)

# %%
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
ang = base_rot["angle"]
axes[0].plot(ang, base_rot["acc"] * 100, "o-", color=TEAL, lw=2.5, ms=8)
axes[0].axhline(10, color=GRAY, ls=":", lw=1.2, label="random (10%)")
axes[0].set(xlabel="rotation (°)", ylabel="accuracy (%)", title="A: test accuracy", ylim=(0, 105))
axes[0].legend(fontsize=9)

axes[1].plot(ang, base_rot["conf"], "o-", color=RED, lw=2.5, ms=8, label="mean confidence")
axes[1].plot(ang, base_rot["acc"], "o--", color=TEAL, lw=2.0, ms=6, label="accuracy (ideal)")
axes[1].fill_between(ang, base_rot["acc"], base_rot["conf"], where=base_rot["conf"] >= base_rot["acc"],
                     alpha=0.2, color=RED, label="overconfidence gap")
axes[1].set(xlabel="rotation (°)", ylabel="probability", title="B: confidence vs. accuracy", ylim=(0, 1.05))
axes[1].legend(fontsize=9)

axes[2].plot(ang, base_rot["entropy"], "o-", color=ORANGE, lw=2.5, ms=8)
axes[2].set(xlabel="rotation (°)", ylabel="mean entropy (nats)", title="C: does the entropy rise?")
for ax in axes:
    ax.axvspan(60, 180, alpha=0.08, color=RED)
    ax.set_xticks(ang)
plt.suptitle("Baseline softmax: the accuracy collapses, the confidence stays high", fontsize=12, fontweight="bold", y=1.02)
plt.tight_layout()
plt.show()

print("overconfidence gap (conf − acc):")
for a, c, ac in zip(ang, base_rot["conf"], base_rot["acc"]):
    print(f"  {a:>3}°  gap = {c - ac:+.3f}  " + "#" * int(abs(c - ac) * 45))

# %% [markdown]
# ## 8. All methods under rotation
#
# *From the HAICON notebook `02_mnist_classification_uq.ipynb`, section 9.*
#
# Now the same sweep for all three methods. Questions:
# - how similar are the accuracies at 0° (same CNN)?
# - how does the entropy of each method change with the rotation?
# - which quantity (entropy or mutual information) changes most with the shift?

# %%
mc_rot = rotation_sweep(lambda X: mc_predict(mc_clf, X, T=MC_T), X_test, y_np)
ens_rot = rotation_sweep(ens_predict, X_test, y_np)

method_styles = {
    "Baseline": (base_rot, RED, "o-"),
    f"MC Dropout (T={MC_T})": (mc_rot, PURPLE, "^-"),
    f"Ensemble (M={N_ENS})": (ens_rot, GREEN, "D-"),
}
panels = [
    ("acc", "accuracy", "Accuracy drops with rotation"),
    ("conf", "mean max confidence", "Does the confidence follow the accuracy?"),
    ("entropy", "mean entropy (nats)", "Total uncertainty"),
    ("mi", "mean mutual information", "Epistemic uncertainty"),
]
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
for ax, (key, ylabel, title) in zip(axes.flatten(), panels):
    for name, (res, col, sty) in method_styles.items():
        ax.plot(res["angle"], res[key], sty, color=col, lw=2.5, ms=7, alpha=0.9, label=name)
    if key == "conf":
        ax.plot(base_rot["angle"], base_rot["acc"], "k:", lw=1.5, label="accuracy (target)")
    ax.axvspan(60, 180, alpha=0.07, color=RED)
    ax.set(xlabel="rotation (°)", ylabel=ylabel, title=title)
    ax.set_xticks(ROTATION_ANGLES)
    ax.legend(fontsize=8)
plt.suptitle("All methods under rotation", fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()
plt.show()

# %% [markdown]
# Things to compare in the plots:
# * Accuracy vs. angle: UQ methods do not change what the classifier predicts, only how sure it is.
# * The baseline has no epistemic part. How does the mutual information of MC Dropout and the ensemble change?
# * Confidence vs. accuracy at strong rotations: compare with the clouds on EuroSAT (notebook 2).
#
# ## Exercises
#
# 1. Change the dropout rate of the MC Dropout model to 0.5. Does the epistemic signal under rotation change?
# 2. Use only 2 ensemble members. How much of the signal is left?
# 3. Instead of rotating, add Gaussian noise of increasing strength to the test images. How do the methods react?
