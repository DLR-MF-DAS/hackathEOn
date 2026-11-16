# %% [markdown]
# # Notebook 2: Uncertainty for land cover classification (EuroSAT)
#
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DLR-MF-DAS/hackathEOn/blob/main/Day2_UQ_and_GeoFMs/02_eurosat_uq.ipynb)
#
# **Session 1, part 2**
#
# What we do:
#
# 1. Look at the data
# 2. Train a small CNN **without dropout**, then **with dropout** (rate 0.1), and watch the training curves
# 3. How well does the softmax confidence match the accuracy? Reliability diagram, ECE
# 4. Four ways to estimate uncertainty: **temperature scaling**, **Deep Ensembles**, **MC Dropout**
#    and **feature-space density** (DDU)
# 5. Which images are uncertain, and why? Aleatoric vs. epistemic examples
# 6. **Clouds**: a distribution shift the model has never seen, analysed in detail
#    (realistic synthetic clouds and evaluations following the HAICON 2026 Trustworthy UQ workshop)
# 7. What changes if the training data **contains clouds**?
#
# The notebook trains 7 small CNNs for 50 epochs each; a GPU (e.g. Colab) makes this much faster.
#
# All random seeds are fixed, so every run gives the same numbers.
#
# **Based on** (existing notebooks this one builds on):
# - HAICON 2026 workshop, [`03_eo_eurosat_uq.ipynb`](https://github.com/ChrisKo94/HAICON_2026_Trustworthy_UQ_Workshop/blob/main/notebooks/03_eo_eurosat_uq.ipynb) (MIT license):
#   cloud generation (SatelliteCloudGenerator settings), cloud-level sweep, ensemble, cloud-aware training,
#   per-image grids. Marked with "as in the HAICON workshop" in the cells below.
# - [SatelliteCloudGenerator](https://github.com/strath-ai/SatelliteCloudGenerator) (Czerkawski et al., 2023) for the clouds.
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
    shell("pip install -q satellite_cloud_generator torchgeo")  # torchgeo: to build the EuroSAT subset

# %%
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from course_utils import (
    CLASSES, DEVICE, set_seed, load_eurosat, scale_to_unit, to_display, show_images, add_clouds, make_cloudy_training_set,
    SmallCNN, train_classifier, train_model, plot_history, predict_logits, mc_dropout_probs,
    predict_features, FeatureDensity, progress,
    entropy, decompose_uncertainty, accuracy, nll, ece, reliability_diagram, auroc,
)

set_seed(0)
DATA_VARIANT = "ms"                    # "ms": all 13 Sentinel-2 bands (default) | "rgb": red, green, blue only
EPOCHS = 50
print("device:", DEVICE)

# %% [markdown]
# ## 1. The data: EuroSAT
#
# [EuroSAT](https://github.com/phelber/EuroSAT) (Helber et al., 2019) is a benchmark for land cover
# classification with **Sentinel-2** images. It has 27,000 image patches from 34 European countries.
#
# * Each patch is **64 × 64 pixels**. With Sentinel-2's 10 m pixels, a patch covers **640 m × 640 m**.
# * Each patch has exactly **one label** out of 10 classes: AnnualCrop, Forest, HerbaceousVegetation,
#   Highway, Industrial, Pasture, PermanentCrop, Residential, River, SeaLake.
# * By default we use all **13 Sentinel-2 bands** (`DATA_VARIANT = "ms"`, Level-1C, from coastal aerosol B01 to
#   short-wave infrared B12). The values are top-of-atmosphere reflectance × 10,000; for the network we divide them
#   by 3000 (`scale_to_unit`), so most land surfaces lie roughly between 0 and 1.
# * With `DATA_VARIANT = "rgb"` (set it at the top) the notebook uses only red, green and blue (0-255, divided by 255).
#   The download is much smaller (~90 MB instead of ~2 GB).
# * The plots show the true-colour bands B04, B03, B02 (red, green, blue).
#
# To keep things fast we use a **balanced subset** (downloaded automatically the first time), split into three parts:
#
# | Split | Images, 13 bands | Images, RGB | Used for |
# |---|---|---|---|
# | train | 1500 (150 per class) | 3000 (300 per class) | fitting the network weights |
# | validation | 500 (50 per class) | 1000 (100 per class) | monitoring training, fitting the temperature, choosing settings |
# | test | 1000 (100 per class) | 1000 (100 per class) | final evaluation only |
#
# The validation set is held out from training. In a real project you would never look at the
# test set while developing the model; here we plot its curve only to show that it behaves like the validation curve.

# %%
data = load_eurosat(DATA_VARIANT)   # downloads the data the first time
x_train, y_train = scale_to_unit(data["x_train"], DATA_VARIANT), data["y_train"]   # (N, 13 or 3, 64, 64)
x_val, y_val = scale_to_unit(data["x_val"], DATA_VARIANT), data["y_val"]
x_test, y_test = scale_to_unit(data["x_test"], DATA_VARIANT), data["y_test"]
print("train", tuple(x_train.shape), " val", tuple(x_val.shape), " test", tuple(x_test.shape))
print("images per class (train):", torch.bincount(y_train).tolist())

# %% [markdown]
# Six examples per class. Note how similar some classes look: Pasture vs. HerbaceousVegetation,
# AnnualCrop vs. PermanentCrop, Highway vs. River. Here even humans are unsure: **aleatoric** uncertainty.

# %%
fig, axes = plt.subplots(10, 6, figsize=(7, 12))
for c in range(10):
    idx = torch.nonzero(y_train == c).flatten()[:6]
    for j, i in enumerate(idx):
        axes[c, j].imshow(data["rgb_train"][i])  # true colour (B04, B03, B02)
        axes[c, j].axis("off")
    axes[c, 0].text(-8, 32, CLASSES[c], ha="right", va="center", fontsize=9)
plt.tight_layout()
plt.show()

# %%
from course_utils import BANDS
bands = BANDS[DATA_VARIANT]
mean_spectrum = torch.stack([x_train[y_train == c].mean(dim=(0, 2, 3)) for c in range(10)])  # (10 classes, bands)
fig, ax = plt.subplots(figsize=(10, 4))
for c in range(10):
    ax.plot(range(len(bands)), mean_spectrum[c], "o-", ms=3, label=CLASSES[c], color=plt.cm.tab10(c))
ax.set_xticks(range(len(bands)))
ax.set_xticklabels(bands)
ax.set_ylabel("mean value (scaled)")
ax.set_title("Mean value per band and class" + (": vegetation is bright in the near infrared (B08), water is dark"
                                                 if DATA_VARIANT == "ms" else ""))
ax.legend(fontsize=8, ncol=2)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 2. Training a CNN without and with dropout
#
# The CNN has 4 convolution blocks (feature extractor) and a small head with 2 linear layers.
# The head has dropout layers; with `p_drop=0` they do nothing.
#
# We train for 50 epochs and record loss and accuracy on train, validation and test after every epoch.
# The images are randomly flipped during training (a satellite image has no "up").
#
# ### 2a. Without dropout

# %%
def make_cnn(p_drop, seed):
    set_seed(seed)  # same seed -> same initial weights
    return SmallCNN(in_channels=x_train.shape[1], num_classes=10, p_drop=p_drop)


eval_sets = {"val": (x_val, y_val), "test": (x_test, y_test)}

model_a, hist_a = train_model(
    f"cnn_nodrop_seed0_e{EPOCHS}",
    lambda: make_cnn(0.0, seed=0),
    lambda m: train_classifier(m, x_train, y_train, epochs=EPOCHS, seed=0, eval_sets=eval_sets),
)
plot_history(hist_a, "CNN without dropout")

# %% [markdown]
# Compare training and validation loss: if the training loss keeps going down while the validation loss rises
# again, the network becomes more and more confident on the training images (**overfitting**). Section 3 looks
# at what this means for the confidence on new images.
#
# ### 2b. With dropout (rate 0.1)

# %%
model_b, hist_b = train_model(
    f"cnn_drop0.1_seed0_e{EPOCHS}",
    lambda: make_cnn(0.1, seed=0),
    lambda m: train_classifier(m, x_train, y_train, epochs=EPOCHS, seed=0, eval_sets=eval_sets),
)
plot_history(hist_b, "CNN with dropout 0.1")

# %%
for name, h in [("without dropout", hist_a), ("with dropout 0.1", hist_b)]:
    best = int(np.argmin(h["val_loss"]))
    print(f"{name:18s} final: val acc {h['val_acc'][-1]:.1%}, val loss {h['val_loss'][-1]:.3f}"
          f"   (lowest val loss {h['val_loss'][best]:.3f} at epoch {best + 1})")

# %% [markdown]
# Dropout is a regulariser. Compare the curves with those of 2a.
# Both models are trained for the full 50 epochs; we do not stop early, so we can see the effect on calibration.

# %% [markdown]
# ## 3. How calibrated is the softmax confidence?
#
# The **confidence** of a prediction is the largest softmax probability.
# If the model says "90%" for 100 images, about 90 of them should be correct.
#
# The **Expected Calibration Error** sums up the gap between confidence and accuracy over bins:
#
# $$\text{ECE} = \sum_{b} \frac{|B_b|}{N}\,\big|\text{acc}(B_b) - \text{conf}(B_b)\big|$$

# %%
probs_a = predict_logits(model_a, x_test).softmax(-1)
probs_b = predict_logits(model_b, x_test).softmax(-1)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
reliability_diagram(probs_a, y_test, ax=axes[0], title="no dropout")
reliability_diagram(probs_b, y_test, ax=axes[1], title="dropout 0.1")
conf = probs_a.max(-1).values
correct = probs_a.argmax(-1) == y_test
axes[2].hist([conf[correct].numpy(), conf[~correct].numpy()], bins=20, label=["correct", "wrong"], stacked=True)
axes[2].set_xlabel("confidence")
axes[2].set_title("no dropout: confidence of correct and wrong predictions")
axes[2].legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# Bars below the diagonal mean the model is **overconfident**. Note how many *wrong*
# predictions still have a confidence close to 1.

# %% [markdown]
# ## 4. Four ways to estimate uncertainty
#
# ### 4a. Temperature scaling (post-hoc)
#
# We divide the logits $z$ by one number $T$ before the softmax: $p = \text{softmax}(z / T)$.
# $T > 1$ makes the predictions less confident. $T$ is fitted on the **validation set**
# by minimizing the negative log-likelihood. The predicted class does not change.

# %%
def fit_temperature(logits_val, y_val):
    log_t = torch.zeros(1, requires_grad=True)  # T = exp(log_t) stays positive
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=200)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(logits_val / log_t.exp(), y_val)
        loss.backward()
        return loss

    opt.step(closure)
    return log_t.exp().item()


T = fit_temperature(predict_logits(model_a, x_val), y_val)
print(f"fitted temperature T = {T:.2f}")
probs_ts = (predict_logits(model_a, x_test) / T).softmax(-1)

# %% [markdown]
# ### 4b. MC Dropout
#
# For the model with dropout we keep dropout on and predict each image $S = 20$ times.
# This gives 20 slightly different probability vectors per image. From them:
#
# * **total** uncertainty: entropy of the mean, $H[\bar p]$
# * **aleatoric** part: mean entropy, $\frac{1}{S}\sum_s H[p_s]$ (each single prediction is already unsure)
# * **epistemic** part: the difference, called mutual information (the samples **disagree**)

# %%
samples_mc = mc_dropout_probs(model_b, x_test, n_samples=20, seed=0)  # (S, N, K)
probs_mc = samples_mc.mean(0)

# %% [markdown]
# ### 4c. Deep Ensemble
#
# We train 5 CNNs without dropout, each with a different seed (different initial weights and data order),
# and average their predictions. The decomposition into aleatoric and epistemic works exactly as for
# MC Dropout. This costs 5 times the training.
# (Member 1 uses seed 0 and the same settings as the network above, so it is identical to `model_a`.)
#
# Set `RUN_ENSEMBLE = False` to skip it; the later sections then use MC Dropout instead.

# %%
RUN_ENSEMBLE = True  # set to False to skip the ensemble

if RUN_ENSEMBLE:
    members = []
    for i, seed in enumerate([0, 1, 2, 3, 4]):
        member, _ = train_model(
            f"cnn_nodrop_seed{seed}_e{EPOCHS}",
            lambda s=seed: make_cnn(0.0, seed=s),
            lambda m, s=seed: train_classifier(m, x_train, y_train, epochs=EPOCHS, seed=s, lr=1e-3, verbose=True),
            verbose=False,
        )
        members.append(member)
    print(f"trained {len(members)} ensemble members")

    def ensemble_probs(x):
        return torch.stack([predict_logits(m, x).softmax(-1) for m in members])  # (M, N, K)

    for c in [0.0, 0.6]:
        xc = add_clouds(x_test, coverage=c, seed=1)
        s = ensemble_probs(xc)
        tot, ale, epi = decompose_uncertainty(s)
        p = s.mean(0)
        print(f"cloud {c:.0%}: acc {accuracy(p, y_test):.1%}  ECE {ece(p, y_test):.3f}  epistemic {epi.mean():.3f}")

    samples_ens = ensemble_probs(x_test)
    probs_ens = samples_ens.mean(0)

# %% [markdown]
# ### 4d. Feature-space density (DDU)
#
# A different idea (Mukhoti et al., 2023, "Deep Deterministic Uncertainty"): look at the **features** that
# the CNN computes before its head (a 64-dimensional vector per image).
#
# 1. Fit one Gaussian per class to the features of the **training** images.
# 2. For a new image, compute how likely its features are under these Gaussians (the **density**).
#
# Low density = "the network has never seen features like these" = **epistemic** uncertainty.
# The softmax entropy of the same network is used for the **aleatoric** part.
# It needs only **one** network and one forward pass, and no retraining.

# %%
density = FeatureDensity().fit(predict_features(model_a, x_train), y_train)
logdens_test = density.log_density(predict_features(model_a, x_test))
logdens_train = density.log_density(predict_features(model_a, x_train))
print(f"log-density: train mean {logdens_train.mean():.1f}, test mean {logdens_test.mean():.1f}")

# %% [markdown]
# ### 4e. Comparison on clean test images

# %%
methods = {
    "softmax, no dropout": probs_a,
    "softmax, dropout 0.1": probs_b,
    "temperature scaling": probs_ts,
    "MC Dropout": probs_mc,
}
if RUN_ENSEMBLE:
    methods["Deep Ensemble (5)"] = probs_ens
print(f"{'method':24s} {'accuracy':>9s} {'NLL':>7s} {'ECE':>7s}")
for name, p in methods.items():
    print(f"{name:24s} {accuracy(p, y_test):9.1%} {nll(p, y_test):7.3f} {ece(p, y_test):7.3f}")
print("(DDU uses the softmax of the no-dropout network for its predictions)")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
reliability_diagram(probs_ts, y_test, ax=axes[0], title="temperature scaling")
reliability_diagram(probs_mc, y_test, ax=axes[1], title="MC Dropout")
if RUN_ENSEMBLE:
    reliability_diagram(probs_ens, y_test, ax=axes[2], title="Deep Ensemble")
else:
    axes[2].axis("off")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 5. Which images are uncertain? Aleatoric vs. epistemic
#
# We use the Deep Ensemble (or MC Dropout, if the ensemble was skipped) and split its uncertainty into
# the aleatoric and the epistemic part.
# The value is shown above every image (in nats; the maximum for 10 classes is $\log 10 \approx 2.3$).

# %%
samples_best = samples_ens if RUN_ENSEMBLE else samples_mc
print("using", "Deep Ensemble" if RUN_ENSEMBLE else "MC Dropout")
total_u, ale_u, epi_u = decompose_uncertainty(samples_best)
pred_u = samples_best.mean(0).argmax(-1)


def show_ranked(score, label, largest=True, n=8):
    idx = score.argsort(descending=largest)[:n]
    titles = [f"true: {CLASSES[y_test[i]][:11]}\npred: {CLASSES[pred_u[i]][:11]}\n{label}={score[i]:.2f}"
              for i in idx]
    show_images(x_test[idx], titles, n=n, ncols=n)


print("HIGHEST aleatoric uncertainty: each member is unsure itself (ambiguous images)")
show_ranked(ale_u, "ale")
print("HIGHEST epistemic uncertainty: the members disagree (unusual images)")
show_ranked(epi_u, "epi")
print("LOWEST aleatoric uncertainty")
show_ranked(ale_u, "ale", largest=False)
print("LOWEST epistemic uncertainty")
show_ranked(epi_u, "epi", largest=False)

# %% [markdown]
# Things to look for:
#
# * High **aleatoric**: are these mixed scenes (a road through fields, a river in a forest) or classes that look alike?
# * High **epistemic**: are these unusual images, e.g. odd colours, haze, rare patterns?
# * Lowest uncertainty: which classes appear here?
#
# Does the uncertainty help to find mistakes? Accuracy per uncertainty quartile:

# %%
for label, score in [("total", total_u), ("aleatoric", ale_u), ("epistemic", epi_u)]:
    q = torch.quantile(score, torch.tensor([0.25, 0.5, 0.75]))
    bins = torch.bucketize(score, q)
    accs = [(pred_u[bins == b] == y_test[bins == b]).float().mean().item() for b in range(4)]
    print(f"{label:10s} accuracy from most certain to least certain quarter: " + "  ".join(f"{a:.1%}" for a in accs))

# %% [markdown]
# ## 6. Clouds: a distribution shift
#
# The training data has (almost) no clouds. We add synthetic clouds to the test images with increasing
# cloud level. This is a typical problem in remote sensing: the model will still give an answer for
# a cloudy image, so we want the uncertainty to warn us.
#
# The clouds come from the **SatelliteCloudGenerator** (Czerkawski et al., 2023), with the same settings as in the
# HAICON 2026 Trustworthy UQ workshop: Perlin-noise cloud shapes with slightly coloured, blurred edges.
# A fixed seed makes the clouds identical in every run.

# %%
levels = [0.0, 0.15, 0.30, 0.45, 0.60, 0.75, 0.90]
examples = torch.cat([add_clouds(x_test[i:i + 1], coverage=c, seed=3) for i in [0, 1, 2] for c in levels])
show_images(examples, [f"cloud level {c:.2f}" for c in levels] * 3, n=21, ncols=7)

# %% [markdown]
# ### 6a. Accuracy, confidence and uncertainty vs. cloud level
#
# We create one cloudy version of the test set per level and evaluate all methods on it.

# %%
cloud_results = {}
for c in progress(levels, desc="cloud levels"):
    xc = add_clouds(x_test, coverage=c, seed=42)
    cloud_results[c] = {
        "x": xc,
        "p_a": predict_logits(model_a, xc).softmax(-1),
        "s_mc": mc_dropout_probs(model_b, xc, n_samples=20, seed=0, show=False),
        "s_ens": ensemble_probs(xc) if RUN_ENSEMBLE else None,
        "dens": density.log_density(predict_features(model_a, xc)),
    }
print(f"{len(levels)} cloud levels done")

sample_methods = [("MC Dropout", "s_mc")] + ([("Deep Ensemble", "s_ens")] if RUN_ENSEMBLE else [])
curves = {}
for name, key in [("softmax (no dropout)", "p_a")] + sample_methods:
    acc, conf, ent = [], [], []
    for c in levels:
        p = cloud_results[c][key]
        p = p if p.dim() == 2 else p.mean(0)
        acc.append(accuracy(p, y_test))
        conf.append(p.max(-1).values.mean().item())
        ent.append(entropy(p).mean().item())
    curves[name] = (acc, conf, ent)

fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))
for name, (acc, conf, ent) in curves.items():
    axes[0].plot(levels, [a * 100 for a in acc], "o-", label=name)
    axes[1].plot(levels, conf, "s-", label=name)
    axes[2].plot(levels, ent, "^-", label=name)
last = list(curves)[-1]
for x_, a in zip(levels, curves[last][0]):
    axes[0].text(x_, a * 100 + 2, f"{a * 100:.0f}%", ha="center", fontsize=8)
axes[0].axhline(10, color="gray", ls=":", label="random guess (10%)")
axes[1].axhline(0.1, color="gray", ls=":")
axes[0].set(ylabel="accuracy (%)", title="Accuracy vs. cloud level", ylim=(0, 105))
axes[1].set(ylabel="mean max probability", title="Confidence stays high although accuracy drops", ylim=(0, 1.05))
axes[2].set(ylabel="mean predictive entropy", title="Total uncertainty vs. cloud level")
for ax in axes:
    ax.set_xlabel("cloud level")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
plt.tight_layout()
plt.show()

# %%
fig, axes = plt.subplots(1, len(sample_methods) + 1, figsize=(6 * (len(sample_methods) + 1), 4))
for ax, (name, key) in zip(axes, sample_methods):
    parts = np.array([[v.mean().item() for v in decompose_uncertainty(cloud_results[c][key])] for c in levels])
    ax.stackplot(levels, parts[:, 1], parts[:, 2], labels=["aleatoric", "epistemic"], alpha=0.7)
    ax.plot(levels, parts[:, 0], "k--", label="total")
    ax.set_title(f"{name}: aleatoric + epistemic")
    ax.set_xlabel("cloud level")
    ax.legend(loc="upper left")
dens_drop = [-(cloud_results[c]["dens"].mean() - logdens_test.mean()).item() for c in levels]
axes[-1].plot(levels, dens_drop, "o-", color="C3")
axes[-1].set_title("DDU: drop in feature log-density")
axes[-1].set_xlabel("cloud level")
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 6b. What does the model predict for cloudy images?
#
# If the model had learned "I don't know", the predictions would spread over all classes.
# Does it, or does it put cloudy images into a few classes?

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 3.5), sharey=True)
for ax, c in zip(axes, [0.45, 0.90]):
    pred = cloud_results[c]["p_a"].argmax(-1)
    ax.bar(range(10), torch.bincount(pred, minlength=10).numpy(), color="C0")
    ax.axhline(100, color="k", ls="--", lw=1, label="true number per class")
    ax.set_xticks(range(10))
    ax.set_xticklabels(CLASSES, rotation=30, ha="right")
    ax.set_title(f"predicted classes at cloud level {c:.2f} (no dropout)")
    ax.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 6c. Can the uncertainty detect cloudy images?
#
# We compare clean test images with the same images at cloud level 0.30 and 0.60.
# Every method gives each image a score (higher = more uncertain). The **AUROC** measures how well the score
# separates clean from cloudy images: 0.5 = random, 1.0 = perfect.

# %%
def scores_for(c):
    r = cloud_results[c]
    out = {
        "1 - max softmax (no dropout)": 1 - r["p_a"].max(-1).values,
        "MC Dropout: total": decompose_uncertainty(r["s_mc"])[0],
        "MC Dropout: epistemic": decompose_uncertainty(r["s_mc"])[2],
    }
    if RUN_ENSEMBLE:
        out["Ensemble: total"] = decompose_uncertainty(r["s_ens"])[0]
        out["Ensemble: epistemic"] = decompose_uncertainty(r["s_ens"])[2]
    out["DDU: - log-density"] = -r["dens"]
    return out


clean_scores, scores_30, scores_60 = scores_for(0.0), scores_for(0.30), scores_for(0.60)
print(f"{'score':32s} {'AUROC level 0.30':>17s} {'AUROC level 0.60':>17s}")
for name in clean_scores:
    print(f"{name:32s} {auroc(clean_scores[name], scores_30[name]):17.3f} {auroc(clean_scores[name], scores_60[name]):17.3f}")

# %%
hist_names = ["1 - max softmax (no dropout)", "MC Dropout: epistemic"] + \
    (["Ensemble: epistemic"] if RUN_ENSEMBLE else []) + ["DDU: - log-density"]
fig, axes = plt.subplots(1, len(hist_names), figsize=(4.5 * len(hist_names), 3.5))
for ax, name in zip(axes, hist_names):
    s0, s6 = clean_scores[name].numpy(), scores_60[name].numpy()
    bins = np.linspace(min(s0.min(), s6.min()), max(s0.max(), s6.max()), 40)
    ax.hist(s0, bins=bins, alpha=0.6, label="clean")
    ax.hist(s6, bins=bins, alpha=0.6, label="cloud level 0.60")
    ax.set_title(name, fontsize=10)
    ax.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 6d. Single images under increasing clouds
#
# Five test images (rows) at four cloud levels (columns). Title: predicted class, confidence and epistemic
# uncertainty. Green frame = correct, red frame = wrong.

# %%
best_key = "s_ens" if RUN_ENSEMBLE else "s_mc"
sel = [20, 220, 420, 620, 820]
show_levels = [0.0, 0.30, 0.60, 0.90]
fig, axes = plt.subplots(len(sel), len(show_levels), figsize=(3 * len(show_levels), 3 * len(sel)))
for r_, i in enumerate(sel):
    for c_, c in enumerate(show_levels):
        s_ = cloud_results[c][best_key][:, i]           # (S, K)
        p_ = s_.mean(0)
        epi = (entropy(p_) - entropy(s_).mean()).item()
        ok = p_.argmax().item() == y_test[i].item()
        ax = axes[r_, c_]
        ax.imshow(to_display(cloud_results[c]["x"][i:i + 1])[0])  # true colour
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"{CLASSES[p_.argmax()]} ({p_.max() * 100:.0f}%)\nepi={epi:.2f}", fontsize=9,
                     color="green" if ok else "red")
        for sp in ax.spines.values():
            sp.set_color("green" if ok else "red")
            sp.set_linewidth(3)
        if c_ == 0:
            ax.set_ylabel(f"true: {CLASSES[y_test[i]]}", fontsize=9)
        if r_ == 0:
            ax.annotate(f"cloud level {c:.2f}", xy=(0.5, 1.35), xycoords="axes fraction", ha="center", fontweight="bold")
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 6e. Using the uncertainty: reject the most uncertain images
#
# In practice we can let the model **refuse** to answer for the most uncertain images.
# We take the clean test images plus, for every test image, one cloudy version with a random cloud level,
# sort all 2000 images by uncertainty and compute the accuracy on the images we keep.
# A good score makes the accuracy rise quickly when we keep fewer images.

# %%
rng = np.random.default_rng(0)
pick = rng.choice(levels[1:], size=len(y_test))  # one random cloud level per test image


def mixed(key):
    """Clean results followed by, for every test image, the result at its randomly picked cloud level."""
    if key == "s_mc" or key == "s_ens":  # shape (S, N, K): images are in dimension 1
        cloudy = torch.stack([cloud_results[c][key][:, i] for i, c in enumerate(pick)], dim=1)
        return torch.cat([cloud_results[0.0][key], cloudy], dim=1)
    cloudy = torch.stack([cloud_results[c][key][i] for i, c in enumerate(pick)])
    return torch.cat([cloud_results[0.0][key], cloudy])


y_mix = torch.cat([y_test, y_test])
p_mix_a, s_mix_mc, dens_mix = mixed("p_a"), mixed("s_mc"), mixed("dens")
mix_scores = {
    "1 - max softmax": (p_mix_a, 1 - p_mix_a.max(-1).values),
    "MC Dropout: total": (s_mix_mc.mean(0), decompose_uncertainty(s_mix_mc)[0]),
    "DDU: - log-density": (p_mix_a, -dens_mix),
}
if RUN_ENSEMBLE:
    s_mix_ens = mixed("s_ens")
    mix_scores["Ensemble: total"] = (s_mix_ens.mean(0), decompose_uncertainty(s_mix_ens)[0])

keep = np.linspace(0.1, 1.0, 19)
plt.figure(figsize=(8, 4.5))
for name, (p, score) in mix_scores.items():
    order = score.argsort()
    correct = (p.argmax(-1) == y_mix)[order].float()
    plt.plot(keep, [correct[: int(k * len(order))].mean().item() for k in keep], "o-", ms=3, label=name)
plt.xlabel("fraction of images kept (most certain first)")
plt.ylabel("accuracy on kept images")
plt.gca().invert_xaxis()
plt.grid(alpha=0.3)
plt.legend()
plt.title("Rejecting uncertain images (clean + cloudy test images)")
plt.show()

# %% [markdown]
# ### What if the uncertainty does not rise under clouds?
#
# * In our CNN, dropout is only in the **head**; the feature extractor is deterministic. If a cloudy image gives
#   features that look like some class, all dropout samples agree.
# * Other options for this kind of shift:
#   1. **Deep Ensembles**: the members have different feature extractors, so they disagree more on unusual images.
#   2. **Feature-space density (DDU)**: directly asks "have I seen features like these?". Works with one network.
#      (Mukhoti et al. use a feature extractor that keeps distances, e.g. spectral normalisation.)
#   3. **Put the shift into the training data**: if clouds are in the training data, the model learns that cloudy
#      images are ambiguous; the uncertainty then becomes **aleatoric** (next section).
#   4. In practice for clouds specifically: use a **cloud mask** (e.g. Sentinel-2 scene classification or
#      s2cloudless) before the classifier.

# %% [markdown]
# ## 7. Training with clouds (cloud-aware model)
#
# As in the HAICON workshop, a fixed share of the training images (`CLOUD_FRAC`, chosen once with a seed)
# gets clouds and **keeps its true label**. Everything else stays the same as for the dropout model.
#
# Hypothesis (see the lecture: "clouds: aleatoric or epistemic?"):
# for this model cloudy images are **known** but ambiguous, so their uncertainty may move from
# **epistemic** to **aleatoric**. We check this below, together with accuracy and confidence under clouds.

# %%
CLOUD_FRAC = 0.05        # share of training images with clouds
CLOUD_LEVEL_TRAIN = 0.9  # cloud level of these images

x_train_mixed, is_cloudy = make_cloudy_training_set(x_train, fraction=CLOUD_FRAC, coverage=CLOUD_LEVEL_TRAIN, seed=0)
print(f"{int(is_cloudy.sum())} of {len(x_train)} training images are cloudy")
show_images(x_train_mixed[is_cloudy][:8], [CLASSES[i] for i in y_train[is_cloudy][:8]], n=8)

model_c, hist_c = train_model(
    f"cnn_drop0.1_clouds{CLOUD_FRAC}_seed0_e{EPOCHS}",
    lambda: make_cnn(0.1, seed=0),
    lambda m: train_classifier(m, x_train_mixed, y_train, epochs=EPOCHS, seed=0, eval_sets=eval_sets),
)
plot_history(hist_c, "CNN with dropout 0.1, trained with clouds")

# %% [markdown]
# ### Baseline vs. cloud-aware model
#
# Baseline (dropout model trained on clean images) vs. cloud-aware model, on clean test images and on
# test images at cloud level 0.9: accuracy, confidence and calibration.

# %%
cmp = {}
for name, model in [("baseline (clean only)", model_b), ("cloud-aware", model_c)]:
    cmp[name] = {}
    for split, c in [("clean", 0.0), ("cloudy (0.9)", 0.90)]:
        s_ = mc_dropout_probs(model, cloud_results[c]["x"], n_samples=20, seed=0)
        p_ = s_.mean(0)
        tot, ale, epi = decompose_uncertainty(s_)
        cmp[name][split] = dict(acc=accuracy(p_, y_test), conf=p_.max(-1).values.mean().item(),
                                ent=tot.mean().item(), ale=ale.mean().item(), epi=epi.mean().item())

splits = ["clean", "cloudy (0.9)"]
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
xpos, w = np.arange(2), 0.35
for ax, (key, label) in zip(axes, [("acc", "accuracy"), ("conf", "mean confidence"), ("ent", "mean entropy")]):
    for k, (name, color) in enumerate([("baseline (clean only)", "gray"), ("cloud-aware", "C2")]):
        vals = [cmp[name][sp][key] for sp in splits]
        bars = ax.bar(xpos + (k - 0.5) * w, vals, w, color=color, label=name)
        for b_, v in zip(bars, vals):
            ax.text(b_.get_x() + b_.get_width() / 2, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(xpos)
    ax.set_xticklabels(splits)
    ax.set_title(label)
axes[0].legend(fontsize=8)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Aleatoric or epistemic? It depends on the training data

# %%
fig, axes = plt.subplots(1, 3, figsize=(17, 4))
rows = []
for name, model in [("baseline (clean only)", model_b), ("cloud-aware", model_c)]:
    acc, ale, epi = [], [], []
    for c in levels:
        s_ = mc_dropout_probs(model, cloud_results[c]["x"], n_samples=20, seed=0)
        tot, a_, e_ = decompose_uncertainty(s_)
        acc.append(accuracy(s_.mean(0), y_test))
        ale.append(a_.mean().item())
        epi.append(e_.mean().item())
    axes[0].plot(levels, acc, "o-", label=name)
    axes[1].plot(levels, ale, "o-", label=name)
    axes[2].plot(levels, epi, "o-", label=name)
    rows.append((name, ale, epi))
for ax, title in zip(axes, ["accuracy", "aleatoric (MC Dropout)", "epistemic (MC Dropout)"]):
    ax.set_title(title)
    ax.set_xlabel("cloud level")
    ax.grid(alpha=0.3)
    ax.legend()
plt.tight_layout()
plt.show()

for name, ale, epi in rows:
    share = [e / (a + e) for a, e in zip(ale, epi)]
    print(f"{name:22s} epistemic share: " + "  ".join(f"{c:.2f}: {s:.0%}" for c, s in zip(levels, share)))

# %% [markdown]
# Compare the two models: which one is more accurate under clouds, and how does the balance between
# aleatoric and epistemic uncertainty change? Whether uncertainty is aleatoric or epistemic is not a property
# of the image alone: **it depends on what the model has seen during training**.
#
# ## Exercises
#
# 1. Fit the temperature on **cloudy** validation images instead. Does calibration on clean data still hold under clouds?
# 2. Use only 3 ensemble members (seeds `[0, 1, 2]`). What changes?
# 3. Set `CLOUD_FRAC = 0.2`. How do accuracy and the aleatoric/epistemic balance under clouds change?
# 4. Use DDU with the features of the cloud-aware model (`model_c`). Does it still detect clouds? Why (not)?
# 5. Replace the clouds by another shift, e.g. darker images (`x_test * 0.5`) or noise. Which method notices it?
