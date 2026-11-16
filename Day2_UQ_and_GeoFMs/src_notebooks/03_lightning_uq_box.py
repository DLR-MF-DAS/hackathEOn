# %% [markdown]
# # Notebook 3: Lightning UQ Box: UQ methods straight out of the box
#
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DLR-MF-DAS/hackathEOn/blob/main/Day2_UQ_and_GeoFMs/03_lightning_uq_box.ipynb)
#
# **Session 1, part 3**
#
# In notebooks 1 and 2 we implemented the methods by hand: training loops, Monte Carlo sampling,
# the variance decomposition. That shows how a method works.
# In practice you rarely want to re-implement it.
# [Lightning UQ Box](https://github.com/lightning-uq-box/lightning-uq-box) wraps many UQ methods as
# PyTorch Lightning modules around *your own* network.
#
# The pattern is always the same:
#
# ```python
# uq_model = SomeUQMethod(my_network, ...)     # wrap the network
# trainer.fit(uq_model, datamodule)            # train (or fit post hoc)
# out = uq_model.predict_step(x)               # dict with "pred", "pred_uct", ...
# ```
#
# The notebook follows the official [Lightning UQ Box tutorials](https://lightning-uq-box.readthedocs.io/en/latest/tutorials/index.html)
# and the HAICON 2026 workshop:
#
# **Part 1: toy regression.** The official *Bayes by Backprop* tutorial. A Bayesian neural network is fiddly to
# write by hand (weight distributions, reparameterisation trick, KL term, sampling at test time); with the library
# it is one class. Then we swap the method for a *Deep Ensemble* (official tutorial) without changing anything else.
#
# **Part 2: EuroSAT.** The same building blocks on satellite images, as in the official classification and
# Earth observation tutorials: we train one ordinary CNN and put several UQ methods on top of it
# (Laplace approximation, SWAG, temperature scaling, conformal prediction with RAPS), plus a Bayesian network.
# Then we compare all of them on clear and on cloudy images.
#
# MC Dropout is covered in notebooks 1 and 2 and left out here.
#
# **Based on** (existing notebooks this one builds on), all from the Lightning UQ Box tutorials (Apache-2.0):
# - Part 1: [`regression/bnn_vi_elbo.ipynb`](https://github.com/lightning-uq-box/lightning-uq-box/blob/main/docs/tutorials/regression/bnn_vi_elbo.ipynb) (Bayes by Backprop, as in the HAICON 2026
#   notebook [`04_lightning_uq_box.ipynb`](https://github.com/ChrisKo94/HAICON_2026_Trustworthy_UQ_Workshop/blob/main/notebooks/04_lightning_uq_box.ipynb)) and
#   [`regression/deep_ensemble.ipynb`](https://github.com/lightning-uq-box/lightning-uq-box/blob/main/docs/tutorials/regression/deep_ensemble.ipynb)
# - Part 2: [`classification/laplace.ipynb`](https://github.com/lightning-uq-box/lightning-uq-box/blob/main/docs/tutorials/classification/laplace.ipynb),
#   [`classification/swag.ipynb`](https://github.com/lightning-uq-box/lightning-uq-box/blob/main/docs/tutorials/classification/swag.ipynb),
#   [`classification/bnn_vi_elbo.ipynb`](https://github.com/lightning-uq-box/lightning-uq-box/blob/main/docs/tutorials/classification/bnn_vi_elbo.ipynb) and
#   [`earth_observation/raps.ipynb`](https://github.com/lightning-uq-box/lightning-uq-box/blob/main/docs/tutorials/earth_observation/raps.ipynb) (RAPS on RESISC45).
#   We apply the same wrappers to our EuroSAT CNN instead of the toy data / RESISC45.
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
    shell("pip install -q lightning-uq-box satellite_cloud_generator torchgeo")  # torchgeo: to build the EuroSAT subset

# %%
import copy
import logging
import warnings
from functools import partial

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from lightning import LightningDataModule, Trainer, seed_everything
from lightning.pytorch.loggers import CSVLogger

from lightning_uq_box.datamodules import ToyHeteroscedasticDatamodule
from lightning_uq_box.models import MLP
from laplace import Laplace
from lightning_uq_box.uq_methods import (NLL, BNN_VI_ELBO_Regression, MVERegression, DeepEnsembleRegression,
                                         DeterministicClassification, LaplaceClassification, SWAGClassification,
                                         BNN_VI_ELBO_Classification, TempScaling, RAPS)
from lightning_uq_box.viz_utils import (plot_calibration_uq_toolbox, plot_predictions_regression,
                                        plot_toy_regression_data, plot_training_metrics)

from course_utils import (CLASSES, set_seed, load_eurosat, scale_to_unit, add_clouds, SmallCNN, accuracy, ece, entropy,
                          quiet_lightning, lightning_bar)

quiet_lightning()  # no start-up messages from Lightning for every Trainer
set_seed(0)
seed_everything(0, workers=True)  # Lightning's way to fix all seeds
warnings.filterwarnings("ignore")
plt.rcParams["figure.figsize"] = [14, 5]

DATA_VARIANT = "ms"                         # "ms": all 13 Sentinel-2 bands (default) | "rgb": red, green, blue only
OUTPUT_DIR = "outputs/03_lightning_uq_box"  # Lightning logs and the checkpoints of the toy ensemble
# default_root_dir: Lightning UQ Box writes preds.csv there during trainer.test()
# GPU if PyTorch can see one, otherwise CPU. Force the CPU with DEVICE = "cpu".
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ACCELERATOR = "gpu" if DEVICE == "cuda" else "cpu"
print("device:", DEVICE)
# deterministic="warn": reproducible where possible; on a GPU a few operations have no deterministic version
# lightning_bar(): one progress line per training run, updated in place (instead of Lightning's own bars)
QUIET = dict(enable_checkpointing=False, enable_progress_bar=False, enable_model_summary=False,
             deterministic="warn", default_root_dir=OUTPUT_DIR, callbacks=[lightning_bar()])

# %% [markdown]
# # Part 1: Bayes by Backprop on a toy regression problem
#
# *Sections 'Theory' to 'Evaluate the predictions' follow the Lightning UQ Box tutorial `regression/bnn_vi_elbo.ipynb` (and the HAICON notebook `04_lightning_uq_box.ipynb`), almost 1:1.*
#
# ## Theory in short
#
# A Bayesian neural network (BNN) learns a **distribution over each weight** instead of one value.
# With mean-field variational inference (VI) this distribution is a Gaussian $q(\theta)$ with a learned
# mean and standard deviation for every weight. It approximates the posterior
#
# $$p(\theta|\mathcal{D}) = \frac{p(Y|\theta, X)\,p(\theta)}{p(Y|X)},$$
#
# with a prior $p(\theta) = \mathcal{N}(0, \lambda)$ on the weights. Training minimises the loss (negative ELBO)
#
# $$\mathcal{L} = \beta\, D_{KL}\big(q(\theta)\,\|\,p(\theta)\big) + \frac{1}{2}\ln\big(2\pi\sigma^2\big) + \frac{1}{2\sigma^2}\big(f_\theta(x)-y\big)^2 :$$
#
# fit the data, but keep the weight distribution close to the prior. $\beta$ weights the two parts.
# The sampling of the weights is made differentiable with the reparameterisation trick (Kingma & Welling, 2013).
#
# At test time we draw $S$ weight samples $\theta_s \sim q(\theta)$. If the network predicts a mean and a
# standard deviation, the prediction is
#
# $$f_m(x) = \frac{1}{S}\sum_{s=1}^S \mu_{\theta_s}(x), \qquad
# \sigma_p(x) = \sqrt{\underbrace{\frac{1}{S}\sum_{s=1}^S \big(\mu_{\theta_s}(x) - f_m(x)\big)^2}_{\text{epistemic}}
# + \underbrace{\frac{1}{S}\sum_{s=1}^S \sigma^2_{\theta_s}(x)}_{\text{aleatoric}}}$$
#
# the same decomposition as for MC Dropout and Deep Ensembles in notebook 1.
#
# **What to watch for**
# 1. The backbone (`MLP`) is an ordinary `nn.Module`, nothing Bayesian about it.
# 2. `BNN_VI_ELBO_Regression(...)` makes chosen layers stochastic via `stochastic_module_names`.
#    Here only the last layer (`[-1]`): a *partially* stochastic BNN.
# 3. `predict_step` returns a dict that already separates aleatoric and epistemic uncertainty.
# 4. How do the uncertainty bands behave where there is no training data?

# %% [markdown]
# ## Data module
#
# A toy regression problem from the library, packed as a
# [Lightning DataModule](https://lightning.ai/docs/pytorch/stable/data/datamodule.html).
# This seems like overkill for a toy problem, but it shows how the pieces fit together for real tasks.

# %%
dm = ToyHeteroscedasticDatamodule(batch_size=32, n_points=500)

X_train, Y_train, train_loader, X_test, Y_test, test_loader, X_gtext, Y_gtext = (
    dm.X_train, dm.Y_train, dm.train_dataloader(), dm.X_test, dm.Y_test, dm.test_dataloader(),
    dm.X_gtext, dm.Y_gtext)

fig = plot_toy_regression_data(X_train, Y_train, X_test, Y_test)

# %% [markdown]
# ## Model
#
# A small multi-layer perceptron with two outputs (mean and standard deviation of a Gaussian).
# Inside the UQ method its last layer is turned into a Bayesian layer.

# %%
network = MLP(n_inputs=1, n_hidden=[50, 50], n_outputs=2, activation_fn=nn.ReLU())
network

# %%
bbp_model = BNN_VI_ELBO_Regression(
    network,
    optimizer=partial(torch.optim.Adam, lr=3e-3),
    criterion=NLL(),
    stochastic_module_names=[-1],
    num_mc_samples_train=10,
    num_mc_samples_test=25,
    burnin_epochs=20,
)

# %% [markdown]
# ## Trainer
#
# The [Lightning Trainer](https://lightning.ai/docs/pytorch/stable/common/trainer.html) runs the training loop.
# The logger writes the training metrics to a CSV file.

# %%
log_dir = os.path.join(OUTPUT_DIR, "bbp_toy")
trainer = Trainer(
    accelerator=ACCELERATOR,
    max_epochs=150,
    logger=CSVLogger(log_dir),
    log_every_n_steps=20,
    **{**QUIET, "default_root_dir": log_dir},
)
trainer.fit(bbp_model, dm)

# %%
fig = plot_training_metrics(os.path.join(log_dir, "lightning_logs"), ["train_loss", "trainRMSE"])

# %% [markdown]
# ## Evaluate the predictions
#
# The data module has two test sets: `X_test` are noisy samples like the training data;
# `X_gtext` ("ground truth extended") is the noise-free function on a wider input range,
# so we can see how the method behaves outside the training data.

# %%
torch.manual_seed(0)
preds = bbp_model.predict_step(X_gtext.to(bbp_model.device))

fig = plot_predictions_regression(
    X_train, Y_train, X_gtext, Y_gtext,
    preds["pred"].squeeze(-1), preds["pred_uct"],
    epistemic=preds["epistemic_uct"], aleatoric=preds["aleatoric_uct"],
    title="Bayes By Backprop MFVI", show_bands=False,
)

# %% [markdown]
# The [uncertainty-toolbox](https://uncertainty-toolbox.github.io/) gives calibration plots for regression:
# does a predicted 90% interval contain 90% of the test points?

# %%
preds = bbp_model.predict_step(X_test.to(bbp_model.device))
fig = plot_calibration_uq_toolbox(
    preds["pred"].cpu().numpy(), preds["pred_uct"].cpu().numpy(),
    Y_test.cpu().numpy(), X_test.cpu().numpy())

# %% [markdown]
# ## Swap the method: a Deep Ensemble
#
# *Follows the Lightning UQ Box tutorial `regression/deep_ensemble.ipynb`.*
#
# Changing the UQ method keeps the data module, the trainer and the plots. Here a Deep Ensemble, as in the official
# [Deep Ensemble tutorial](https://lightning-uq-box.readthedocs.io/en/latest/tutorials/regression/deep_ensemble.html):
# we train 5 networks that predict mean and variance (`MVERegression`, "mean variance estimation"), save each one
# as a checkpoint, and combine them with `DeepEnsembleRegression`.

# %%
n_ensembles = 5
members = []
for i in range(n_ensembles):
    seed_everything(i, workers=True)  # a different initialisation for every member
    member = MVERegression(MLP(n_inputs=1, n_hidden=[50, 50], n_outputs=2, activation_fn=nn.Tanh()),
                           optimizer=partial(torch.optim.Adam, lr=1e-2), burnin_epochs=20)
    trainer = Trainer(accelerator=ACCELERATOR, max_epochs=150, limit_val_batches=0, num_sanity_val_steps=0,
                      logger=False, **QUIET)
    trainer.fit(member, dm)
    ckpt = os.path.join(OUTPUT_DIR, f"toy_ensemble_member_{i}.ckpt")
    trainer.save_checkpoint(ckpt)
    members.append({"base_model": member, "ckpt_path": ckpt})

deep_ens = DeepEnsembleRegression(members)

# %%
preds = deep_ens.predict_step(X_gtext.to(deep_ens.device))
fig = plot_predictions_regression(
    X_train, Y_train, X_gtext, Y_gtext,
    preds["pred"], preds["pred_uct"],
    epistemic=preds["epistemic_uct"], aleatoric=preds["aleatoric_uct"],
    title="Deep Ensemble (5 members)", show_bands=False,
)

# %% [markdown]
# Compare the epistemic part with Bayes by Backprop, in particular where there is no training data.
# Other regression methods work the same way, e.g. `SWAGRegression`, `LaplaceRegression`, `SNGPRegression`,
# `QuantileRegression`, `DERModel` (deep evidential regression).

# %% [markdown]
# # Part 2: the same on EuroSAT
#
# Now a real remote sensing task: EuroSAT scene classification with the small CNN from notebook 2
# (64 × 64 pixels, 10 classes; 13 Sentinel-2 bands, or red / green / blue with `DATA_VARIANT = "rgb"`).
#
# ## Data module
#
# Lightning UQ Box expects each batch as a dictionary with the keys `"input"` and `"target"`.
# Several methods (BNN, Laplace) also need a `LightningDataModule`, so we write a small one.
# Like the official RAPS tutorial, it has an extra **calibration** set for the post-hoc methods:
# here we split our validation images into calibration (half) and validation (half).

# %%
class DictDataset(Dataset):
    def __init__(self, x, y):
        self.x, self.y = x, torch.as_tensor(y)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return {"input": self.x[i], "target": self.y[i]}


class EuroSATDataModule(LightningDataModule):
    def __init__(self, splits, batch_size=64):
        super().__init__()
        self.splits, self.batch_size = splits, batch_size

    def train_dataloader(self):
        return DataLoader(DictDataset(*self.splits["train"]), batch_size=self.batch_size, shuffle=True,
                          generator=torch.Generator().manual_seed(0))

    def val_dataloader(self):
        return DataLoader(DictDataset(*self.splits["val"]), batch_size=256)

    def calibration_dataloader(self):
        return DataLoader(DictDataset(*self.splits["calib"]), batch_size=256)

    def test_dataloader(self):
        return DataLoader(DictDataset(*self.splits["test"]), batch_size=256)


data = load_eurosat(DATA_VARIANT)  # downloads the data the first time; scaled as in notebook 2
x_val, y_val = scale_to_unit(data["x_val"], DATA_VARIANT), data["y_val"]
half = len(x_val) // 2
splits = {"train": (scale_to_unit(data["x_train"], DATA_VARIANT), data["y_train"]),
          "val": (x_val[:half], y_val[:half]),
          "calib": (x_val[half:], y_val[half:]),
          "test": (scale_to_unit(data["x_test"], DATA_VARIANT), data["y_test"])}
eurosat_dm = EuroSATDataModule(splits)
x_test, y_test = splits["test"]
x_cloudy = add_clouds(x_test, coverage=0.6, seed=1)  # the same test images with clouds (as in notebook 2)
print({k: len(v[0]) for k, v in splits.items()})


def train(uq_model, epochs=10):
    """Train a Lightning UQ Box model on EuroSAT."""
    seed_everything(0, workers=True)
    Trainer(max_epochs=epochs, accelerator=ACCELERATOR, logger=False, **QUIET).fit(uq_model, datamodule=eurosat_dm)
    return uq_model.eval()

# %% [markdown]
# ## A deterministic CNN
#
# *Pattern from the Lightning UQ Box tutorials `classification/laplace.ipynb` and `classification/swag.ipynb` (train a deterministic model first).*
#
# As in the official classification tutorials, everything starts with an ordinary network,
# wrapped in `DeterministicClassification`. Most UQ methods below are built **on top of this one trained network**.

# %%
det_model = DeterministicClassification(
    SmallCNN(in_channels=splits['train'][0].shape[1], num_classes=10),
    loss_fn=nn.CrossEntropyLoss(),
    optimizer=partial(torch.optim.Adam, lr=1e-3),
)
det_model = train(det_model)
cnn = det_model.model.eval()  # the plain PyTorch network inside

# %% [markdown]
# ## Laplace approximation (last layer)
#
# *Follows the Lightning UQ Box tutorial `classification/laplace.ipynb`; the prior-precision tuning uses the Laplace library.*
#
# The Laplace approximation puts a Gaussian around the trained weights. Its width comes from the
# curvature of the loss: where the loss is flat, many weights are about as good, so the Gaussian is wide.
#
# $$p(\theta \mid \mathcal{D}) \approx \mathcal{N}\big(\theta^*,\ H^{-1}\big), \qquad
# H = \nabla^2_\theta\, \mathcal{L}(\theta^*) + \text{prior precision}$$
#
# Here only for the last layer (650 weights), so the full Hessian is small and cheap.
# Lightning UQ Box wraps the [Laplace library](https://aleximmer.github.io/Laplace/); as in the official
# tutorial, `trainer.test` fits the approximation on the training data.

# %%
la = Laplace(cnn, "classification", subset_of_weights="last_layer", hessian_structure="full")
laplace_model = LaplaceClassification(laplace_model=la)
Trainer(accelerator=ACCELERATOR, logger=False, **QUIET).test(laplace_model, datamodule=eurosat_dm, verbose=False)

# The width of the Gaussian depends on the prior precision (how strongly the weights are pulled to 0).
# We tune it with the marginal likelihood ("empirical Bayes"), as recommended by the Laplace library.
with torch.inference_mode(False):
    laplace_model.laplace_model.optimize_prior_precision(method="marglik")
print(f"prior precision: {laplace_model.laplace_model.prior_precision.item():.2f}")

# %% [markdown]
# ## SWAG (last layer)
#
# *Follows the Lightning UQ Box tutorial `classification/swag.ipynb`.*
#
# SWAG (official [SWAG tutorial](https://lightning-uq-box.readthedocs.io/en/latest/tutorials/classification/swag.html))
# continues training the trained network with a small learning rate, keeps snapshots of the weights and fits a
# Gaussian to them (mean, diagonal and low-rank covariance). At prediction time it samples weights from this Gaussian.
# We again only make the last layer stochastic.

# %%
swag_model = SWAGClassification(
    copy.deepcopy(cnn),
    max_swag_snapshots=20,
    snapshot_freq=5,
    num_mc_samples=20,
    swag_lr=1e-3,
    loss_fn=nn.CrossEntropyLoss(),
    stochastic_module_names=[-1],
)
seed_everything(0, workers=True)
Trainer(max_epochs=3, accelerator=ACCELERATOR, logger=False, **QUIET).fit(swag_model, datamodule=eurosat_dm)
swag_model.eval()
print(f"SWAG: {swag_model.num_tracked} snapshots")

# %% [markdown]
# ## Bayes by Backprop for classification
#
# *Follows the Lightning UQ Box tutorial `classification/bnn_vi_elbo.ipynb`.*
#
# The Bayesian method from part 1, now for classification. Unlike Laplace and SWAG it needs its own training run.
# Again only the last layer is stochastic.

# %%
bnn_model = BNN_VI_ELBO_Classification(
    SmallCNN(in_channels=splits['train'][0].shape[1], num_classes=10),
    criterion=nn.CrossEntropyLoss(),
    stochastic_module_names=[-1],
    num_mc_samples_train=5,
    num_mc_samples_test=20,
    optimizer=partial(torch.optim.Adam, lr=1e-3),
)
bnn_model = train(bnn_model)

# %% [markdown]
# ## Temperature scaling
#
# Post-hoc: one number $T$ is fitted on the calibration set, the network stays fixed:
# $p = \mathrm{softmax}(z / T)$. `trainer.fit` only fits $T$.

# %%
ts_model = TempScaling(copy.deepcopy(cnn).eval())
Trainer(max_epochs=1, accelerator=ACCELERATOR, logger=False, **QUIET).fit(ts_model, train_dataloaders=eurosat_dm.calibration_dataloader())
print(f"temperature: {ts_model.temperature.item():.2f}")

# %% [markdown]
# ## Compare on clear and cloudy images
#
# All methods return a dictionary from `predict_step`. `pred` holds the class probabilities
# (for temperature scaling: the scaled logits). We compute accuracy, calibration error (ECE) and the mean
# predictive entropy, on clear test images and on the same images with clouds.

# %%
def probs_of(model, x, seed=0):
    # After fit/test, Lightning moves models back to the CPU. Move model and images to DEVICE.
    # (Laplace keeps its own copy of the network on the GPU if there is one.)
    model.to(DEVICE)
    x = x.to(DEVICE)
    torch.manual_seed(seed)  # the samples of Laplace, SWAG and the BNN are random
    with torch.no_grad():
        if model is cnn:
            return cnn(x).softmax(-1).cpu()
        out = model.predict_step(x)
    p = out["pred"].cpu()
    return p.softmax(-1) if model is ts_model else p


methods = {"CNN (softmax)": cnn, "Temperature scaling": ts_model, "Laplace (last layer)": laplace_model,
           "SWAG (last layer)": swag_model, "BNN (VI, last layer)": bnn_model}
results = {}
print(f"{'method':22s} {'images':8s} {'accuracy':>9s} {'ECE':>7s} {'entropy':>8s}")
for name, model in methods.items():
    for label, x in [("clear", x_test), ("cloudy", x_cloudy)]:
        p = probs_of(model, x)
        results[(name, label)] = p
        print(f"{name:22s} {label:8s} {accuracy(p, y_test):9.1%} {ece(p, y_test):7.3f} {entropy(p).mean().item():8.3f}")

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
names = list(methods)
for ax, metric, fn in [(axes[0], "accuracy", accuracy), (axes[1], "ECE", ece)]:
    for k, label in enumerate(["clear", "cloudy"]):
        ax.bar(np.arange(len(names)) + 0.4 * k, [fn(results[(n, label)], y_test) for n in names], width=0.4,
               label=label)
    ax.set_xticks(np.arange(len(names)) + 0.2)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_title(metric)
    ax.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# Things to compare:
# * Laplace, SWAG and temperature scaling share the same trained network. How do accuracy and calibration differ?
# * Methods that only make the last layer stochastic see the clouds only through the features of the CNN.
#   How much does their entropy change under clouds? Compare with notebook 2.

# %% [markdown]
# ## Conformal prediction (RAPS)
#
# *Follows the Lightning UQ Box tutorial `earth_observation/raps.ipynb` (calibration set, RAPS on top of a trained network).*
#
# As in the official [Earth observation tutorial](https://lightning-uq-box.readthedocs.io/en/latest/tutorials/earth_observation/raps.html)
# (RESISC45 with a pretrained ResNet): conformal prediction gives a **set of classes** with a guarantee.
# On data like the calibration data, the true class is in the set with probability at least $1-\alpha$.
#
# Intuition: on the calibration images we look at how much probability mass is needed until the true class
# is included, and take the $(1-\alpha)$ quantile $\hat q$ of these values. A new image gets all classes
# (sorted by probability) until the mass $\hat q$ is reached. Uncertain images get larger sets.
# RAPS adds a small penalty for large sets.

# %%
raps = RAPS(copy.deepcopy(cnn).eval(), alpha=0.1)
Trainer(max_epochs=1, accelerator=ACCELERATOR, logger=False, **QUIET).fit(raps, train_dataloaders=eurosat_dm.calibration_dataloader())


def coverage_and_size(x, y):
    raps.to(DEVICE)
    out = raps.predict_step(x.to(DEVICE))
    covered = [int(y[i]) in [int(c) for c in s] for i, s in enumerate(out["pred_set"])]
    return sum(covered) / len(covered), out["size"].float().mean().item(), out


for c in [0.0, 0.3, 0.6]:
    cov, size, _ = coverage_and_size(add_clouds(x_test, coverage=c, seed=1), y_test)
    print(f"cloud cover {c:.0%}: coverage {cov:.1%}   mean set size {size:.2f}")

# %% [markdown]
# * The guarantee holds for data like the calibration data. Compare the coverage on clear images with the 90% target.
# * Cloudy images are not like the calibration data, so the guarantee does not apply. Look at coverage and set size.
#
# A few example prediction sets:

# %%
_, _, out = coverage_and_size(x_test[:8], y_test[:8])
for i, s in enumerate(out["pred_set"]):
    print(f"true: {CLASSES[y_test[i]]:22s} set: {[CLASSES[int(c)] for c in s]}")

# %% [markdown]
# ## More methods
#
# Swapping the method is mostly a one-line change. More classification methods in `lightning_uq_box.uq_methods`,
# each with an official tutorial:
#
# | Method | Class |
# |---|---|
# | Deep Ensembles | `DeepEnsembleClassification` |
# | SNGP | `SNGPClassification` |
# | Deep kernel learning | `DKLClassification` |
# | DDU (feature density) | `DDUClassification` |
# | Masksembles | `MasksemblesClassification` |
#
# The library also supports segmentation and pixel-wise regression. See the
# [tutorials](https://lightning-uq-box.readthedocs.io/en/latest/tutorials/index.html), e.g. land cover segmentation
# and change detection.
#
# ## Exercises
#
# 1. Make SWAG stochastic in all layers (`stochastic_module_names=None`). Does the uncertainty on cloudy images rise?
# 2. Change `alpha` of RAPS to 0.05. What happens to the set size?
# 3. Fit RAPS on *cloudy* calibration images. Does the coverage on cloudy test images recover?
# 4. Build a `DeepEnsembleClassification` from 3 `DeterministicClassification` models (see the Deep Ensemble
#    example in part 1). Compare it on cloudy images.
