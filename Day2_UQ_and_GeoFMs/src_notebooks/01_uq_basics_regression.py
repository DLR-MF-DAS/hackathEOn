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
# **Part 1 · Uncertainty Quantification · notebook 1 of 3**
#
# # Notebook 1: Uncertainty quantification in regression
#
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DLR-MF-DAS/hackathEOn/blob/main/Day2_UQ_and_GeoFMs/01_uq_basics_regression.ipynb)
#
# This notebook follows the regression notebook of the HAICON 2026 workshop
# ["Uncertainty Quantification for Neural Networks"](https://github.com/ChrisKo94/HAICON_2026_Trustworthy_UQ_Workshop)
# (J. Gawlikowski, C. Schweden, DLR).
#
# **Learning goals**
# - Understand **aleatoric** (data noise) and **epistemic** (model / knowledge) uncertainty in regression
# - Train models that learn the aleatoric uncertainty
# - Compare several approaches for aleatoric and epistemic uncertainty
#
# **Methods**
#
# | Method | Aleatoric | Epistemic | Distribution assumption |
# |--------|:---------:|:---------:|------------------------|
# | Plain MSE network | no | no | none |
# | Gaussian NLL | yes | no | Gaussian |
# | Quantile regression | yes | no | none (distribution-free) |
# | MC Dropout | yes | yes | Gaussian |
# | Deep Ensemble | yes | yes | Gaussian (approx.) |
# | SNGP | no (here) | yes | Gaussian process |
#
# A Bayesian neural network (Bayes by Backprop) follows in notebook 3 with Lightning UQ Box.
#
# Everything runs on a CPU. All random seeds are fixed, so every run gives the same result.
#
# **Based on** (existing notebooks this one builds on):
# - HAICON 2026 workshop, [`01_regression_uq.ipynb`](https://github.com/ChrisKo94/HAICON_2026_Trustworthy_UQ_Workshop/blob/main/notebooks/01_regression_uq.ipynb) (MIT license, J. Gawlikowski, C. Schweden):
#   data, models, training loop, Gaussian NLL, quantile regression, MC Dropout, Deep Ensemble, SNGP and the final
#   comparison are taken over almost 1:1. Changes: fixed seeds, three-panel aleatoric / epistemic / total plots,
#   corrected formula of the toy function, coverage check at the end.
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
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from course_utils import set_seed, progress

SEED = 42
set_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# %% [markdown]
# ## 1. The toy dataset
#
# *Adapted from the HAICON notebook `01_regression_uq.ipynb`, section 1.*
#
# We create the data ourselves, so we know the truth. The targets are a smooth function plus noise:
#
# $$y = \mu(x) + \varepsilon(x), \qquad \mu(x) = 0.25\,x + 0.01\,x^2 + 0.5\,x\,\sin(0.5\,x), \qquad
# \varepsilon(x) \sim \mathcal{N}\big(0,\ \sigma(x)^2\big)$$
#
# $$\sigma(x) = 0.1 + 0.3\,e^{-0.5\,(x+1.7)^2} + 0.6\,e^{-0.5\,(x-3.5)^2}$$
#
# * **The noise level changes with $x$** (heteroscedastic noise). It has two bumps, at $x = -1.7$ and at $x = 3.5$.
#   There the points scatter much more around the curve. This is **aleatoric** uncertainty:
#   it is in the data, and more data would not remove it.
#   A remote sensing analogue: a biomass estimate is noisier for dense forest than for grassland.
# * **The training inputs only cover $x \in [-4.5, 4.5]$.** We test on $[-7, 7]$.
#   Outside the training range every prediction is a guess. This is **epistemic** uncertainty:
#   it comes from missing knowledge, and data in these regions would reduce it.
#   A remote sensing analogue: a region, season or sensor the model never saw.
#
# We draw 250 training points with $x$ uniformly distributed in $[-4.5, 4.5]$.

# %%
def true_mean(x):
    return 0.25 * x + 0.01 * x**2 + 0.5 * x * np.sin(0.5 * x)


def true_std(x):
    b1 = 0.3 * np.exp(-0.5 * (x + 1.7) ** 2)
    b2 = 0.6 * np.exp(-0.5 * (x - 3.5) ** 2)
    return 0.1 + b1 + b2


def make_train(n=160, seed=SEED):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-4.5, 4.5, n).astype(np.float32)
    y = (true_mean(x) + rng.normal(0, true_std(x))).astype(np.float32)
    return x, y


x_tr, y_tr = make_train(250)
x_te = np.linspace(-7, 7, 700).astype(np.float32)

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 4))

ax = axes[0]
ax.scatter(x_tr, y_tr, s=8, alpha=0.5, color="blue", label="training points")
ax.plot(x_te, true_mean(x_te), color="green", lw=2.5, label=r"$\mu(x)$")
ax.fill_between(x_te, true_mean(x_te) - 2 * true_std(x_te), true_mean(x_te) + 2 * true_std(x_te),
                alpha=0.2, color="green", label=r"$\mu(x)\pm2\sigma(x)$")
ax.set(xlabel="x", ylabel="y", title="Regression dataset", xlim=(-7, 7))
ax.legend(fontsize=9)

ax = axes[1]
ax.fill_between(x_te, 0, true_std(x_te), alpha=0.3, color="purple")
ax.plot(x_te, true_std(x_te), color="purple", lw=2.5, label=r"true $\sigma(x)$")
ax.axvspan(-7, -4.5, color="gray", alpha=0.15, label="no training data")
ax.axvspan(4.5, 7, color="gray", alpha=0.15)
ax.set(xlabel="x", ylabel=r"$\sigma(x)$", title="Noise standard deviation (aleatoric)", xlim=(-7, 7))
ax.legend(fontsize=9)
plt.show()

# %% [markdown]
# ## 2. Training setup
#
# *From the HAICON notebook `01_regression_uq.ipynb`, section 2 (with a seeded data loader).*
#
# All models use the same training loop: Adam, mini-batches of 32, 1000 epochs.
# The loss function is passed as an argument, so we only have to write a new loss for each method.

# %%
x_tr_t = torch.from_numpy(x_tr).unsqueeze(1).to(device)
y_tr_t = torch.from_numpy(y_tr).unsqueeze(1).to(device)
x_te_t = torch.from_numpy(x_te).unsqueeze(1).to(device)


TRAINING_CURVES = {}  # loss per epoch of every model, plotted at the end of the notebook


def fit(model, loss_fn, n_epochs=1000, lr=1e-4, bs=32, seed=SEED, name="training", verbose=True):
    """Train `model`; one progress line while training, one summary line at the end (verbose=True)."""
    torch.manual_seed(seed)
    opt = optim.Adam(model.parameters(), lr=lr)
    dl = DataLoader(TensorDataset(x_tr_t, y_tr_t), batch_size=bs, shuffle=True,
                    generator=torch.Generator().manual_seed(seed))
    hist = []
    bar = progress(range(n_epochs), desc=name, unit="epoch")
    for ep in bar:
        model.train()
        ep_loss = 0.0
        for xb, yb in dl:
            opt.zero_grad()
            loss = loss_fn(model, xb, yb)
            loss.backward()
            opt.step()
            ep_loss += loss.item()
        hist.append(ep_loss / len(dl))
        if ep % 20 == 0:
            bar.set_postfix_str(f"loss {hist[-1]:.4f}")
    bar.close()
    TRAINING_CURVES[name] = hist  # no return value: Jupyter would print the whole list below the cell
    if verbose:
        print(f"{name}: final loss {hist[-1]:.4f}")


def plot_decomposition(mu, s_al, s_ep, title):
    """Aleatoric, epistemic and total uncertainty as ±2σ bands, plus all three σ curves."""
    s_tot = np.sqrt(s_al**2 + s_ep**2)
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.2), sharex=True)
    for ax, s, color, name in [(axes[0], s_al, "orange", "Aleatoric"), (axes[1], s_ep, "purple", "Epistemic"),
                               (axes[2], s_tot, "red", "Total")]:
        ax.scatter(x_tr, y_tr, s=8, alpha=0.4, color="blue", zorder=4)
        ax.plot(x_te, true_mean(x_te), color="green", lw=2, ls="--")
        ax.plot(x_te, mu, color="black", lw=2)
        ax.fill_between(x_te, mu - 2 * s, mu + 2 * s, alpha=0.35, color=color)
        ax.set(xlabel="x", ylabel="y", title=f"{name} uncertainty (±2σ)", xlim=(-7, 7), ylim=(-4, 6))
    ax = axes[3]
    ax.plot(x_te, s_al, color="orange", lw=2.5, label="aleatoric σ")
    ax.plot(x_te, s_ep, color="purple", lw=2.5, label="epistemic σ")
    ax.plot(x_te, s_tot, color="red", lw=2, ls="--", label="total σ")
    ax.plot(x_te, true_std(x_te), color="green", lw=2, ls=":", label="true noise σ")
    ax.set(xlabel="x", ylabel="σ", title="Decomposition", xlim=(-7, 7))
    ax.legend(fontsize=8)
    plt.suptitle(title, fontsize=13, fontweight="bold", y=1.03)
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ## 3. A standard network: no uncertainty
#
# *From the HAICON notebook `01_regression_uq.ipynb`, section 2.1.*
#
# A standard network with one output $f_\theta(x) = \hat y$, trained with the mean squared error
#
# $$\mathcal{L} = \frac{1}{N}\sum_{i=1}^N (y_i - \hat y_i)^2 .$$
#
# This model gives only a *point prediction*, no uncertainty.

# %%
def make_mlp(out_dim=1, hidden=64, n_layers=2, dropout=0.1):
    layers = [nn.Linear(1, hidden), nn.GELU()]
    for _ in range(n_layers - 1):
        layers += [nn.Linear(hidden, hidden), nn.GELU()]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
    layers.append(nn.Linear(hidden, out_dim))
    return nn.Sequential(*layers)


def mse_loss(model, xb, yb):
    return nn.functional.mse_loss(model(xb), yb)


set_seed(SEED)
det_mlp = make_mlp().to(device)
fit(det_mlp, mse_loss, name="MSE network")

# %%
det_mlp.eval()
with torch.no_grad():
    y_det = det_mlp(x_te_t).cpu().squeeze().numpy()

fig, ax = plt.subplots(figsize=(8, 4))
ax.scatter(x_tr, y_tr, s=12, alpha=0.4, color="blue", zorder=-1)
ax.plot(x_te, true_mean(x_te), color="green", lw=2, ls="--", label=r"$\mu(x)$")
ax.plot(x_te, y_det, color="red", lw=2.5, label="MSE network")
ax.set(xlabel="x", ylabel="y", title="Standard network: point predictions only", xlim=(-7, 7))
ax.legend(fontsize=9)
plt.show()

# %% [markdown]
# The fit on the data is fine. But nothing tells us that the predictions at $|x| > 4.5$ are pure
# extrapolation, or that the points near $x = 3.5$ scatter a lot.

# %% [markdown]
# ## 4. Gaussian negative log-likelihood (aleatoric uncertainty)
#
# *From the HAICON notebook `01_regression_uq.ipynb`, section 2.2.*
#
# > **In short:** The network predicts a mean and a variance and is trained with the Gaussian negative log-likelihood. It captures the noise in the data (aleatoric), but one network with one set of weights cannot know what it has not seen.
# >
# > **Read more:** [Kendall & Gal (2017), What uncertainties do we need in Bayesian deep learning for computer vision?](https://arxiv.org/abs/1703.04977)
#
# Instead of a single number we predict a **Gaussian distribution**:
#
# $$f_\theta(x) \rightarrow \bigl(\mu_\theta(x),\; \log\sigma^2_\theta(x)\bigr)$$
#
# and train with the Gaussian negative log-likelihood (NLL):
#
# $$\mathcal{L} = \frac{1}{N}\sum_{i=1}^N \left[\frac{(y_i - \mu_i)^2}{2\sigma_i^2} + \frac{1}{2}\log\sigma_i^2\right]$$
#
# If the error is large, the network can "explain" it with a larger $\sigma^2$, but it pays for that
# with the $\log \sigma^2$ term. The model **learns the data noise**, i.e. the aleatoric uncertainty only.
# (We predict $\log\sigma^2$ because it can be any number, while $\sigma^2$ must be positive.)

# %%
class GaussianMLP(nn.Module):
    def __init__(self, hidden=64, n_layers=2, dropout=0.1):
        super().__init__()
        layers = [nn.Linear(1, hidden), nn.GELU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.GELU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        self.backbone = nn.Sequential(*layers)
        self.mu_head = nn.Linear(hidden, 1)
        self.lv_head = nn.Linear(hidden, 1)  # log-variance

    def forward(self, x):
        h = self.backbone(x)
        return self.mu_head(h), torch.clamp(self.lv_head(h), -10, 4)


def gauss_nll(model, xb, yb):
    mu, lv = model(xb)
    return 0.5 * (lv + (yb - mu).pow(2) / lv.exp()).mean()


set_seed(SEED)
model_gauss = GaussianMLP().to(device)
fit(model_gauss, gauss_nll, name="Gaussian NLL")

# %%
model_gauss.eval()
with torch.no_grad():
    mu_g, lv_g = model_gauss(x_te_t)
mu_g = mu_g.cpu().squeeze().numpy()
sigma_g = lv_g.mul(0.5).exp().cpu().squeeze().numpy()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
ax = axes[0]
ax.scatter(x_tr, y_tr, s=12, alpha=0.4, color="blue", zorder=4, label="train")
ax.plot(x_te, true_mean(x_te), color="green", lw=2, ls="--", label=r"true $\mu(x)$")
ax.plot(x_te, mu_g, color="red", lw=2.5, label="predicted mean")
ax.fill_between(x_te, mu_g - 2 * sigma_g, mu_g + 2 * sigma_g, alpha=0.3, color="orange",
                label=r"predicted $\pm 2\sigma(x)$")
ax.fill_between(x_te, true_mean(x_te) - 2 * true_std(x_te), true_mean(x_te) + 2 * true_std(x_te),
                alpha=0.12, color="green", label=r"true $\pm 2\sigma(x)$")
ax.set(xlabel="x", ylabel="y", title="Gaussian NLL (aleatoric uncertainty)", xlim=(-7, 7))
ax.legend(fontsize=8)

ax = axes[1]
ax.plot(x_te, true_std(x_te), color="green", lw=2.5, ls="--", label=r"true $\sigma(x)$")
ax.plot(x_te, sigma_g, color="orange", lw=2.5, label=r"predicted $\sigma(x)$")
ax.set(xlabel="x", ylabel=r"$\sigma(x)$", title="Predicted vs. true noise std", xlim=(-7, 7))
ax.legend(fontsize=9)
plt.tight_layout()
plt.show()

# %% [markdown]
# Inside the training range the predicted noise level follows the two bumps.
# Outside the training range the predicted $\sigma$ is whatever the network extrapolates:
# a single network cannot tell us **what it does not know**.

# %% [markdown]
# ## 5. Quantile regression: no distribution assumption
#
# *From the HAICON notebook `01_regression_uq.ipynb`, section 2.3.*
#
# > **In short:** The network predicts quantiles directly (pinball loss), without assuming a Gaussian. Like the Gaussian NLL it only covers aleatoric uncertainty: nothing widens the intervals outside the data.
# >
# > **Read more:** [Koenker & Hallock (2001), Quantile regression](https://doi.org/10.1257/jep.15.4.143); for neural networks: [Tagasovska & Lopez-Paz (2019)](https://arxiv.org/abs/1811.00908)
#
# Instead of a Gaussian, we predict **quantiles** directly. The $q$-quantile is the value below which
# a fraction $q$ of the data lies. The loss for quantile $q \in (0, 1)$ is the **pinball loss**
# (with $u = y - \hat y_q$):
#
# $$\rho_q(u) = \begin{cases} q\,u & u \geq 0 \\ (q-1)\,u & u < 0 \end{cases}$$
#
# For $q = 0.9$, points above the prediction cost 9 times more than points below it,
# so the prediction moves up until 90% of the points are below it.
#
# One network outputs several quantiles at once: `[0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]`.
# The interval between the 0.05- and the 0.95-quantile is a 90% prediction interval (PI).
# This works with **any noise distribution**, but it still only captures aleatoric uncertainty.

# %%
QUANTS = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
N_Q = len(QUANTS)


class QReg(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, N_Q))

    def forward(self, x):
        return self.net(x)


def pinball(model, xb, yb):
    q_pred = model(xb)                      # (B, N_Q)
    q_t = torch.tensor(QUANTS, device=xb.device, dtype=xb.dtype)
    u = yb.expand_as(q_pred) - q_pred
    return torch.max((q_t - 1) * u, q_t * u).mean()


set_seed(SEED)
model_qreg = QReg().to(device)
fit(model_qreg, pinball, n_epochs=2000, name="quantile regression")

# %%
model_qreg.eval()
with torch.no_grad():
    q_preds = model_qreg(x_te_t).cpu().numpy()  # (700, 7)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
ax = axes[0]
ax.scatter(x_tr, y_tr, s=10, alpha=0.4, color="blue", zorder=4)
ax.plot(x_te, true_mean(x_te), color="green", lw=2, ls="--", label=r"true $\mu(x)$")
ax.plot(x_te, q_preds[:, 3], color="red", lw=2.5, label="median (q=0.5)")
for (lo, hi), col, lbl in [((0, 6), "orange", "90% PI"), ((1, 5), "purple", "80% PI"), ((2, 4), "blue", "50% PI")]:
    ax.fill_between(x_te, q_preds[:, lo], q_preds[:, hi], alpha=0.22, color=col, label=lbl)
ax.set(xlabel="x", ylabel="y", title="Quantile regression: prediction intervals", xlim=(-7, 7))
ax.legend(fontsize=9)

ax = axes[1]
ax.plot(x_te, q_preds[:, 6] - q_preds[:, 0], color="orange", lw=2.5, label="90% PI width")
ax.plot(x_te, q_preds[:, 5] - q_preds[:, 1], color="purple", lw=2, label="80% PI width")
ax.plot(x_te, 2 * 1.645 * true_std(x_te), color="green", lw=2, ls="--", label="true 90% PI width")
ax.set(xlabel="x", ylabel="PI width", title="Interval width follows the noise level", xlim=(-7, 7))
ax.legend(fontsize=9)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 6. MC Dropout
#
# *From the HAICON notebook `01_regression_uq.ipynb`, section 2.4 (plots extended to aleatoric / epistemic / total).*
#
# > **In short:** Dropout stays on at test time; each forward pass is a slightly different network, and their disagreement is the epistemic part. No change to training, but $T$ forward passes per input, and the result depends on dropout rate and position.
# >
# > **Read more:** [Gal & Ghahramani (2016), Dropout as a Bayesian approximation](https://arxiv.org/abs/1506.02142)
#
# **Dropout** randomly sets activations to zero during training.
# For MC Dropout we keep dropout **switched on at test time** and run $T$ forward passes.
# Every pass uses a slightly different network:
#
# $$\hat{\mu}(x) = \frac{1}{T}\sum_t \mu_t(x), \qquad
# \underbrace{\hat{\sigma}^2_{\text{epi}}(x) = \frac{1}{T}\sum_t \big(\mu_t(x) - \hat\mu(x)\big)^2}_{\text{epistemic: the passes disagree}}, \qquad
# \underbrace{\hat{\sigma}^2_{\text{ale}}(x) = \frac{1}{T}\sum_t \sigma_t^2(x)}_{\text{aleatoric: average predicted noise}}$$
#
# The network is the same `GaussianMLP` as above, only with a higher dropout rate (0.3).
# `model.train()` keeps dropout active at inference. Careful: this also switches BatchNorm layers
# to training mode (our network has none).

# %%
def mc_predict(model, x_t, T=200, seed=SEED):
    model.train()  # training mode keeps dropout active
    torch.manual_seed(seed)
    mus, sigmas = [], []
    with torch.no_grad():
        for _ in range(T):
            mu, lv = model(x_t)
            mus.append(mu.cpu().squeeze().numpy())
            sigmas.append(lv.mul(0.5).exp().cpu().squeeze().numpy())
    mus, sigmas = np.stack(mus), np.stack(sigmas)  # (T, N)
    mu_hat = mus.mean(0)
    var_epi = mus.var(0)
    var_ale = (sigmas**2).mean(0)
    return mu_hat, np.sqrt(var_ale), np.sqrt(var_epi), mus


set_seed(SEED)
mc_model = GaussianMLP(hidden=64, n_layers=2, dropout=0.3).to(device)
fit(mc_model, gauss_nll, name="MC Dropout")

# %%
mu_mc, sig_al, sig_ep, mc_samples = mc_predict(mc_model, x_te_t, T=300)
sig_tot = np.sqrt(sig_al**2 + sig_ep**2)
plot_decomposition(mu_mc, sig_al, sig_ep, "MC Dropout: total = aleatoric + epistemic")

# %% [markdown]
# Look at the three panels: where does the aleatoric part rise, and how does the epistemic part behave
# outside the training range? Compare with the Deep Ensemble below.

# %% [markdown]
# ## 7. Deep Ensembles
#
# *From the HAICON notebook `01_regression_uq.ipynb`, section 2.5.*
#
# > **In short:** Several networks trained from different random initialisations; their average is the prediction, their disagreement the epistemic part. Costs $M$ times training and prediction ($M = 5$ is common).
# >
# > **Read more:** [Lakshminarayanan et al. (2017), Simple and scalable predictive uncertainty estimation using deep ensembles](https://arxiv.org/abs/1612.01474)
#
# We train $M = 5$ networks from **different random initialisations**. The prediction is a mixture:
#
# $$p(y|x) \approx \frac{1}{M}\sum_{m=1}^M \mathcal{N}\big(y;\mu_m(x),\sigma_m^2(x)\big)$$
#
# and its variance splits in the same two parts (law of total variance):
#
# $$\sigma^2_{\text{total}} = \underbrace{\frac{1}{M}\sum_m\sigma_m^2}_{\text{aleatoric}}
# + \underbrace{\frac{1}{M}\sum_m(\mu_m-\bar\mu)^2}_{\text{epistemic}}$$
#
# **Pros**: simple to implement, no change to the network. **Cons**: $M$ times the training and inference cost.

# %%
num_members = 5


def ens_predict(models, x_t):
    mus, sigs = [], []
    for mdl in models:
        mdl.eval()
        with torch.no_grad():
            mu, lv = mdl(x_t)
        mus.append(mu.cpu().squeeze().numpy())
        sigs.append(lv.mul(0.5).exp().cpu().squeeze().numpy())
    mus, sigs = np.stack(mus), np.stack(sigs)
    return mus.mean(0), np.sqrt((sigs**2).mean(0)), np.sqrt(mus.var(0)), mus


ensemble = []
for i in range(num_members):
    set_seed(i * 777)  # a different initialisation for every member
    m = GaussianMLP(hidden=32, n_layers=2).to(device)
    fit(m, gauss_nll, seed=i * 777, name=f"ensemble member {i + 1}", verbose=False)
    ensemble.append(m)
print(f"Deep Ensemble: {num_members} members, final loss " +
      ", ".join(f"{TRAINING_CURVES[f'ensemble member {i + 1}'][-1]:.4f}" for i in range(num_members)))

# %%
mu_ens, sal_ens, sep_ens, all_mus = ens_predict(ensemble, x_te_t)
stot_ens = np.sqrt(sal_ens**2 + sep_ens**2)

fig, ax = plt.subplots(figsize=(8, 4))
for m_pred in all_mus:
    ax.plot(x_te, m_pred, alpha=0.5, lw=1, color="blue")
ax.plot(x_te, mu_ens, color="red", lw=2.5, label=f"ensemble mean (M={num_members})")
ax.plot(x_te, true_mean(x_te), color="green", lw=2, ls="--")
ax.scatter(x_tr, y_tr, s=8, alpha=0.3, color="gray")
ax.set(title="The members agree on the data and disagree outside", xlabel="x", ylabel="y", xlim=(-7, 7))
ax.legend(fontsize=9)
plt.show()

plot_decomposition(mu_ens, sal_ens, sep_ens, "Deep Ensemble: total = aleatoric + epistemic")

# %% [markdown]
# ## 8. SNGP: Spectral-normalized Neural Gaussian Process
#
# *From the HAICON notebook `01_regression_uq.ipynb`, section 2.6.*
#
# > **In short:** One network, one forward pass: spectral normalisation keeps distances in feature space, and a Gaussian-process output layer turns the distance to the training data into uncertainty.
# >
# > **Read more:** [Liu et al. (2020), Simple and principled uncertainty estimation with deterministic deep learning via distance awareness](https://arxiv.org/abs/2006.10108)
#
# Standard networks can be confidently wrong **far from the training data**.
# SNGP (Liu et al., 2020) makes the uncertainty depend on the **distance to the training data**:
#
# 1. **Spectral normalization** of all weight matrices: inputs that are close stay close in feature space,
#    and inputs that are far apart stay far apart.
# 2. **A Gaussian process (GP) output layer**, approximated with random Fourier features:
#
# $$\Phi(x) = \sqrt{\tfrac{2}{D}}\cos\!\left(W_{\text{rff}}\,h(x) + b\right), \qquad
# f(x) = \beta^T\Phi(x), \qquad \text{Var}[f(x)] = \Phi(x)^T\,\Sigma\,\Phi(x)$$
#
# The covariance $\Sigma$ is computed in one pass over the training data after training.
# Where the features of a test point are unlike all training features, the variance is large.
#
# Our simplified version is trained with the MSE, so it only gives the **epistemic** part.

# %%
from torch.nn.utils import spectral_norm


class SNGP(nn.Module):
    """Simplified SNGP for regression."""

    def __init__(self, hidden=64, n_layers=2, D=1024, ridge=1e-2):
        super().__init__()
        layers, h = [], 1
        for _ in range(n_layers):
            layers += [spectral_norm(nn.Linear(h, hidden)), nn.GELU()]
            h = hidden
        self.backbone = nn.Sequential(*layers)
        self.register_buffer("W_rff", torch.randn(hidden, D))      # fixed random Fourier features
        self.register_buffer("b_rff", torch.rand(D) * 2 * np.pi)
        self.output = nn.Linear(D, 1, bias=False)
        self.register_buffer("prec", torch.eye(D) * ridge)          # precision matrix Σ^-1
        self.ridge, self.D = ridge, D

    def phi(self, x):
        h = self.backbone(x)
        return (2.0 / self.D) ** 0.5 * torch.cos(h @ self.W_rff + self.b_rff)

    def forward(self, x, return_var=False):
        p = self.phi(x)
        mean = self.output(p)
        if return_var:
            var = (p @ torch.linalg.inv(self.prec) * p).sum(-1, keepdim=True)  # φ^T Σ φ
            return mean, var
        return mean

    def update_prec(self, loader):
        """One pass over the training data to compute the precision matrix."""
        self.prec.data = torch.eye(self.D, device=self.prec.device) * self.ridge
        self.eval()
        with torch.no_grad():
            for xb, _ in loader:
                p = self.phi(xb.to(self.prec.device))
                self.prec.data += p.T @ p


def sngp_mse(model, xb, yb):
    return nn.functional.mse_loss(model(xb), yb)


set_seed(SEED)
model_sngp = SNGP(hidden=64, n_layers=2, D=512).to(device)
fit(model_sngp, sngp_mse, n_epochs=2500, lr=3e-4, bs=64, name="SNGP")
model_sngp.update_prec(DataLoader(TensorDataset(x_tr_t, y_tr_t), batch_size=64))  # GP posterior after training

# %%
model_sngp.eval()
with torch.no_grad():
    mu_sn, var_sn = model_sngp(x_te_t, return_var=True)
mu_sn = mu_sn.cpu().squeeze().numpy()
std_sn = var_sn.sqrt().cpu().squeeze().numpy()
# The GP variance has no fixed scale here (no noise model, no tuned prior).
# We rescale it for plotting, only its shape matters.
std_sn = std_sn / (std_sn[np.abs(x_te) > 4.5].mean() + 1e-8) * 0.4

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
ax = axes[0]
ax.scatter(x_tr, y_tr, s=10, alpha=0.4, color="blue", zorder=4)
ax.plot(x_te, true_mean(x_te), color="green", lw=2, ls="--")
ax.plot(x_te, mu_sn, color="red", lw=2.5, label="SNGP mean")
ax.fill_between(x_te, mu_sn - 2 * std_sn, mu_sn + 2 * std_sn, alpha=0.35, color="purple", label="SNGP ±2σ")
ax.set(title="SNGP: distance-aware uncertainty", xlabel="x", ylabel="y", xlim=(-7, 7))
ax.legend(fontsize=9)

ax = axes[1]
ax.plot(x_te, std_sn, color="purple", lw=2.5, label="SNGP (rescaled)")
ax.plot(x_te, sep_ens, color="red", lw=2, ls="--", label="Ensemble epistemic")
ax.plot(x_te, sig_ep, color="orange", lw=2, ls=":", label="MC Dropout epistemic")
ax.set(title="Epistemic uncertainty compared", xlabel="x", ylabel="σ", xlim=(-7, 7))
ax.legend(fontsize=9)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 9. All methods side by side
#
# *From the HAICON notebook `01_regression_uq.ipynb`, section 3.*
#
# | Method | Aleatoric | Epistemic | Cost |
# |---|:---:|:---:|---|
# | Plain MSE network | no | no | low |
# | Gaussian NLL | yes | no | low |
# | Quantile regression | yes | no | low |
# | MC Dropout | yes | yes | low (T forward passes) |
# | Deep Ensemble | yes | yes | high (M networks) |
# | SNGP | no (here) | yes | medium |

# %%
methods = [
    ("1. MSE network", y_det, np.zeros_like(x_te), "gray", "no uncertainty"),
    ("2. Gaussian NLL", mu_g, sigma_g, "orange", "aleatoric only"),
    ("3. Quantile regression", q_preds[:, 3], None, "orange", "aleatoric only"),
    ("4. MC Dropout", mu_mc, sig_tot, "purple", "aleatoric + epistemic"),
    ("5. Deep Ensemble", mu_ens, stot_ens, "green", "aleatoric + epistemic"),
    ("6. SNGP", mu_sn, std_sn, "red", "epistemic, distance-aware"),
]
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
for ax, (name, mu, sigma, color, tag) in zip(axes.flatten(), methods):
    ax.scatter(x_tr, y_tr, s=7, alpha=0.25, color="gray", zorder=2)
    ax.plot(x_te, true_mean(x_te), "k:", lw=1.2, alpha=0.5)
    ax.plot(x_te, mu, color=color, lw=2.5)
    if sigma is None:  # quantile regression: show the 90% interval
        ax.fill_between(x_te, q_preds[:, 0], q_preds[:, 6], alpha=0.3, color=color)
    elif sigma.max() > 1e-5:
        ax.fill_between(x_te, mu - 2 * sigma, mu + 2 * sigma, alpha=0.3, color=color)
    ax.axvspan(-7, -4.5, color="gray", alpha=0.1)
    ax.axvspan(4.5, 7, color="gray", alpha=0.1)
    ax.set_title(f"{name}\n{tag}", fontsize=11, fontweight="bold")
    ax.set(xlim=(-7, 7), ylim=(-3, 6), xlabel="x", ylabel="y")
plt.suptitle("Regression UQ: all methods compared (grey: no training data)", fontsize=14, y=1.01, fontweight="bold")
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Are the intervals any good?
#
# A 90% interval should contain about 90% of new data points. We check this on new data
# **inside** the training range and **outside** of it ($4.5 < |x| < 7$).
# For the Gaussian methods the 90% interval is $\mu \pm 1.645\,\sigma$, for quantile regression it is
# the interval between the 0.05- and the 0.95-quantile. (SNGP is left out: its scale was set by hand.)

# %%
rng = np.random.default_rng(1)
x_in = rng.uniform(-4.5, 4.5, 2000).astype(np.float32)
x_out = np.concatenate([rng.uniform(-7, -4.5, 1000), rng.uniform(4.5, 7, 1000)]).astype(np.float32)


def coverage(lo, hi, x):
    y = true_mean(x) + rng.normal(0, true_std(x))
    return np.mean((y >= np.interp(x, x_te, lo)) & (y <= np.interp(x, x_te, hi)))


intervals = {
    "Gaussian NLL": (mu_g - 1.645 * sigma_g, mu_g + 1.645 * sigma_g),
    "Quantile regression": (q_preds[:, 0], q_preds[:, 6]),
    "MC Dropout": (mu_mc - 1.645 * sig_tot, mu_mc + 1.645 * sig_tot),
    "Deep Ensemble": (mu_ens - 1.645 * stot_ens, mu_ens + 1.645 * stot_ens),
}
print(f"{'method':22s} {'coverage inside':>16s} {'coverage outside':>17s}   (target 90%)")
for name, (lo, hi) in intervals.items():
    print(f"{name:22s} {coverage(lo, hi, x_in):16.1%} {coverage(lo, hi, x_out):17.1%}")

# %%
groups = {}
for name, hist in TRAINING_CURVES.items():
    groups.setdefault(name.split(" member")[0] if name.startswith("ensemble") else name, []).append(hist)
fig, axes = plt.subplots(1, len(groups), figsize=(3.2 * len(groups), 2.8))
for ax, (name, hists) in zip(np.atleast_1d(axes), groups.items()):
    for h in hists:
        ax.plot(h, lw=1)
    ax.set_title("Deep Ensemble" if name == "ensemble" else name, fontsize=10)
    ax.set_xlabel("epoch")
np.atleast_1d(axes)[0].set_ylabel("training loss")
plt.tight_layout()
plt.show()
