"""Small helper functions shared by the course notebooks.

Kept in one file on purpose, so that it is easy to read.
"""

import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm as _tqdm

CLASSES = [
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
    "Pasture", "PermanentCrop", "Residential", "River", "SeaLake",
]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _in_notebook():
    """True in Jupyter, VS Code notebooks and Colab (where IPython display handles exist)."""
    try:
        from IPython import get_ipython
        shell = get_ipython()
        return shell is not None and type(shell).__name__ in ("ZMQInteractiveShell", "Shell")
    except Exception:
        return False


class _NotebookBar(_tqdm):
    """tqdm progress bar that writes into ONE notebook output and replaces it on every update.

    Text bars with carriage returns become a long list of lines in some notebook front-ends;
    an IPython display handle is updated in place everywhere (JupyterLab, VS Code, Colab).
    """

    _free = {}                        # (cell execution, nesting level) -> output line that can be reused
    _depth = {"cell": None, "level": 0}

    def __init__(self, *args, **kwargs):
        import io
        from IPython import get_ipython
        from IPython.display import HTML, display
        cell = get_ipython().execution_count
        if _NotebookBar._depth["cell"] != cell:  # new cell: start again at nesting level 0
            _NotebookBar._depth.update(cell=cell, level=0)
            _NotebookBar._free.clear()
        self._key = (cell, _NotebookBar._depth["level"])
        _NotebookBar._depth["level"] += 1
        # bars at the same nesting level in one cell share one output line, so loops leave no empty lines behind
        self._handle = _NotebookBar._free.pop(self._key, None) or display(HTML(""), display_id=True)
        self._released = False
        kwargs["file"] = io.StringIO()  # tqdm's own text output goes nowhere
        kwargs.setdefault("ascii", False)  # block characters, also with the dummy file
        super().__init__(*args, **kwargs)

    def close(self):
        super().close()
        if not getattr(self, "_released", True):
            self._released = True
            _NotebookBar._depth["level"] = max(0, _NotebookBar._depth["level"] - 1)
            if not self.leave:
                _NotebookBar._free[self._key] = self._handle

    def display(self, msg=None, pos=None):
        from IPython.display import HTML, Pretty
        if getattr(self, "_handle", None) is None:
            return True
        text = self.__str__() if msg is None else msg
        self._handle.update(Pretty(text) if text else HTML(""))
        return True


def progress(iterable=None, desc=None, total=None, leave=False, **kwargs):
    """A progress bar on ONE line that is updated in place (Jupyter, VS Code, Colab and the terminal).

    leave=False (default): the line disappears when the loop is done, so long runs do not fill the output.
    leave=True: the final state stays as one line.
    """
    bar = _NotebookBar if _in_notebook() else _tqdm
    return bar(iterable, desc=desc, total=total, leave=leave, file=sys.stdout, ncols=110,
               mininterval=0.5, **kwargs)


def quiet_lightning():
    """Hide Lightning's start-up messages (GPU available, tips, ...), which repeat for every Trainer."""
    import logging
    import warnings
    for name in ["lightning", "lightning.pytorch", "lightning.fabric", "pytorch_lightning", "lightning_fabric"]:
        logging.getLogger(name).setLevel(logging.ERROR)
    warnings.filterwarnings("ignore", module="lightning")
    warnings.filterwarnings("ignore", module="pytorch_lightning")


def lightning_bar(desc="training"):
    """Lightning callback: one progress line over the epochs (use with enable_progress_bar=False)."""
    from lightning.pytorch.callbacks import Callback

    class _EpochBar(Callback):
        def on_train_start(self, trainer, pl_module):
            self.bar = progress(total=max(trainer.max_epochs, 1), desc=desc, unit="epoch")

        def on_train_epoch_end(self, trainer, pl_module):
            metrics = {k: float(v) for k, v in trainer.callback_metrics.items()
                       if hasattr(v, "numel") and v.numel() == 1}
            self.bar.set_postfix_str("  ".join(f"{k} {v:.3f}" for k, v in list(metrics.items())[:3]), refresh=False)
            self.bar.update()

        def on_train_end(self, trainer, pl_module):
            self.bar.close()

    return _EpochBar()


def set_seed(seed=0):
    """Fix all random number generators, so that every run gives the same results.

    Results are identical when you run a notebook again on the same machine.
    On different hardware (CPU vs. GPU) tiny numerical differences can remain.
    """
    import os
    import random
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # needed for deterministic results on GPUs
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_npz(path):
    """Load an .npz file into a dict of numpy arrays."""
    d = np.load(path)
    return {k: d[k] for k in d.files}


# The course data comes in two variants (see scripts/prepare_data.py):
#   "ms":  data/eurosat_ms.npz   3000 images, all 13 Sentinel-2 bands, raw values (default; ~2 GB download)
#   "rgb": data/eurosat_rgb.npz  5000 images, red / green / blue, values 0-255 (~90 MB download)
DATA_FILES = {"ms": "data/eurosat_ms.npz", "rgb": "data/eurosat_rgb.npz"}

# Notebook 4 writes the TerraMind embeddings here; notebooks 5 and 6 read them.
EMBEDDINGS_FILE = "outputs/04_terramind_embeddings/embeddings_terramind_tiny.npz"

# Sentinel-2 bands in the order stored in eurosat_ms.npz (= TerraMind's S2L1C order)
MS_BANDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B10", "B11", "B12"]
BANDS = {"ms": MS_BANDS, "rgb": ["red (B04)", "green (B03)", "blue (B02)"]}

# The CNNs see x / SCALE: roughly 0-1 for most land surfaces
SCALE = {"ms": 3000.0, "rgb": 255.0}
# Brightness of a thick synthetic cloud in the units of the raw data
CLOUD = {"ms": 6000.0, "rgb": 255.0}


def download_eurosat(variant="ms"):
    """Download (or build) data/eurosat_<variant>.npz. Runs scripts/download_data.py in this process,
    so that its progress bars show up in the notebook."""
    import os
    import runpy
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts", "download_data.py")
    argv = sys.argv
    try:
        sys.argv = [script, "--variant", variant]
        runpy.run_path(script, run_name="__main__")
    finally:
        sys.argv = argv


def ms_to_rgb_uint8(x_ms):
    """(N, 13, H, W) raw Sentinel-2 values -> (N, H, W, 3) uint8 true-colour view (B04, B03, B02) for plots."""
    rgb = np.asarray(x_ms)[:, [3, 2, 1]].astype(np.float32) / SCALE["ms"]
    return (np.clip(rgb, 0, 1) * 255).astype(np.uint8).transpose(0, 2, 3, 1)


def load_eurosat(variant="ms"):
    """Load the EuroSAT subset; downloads it first if it is not there yet.

    variant: "ms" (all 13 Sentinel-2 bands, default) or "rgb" (red, green, blue)
    Returns a dict with
      variant          "ms" or "rgb"
      x_train, ...     float tensors (N, C, 64, 64) in the units of the data: raw values (ms) or 0-255 (rgb)
      rgb_train, ...   uint8 arrays (N, 64, 64, 3), a true-colour view of the same images (for plots)
      y_train, ...     long tensors (N,)
    """
    import os
    if variant not in DATA_FILES:
        raise ValueError(f'variant must be "ms" or "rgb", not {variant!r}')
    if not os.path.exists(DATA_FILES[variant]):
        print(f"{DATA_FILES[variant]} not found: downloading it (only once)")
        download_eurosat(variant)
    raw = load_npz(DATA_FILES[variant])
    out = {"variant": variant}
    for s in ["train", "val", "test"]:
        x = raw[f"x_{s}"]
        if variant == "rgb":
            out[f"rgb_{s}"] = x
            out[f"x_{s}"] = torch.from_numpy(x).permute(0, 3, 1, 2).float()
        else:
            out[f"rgb_{s}"] = ms_to_rgb_uint8(x)
            out[f"x_{s}"] = torch.from_numpy(x.astype(np.float32))
        out[f"y_{s}"] = torch.from_numpy(raw[f"y_{s}"])
    n = {s: len(out[f"y_{s}"]) for s in ["train", "val", "test"]}
    name = "13 Sentinel-2 bands" if variant == "ms" else "RGB"
    print(f"EuroSAT, {name} ({out['x_train'].shape[1]} channels), images per split: {n}")
    return out


def scale_to_unit(x, variant="ms"):
    """Data units -> roughly [0, 1], the input of the CNNs: /3000 for raw Sentinel-2 values, /255 for RGB."""
    return x / SCALE[variant]


def to_display(x):
    """(N, C, H, W) tensor scaled with scale_to_unit -> (N, H, W, 3) array for imshow (true colour)."""
    if x.shape[1] == 13:
        x = x[:, [3, 2, 1]]  # B04, B03, B02
    return x.permute(0, 2, 3, 1).clamp(0, 1).cpu().numpy()


def show_images(x, titles=None, n=8, ncols=8):
    """Show a few images as true colour.

    x: (N, 13 or 3, H, W) tensor scaled with scale_to_unit (as the CNNs see it), or (N, H, W, 3) array in [0, 1].
    """
    if isinstance(x, torch.Tensor):
        x = to_display(x)
    n = min(n, len(x))
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(1.8 * ncols, 1.9 * nrows))
    for i, ax in enumerate(np.atleast_1d(axes).ravel()):
        ax.axis("off")
        if i < n:
            ax.imshow(np.clip(x[i], 0, 1))
            if titles is not None:
                ax.set_title(titles[i], fontsize=8)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Synthetic clouds (a simple distribution shift)
# ---------------------------------------------------------------------------

def add_clouds(x, coverage=0.5, cloud_value=None, seed=0):
    """Add realistic synthetic clouds to a batch of images (N, C, H, W).

    Uses the SatelliteCloudGenerator (Czerkawski et al., 2023; pip package `satellite_cloud_generator`)
    with the same settings as the HAICON 2026 Trustworthy UQ workshop: Perlin-noise clouds with
    slightly coloured, blurred edges.

    coverage:    cloud level between 0 (no clouds) and 1 (fully covered); roughly the share of the
                 image hidden by clouds (the generator's `min_lvl`)
    cloud_value: brightness of a thick cloud in the units of x. Default (None): for images scaled with
                 scale_to_unit, i.e. 2 for 13 bands and 1 for RGB. For raw data use CLOUD[variant].
    seed:        fixes the cloud pattern, so results are reproducible
    """
    if coverage <= 0:
        return x.clone()
    if cloud_value is None:
        cloud_value = CLOUD["ms"] / SCALE["ms"] if x.shape[1] == 13 else 1.0
    from satellite_cloud_generator import add_cloud
    out = []
    batch_size = max(1, 192 // x.shape[1])  # the generator needs a lot of memory: ~15 images with 13 bands
    for start in range(0, len(x), batch_size):
        with torch.random.fork_rng(devices=[]):  # own seed per batch, global random state untouched
            torch.manual_seed(seed + start)
            cloudy, _ = add_cloud(
                x[start:start + batch_size].float() / cloud_value, min_lvl=coverage, max_lvl=1,
                const_scale=True, decay_factor=1.0, clear_threshold=[0.0, 0.2], locality_degree=1,
                cloud_color=True, channel_offset=2, blur_scaling=2, return_cloud=True,
            )
        out.append(cloudy * cloud_value)
    return torch.cat(out)


def make_cloudy_training_set(x, fraction=0.05, coverage=0.9, cloud_value=None, seed=0):
    """A fixed, reproducible mix of clean and cloudy training images (as in the HAICON workshop).

    A random `fraction` of the images (chosen once, with `seed`) gets clouds; labels stay the same.
    Returns the mixed images and a boolean mask of the cloudy ones.
    """
    rng = np.random.default_rng(seed)
    is_cloudy = torch.from_numpy(rng.random(len(x)) < fraction)
    x_mixed = x.clone()
    idx = torch.nonzero(is_cloudy).flatten()
    x_mixed[idx] = add_clouds(x[idx], coverage=coverage, cloud_value=cloud_value, seed=seed)
    return x_mixed, is_cloudy


# ---------------------------------------------------------------------------
# Models and training
# ---------------------------------------------------------------------------

class SmallCNN(nn.Module):
    """A small CNN for 64x64 images: 4 conv blocks, then a head with dropout.

    p_drop=0 gives a normal (deterministic) network; p_drop>0 allows MC Dropout.
    """

    def __init__(self, in_channels=13, num_classes=10, p_drop=0.0, width=16):
        super().__init__()

        def block(c_in, c_out):
            return nn.Sequential(
                nn.Conv2d(c_in, c_out, 3, padding=1), nn.BatchNorm2d(c_out), nn.ReLU(),
                nn.MaxPool2d(2),
            )

        w = width
        self.features = nn.Sequential(
            block(in_channels, w), block(w, 2 * w), block(2 * w, 4 * w), block(4 * w, 4 * w),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )
        self.head = nn.Sequential(
            nn.Dropout(p_drop), nn.Linear(4 * w, 4 * w), nn.ReLU(),
            nn.Dropout(p_drop), nn.Linear(4 * w, num_classes),
        )

    def forward(self, x):
        return self.head(self.features(x))


@torch.no_grad()
def evaluate(model, x, y, batch_size=256):
    """Loss and accuracy on a dataset (dropout off)."""
    logits = predict_logits(model, x, batch_size)
    return F.cross_entropy(logits, y).item(), (logits.argmax(-1) == y).float().mean().item()


def train_classifier(model, x, y, epochs=10, lr=1e-3, batch_size=64, augment=True, seed=0,
                     eval_sets=None, transform=None, verbose=True, desc=None):
    """Plain training loop. x: float tensor (N, C, H, W), y: long tensor (N,).

    eval_sets: optional dict like {"val": (x_val, y_val)}; loss and accuracy are recorded every epoch.
    transform: optional function(xb, generator) -> xb applied to each training batch (e.g. clouds).
    Returns the history: a dict of lists ("train_loss", "train_acc", "val_loss", ...).
    """
    g = torch.Generator().manual_seed(seed)  # controls data order and augmentation
    model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    history = {"train_loss": [], "train_acc": []}
    for name in (eval_sets or {}):
        history[f"{name}_loss"], history[f"{name}_acc"] = [], []
    n = len(x)
    bar = progress(range(epochs), desc=desc or _TRAINING_NAME or "training", unit="epoch", disable=not verbose)
    for epoch in bar:
        model.train()
        perm = torch.randperm(n, generator=g)
        total, correct = 0.0, 0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            xb, yb = x[idx], y[idx]
            if transform is not None:
                xb = transform(xb, g)
            if augment:  # random flips: a satellite image has no "up"
                if torch.rand(1, generator=g) < 0.5:
                    xb = xb.flip(-1)
                if torch.rand(1, generator=g) < 0.5:
                    xb = xb.flip(-2)
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
            correct += (logits.argmax(-1) == yb).sum().item()
        history["train_loss"].append(total / n)
        history["train_acc"].append(correct / n)
        for name, (xe, ye) in (eval_sets or {}).items():
            l, a = evaluate(model, xe, ye)
            history[f"{name}_loss"].append(l)
            history[f"{name}_acc"].append(a)
        bar.set_postfix_str("  ".join(f"{k} {v[-1]:.3f}" for k, v in history.items()))
    bar.close()
    return history


_TRAINING_NAME = None  # shown in the progress bar while train_model trains a model


def train_model(name, make_model, train_fn, verbose=True):
    """Train a fresh model and return it with its training curves.

    name:       shown in the progress bar
    make_model: function that returns a fresh (untrained) model
    train_fn:   function(model) -> history, does the training
    verbose:    print one summary line when done
    """
    global _TRAINING_NAME
    model = make_model()
    _TRAINING_NAME, t0 = name, time.time()
    try:
        history = train_fn(model)
    finally:
        _TRAINING_NAME = None
    if verbose:
        last = "  ".join(f"{k} {v[-1]:.3f}" for k, v in history.items() if v and k.startswith("val"))
        print(f"trained {name}  ({last}, {time.time() - t0:.0f} s)" if last else f"trained {name}  ({time.time() - t0:.0f} s)")
    return model.to(DEVICE).eval(), history


def plot_history(history, title=""):
    """Loss and accuracy curves for all recorded datasets."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    for metric, ax in zip(["loss", "acc"], axes):
        for key, values in history.items():
            if key.endswith(metric):
                ax.plot(range(1, len(values) + 1), values, label=key.replace(f"_{metric}", ""))
        ax.set_xlabel("epoch")
        ax.set_ylabel("cross-entropy loss" if metric == "loss" else "accuracy")
        ax.legend()
        ax.grid(alpha=0.3)
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


@torch.no_grad()
def predict_logits(model, x, batch_size=256, dropout=False):
    """Logits for all images. With dropout=True, dropout stays active (MC Dropout)."""
    model.to(DEVICE).eval()
    if dropout:
        for m in model.modules():
            if isinstance(m, nn.Dropout):
                m.train()
    out = [model(x[i:i + batch_size].to(DEVICE)).cpu() for i in range(0, len(x), batch_size)]
    model.eval()
    return torch.cat(out)


def mc_dropout_probs(model, x, n_samples=20, seed=0, show=True):
    """Returns probabilities of shape (n_samples, N, num_classes). Seeded, so always the same."""
    torch.manual_seed(seed)
    return torch.stack([predict_logits(model, x, dropout=True).softmax(-1)
                        for _ in progress(range(n_samples), desc="MC Dropout samples", disable=not show)])


@torch.no_grad()
def predict_features(model, x, batch_size=256):
    """Output of the feature extractor (the input of the head), shape (N, D)."""
    model.to(DEVICE).eval()
    return torch.cat([model.features(x[i:i + batch_size].to(DEVICE)).cpu() for i in range(0, len(x), batch_size)])


class FeatureDensity:
    """Gaussian density in feature space, one Gaussian per class (the idea of DDU, Mukhoti et al. 2023).

    Fit on the features of the training images. For a new image, a low density means
    "nothing like this was seen during training": a measure of epistemic uncertainty
    that needs only one forward pass and no retraining.
    """

    def fit(self, features, y, num_classes=10, jitter=1e-4):
        f = features.double()
        self.means, self.covs = [], []
        for c in range(num_classes):
            fc = f[y == c]
            mu = fc.mean(0)
            cov = torch.cov(fc.T) + jitter * torch.eye(f.shape[1], dtype=f.dtype)
            self.means.append(mu)
            self.covs.append(cov)
        self.priors = torch.bincount(y, minlength=num_classes).double() / len(y)
        self.dists = [torch.distributions.MultivariateNormal(m, covariance_matrix=c)
                      for m, c in zip(self.means, self.covs)]
        return self

    def log_density(self, features):
        f = features.double()
        logp = torch.stack([d.log_prob(f) for d in self.dists], dim=1) + self.priors.log()
        return torch.logsumexp(logp, dim=1).float()


# ---------------------------------------------------------------------------
# Uncertainty measures and metrics
# ---------------------------------------------------------------------------

def entropy(p, eps=1e-12):
    """Entropy of probability vectors along the last axis."""
    return -(p * (p + eps).log()).sum(-1)


def decompose_uncertainty(prob_samples):
    """prob_samples: (S, N, K) from MC Dropout or an ensemble.

    total     = H[ mean_s p_s ]
    aleatoric = mean_s H[ p_s ]      (expected entropy)
    epistemic = total - aleatoric    (mutual information)
    """
    total = entropy(prob_samples.mean(0))
    aleatoric = entropy(prob_samples).mean(0)
    return total, aleatoric, total - aleatoric


def accuracy(probs, y):
    return (probs.argmax(-1) == y).float().mean().item()


def nll(probs, y, eps=1e-12):
    return -(probs[torch.arange(len(y)), y] + eps).log().mean().item()


def ece(probs, y, n_bins=15):
    """Expected Calibration Error: average |accuracy - confidence| over confidence bins."""
    conf, pred = probs.max(-1)
    correct = (pred == y).float()
    edges = torch.linspace(0, 1, n_bins + 1)
    err = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = (conf > lo) & (conf <= hi)
        if in_bin.any():
            err += in_bin.float().mean() * (correct[in_bin].mean() - conf[in_bin].mean()).abs()
    return float(err)


def reliability_diagram(probs, y, ax=None, n_bins=10, title=""):
    """Bar plot of accuracy per confidence bin. A perfectly calibrated model lies on the diagonal."""
    conf, pred = probs.max(-1)
    correct = (pred == y).float()
    edges = torch.linspace(0, 1, n_bins + 1)
    accs, centers = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = (conf > lo) & (conf <= hi)
        centers.append(float((lo + hi) / 2))
        accs.append(float(correct[in_bin].mean()) if in_bin.any() else 0.0)
    if ax is None:
        _, ax = plt.subplots(figsize=(4, 4))
    ax.bar(centers, accs, width=1 / n_bins, edgecolor="k", alpha=0.7, label="accuracy")
    ax.plot([0, 1], [0, 1], "k--", label="perfect calibration")
    ax.set_xlabel("confidence")
    ax.set_ylabel("accuracy")
    ax.set_title(f"{title}  ECE={ece(probs, y):.3f}")
    ax.legend(loc="upper left", fontsize=8)
    return ax


def auroc(scores_in, scores_out):
    """AUROC for separating in-distribution (low score) from shifted data (high score)."""
    from sklearn.metrics import roc_auc_score
    s = np.concatenate([np.asarray(scores_in), np.asarray(scores_out)])
    t = np.concatenate([np.zeros(len(scores_in)), np.ones(len(scores_out))])
    return roc_auc_score(t, s)


# ---------------------------------------------------------------------------
# TerraMind (Session 2)
# ---------------------------------------------------------------------------

# TerraMind input for each data variant: modality name and key of its pre-training statistics
TERRAMIND_INPUT = {
    "ms": ("S2L1C", "untok_sen2l1c@224"),   # Sentinel-2 L1C, 13 bands, raw values
    "rgb": ("RGB", "untok_sen2rgb@224"),    # Sentinel-2 RGB, values 0-255
}


def load_terramind(variant="ms", name="terramind_v1_tiny", pretrained=True):
    """TerraMind encoder from TerraTorch, with the input modality that fits the data variant."""
    from terratorch.registry import BACKBONE_REGISTRY
    modality, _ = TERRAMIND_INPUT[variant]
    model = BACKBONE_REGISTRY.build(name, pretrained=pretrained, modalities=[modality])
    return model.eval().to(DEVICE)


def terramind_stats(variant="ms"):
    """Mean / std of the TerraMind pre-training data for this input (shape (1, C, 1, 1))."""
    from terratorch.models.backbones.terramind.model.terramind_register import (
        v1_pretraining_mean, v1_pretraining_std,
    )
    _, key = TERRAMIND_INPUT[variant]
    mean = torch.tensor(v1_pretraining_mean[key]).view(1, -1, 1, 1)
    std = torch.tensor(v1_pretraining_std[key]).view(1, -1, 1, 1)
    return mean, std


@torch.no_grad()
def terramind_embed(model, x, variant="ms", batch_size=32, size=224, desc="TerraMind embeddings"):
    """Images (N, C, 64, 64) in the units of the data -> one 192-d (tiny) vector per image.

    Steps: standardise with pre-training stats, resize to 224x224,
    run the encoder, take the last layer and average over all patch tokens.
    """
    modality, _ = TERRAMIND_INPUT[variant]
    mean, std = terramind_stats(variant)
    out = []
    for i in progress(range(0, len(x), batch_size), desc=desc, unit="batch"):
        xb = (x[i:i + batch_size].float() - mean) / std
        xb = F.interpolate(xb, size=(size, size), mode="bilinear", align_corners=False)
        tokens = model({modality: xb.to(DEVICE)})[-1]  # (B, 196, D)
        out.append(tokens.mean(1).cpu())
    return torch.cat(out)


def compute_terramind_embeddings(variant="ms", path=EMBEDDINGS_FILE):
    """What notebook 4 does: TerraMind-tiny embeddings of the EuroSAT splits and of cloudy test images
    (30 % and 60 % cloud cover), saved to `path`."""
    import os
    data = load_eurosat(variant)
    model = load_terramind(variant, "terramind_v1_tiny", pretrained=True)
    out = {}
    for s in ["train", "val", "test"]:
        out[f"emb_{s}"] = terramind_embed(model, data[f"x_{s}"], variant, desc=f"embedding {s}").numpy()
        out[f"y_{s}"] = data[f"y_{s}"].numpy()
    for c in [0.3, 0.6]:
        x_cloudy = add_clouds(data["x_test"], coverage=c, cloud_value=CLOUD[variant], seed=1)
        out[f"emb_test_cloud{round(c * 100)}"] = terramind_embed(
            model, x_cloudy, variant, desc=f"embedding test, {c:.0%} clouds").numpy()
    out["variant"] = np.array(variant)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, **out)
    print(f"saved {path}")
    return out


def load_embeddings(variant=None, path=EMBEDDINGS_FILE):
    """The embeddings written by notebook 4. If they are missing (e.g. in Colab, where every notebook runs on its
    own machine) or were computed from another data variant, they are computed here (a few minutes on a CPU).

    variant: None = use the saved file whatever its data variant ("ms" if nothing is saved); "ms" or "rgb" = require it
    """
    import os
    if os.path.exists(path):
        d = load_npz(path)
        if variant is None or str(d["variant"]) == variant:
            print(f"loaded {path} (data variant: {d['variant']})")
            return d
    print("TerraMind embeddings not found: computing them as in notebook 4")
    return compute_terramind_embeddings(variant or "ms", path)
