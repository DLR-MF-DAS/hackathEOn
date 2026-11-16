"""Prepare the EuroSAT subset used in the course. Two variants:

    python scripts/prepare_data.py                  (default: --variant ms)
        Downloads the multispectral EuroSAT (all 13 Sentinel-2 bands, ~2 GB) via TorchGeo.
        Writes data/eurosat_ms.npz: 3000 images (150 / 50 / 100 per class), 13x64x64,
        raw Level-1C values (int16), ~250 MB.

    python scripts/prepare_data.py --variant rgb
        Downloads the RGB version of EuroSAT (27,000 JPEGs, ~90 MB) via torchvision.
        Writes data/eurosat_rgb.npz: 5000 images (300 / 100 / 100 per class), 64x64x3, uint8, ~50 MB.

Both files contain x_train, y_train, x_val, y_val, x_test, y_test, balanced over the 10 classes.

The bands in eurosat_ms.npz are stored in TerraMind's S2L1C order:
B01 B02 B03 B04 B05 B06 B07 B08 B8A B09 B10 B11 B12
(TorchGeo returns B8A last, so we move it.)

After that, the download in data/raw/ can be deleted.

For testing the notebooks without internet access:

    python scripts/prepare_data.py --synthetic            (or --variant rgb --synthetic)
"""

import argparse
import os
import sys

try:
    import numpy as np
except ModuleNotFoundError:
    raise SystemExit(f"numpy not found for {sys.executable}.\n"
                     "Run this script inside the course environment, e.g. "
                     "`uv run python scripts/download_data.py`, or activate .venv first.")

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")

# images per class and split
SIZES = {
    "ms": {"train": 150, "val": 50, "test": 100},    # 3000 images
    "rgb": {"train": 300, "val": 100, "test": 100},  # 5000 images
}

# TorchGeo order B01..B08, B09, B10, B11, B12, B8A  ->  B01..B08, B8A, B09, B10, B11, B12
TORCHGEO_TO_TERRAMIND = [0, 1, 2, 3, 4, 5, 6, 7, 12, 8, 9, 10, 11]


def balanced_splits(labels, sizes, seed=0):
    """Disjoint, class-balanced random indices for train / val / test."""
    rng = np.random.default_rng(seed)
    per_class = {c: rng.permutation(np.where(labels == c)[0]) for c in range(10)}
    out, start = {}, 0
    for split, n in sizes.items():
        idx = np.concatenate([per_class[c][start:start + n] for c in range(10)])
        out[split] = rng.permutation(idx)
        start += n
    return out


def rgb_from_torchvision():
    from torchvision.datasets import EuroSAT

    ds = EuroSAT(root=os.path.join(DATA, "raw"), download=True)
    labels = np.array(ds.targets)
    out = {}
    for split, idx in balanced_splits(labels, SIZES["rgb"]).items():
        out[f"x_{split}"] = np.stack([np.asarray(ds[int(i)][0].convert("RGB")) for i in idx])  # (N, 64, 64, 3)
        out[f"y_{split}"] = labels[idx].astype(np.int64)
        print(f"{split}: {len(idx)} images")
    return out


def ms_from_torchgeo():
    try:
        from torchgeo.datasets import EuroSAT
    except ImportError:
        raise SystemExit("The multispectral data needs TorchGeo (part of the course environment; in Colab: pip install torchgeo)")

    root = os.path.join(DATA, "raw")
    # TorchGeo comes with its own train/val/test lists; we pool them and make our own split
    datasets = [EuroSAT(root=root, split=s, bands=EuroSAT.all_band_names, download=True)
                for s in ["train", "val", "test"]]
    items = [(d, i) for d in datasets for i in range(len(d))]
    labels = np.concatenate([np.array(d.targets) for d in datasets])
    out = {}
    for split, idx in balanced_splits(labels, SIZES["ms"]).items():
        x = np.stack([items[int(i)][0][items[int(i)][1]]["image"].numpy() for i in idx])
        x = x[:, TORCHGEO_TO_TERRAMIND]
        out[f"x_{split}"] = np.clip(x, 0, 32767).astype(np.int16)  # (N, 13, 64, 64)
        out[f"y_{split}"] = labels[idx].astype(np.int64)
        print(f"{split}: {len(idx)} images")
    return out


def synthetic(variant):
    """Fake data with the same shapes, only for testing the code."""
    rng = np.random.default_rng(0)
    channels, hi = (13, 6000) if variant == "ms" else (3, 255)
    base = rng.uniform(0.15 * hi, 0.8 * hi, size=(10, channels))  # one mean value per class and channel
    out = {}
    for split, n in SIZES[variant].items():
        y = rng.permutation(np.repeat(np.arange(10), n))
        shift = rng.normal(0, 0.06 * hi, size=(len(y), channels, 1, 1))
        x = base[y][:, :, None, None] + shift + rng.normal(0, 0.1 * hi, size=(len(y), channels, 64, 64))
        x = np.clip(x, 0, hi)
        if variant == "ms":
            out[f"x_{split}"] = x.astype(np.int16)
        else:
            out[f"x_{split}"] = x.transpose(0, 2, 3, 1).astype(np.uint8)
        out[f"y_{split}"] = y.astype(np.int64)
    return out


def build(variant="ms", fake=False):
    """Build data/eurosat_<variant>.npz (fake=True: synthetic data for testing)."""
    os.makedirs(DATA, exist_ok=True)
    if fake:
        data = synthetic(variant)
    else:
        data = ms_from_torchgeo() if variant == "ms" else rgb_from_torchvision()
    path = os.path.join(DATA, f"eurosat_{variant}.npz")
    np.savez_compressed(path, **data)
    print(f"wrote data/eurosat_{variant}.npz ({os.path.getsize(path) / 1e6:.0f} MB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["ms", "rgb"], default="ms",
                        help="ms: all 13 bands, ~2 GB download (default); rgb: ~90 MB download")
    parser.add_argument("--synthetic", action="store_true", help="write fake data for testing")
    args = parser.parse_args()
    build(args.variant, args.synthetic)
