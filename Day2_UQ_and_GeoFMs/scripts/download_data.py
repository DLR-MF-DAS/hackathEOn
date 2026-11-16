"""Get the course data. Choose a variant:

    uv run python scripts/download_data.py                  # 13 Sentinel-2 bands (default): ~2 GB download, ~250 MB on disk
    uv run python scripts/download_data.py --variant rgb    # RGB: ~90 MB download, ~50 MB on disk

The notebooks call this automatically when the data is missing (also in Colab).

If the instructor has uploaded the prepared .npz files (e.g. as a GitHub release asset), put the links
into URLS below and the files are downloaded directly. Otherwise the subset is built from the full
EuroSAT (scripts/prepare_data.py).
"""

import argparse
import os
import runpy
import sys
import urllib.request

from tqdm import tqdm

# optional (instructor): links to uploaded copies of the prepared files
URLS = {
    "ms": "",    # data/eurosat_ms.npz  (~250 MB)
    "rgb": "",   # data/eurosat_rgb.npz (~50 MB)
}

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")


def main(variant="ms"):
    target = os.path.join(DATA, f"eurosat_{variant}.npz")
    os.makedirs(DATA, exist_ok=True)
    if os.path.exists(target):
        print(f"data/eurosat_{variant}.npz is already there")
        return

    if URLS[variant]:
        print(f"downloading eurosat_{variant}.npz ...")
        with tqdm(unit="B", unit_scale=True, unit_divisor=1024, file=sys.stdout, ncols=100, mininterval=0.5) as bar:
            def hook(blocks, block_size, total):
                bar.total = total if total > 0 else None
                bar.update(blocks * block_size - bar.n)
            urllib.request.urlretrieve(URLS[variant], target + ".part", reporthook=hook)
        os.replace(target + ".part", target)  # only a complete download gets the final name
        print("done")
    else:
        size = "~2 GB" if variant == "ms" else "~90 MB"
        print(f"no download link set: building the subset from EuroSAT ({size} download, a few minutes)")
        prepare = runpy.run_path(os.path.join(HERE, "prepare_data.py"))  # same process: progress is visible
        prepare["build"](variant)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["ms", "rgb"], default="ms",
                        help="ms: all 13 bands (default); rgb: red, green, blue only")
    main(parser.parse_args().variant)
