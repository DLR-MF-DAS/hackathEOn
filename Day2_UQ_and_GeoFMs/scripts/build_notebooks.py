"""Rebuild the .ipynb notebooks from the scripts in src_notebooks/ (needs `pip install jupytext`).

    python scripts/build_notebooks.py

Each notebook gets the standard kernel "Python 3 (ipykernel)". Started with
`uv run jupyter lab`, that kernel is the course environment, so Jupyter does not
ask which kernel to use (and nobody picks a kernel from another project by accident).
"""

import glob
import os

import jupytext

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")

KERNEL = {
    "kernelspec": {"name": "python3", "display_name": "Python 3 (ipykernel)", "language": "python"},
    "language_info": {"name": "python"},
}

for src in sorted(glob.glob(os.path.join(ROOT, "src_notebooks", "*.py"))):
    nb = jupytext.read(src)
    nb.metadata = dict(KERNEL)
    out = os.path.join(ROOT, os.path.basename(src).replace(".py", ".ipynb"))
    jupytext.write(nb, out, fmt="ipynb")
    print("wrote", os.path.basename(out))
