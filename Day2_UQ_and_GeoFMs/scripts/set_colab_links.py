"""Write the repository address into all notebooks and the README (Colab links and the Colab setup cell).

    python scripts/set_colab_links.py https://github.com/<owner>/<repo> <folder> [branch]

<folder> is the path of this course folder inside the repository, e.g. "Day2_UQ_and_GeoFMs".
Run it again whenever the repository or the folder changes. Then commit the notebooks and the README.
"""

import glob
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    repo = sys.argv[1].rstrip("/").removesuffix(".git")
    folder = sys.argv[2].strip("/")
    branch = sys.argv[3] if len(sys.argv) > 3 else "main"
    m = re.match(r"https://github\.com/([^/]+)/([^/]+)$", repo)
    if not m:
        sys.exit("The repository must look like https://github.com/<owner>/<repo>")
    owner, name = m.groups()
    colab = f"https://colab.research.google.com/github/{owner}/{name}/blob/{branch}/{folder}/"

    files = (glob.glob(os.path.join(ROOT, "*.ipynb")) + glob.glob(os.path.join(ROOT, "src_notebooks", "*.py"))
             + [os.path.join(ROOT, "README.md")])
    for path in files:
        text = open(path, encoding="utf-8").read()
        new = re.sub(r"https://colab\.research\.google\.com/github/[^/\s)\"]+/[^/\s)\"]+/blob/[^/\s)\"]+/"
                     r"(?:[^\s)\"]*/)?([0-9][^/\s)\"]*\.ipynb)", lambda mm: colab + mm.group(1), text)
        # REPO = "..." and FOLDER = "..." in the Colab cell (in .ipynb files the quotes are escaped)
        new = re.sub(r'(REPO = \\?")[^"\\]*(\\?")', lambda mm: mm.group(1) + repo + mm.group(2), new)
        new = re.sub(r'(FOLDER = \\?")[^"\\]*(\\?")', lambda mm: mm.group(1) + folder + mm.group(2), new)
        if new != text:
            open(path, "w", encoding="utf-8").write(new)
            print("updated", os.path.relpath(path, ROOT))
    print(f"\nColab links now point to {colab}<notebook>.ipynb")
    print("The repository must be public (or participants must allow Colab to access it).")


if __name__ == "__main__":
    main()
