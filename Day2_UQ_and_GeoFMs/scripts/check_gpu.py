"""Check whether PyTorch can use an NVIDIA GPU, and say what to do if not.

    uv run --no-sync python scripts/check_gpu.py     (or: python scripts/check_gpu.py in the activated environment)
"""

import shutil
import subprocess
import sys


def nvidia_smi():
    """Return (ok, text) from `nvidia-smi`, or (None, reason) if the tool is not installed."""
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return None, "nvidia-smi not found (no NVIDIA driver installed, or no NVIDIA GPU)"
    try:
        out = subprocess.run([exe], capture_output=True, text=True, timeout=20)
    except Exception as err:  # noqa: BLE001
        return False, f"nvidia-smi could not run: {err}"
    text = (out.stdout or "") + (out.stderr or "")
    return out.returncode == 0, text.strip()


def main():
    print(f"Python:  {sys.version.split()[0]}  ({sys.executable})")
    try:
        import torch
    except ImportError:
        print("PyTorch is not installed in this Python. Run `uv sync` in the course folder and start with `uv run`.")
        return 1

    build = torch.version.cuda
    print(f"PyTorch: {torch.__version__}  ({'CUDA ' + build + ' build' if build else 'CPU-only build'})")

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"GPU found: {i}: {torch.cuda.get_device_name(i)}")
        x = torch.randn(256, 256, device="cuda")
        print("test on the GPU: ok" if torch.isfinite(x @ x).all().item() else "test on the GPU: FAILED")
        print("The notebooks will use the GPU.")
        return 0

    print("No GPU available to PyTorch. The notebooks will use the CPU (enough for the course).")
    ok, text = nvidia_smi()
    if ok is None:
        print(f"- {text}.")
        print("  If you have an NVIDIA GPU: install the NVIDIA driver first (README, 'Using an NVIDIA GPU').")
    elif not ok:
        print("- The NVIDIA driver is not working. nvidia-smi says:")
        print("    " + text.splitlines()[0] if text else "    (no output)")
        print("  Fix the driver first (README, 'Using an NVIDIA GPU', step 1), then run this check again.")
    else:
        line = next((l for l in text.splitlines() if "CUDA Version" in l), "")
        print("- The NVIDIA driver works:", line.strip(" |") or "(see nvidia-smi)")
        if build is None:
            print("  But this is a CPU-only PyTorch build. Install the CUDA build:")
            print("  uv: `uv sync --extra cuda`, then start Jupyter with `uv run --no-sync jupyter lab`.")
        else:
            print(f"  PyTorch was built for CUDA {build}, which may be newer than your driver supports.")
            print("  Update the NVIDIA driver (nvidia-smi should show CUDA Version 13.x).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
