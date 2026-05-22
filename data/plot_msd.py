"""
Plot MSD or COM MSD vs time from a .npy file.

The archive `msd_archive.tar.gz` in this folder contains, for each temperature
and trajectory, two files:
    - msd.npy      : single-particle mean squared displacement
    - com_msd.npy  : collective (center-of-mass) mean squared displacement

Both are 1D float arrays of length N (here N = 10000), covering a 2 ns
trajectory (i.e. one data point every 0.2 ps). Units: Å^2.

Usage
-----
    python plot_msd.py path/to/msd.npy
    python plot_msd.py path/to/com_msd.npy -o my_plot.png
    python plot_msd.py path/to/msd.npy --total-ns 2.0 --loglog
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt


def plot_msd(npy_path, total_ns=2.0, loglog=False, out_path=None):
    msd = np.load(npy_path)
    n = len(msd)
    t = np.linspace(0, total_ns, n)  # time axis in ns

    # Infer a sensible label/title from the filename
    base = os.path.basename(npy_path).lower()
    if "com" in base:
        ylabel = "COM MSD (Å²)"
        title = "Collective (COM) MSD vs Time"
    else:
        ylabel = "MSD (Å²)"
        title = "Single-particle MSD vs Time"

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(t, msd, lw=1.2, color="#1f77b4")
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3)

    if loglog:
        ax.set_xscale("log")
        ax.set_yscale("log")
    else:
        ax.set_xlim(0, total_ns)
        ax.set_ylim(0, None)

    fig.tight_layout()

    if out_path is None:
        out_path = os.path.splitext(npy_path)[0] + ".png"
    plt.plot()
    #fig.savefig(out_path, dpi=150)
    #print(f"Saved plot to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot MSD or COM MSD from a .npy file.")
    parser.add_argument("npy_path", help="Path to msd.npy or com_msd.npy")
    parser.add_argument(
        "--total-ns", type=float, default=2.0,
        help="Total trajectory length in ns (default: 2.0)",
    )
    parser.add_argument(
        "--loglog", action="store_true",
        help="Use log-log axes (useful for checking diffusive slope = 1)",
    )
    parser.add_argument(
        "-o", "--output", default=None,
        help="Output image path (default: same name as input, .png extension)",
    )
    args = parser.parse_args()

    plot_msd(args.npy_path, total_ns=args.total_ns,
             loglog=args.loglog, out_path=args.output)


if __name__ == "__main__":
    main()
