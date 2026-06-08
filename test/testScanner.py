import argparse
import os
import sys

import numpy as np


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

os.environ.setdefault(
    "MPLCONFIGDIR", os.path.join(ROOT_DIR, "test", "output", "mplconfig"))

import matplotlib.pyplot as plt

from PETnerf.PETScanner import PETScanner
from PETnerf.config import get_scanner_config


def set_axes_equal(ax):
    limits = np.array([
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d(),
    ])
    centers = limits.mean(axis=1)
    radius = 0.5 * np.max(limits[:, 1] - limits[:, 0])

    ax.set_xlim3d([centers[0] - radius, centers[0] + radius])
    ax.set_ylim3d([centers[1] - radius, centers[1] + radius])
    ax.set_zlim3d([centers[2] - radius, centers[2] + radius])


def visualize_scanner(scanner_name, output_path=None, show=False):
    scanner = PETScanner(get_scanner_config(scanner_name))
    centers = scanner.get_all_crystal_centers()

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    ring_num = scanner.get_ring_num()
    crystal_num_one_ring = scanner.get_crystal_num_one_ring()
    ring_ids = np.arange(scanner.get_crystal_num()) // crystal_num_one_ring

    scatter = ax.scatter(
        centers[:, 0],
        centers[:, 1],
        centers[:, 2],
        c=ring_ids,
        cmap="viridis",
        s=1.2,
        alpha=0.85,
        linewidths=0,
    )

    ax.set_title(
        f"{scanner_name} crystal centers "
        f"({scanner.get_crystal_num()} crystals, {ring_num} rings)"
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.view_init(elev=22, azim=38)
    set_axes_equal(ax)

    cbar = fig.colorbar(scatter, ax=ax, shrink=0.7, pad=0.08)
    cbar.set_label("ring id")
    fig.tight_layout()

    if output_path is not None:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path, dpi=220)
        print(f"Saved scanner visualization to {output_path}")

    print(f"scanner: {scanner_name}")
    print(f"crystal_num_one_ring: {scanner.get_crystal_num_one_ring()}")
    print(f"ring_num: {scanner.get_ring_num()}")
    print(f"bin_num: {scanner.get_bin_num()}")
    print(f"view_num: {scanner.get_view_num()}")
    print(f"slice_num: {scanner.get_slice_num()}")
    print(f"lor_num: {scanner.get_lor_num()}")
    print(f"total_crystal_num: {scanner.get_crystal_num()}")
    print(f"first_center: {centers[0]}")
    print(f"last_center: {centers[-1]}")

    lor_id = 1
    crystal_id1, crystal_id2 = scanner.get_crystal_id_from_lor_id(lor_id)
    crystal1 = scanner.get_crystal_center_from_global_id(crystal_id1)
    crystal2 = scanner.get_crystal_center_from_global_id(crystal_id2)
    print(f"lor_id: {lor_id}")
    print(f"lor_crystal_ids: {crystal_id1}, {crystal_id2}")
    print(f"lor_crystal1_center: {crystal1}")
    print(f"lor_crystal2_center: {crystal2}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scanner", default="DigitMI2D")
    parser.add_argument(
        "--output",
        default=os.path.join(ROOT_DIR, "test", "output", "DigitMI930.png"),
    )
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    visualize_scanner(args.scanner, args.output, args.show)


if __name__ == "__main__":
    main()
