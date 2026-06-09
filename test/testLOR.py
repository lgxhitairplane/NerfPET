import argparse
import os
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle, Rectangle


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from PETnerf.PETScanner import PETScanner  # noqa: E402
from PETnerf.config import get_scanner_config  # noqa: E402


DEFAULT_SCANNER = "DigitMI2D"
DEFAULT_OUTPUT = os.path.join(
    ROOT_DIR, "test", "output", "DigitMI2D_lors.png")


def parse_bounds(value):
    parts = [float(x) for x in str(value).split(",")]
    if len(parts) != 4:
        raise ValueError("Expected xmin,xmax,ymin,ymax.")
    return parts


def parse_cylinders(value):
    if value is None or str(value).strip() == "":
        return []
    cylinders = []
    for item in str(value).split(";"):
        parts = [float(x) for x in item.split(",")]
        if len(parts) not in (3, 4):
            raise ValueError(
                "Expected cylinder as cx,cy,radius[,value], got {}".format(
                    item))
        cx, cy, radius = parts[:3]
        value = parts[3] if len(parts) == 4 else 1.0
        cylinders.append((cx, cy, radius, value))
    return cylinders


def sample_indices(count, max_items=None):
    if max_items is None or max_items <= 0 or max_items >= count:
        return np.arange(count, dtype=np.int64)
    return np.linspace(0, count - 1, max_items, dtype=np.int64)


def equal_xy_limits(ax, xy, radius=None, bounds=None, padding=0.04):
    if bounds is not None:
        xmin, xmax, ymin, ymax = bounds
    else:
        xmin, ymin = xy.min(axis=0)
        xmax, ymax = xy.max(axis=0)
        if radius is not None:
            xmin = min(xmin, -radius)
            xmax = max(xmax, radius)
            ymin = min(ymin, -radius)
            ymax = max(ymax, radius)

    cx = 0.5 * (xmin + xmax)
    cy = 0.5 * (ymin + ymax)
    span = max(xmax - xmin, ymax - ymin)
    span = span * (1.0 + padding)
    ax.set_xlim(cx - 0.5 * span, cx + 0.5 * span)
    ax.set_ylim(cy - 0.5 * span, cy + 0.5 * span)


def plot_lors(
    scanner_name,
    output_path,
    max_lors=None,
    line_alpha=0.035,
    line_width=0.25,
    crystal_size=8.0,
    bounds=None,
    cylinders=None,
    dpi=220,
):
    scanner = PETScanner(get_scanner_config(scanner_name))
    lor_num = scanner.get_lor_num()
    lor_ids = sample_indices(lor_num, max_lors)
    crystal1, crystal2 = scanner.get_lor_endpoint_array(lor_ids)

    xy1 = crystal1[:, :2]
    xy2 = crystal2[:, :2]
    segments = np.stack([xy1, xy2], axis=1)
    centers = scanner.get_all_crystal_centers()
    centers_xy = centers[:, :2]

    radius = scanner.get_radius()
    all_xy = np.concatenate([xy1, xy2, centers_xy], axis=0)

    fig, ax = plt.subplots(figsize=(7.2, 7.2), dpi=dpi)
    lines = LineCollection(
        segments,
        colors=(0.06, 0.24, 0.55, line_alpha),
        linewidths=line_width,
        rasterized=True,
    )
    ax.add_collection(lines)
    ax.scatter(
        centers_xy[:, 0],
        centers_xy[:, 1],
        s=crystal_size,
        c="tab:red",
        linewidths=0,
        label="crystal centers",
        zorder=3,
    )

    ax.add_patch(
        Circle(
            (0.0, 0.0),
            radius,
            fill=False,
            edgecolor="black",
            linewidth=1.0,
            linestyle="--",
            label="scanner radius",
        )
    )
    if bounds is not None:
        xmin, xmax, ymin, ymax = bounds
        ax.add_patch(
            Rectangle(
                (xmin, ymin),
                xmax - xmin,
                ymax - ymin,
                fill=False,
                edgecolor="tab:green",
                linewidth=1.2,
                label="bounds",
            )
        )

    if cylinders:
        values = np.asarray([c[3] for c in cylinders], dtype=np.float32)
        vmin = float(values.min())
        vmax = float(values.max())
        denom = max(vmax - vmin, 1e-6)
        cmap = plt.get_cmap("plasma")
        for idx, (cx, cy, cyl_radius, value) in enumerate(cylinders):
            color = cmap((value - vmin) / denom)
            ax.add_patch(
                Circle(
                    (cx, cy),
                    cyl_radius,
                    fill=True,
                    facecolor=color,
                    edgecolor="black",
                    linewidth=1.0,
                    alpha=0.36,
                    label="demo cylinders" if idx == 0 else None,
                    zorder=2,
                )
            )

    equal_xy_limits(ax, all_xy, radius=radius, bounds=bounds)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_title(
        "{} LORs in XY plane ({} / {})".format(
            scanner_name, len(lor_ids), lor_num))
    ax.grid(True, linewidth=0.4, alpha=0.35)
    ax.legend(loc="upper right")
    fig.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)

    lengths = np.linalg.norm(crystal2 - crystal1, axis=-1)
    print("scanner:", scanner_name)
    print("crystals:", scanner.get_crystal_num())
    print("LORs plotted:", len(lor_ids), "of", lor_num)
    print("scanner_radius:", radius)
    print("endpoint_x_range:", float(all_xy[:, 0].min()), float(all_xy[:, 0].max()))
    print("endpoint_y_range:", float(all_xy[:, 1].min()), float(all_xy[:, 1].max()))
    print("LOR_length_range:", float(lengths.min()), float(lengths.max()))
    if cylinders:
        print("demo_cylinders:", len(cylinders))
    print("saved:", output_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scanner", default=DEFAULT_SCANNER)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--max_lors",
        type=int,
        default=0,
        help="plot at most this many LORs; 0 means all LORs",
    )
    parser.add_argument("--line_alpha", type=float, default=0.035)
    parser.add_argument("--line_width", type=float, default=0.25)
    parser.add_argument("--crystal_size", type=float, default=8.0)
    parser.add_argument(
        "--bounds",
        default=None,
        help="optional xmin,xmax,ymin,ymax box to draw and use as view range",
    )
    parser.add_argument(
        "--demo_cylinders",
        default=None,
        help="semicolon list cx,cy,radius,value;... to overlay demo cylinders",
    )
    parser.add_argument("--dpi", type=int, default=220)
    args = parser.parse_args()

    plot_lors(
        scanner_name=args.scanner,
        output_path=args.output,
        max_lors=args.max_lors,
        line_alpha=args.line_alpha,
        line_width=args.line_width,
        crystal_size=args.crystal_size,
        bounds=parse_bounds(args.bounds) if args.bounds is not None else None,
        cylinders=parse_cylinders(args.demo_cylinders),
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
