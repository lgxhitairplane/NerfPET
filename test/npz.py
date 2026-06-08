import argparse
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_OUTPUT_DIR = "logs/DigitMI2D"
DEFAULT_NPZ_PATH = os.path.join(DEFAULT_OUTPUT_DIR, "reconstruction_image.npz")


def voxel_spacing(axis_values):
    if axis_values.shape[0] < 2:
        return 1.0
    return float(abs(axis_values[1] - axis_values[0]))


def load_volume(npz_path):
    data = np.load(npz_path)
    activity_key = "activity" if "activity" in data else "intensity"
    mu_key = "mu" if "mu" in data else "density"
    return {
        "activity": data[activity_key],
        "mu": data[mu_key],
        "x": data["x"],
        "y": data["y"],
        "z": data["z"],
    }


def save_amide_raw(volume, output_path):
    """Write a float32 raw file with x as the fastest-changing dimension."""
    amide_order = np.asarray(volume.transpose(2, 1, 0), dtype="<f4")
    amide_order.tofile(output_path)
    print("saved", output_path)


def save_amide_import_notes(output_path, shape, spacing):
    nx, ny, nz = shape
    sx, sy, sz = spacing
    text = (
        "AMIDE raw import parameters\n"
        "---------------------------\n"
        "Data type: 32-bit float\n"
        "Byte order: little endian\n"
        "Header size: 0 bytes\n"
        "Dimensions:\n"
        "  x: {nx}\n"
        "  y: {ny}\n"
        "  z: {nz}\n"
        "Voxel size / spacing:\n"
        "  x: {sx:.8g}\n"
        "  y: {sy:.8g}\n"
        "  z: {sz:.8g}\n"
        "File voxel order: x fastest, then y, then z\n"
        "Source numpy order: volume[x, y, z]\n"
    ).format(nx=nx, ny=ny, nz=nz, sx=sx, sy=sy, sz=sz)
    with open(output_path, "w", encoding="ascii") as f:
        f.write(text)
    print("saved", output_path)


def save_slice(volume, axis, index, output_path, title, cmap="hot"):
    if axis == 0:
        image = volume[index, :, :].T
        xlabel, ylabel = "y", "z"
    elif axis == 1:
        image = volume[:, index, :].T
        xlabel, ylabel = "x", "z"
    elif axis == 2:
        image = volume[:, :, index].T
        xlabel, ylabel = "x", "y"
    else:
        raise ValueError("axis must be 0, 1, or 2")

    fig, ax = plt.subplots(figsize=(6, 5), dpi=160)
    im = ax.imshow(image, cmap=cmap, origin="lower")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print("saved", output_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz_path", default=DEFAULT_NPZ_PATH)
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    npz_path = args.npz_path
    if args.output_dir is None:
        output_dir = os.path.dirname(npz_path) or DEFAULT_OUTPUT_DIR
    else:
        output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    volume = load_volume(npz_path)
    activity = volume["activity"]
    mu = volume["mu"]
    x = volume["x"]
    y = volume["y"]
    z = volume["z"]

    print("npz:", npz_path)
    print("activity_shape:", activity.shape)
    print("activity_range:", float(activity.min()), float(activity.max()))
    print("mu_range:", float(mu.min()), float(mu.max()))

    spacing = (voxel_spacing(x), voxel_spacing(y), voxel_spacing(z))
    save_amide_raw(activity, os.path.join(output_dir, "activity_float32_amide.raw"))
    save_amide_raw(mu, os.path.join(output_dir, "mu_float32_amide.raw"))
    save_amide_import_notes(
        os.path.join(output_dir, "amide_raw_import.txt"),
        activity.shape,
        spacing,
    )

    mid = [size // 2 for size in activity.shape]
    save_slice(
        activity,
        0,
        mid[0],
        os.path.join(output_dir, "activity_yz.png"),
        "activity x-mid",
    )
    save_slice(
        activity,
        1,
        mid[1],
        os.path.join(output_dir, "activity_xz.png"),
        "activity y-mid",
    )
    save_slice(
        activity,
        2,
        mid[2],
        os.path.join(output_dir, "activity_xy.png"),
        "activity z-mid",
    )
    save_slice(
        mu,
        2,
        mid[2],
        os.path.join(output_dir, "mu_xy.png"),
        "mu z-mid",
        cmap="viridis",
    )


if __name__ == "__main__":
    main()
