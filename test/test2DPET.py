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


DEFAULT_SCANNER = "DigitMI2D"
DEFAULT_EXPNAME = "DigitMI2D"
DEFAULT_SINOGRAM_SHAPE = (287, 144, 1)


def parse_int3(value):
    parts = [int(x) for x in str(value).split(",")]
    if len(parts) == 1:
        return parts * 3
    if len(parts) != 3:
        raise ValueError("Expected one int or three comma-separated ints.")
    return parts


def parse_float6(value):
    parts = [float(x) for x in str(value).split(",")]
    if len(parts) != 6:
        raise ValueError("Expected xmin,xmax,ymin,ymax,zmin,zmax.")
    return parts


def scanner_shape(scanner):
    return scanner.get_bin_num(), scanner.get_view_num(), scanner.get_slice_num()


def ensure_parent_dir(path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def load_sinogram(path, shape=DEFAULT_SINOGRAM_SHAPE, dtype="float32",
                  header_bytes=0):
    dtype = np.dtype(dtype)
    shape = tuple(int(x) for x in shape)
    expected_values = int(np.prod(shape))

    if path.endswith(".npy"):
        sino = np.load(path).astype(np.float32)
    elif path.endswith(".npz"):
        data = np.load(path)
        key = "sinogram" if "sinogram" in data else data.files[0]
        sino = data[key].astype(np.float32)
    else:
        payload_bytes = os.path.getsize(path) - header_bytes
        expected_bytes = expected_values * dtype.itemsize
        if payload_bytes != expected_bytes:
            raise ValueError(
                "{} contains {} payload bytes, but shape {} with dtype {} "
                "expects {} bytes.".format(
                    path, payload_bytes, shape, dtype, expected_bytes))
        sino = np.memmap(
            path,
            dtype=dtype,
            mode="r",
            offset=header_bytes,
            shape=shape,
        ).astype(np.float32)

    if sino.size != expected_values:
        raise ValueError(
            "Sinogram has {} values, but requested shape {} has {}.".format(
                sino.size, shape, expected_values))
    return np.reshape(sino, shape).astype(np.float32)


def sinogram_to_pet_npz(sinogram, scanner, output_path,
                        scanner_name=DEFAULT_SCANNER):
    expected_shape = scanner_shape(scanner)
    if tuple(sinogram.shape) != expected_shape:
        raise ValueError(
            "Sinogram shape {} does not match scanner shape {} "
            "(bin, view, slice).".format(sinogram.shape, expected_shape))

    targets = np.ascontiguousarray(sinogram.reshape([-1]), dtype=np.float32)
    lor_ids = np.arange(targets.shape[0], dtype=np.int64)
    crystal1, crystal2 = scanner.get_lor_endpoint_array(lor_ids)

    ensure_parent_dir(output_path)
    np.savez_compressed(
        output_path,
        crystal1=crystal1.astype(np.float32),
        crystal2=crystal2.astype(np.float32),
        targets=targets,
        sinogram_shape=np.asarray(sinogram.shape, dtype=np.int32),
        scanner=np.asarray(scanner_name),
    )
    return output_path


def prepare_npz(path, scanner_name=DEFAULT_SCANNER, shape=DEFAULT_SINOGRAM_SHAPE,
                dtype="float32", header_bytes=0, output_path=None):
    scanner = PETScanner(get_scanner_config(scanner_name))
    sinogram = load_sinogram(
        path,
        shape=shape,
        dtype=dtype,
        header_bytes=header_bytes)

    if output_path is None:
        output_path = os.path.join(
            ROOT_DIR, "logs", DEFAULT_EXPNAME, "sinogram_lors.npz")

    npz_path = sinogram_to_pet_npz(
        sinogram,
        scanner,
        output_path,
        scanner_name=scanner_name)
    print("Prepared 2D PET npz")
    print("  scanner:", scanner_name)
    print("  sinogram:", path)
    print("  shape:", sinogram.shape)
    print("  output:", npz_path)
    print("  target range:", float(sinogram.min()), float(sinogram.max()))
    return npz_path


def plot_sinogram(path, scanner_name=DEFAULT_SCANNER,
                  shape=DEFAULT_SINOGRAM_SHAPE, dtype="float32",
                  header_bytes=0, output_path=None):
    scanner = PETScanner(get_scanner_config(scanner_name))
    sinogram = load_sinogram(
        path,
        shape=shape,
        dtype=dtype,
        header_bytes=header_bytes)
    expected_shape = scanner_shape(scanner)
    if tuple(sinogram.shape) != expected_shape:
        raise ValueError(
            "Sinogram shape {} does not match scanner shape {}.".format(
                sinogram.shape, expected_shape))

    if output_path is None:
        output_path = os.path.join(
            ROOT_DIR, "test", "output", "DigitMI2D_sinogram.png")
    ensure_parent_dir(output_path)

    image = sinogram[:, :, 0]
    fig, ax = plt.subplots(figsize=(9, 6))
    im = ax.imshow(image, origin="lower", aspect="auto", cmap="magma")
    ax.set_title("DigitMI2D sinogram")
    ax.set_xlabel("view")
    ax.set_ylabel("bin")
    fig.colorbar(im, ax=ax, label="count")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)

    print("scanner:", scanner_name)
    print("sinogram:", path)
    print("shape:", sinogram.shape)
    print("scanner_shape:", expected_shape)
    print("saved_plot:", output_path)
    print("range:", float(sinogram.min()), float(sinogram.max()))


def run_experiment(
    path,
    scanner_name=DEFAULT_SCANNER,
    shape=DEFAULT_SINOGRAM_SHAPE,
    dtype="float32",
    header_bytes=0,
    expname=DEFAULT_EXPNAME,
    basedir=os.path.join(ROOT_DIR, "logs"),
    npz_output=None,
    n_iters=200000,
    n_rand=2048,
    n_samples=0,
    n_importance=0,
    sample_step=1.0,
    lrate=5e-4,
    lrate_decay=250,
    netchunk=1024 * 64,
    raw_activation="softplus",
    density_max=0.015,
    fov_size=None,
    fov_center="0,0,0",
    fov_bounds=None,
    i_print=100,
    i_weights=20000,
    random_seed=None,
):
    from PETnerf.run_PET_nerf import train

    scanner = PETScanner(get_scanner_config(scanner_name))
    if npz_output is None:
        npz_output = os.path.join(basedir, expname, "sinogram_lors.npz")
    datadir = prepare_npz(
        path,
        scanner_name=scanner_name,
        shape=shape,
        dtype=dtype,
        header_bytes=header_bytes,
        output_path=npz_output)

    argv = [
        "--scanner", scanner_name,
        "--datadir", datadir,
        "--expname", expname,
        "--basedir", basedir,
        "--N_iters", str(n_iters),
        "--N_rand", str(n_rand),
        "--N_samples", str(n_samples),
        "--N_importance", str(n_importance),
        "--sample_step", str(sample_step),
        "--lrate", str(lrate),
        "--lrate_decay", str(lrate_decay),
        "--netchunk", str(netchunk),
        "--raw_activation", raw_activation,
        "--density_max", str(density_max),
        "--i_print", str(i_print),
        "--i_weights", str(i_weights),
    ]
    if random_seed is not None:
        argv.extend(["--random_seed", str(random_seed)])
    if fov_size is not None:
        argv.extend(["--fov_size", str(fov_size)])
        argv.extend(["--fov_center", str(fov_center)])
    if fov_bounds is not None:
        argv.extend(["--fov_bounds", str(fov_bounds)])

    print("Starting DigitMI2D PET-NeRF experiment with args:")
    print(" ".join(argv))
    print("scanner_shape:", scanner_shape(scanner))
    return train(argv)


def export_reconstruction_image(
    weights_path,
    scanner_name=DEFAULT_SCANNER,
    expname=DEFAULT_EXPNAME,
    basedir=os.path.join(ROOT_DIR, "logs"),
    output_path=None,
    resolution="320,320,1",
    bounds=None,
    netchunk=1024 * 64,
    raw_activation="softplus",
    density_max=0.015,
):
    from PETnerf.run_PET_nerf import train

    resolution_parts = parse_int3(resolution)
    if resolution_parts[2] != 1:
        raise ValueError("2D export expects z resolution 1, got {}".format(
            resolution))

    if bounds is None:
        radius = PETScanner(get_scanner_config(scanner_name)).get_radius()
        bounds = "{},{},{},{},-0.5,0.5".format(-radius, radius, -radius, radius)
    else:
        parse_float6(bounds)

    argv = [
        "--scanner", scanner_name,
        "--expname", expname,
        "--basedir", basedir,
        "--netchunk", str(netchunk),
        "--export_volume",
        "--ft_weights", weights_path,
        "--volume_resolution", str(resolution),
        "--volume_bounds={}".format(bounds),
        "--volume_direction_mode", "z",
        "--raw_activation", raw_activation,
        "--density_max", str(density_max),
    ]
    if output_path is not None:
        argv.extend(["--volume_output", output_path])

    print("Exporting DigitMI2D PET-NeRF reconstruction image with args:")
    print(" ".join(argv))
    return train(argv)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["train", "inspect", "prepare", "export"],
        default="train")
    parser.add_argument("--path", "--sinogram", dest="path", required=False)
    parser.add_argument("--scanner", default=DEFAULT_SCANNER)
    parser.add_argument("--shape", default="287,144,1")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--header_bytes", type=int, default=0)
    parser.add_argument("--output", default=None)
    parser.add_argument("--npz_output", default=None)
    parser.add_argument("--expname", default=DEFAULT_EXPNAME)
    parser.add_argument("--basedir", default=os.path.join(ROOT_DIR, "logs"))
    parser.add_argument("--N_iters", type=int, default=200000)
    parser.add_argument("--N_rand", type=int, default=2048)
    parser.add_argument("--N_samples", type=int, default=0)
    parser.add_argument("--N_importance", type=int, default=0)
    parser.add_argument("--sample_step", type=float, default=1.0)
    parser.add_argument("--lrate", type=float, default=5e-4)
    parser.add_argument("--lrate_decay", type=int, default=250)
    parser.add_argument("--netchunk", type=int, default=1024 * 64)
    parser.add_argument("--raw_activation", choices=["softplus", "relu"],
                        default="softplus")
    parser.add_argument("--density_max", type=float, default=0.015)
    parser.add_argument("--fov_size", default=None)
    parser.add_argument("--fov_center", default="0,0,0")
    parser.add_argument("--fov_bounds", default=None)
    parser.add_argument("--i_print", type=int, default=100)
    parser.add_argument("--i_weights", type=int, default=20000)
    parser.add_argument("--random_seed", type=int, default=None)
    parser.add_argument("--ft_weights", default=None)
    parser.add_argument("--image_output", "--volume_output",
                        dest="image_output", default=None)
    parser.add_argument("--image_resolution", "--volume_resolution",
                        dest="image_resolution", default="320,320,1")
    parser.add_argument("--image_bounds", "--volume_bounds",
                        dest="image_bounds", default=None)
    args = parser.parse_args()

    shape = tuple(parse_int3(args.shape))

    if args.mode in ["train", "inspect", "prepare"] and args.path is None:
        raise ValueError("--path/--sinogram is required for this mode.")

    if args.mode == "inspect":
        plot_sinogram(
            args.path,
            scanner_name=args.scanner,
            shape=shape,
            dtype=args.dtype,
            header_bytes=args.header_bytes,
            output_path=args.output)
    elif args.mode == "prepare":
        prepare_npz(
            args.path,
            scanner_name=args.scanner,
            shape=shape,
            dtype=args.dtype,
            header_bytes=args.header_bytes,
            output_path=args.npz_output)
    elif args.mode == "export":
        if args.ft_weights is None:
            raise ValueError("--ft_weights is required in export mode.")
        export_reconstruction_image(
            args.ft_weights,
            scanner_name=args.scanner,
            expname=args.expname,
            basedir=args.basedir,
            output_path=args.image_output,
            resolution=args.image_resolution,
            bounds=args.image_bounds,
            netchunk=args.netchunk,
            raw_activation=args.raw_activation,
            density_max=args.density_max)
    else:
        run_experiment(
            args.path,
            scanner_name=args.scanner,
            shape=shape,
            dtype=args.dtype,
            header_bytes=args.header_bytes,
            expname=args.expname,
            basedir=args.basedir,
            npz_output=args.npz_output,
            n_iters=args.N_iters,
            n_rand=args.N_rand,
            n_samples=args.N_samples,
            n_importance=args.N_importance,
            sample_step=args.sample_step,
            lrate=args.lrate,
            lrate_decay=args.lrate_decay,
            netchunk=args.netchunk,
            raw_activation=args.raw_activation,
            density_max=args.density_max,
            fov_size=args.fov_size,
            fov_center=args.fov_center,
            fov_bounds=args.fov_bounds,
            i_print=args.i_print,
            i_weights=args.i_weights,
            random_seed=args.random_seed)


if __name__ == "__main__":
    main()
