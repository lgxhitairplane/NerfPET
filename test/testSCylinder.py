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


SCYLINDER_MICH = "/home/notfound404/lgxGate/IQ_TrueCoin/mich/SCylinder80.mich"

# The current scanner configuration used by SCylinder80.mich.
testScannerName = "DigitMI930"
testConfig = get_scanner_config(testScannerName)


def load_mich_counts(path, scanner, dtype="float32", header_bytes=0):
    dtype = np.dtype(dtype)
    lor_num = scanner.get_lor_num()
    payload_bytes = os.path.getsize(path) - header_bytes
    expected_bytes = lor_num * dtype.itemsize

    if payload_bytes != expected_bytes:
        raise ValueError(
            f"{path} contains {payload_bytes} payload bytes, but "
            f"selected scanner expects {expected_bytes} bytes "
            f"({lor_num} LORs * {dtype.itemsize} bytes)."
        )

    return np.memmap(
        path,
        dtype=dtype,
        mode="r",
        offset=header_bytes,
        shape=(lor_num,),
    )


def first_nonzero_lor_ids(counts, max_items=32, chunk_size=1_000_000):
    lor_ids = []
    for start in range(0, counts.shape[0], chunk_size):
        stop = min(start + chunk_size, counts.shape[0])
        local = np.flatnonzero(counts[start:stop] > 0)
        if local.size:
            need = max_items - len(lor_ids)
            lor_ids.extend((local[:need] + start).astype(np.int64).tolist())
            if len(lor_ids) >= max_items:
                break
    return np.array(lor_ids, dtype=np.int64)


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


def plot_lor_samples(scanner, lor_ids, counts, output_path):
    crystal1, crystal2 = scanner.get_lor_endpoint_array(lor_ids)

    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")

    for idx, lor_id in enumerate(lor_ids):
        p1 = crystal1[idx]
        p2 = crystal2[idx]
        value = float(counts[lor_id])
        alpha = min(1.0, 0.2 + value / max(float(counts[lor_ids].max()), 1.0))
        ax.plot(
            [p1[0], p2[0]],
            [p1[1], p2[1]],
            [p1[2], p2[2]],
            color="tab:red",
            alpha=alpha,
            linewidth=0.8,
        )

    ax.scatter(crystal1[:, 0], crystal1[:, 1], crystal1[:, 2],
               s=8, color="tab:blue", label="crystal1")
    ax.scatter(crystal2[:, 0], crystal2[:, 1], crystal2[:, 2],
               s=8, color="tab:green", label="crystal2")
    ax.set_title(f"SCylinder80 sampled nonzero LORs ({len(lor_ids)})")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.legend(loc="upper right")
    ax.view_init(elev=24, azim=35)
    set_axes_equal(ax)
    fig.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def inspect_scylinder(
    path=SCYLINDER_MICH,
    config=testConfig,
    scanner_name=testScannerName,
    dtype="float32",
    header_bytes=0,
    sample_count=64,
    output_path=os.path.join(ROOT_DIR, "test", "output", "SCylinder80_lors.png"),
):
    scanner = PETScanner(config)
    counts = load_mich_counts(path, scanner, dtype=dtype,
                              header_bytes=header_bytes)
    lor_ids = first_nonzero_lor_ids(counts, max_items=sample_count)

    print("scanner:", scanner_name)
    print("mich:", path)
    print("dtype:", dtype)
    print("crystal_num:", scanner.get_crystal_num())
    print("crystal_num_one_ring:", scanner.get_crystal_num_one_ring())
    print("ring_num:", scanner.get_ring_num())
    print("lor_num:", scanner.get_lor_num())
    print("file_values:", counts.shape[0])
    print("first_values:", np.asarray(counts[:10]))
    print("sample_nonzero_lor_ids:", lor_ids[:10])

    if lor_ids.size:
        crystal1, crystal2 = scanner.get_lor_endpoint_array(lor_ids[:5])
        print("sample_counts:", np.asarray(counts[lor_ids[:5]]))
        print("sample_crystal1:", crystal1)
        print("sample_crystal2:", crystal2)
        plot_lor_samples(scanner, lor_ids, counts, output_path)
        print("saved_plot:", output_path)
    else:
        print("No nonzero LOR counts found in sampled scan.")

    return scanner, counts, lor_ids


def run_experiment(
    path=SCYLINDER_MICH,
    scanner_name=testScannerName,
    dtype="float32",
    header_bytes=0,
    expname="D930IQ",
    basedir=os.path.join(ROOT_DIR, "logs"),
    n_iters=1000000,
    n_rand=4096,
    n_samples=0,
    n_importance=0,
    sample_step=0.5,
    lrate=5e-4,
    lrate_decay=250,
    netchunk=1024 * 64,
    raw_activation="softplus",
    density_max=0.015,
    bin_cut_ratio=0.25,
    fov_size=None,
    fov_center="0,0,0",
    fov_bounds=None,
    allow_repeated_lor_batch=False,
    i_print=100,
    i_weights=20000,
    random_seed=None,
):
    from PETnerf.run_PET_nerf import train

    argv = [
        "--scanner", scanner_name,
        "--datadir", path,
        "--mich_dtype", dtype,
        "--mich_header_bytes", str(header_bytes),
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
        "--bin_cut_ratio", str(bin_cut_ratio),
        "--i_print", str(i_print),
        "--i_weights", str(i_weights),
    ]
    if allow_repeated_lor_batch:
        argv.append("--allow_repeated_lor_batch")
    if random_seed is not None:
        argv.extend(["--random_seed", str(random_seed)])
    if fov_size is not None:
        argv.extend(["--fov_size", str(fov_size)])
        argv.extend(["--fov_center", str(fov_center)])
    if fov_bounds is not None:
        argv.extend(["--fov_bounds", str(fov_bounds)])

    print("Starting SCylinder PET-NeRF experiment with args:")
    print(" ".join(argv))
    return train(argv)


def export_reconstruction_volume(
    weights_path,
    scanner_name=testScannerName,
    expname="D930IQ",
    basedir=os.path.join(ROOT_DIR, "logs"),
    output_path=None,
    resolution="128",
    bounds=None,
    direction_mode="six",
    netchunk=1024 * 64,
    raw_activation="softplus",
    density_max=0.015,
):
    from PETnerf.run_PET_nerf import train

    argv = [
        "--scanner", scanner_name,
        "--expname", expname,
        "--basedir", basedir,
        "--netchunk", str(netchunk),
        "--export_volume",
        "--ft_weights", weights_path,
        "--volume_resolution", str(resolution),
        "--volume_direction_mode", direction_mode,
        "--raw_activation", raw_activation,
        "--density_max", str(density_max),
    ]
    if output_path is not None:
        argv.extend(["--volume_output", output_path])
    if bounds is not None:
        argv.extend(["--volume_bounds", bounds])

    print("Exporting SCylinder PET-NeRF reconstruction volume with args:")
    print(" ".join(argv))
    return train(argv)


def render_reconstruction_views(
    weights_path,
    scanner_name=testScannerName,
    expname="D930IQ",
    basedir=os.path.join(ROOT_DIR, "logs"),
    output_path=None,
    bounds=None,
    render_frames=40,
    render_topdown_frames=12,
    render_hw="256,256",
    render_fov=45.0,
    render_radius=None,
    render_elevation=20.0,
    render_samples=192,
    render_chunk=1024,
    netchunk=1024 * 64,
    raw_activation="softplus",
    density_max=0.015,
):
    from PETnerf.run_PET_nerf import train

    argv = [
        "--scanner", scanner_name,
        "--expname", expname,
        "--basedir", basedir,
        "--netchunk", str(netchunk),
        "--render_views",
        "--ft_weights", weights_path,
        "--render_frames", str(render_frames),
        "--render_topdown_frames", str(render_topdown_frames),
        "--render_hw", render_hw,
        "--render_fov", str(render_fov),
        "--render_elevation", str(render_elevation),
        "--render_samples", str(render_samples),
        "--render_chunk", str(render_chunk),
        "--raw_activation", raw_activation,
        "--density_max", str(density_max),
    ]
    if output_path is not None:
        argv.extend(["--render_output", output_path])
    if bounds is not None:
        argv.extend(["--volume_bounds", bounds])
    if render_radius is not None:
        argv.extend(["--render_radius", str(render_radius)])

    print("Rendering SCylinder PET-NeRF reconstruction views with args:")
    print(" ".join(argv))
    return train(argv)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "inspect", "export", "render"],
                        default="train")
    parser.add_argument("--path", "--datadir", dest="path",
                        default=SCYLINDER_MICH)
    parser.add_argument("--dtype", "--mich_dtype", dest="dtype",
                        default="float32")
    parser.add_argument("--header_bytes", "--mich_header_bytes",
                        dest="header_bytes", type=int, default=0)
    parser.add_argument("--scanner", default=testScannerName)
    parser.add_argument("--sample_count", type=int, default=64)
    parser.add_argument(
        "--output",
        default=os.path.join(ROOT_DIR, "test", "output", "SCylinder80_lors.png"),
    )
    parser.add_argument("--expname", default="D930IQ")
    parser.add_argument("--basedir", default=os.path.join(ROOT_DIR, "logs"))
    parser.add_argument("--N_iters", type=int, default=1000000)
    parser.add_argument("--N_rand", type=int, default=4096)
    parser.add_argument("--N_samples", type=int, default=0)
    parser.add_argument("--N_importance", type=int, default=0)
    parser.add_argument("--sample_step", type=float, default=0.5)
    parser.add_argument("--lrate", type=float, default=5e-4)
    parser.add_argument("--lrate_decay", type=int, default=250)
    parser.add_argument("--netchunk", type=int, default=1024 * 64)
    parser.add_argument("--raw_activation", choices=["softplus", "relu"],
                        default="softplus")
    parser.add_argument("--density_max", type=float, default=0.015)
    parser.add_argument("--bin_cut_ratio", type=float, default=0.25)
    parser.add_argument("--fov_size", default=None)
    parser.add_argument("--fov_center", default="0,0,0")
    parser.add_argument("--fov_bounds", default=None)
    parser.add_argument("--allow_repeated_lor_batch", action="store_true")
    parser.add_argument("--i_print", type=int, default=100)
    parser.add_argument("--i_weights", type=int, default=20000)
    parser.add_argument("--random_seed", type=int, default=None)
    parser.add_argument("--ft_weights", default=None)
    parser.add_argument("--volume_output", default=None)
    parser.add_argument("--volume_resolution", default="128")
    parser.add_argument("--volume_bounds", default=None)
    parser.add_argument("--volume_direction_mode",
                        choices=["z", "xyz", "six"], default="six")
    parser.add_argument("--render_output", default=None)
    parser.add_argument("--render_frames", type=int, default=40)
    parser.add_argument("--render_topdown_frames", type=int, default=12)
    parser.add_argument("--render_hw", default="256,256")
    parser.add_argument("--render_fov", type=float, default=45.0)
    parser.add_argument("--render_radius", type=float, default=None)
    parser.add_argument("--render_elevation", type=float, default=20.0)
    parser.add_argument("--render_samples", type=int, default=192)
    parser.add_argument("--render_chunk", type=int, default=1024)
    args = parser.parse_args()

    if args.mode == "inspect":
        inspect_scylinder(
            path=args.path,
            config=get_scanner_config(args.scanner),
            scanner_name=args.scanner,
            dtype=args.dtype,
            header_bytes=args.header_bytes,
            sample_count=args.sample_count,
            output_path=args.output,
        )
    elif args.mode == "export":
        if args.ft_weights is None:
            raise ValueError("--ft_weights is required in export mode.")
        export_reconstruction_volume(
            weights_path=args.ft_weights,
            scanner_name=args.scanner,
            expname=args.expname,
            basedir=args.basedir,
            output_path=args.volume_output,
            resolution=args.volume_resolution,
            bounds=args.volume_bounds,
            direction_mode=args.volume_direction_mode,
            netchunk=args.netchunk,
            raw_activation=args.raw_activation,
            density_max=args.density_max,
        )
    elif args.mode == "render":
        if args.ft_weights is None:
            raise ValueError("--ft_weights is required in render mode.")
        render_reconstruction_views(
            weights_path=args.ft_weights,
            scanner_name=args.scanner,
            expname=args.expname,
            basedir=args.basedir,
            output_path=args.render_output,
            bounds=args.volume_bounds,
            render_frames=args.render_frames,
            render_topdown_frames=args.render_topdown_frames,
            render_hw=args.render_hw,
            render_fov=args.render_fov,
            render_radius=args.render_radius,
            render_elevation=args.render_elevation,
            render_samples=args.render_samples,
            render_chunk=args.render_chunk,
            netchunk=args.netchunk,
            raw_activation=args.raw_activation,
            density_max=args.density_max,
        )
    else:
        run_experiment(
            path=args.path,
            scanner_name=args.scanner,
            dtype=args.dtype,
            header_bytes=args.header_bytes,
            expname=args.expname,
            basedir=args.basedir,
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
            bin_cut_ratio=args.bin_cut_ratio,
            fov_size=args.fov_size,
            fov_center=args.fov_center,
            fov_bounds=args.fov_bounds,
            allow_repeated_lor_batch=args.allow_repeated_lor_batch,
            i_print=args.i_print,
            i_weights=args.i_weights,
            random_seed=args.random_seed,
        )


if __name__ == "__main__":
    main()
