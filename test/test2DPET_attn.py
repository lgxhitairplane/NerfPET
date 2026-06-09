import argparse
import os
import sys


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from test2DPET import (  # noqa: E402
    DEFAULT_SCANNER,
    DEFAULT_SINOGRAM_SHAPE,
    ensure_parent_dir,
    parse_float6,
    parse_int3,
    plot_sinogram,
    prepare_npz,
)
from PETnerf.PETScanner import PETScanner  # noqa: E402
from PETnerf.config import get_scanner_config  # noqa: E402


DEFAULT_EXPNAME = "DigitMI2D_attn"


def parse_int2_or_3(value):
    parts = [int(x) for x in str(value).split(",")]
    if len(parts) not in (2, 3):
        raise ValueError("Expected height,width or height,width,1.")
    return parts


def parse_float4(value):
    parts = [float(x) for x in str(value).split(",")]
    if len(parts) != 4:
        raise ValueError("Expected xmin,xmax,ymin,ymax.")
    return parts


def default_attn_bounds(scanner_name):
    radius = PETScanner(get_scanner_config(scanner_name)).get_radius()
    return "{},{},{},{}".format(-radius, radius, -radius, radius)


def join_negative_option_values(argv, option_names):
    """Allow argparse options whose comma-list value starts with a minus sign."""
    fixed = []
    i = 0
    option_names = set(option_names)
    while i < len(argv):
        if (
            argv[i] in option_names
            and i + 1 < len(argv)
            and argv[i + 1].startswith("-")
            and "," in argv[i + 1]
        ):
            fixed.append("{}={}".format(argv[i], argv[i + 1]))
            i += 2
        else:
            fixed.append(argv[i])
            i += 1
    return fixed


def add_attn_args(argv, attn_image, attn_shape=None, attn_dtype="float32",
                  attn_header_bytes=0, attn_bounds=None,
                  scanner_name=DEFAULT_SCANNER,
                  attn_activity_soft_mask=False):
    if attn_image is None:
        raise ValueError("--attn_image is required for test2DPET_attn.")

    if attn_bounds is None:
        attn_bounds = default_attn_bounds(scanner_name)
    else:
        parse_float4(attn_bounds)

    argv.extend([
        "--attn_image", attn_image,
        "--attn_dtype", attn_dtype,
        "--attn_header_bytes", str(attn_header_bytes),
        "--attn_bounds={}".format(attn_bounds),
    ])
    if attn_shape is not None:
        parse_int2_or_3(attn_shape)
        argv.extend(["--attn_shape", str(attn_shape)])
    if attn_activity_soft_mask:
        argv.append("--attn_activity_soft_mask")
    return argv


def run_experiment(
    path,
    attn_image,
    scanner_name=DEFAULT_SCANNER,
    shape=DEFAULT_SINOGRAM_SHAPE,
    dtype="float32",
    header_bytes=0,
    attn_shape=None,
    attn_dtype="float32",
    attn_header_bytes=0,
    attn_bounds=None,
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
    netdepth=4,
    netwidth=128,
    netdepth_fine=4,
    netwidth_fine=128,
    net_skips="2",
    netchunk=1024 * 64,
    raw_activation="softplus",
    attn_activity_soft_mask=False,
    fov_size=None,
    fov_center="0,0,0",
    fov_bounds=None,
    i_print=100,
    i_weights=20000,
    random_seed=None,
):
    from PETnerf.run_PET_nerf import train

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
        "--netdepth", str(netdepth),
        "--netwidth", str(netwidth),
        "--netdepth_fine", str(netdepth_fine),
        "--netwidth_fine", str(netwidth_fine),
        "--net_skips", str(net_skips),
        "--netchunk", str(netchunk),
        "--raw_activation", raw_activation,
        "--i_print", str(i_print),
        "--i_weights", str(i_weights),
    ]
    add_attn_args(
        argv,
        attn_image,
        attn_shape=attn_shape,
        attn_dtype=attn_dtype,
        attn_header_bytes=attn_header_bytes,
        attn_bounds=attn_bounds,
        scanner_name=scanner_name,
        attn_activity_soft_mask=attn_activity_soft_mask)
    if random_seed is not None:
        argv.extend(["--random_seed", str(random_seed)])
    if fov_size is not None:
        argv.extend(["--fov_size", str(fov_size)])
        argv.extend(["--fov_center", str(fov_center)])
    if fov_bounds is not None:
        parse_float6(fov_bounds)
        argv.extend(["--fov_bounds", str(fov_bounds)])

    print("Starting DigitMI2D PET-NeRF fixed-attn experiment with args:")
    print(" ".join(argv))
    return train(argv)


def export_reconstruction_image(
    weights_path,
    attn_image,
    scanner_name=DEFAULT_SCANNER,
    expname=DEFAULT_EXPNAME,
    basedir=os.path.join(ROOT_DIR, "logs"),
    output_path=None,
    resolution="320,320,1",
    bounds=None,
    netchunk=1024 * 64,
    netdepth=4,
    netwidth=128,
    netdepth_fine=4,
    netwidth_fine=128,
    net_skips="2",
    raw_activation="softplus",
    attn_activity_soft_mask=False,
    attn_shape=None,
    attn_dtype="float32",
    attn_header_bytes=0,
    attn_bounds=None,
):
    from PETnerf.run_PET_nerf import train

    resolution_parts = parse_int3(resolution)
    if resolution_parts[2] != 1:
        raise ValueError("2D export expects z resolution 1, got {}".format(
            resolution))

    if bounds is None:
        radius = PETScanner(get_scanner_config(scanner_name)).get_radius()
        bounds = "{},{},{},{},-0.5,0.5".format(
            -radius, radius, -radius, radius)
    else:
        parse_float6(bounds)

    argv = [
        "--scanner", scanner_name,
        "--expname", expname,
        "--basedir", basedir,
        "--netchunk", str(netchunk),
        "--netdepth", str(netdepth),
        "--netwidth", str(netwidth),
        "--netdepth_fine", str(netdepth_fine),
        "--netwidth_fine", str(netwidth_fine),
        "--net_skips", str(net_skips),
        "--export_volume",
        "--ft_weights", weights_path,
        "--volume_resolution", str(resolution),
        "--volume_bounds={}".format(bounds),
        "--volume_direction_mode", "z",
        "--raw_activation", raw_activation,
    ]
    add_attn_args(
        argv,
        attn_image,
        attn_shape=attn_shape,
        attn_dtype=attn_dtype,
        attn_header_bytes=attn_header_bytes,
        attn_bounds=attn_bounds,
        scanner_name=scanner_name,
        attn_activity_soft_mask=attn_activity_soft_mask)
    if output_path is not None:
        ensure_parent_dir(output_path)
        argv.extend(["--volume_output", output_path])

    print("Exporting DigitMI2D fixed-attn reconstruction image with args:")
    print(" ".join(argv))
    return train(argv)


def main():
    sys.argv[1:] = join_negative_option_values(
        sys.argv[1:],
        ["--attn_bounds", "--fov_bounds", "--image_bounds", "--volume_bounds"])

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["train", "inspect", "prepare", "export"],
        default="train")
    parser.add_argument("--path", "--sinogram", dest="path", required=False)
    parser.add_argument("--attn_image", "--mumap", dest="attn_image",
                        default=None)
    parser.add_argument("--attn_shape", default=None)
    parser.add_argument("--attn_dtype", default="float32")
    parser.add_argument("--attn_header_bytes", type=int, default=0)
    parser.add_argument("--attn_bounds", default=None)
    parser.add_argument("--attn_activity_soft_mask", action="store_true")
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
    parser.add_argument("--netdepth", type=int, default=4)
    parser.add_argument("--netwidth", type=int, default=128)
    parser.add_argument("--netdepth_fine", type=int, default=4)
    parser.add_argument("--netwidth_fine", type=int, default=128)
    parser.add_argument("--net_skips", default="2")
    parser.add_argument("--netchunk", type=int, default=1024 * 64)
    parser.add_argument("--raw_activation", choices=["softplus", "relu"],
                        default="softplus")
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
    if args.mode in ["train", "export"] and args.attn_image is None:
        raise ValueError("--attn_image/--mumap is required for this mode.")

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
            args.attn_image,
            scanner_name=args.scanner,
            expname=args.expname,
            basedir=args.basedir,
            output_path=args.image_output,
            resolution=args.image_resolution,
            bounds=args.image_bounds,
            netchunk=args.netchunk,
            netdepth=args.netdepth,
            netwidth=args.netwidth,
            netdepth_fine=args.netdepth_fine,
            netwidth_fine=args.netwidth_fine,
            net_skips=args.net_skips,
            raw_activation=args.raw_activation,
            attn_activity_soft_mask=args.attn_activity_soft_mask,
            attn_shape=args.attn_shape,
            attn_dtype=args.attn_dtype,
            attn_header_bytes=args.attn_header_bytes,
            attn_bounds=args.attn_bounds)
    else:
        run_experiment(
            args.path,
            args.attn_image,
            scanner_name=args.scanner,
            shape=shape,
            dtype=args.dtype,
            header_bytes=args.header_bytes,
            attn_shape=args.attn_shape,
            attn_dtype=args.attn_dtype,
            attn_header_bytes=args.attn_header_bytes,
            attn_bounds=args.attn_bounds,
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
            netdepth=args.netdepth,
            netwidth=args.netwidth,
            netdepth_fine=args.netdepth_fine,
            netwidth_fine=args.netwidth_fine,
            net_skips=args.net_skips,
            netchunk=args.netchunk,
            raw_activation=args.raw_activation,
            attn_activity_soft_mask=args.attn_activity_soft_mask,
            fov_size=args.fov_size,
            fov_center=args.fov_center,
            fov_bounds=args.fov_bounds,
            i_print=args.i_print,
            i_weights=args.i_weights,
            random_seed=args.random_seed)


if __name__ == "__main__":
    main()
