import argparse
import os
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from PETnerf.PETScanner import PETScanner
from PETnerf.config import get_scanner_config


DEFAULT_SCANNER = "DigitMI2D"
DEFAULT_SHAPE = (287, 144, 1)


def parse_int3(value):
    parts = [int(x) for x in str(value).split(",")]
    if len(parts) == 1:
        return parts * 3
    if len(parts) != 3:
        raise ValueError("Expected one int or three comma-separated ints.")
    return parts


def parse_float4(value):
    parts = [float(x) for x in str(value).split(",")]
    if len(parts) != 4:
        raise ValueError("Expected xmin,xmax,ymin,ymax.")
    return parts


def parse_cylinder_specs(value):
    """Parse x,y,r,value;... cylinder specs."""
    specs = []
    for item in str(value).split(";"):
        item = item.strip()
        if not item:
            continue
        parts = [float(x) for x in item.split(",")]
        if len(parts) not in (3, 4):
            raise ValueError(
                "Each cylinder expects x,y,r or x,y,r,value.")
        if parts[2] <= 0.:
            raise ValueError("Cylinder radius must be positive.")
        if len(parts) == 3:
            parts.append(1.)
        specs.append(tuple(parts))
    if not specs:
        raise ValueError("At least one cylinder is required.")
    return specs


def ensure_parent_dir(path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def load_projection(path, shape=DEFAULT_SHAPE, dtype="float32", header_bytes=0):
    shape = tuple(int(x) for x in shape)
    dtype = np.dtype(dtype)
    expected_values = int(np.prod(shape))

    if path.endswith(".npy"):
        projection = np.load(path).astype(np.float32)
    elif path.endswith(".npz"):
        data = np.load(path)
        if "sinogram" in data:
            projection = data["sinogram"].astype(np.float32)
        elif "targets" in data:
            projection = data["targets"].astype(np.float32).reshape(shape)
        else:
            projection = data[data.files[0]].astype(np.float32)
    else:
        payload_bytes = os.path.getsize(path) - header_bytes
        expected_bytes = expected_values * dtype.itemsize
        if payload_bytes != expected_bytes:
            raise ValueError(
                "{} contains {} payload bytes, but shape {} with dtype {} "
                "expects {} bytes.".format(
                    path, payload_bytes, shape, dtype, expected_bytes))
        projection = np.memmap(
            path,
            dtype=dtype,
            mode="r",
            offset=header_bytes,
            shape=shape,
        ).astype(np.float32)

    if projection.size != expected_values:
        raise ValueError(
            "Projection has {} values, but shape {} expects {}.".format(
                projection.size, shape, expected_values))
    return projection.reshape(shape).astype(np.float32)


def canonical_line_from_endpoints(crystal1, crystal2):
    xy1 = crystal1[:, :2].astype(np.float64)
    xy2 = crystal2[:, :2].astype(np.float64)
    direction = xy2 - xy1
    length = np.linalg.norm(direction, axis=1)
    if np.any(length <= 0.):
        raise ValueError("Found zero-length LOR.")

    direction = direction / length[:, None]
    normal = np.stack([-direction[:, 1], direction[:, 0]], axis=1)
    theta = np.arctan2(normal[:, 1], normal[:, 0])
    s = np.sum(normal * xy1, axis=1)

    low = theta < 0.
    theta[low] += np.pi
    s[low] *= -1.

    high = theta >= np.pi
    theta[high] -= np.pi
    s[high] *= -1.

    return theta.astype(np.float64), s.astype(np.float64)


def axial_angle_mean(theta):
    """Mean unoriented line-normal angle in [0, pi)."""
    angle = 0.5 * np.arctan2(
        np.mean(np.sin(2. * theta)),
        np.mean(np.cos(2. * theta)))
    if angle < 0.:
        angle += np.pi
    return angle


def digitmi2d_line_geometry(scanner):
    bin_num = scanner.get_bin_num()
    view_num = scanner.get_view_num()
    slice_num = scanner.get_slice_num()
    if slice_num != 1:
        raise ValueError("This simple FBP helper expects a 2D scanner.")

    lor_ids = np.arange(scanner.get_lor_num(), dtype=np.int64)
    crystal1, crystal2 = scanner.get_lor_endpoint_array(lor_ids)
    theta, s = canonical_line_from_endpoints(crystal1, crystal2)

    theta = theta.reshape(slice_num, view_num, bin_num)[0].T
    s = s.reshape(slice_num, view_num, bin_num)[0].T
    view_theta = np.asarray(
        [axial_angle_mean(theta[:, view]) for view in range(view_num)],
        dtype=np.float64)

    for view in range(view_num):
        # Lines are unoriented: (theta, s) and (theta + pi, -s) are identical.
        # Align every bin's signed detector coordinate to the chosen view angle.
        reverse = np.cos(theta[:, view] - view_theta[view]) < 0.
        s[reverse, view] *= -1.

    return view_theta, s


def resample_views_to_uniform_s(sinogram2d, s_by_bin_view, output_bins=None):
    bin_num, view_num = sinogram2d.shape
    if output_bins is None:
        output_bins = bin_num

    s_min = float(np.max(np.min(s_by_bin_view, axis=0)))
    s_max = float(np.min(np.max(s_by_bin_view, axis=0)))
    if s_min >= s_max:
        raise ValueError("Could not find common detector coordinate range.")

    s_uniform = np.linspace(s_min, s_max, output_bins, dtype=np.float64)
    uniform = np.zeros((output_bins, view_num), dtype=np.float32)

    for view in range(view_num):
        order = np.argsort(s_by_bin_view[:, view])
        s_view = s_by_bin_view[order, view]
        p_view = sinogram2d[order, view]
        uniform[:, view] = np.interp(
            s_uniform,
            s_view,
            p_view,
            left=0.,
            right=0.).astype(np.float32)

    return uniform, s_uniform


def project_uniform_cylinder(s_by_bin_view, theta, radius, center=(0., 0.),
                             value=1.):
    """Analytic line integrals through a uniform 2D cylinder.

    For a centered cylinder this is the exact parallel-beam projection:
    p(s) = 2 * value * sqrt(radius^2 - s^2).  For a shifted cylinder, s is
    offset per view by the cylinder center projected onto that view normal.
    """
    cx, cy = center
    bin_num, view_num = s_by_bin_view.shape
    center_offset = cx * np.cos(theta) + cy * np.sin(theta)
    distance = s_by_bin_view - center_offset[None, :]
    inside = np.abs(distance) <= radius
    projection = np.zeros((bin_num, view_num), dtype=np.float32)
    projection[inside] = (
        2. * value * np.sqrt(radius * radius - distance[inside] ** 2)
    )
    return projection


def project_cylinders(s_by_bin_view, theta, cylinder_specs):
    projection = np.zeros_like(s_by_bin_view, dtype=np.float32)
    for cx, cy, radius, value in cylinder_specs:
        projection += project_uniform_cylinder(
            s_by_bin_view,
            theta,
            radius=radius,
            center=(cx, cy),
            value=value)
    return projection


def attenuation_factor_from_cylinders(s_by_bin_view, theta, attenuation_specs):
    """Return PET attenuation factor exp(- integral mu dl) for each LOR."""
    if not attenuation_specs:
        return np.ones_like(s_by_bin_view, dtype=np.float32)
    mu_integral = project_cylinders(s_by_bin_view, theta, attenuation_specs)
    return np.exp(-mu_integral).astype(np.float32)


def apply_attenuation_correction(sinogram2d, attenuation_factor, eps=1e-6):
    return sinogram2d / np.maximum(attenuation_factor, eps)


def ramp_filter(sinogram, ds, filter_name="ramp"):
    n_det, _ = sinogram.shape
    n_fft = max(64, 1 << int(np.ceil(np.log2(2 * n_det))))
    freqs = np.fft.rfftfreq(n_fft, d=ds)
    filt = np.abs(freqs)

    if filter_name == "hann":
        filt *= 0.5 + 0.5 * np.cos(np.pi * freqs / np.max(freqs))
    elif filter_name != "ramp":
        raise ValueError("Unknown filter '{}'. Use ramp or hann.".format(
            filter_name))

    padded = np.zeros((n_fft, sinogram.shape[1]), dtype=np.float32)
    padded[:n_det] = sinogram
    spectrum = np.fft.rfft(padded, axis=0)
    filtered = np.fft.irfft(spectrum * filt[:, None], n=n_fft, axis=0)
    return filtered[:n_det].astype(np.float32)


def filtered_backprojection(filtered_sino, s_uniform, theta, image_size,
                            bounds):
    xmin, xmax, ymin, ymax = bounds
    xs = np.linspace(xmin, xmax, image_size, dtype=np.float32)
    ys = np.linspace(ymin, ymax, image_size, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys, indexing="ij")
    recon = np.zeros_like(xx, dtype=np.float64)

    for view, angle in enumerate(theta):
        detector_coord = xx * np.cos(angle) + yy * np.sin(angle)
        values = np.interp(
            detector_coord.ravel(),
            s_uniform,
            filtered_sino[:, view],
            left=0.,
            right=0.)
        recon += values.reshape(recon.shape)

    recon *= np.pi / (2. * len(theta))
    recon = np.maximum(recon, 0.)
    return recon.astype(np.float32), xs, ys


def save_outputs(recon, xs, ys, output_npz, output_png):
    ensure_parent_dir(output_npz)
    np.savez_compressed(
        output_npz,
        reconstruction=recon,
        activity=recon[:, :, None],
        intensity=recon[:, :, None],
        x=xs,
        y=ys,
        z=np.asarray([0.], dtype=np.float32),
    )
    print("saved", output_npz)

    ensure_parent_dir(output_png)
    fig, ax = plt.subplots(figsize=(6, 5), dpi=180)
    im = ax.imshow(
        recon.T,
        origin="lower",
        cmap="hot",
        extent=[xs[0], xs[-1], ys[0], ys[-1]])
    ax.set_title("DigitMI2D FBP")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(output_png)
    plt.close(fig)
    print("saved", output_png)


def voxel_spacing(axis_values):
    if len(axis_values) < 2:
        return 1.0
    return float(abs(axis_values[1] - axis_values[0]))


def save_amide_raw(volume, output_path):
    """Write a 3D float32 volume with x as the fastest-changing dimension."""
    ensure_parent_dir(output_path)
    amide_order = np.asarray(volume.transpose(2, 1, 0), dtype="<f4")
    amide_order.tofile(output_path)
    print("saved", output_path)


def save_amide_notes(output_path, shape, spacing, title):
    ensure_parent_dir(output_path)
    nx, ny, nz = shape
    sx, sy, sz = spacing
    text = (
        "{title}\n"
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
    ).format(title=title, nx=nx, ny=ny, nz=nz, sx=sx, sy=sy, sz=sz)
    with open(output_path, "w", encoding="ascii") as f:
        f.write(text)
    print("saved", output_path)


def save_sinogram_amide(sinogram2d, s_by_bin_view, s_uniform, theta, args,
                        prefix=""):
    raw_output = add_prefix_to_path(args.sinogram_raw_output, prefix)
    notes_output = add_prefix_to_path(args.sinogram_notes_output, prefix)
    uniform_raw_output = add_prefix_to_path(
        args.uniform_sinogram_raw_output, prefix)
    uniform_notes_output = add_prefix_to_path(
        args.uniform_sinogram_notes_output, prefix)

    sinogram_volume = sinogram2d[:, :, None].astype(np.float32)
    save_amide_raw(sinogram_volume, raw_output)
    save_amide_notes(
        notes_output,
        sinogram_volume.shape,
        (1., 1., 1.),
        "AMIDE raw import parameters for original sinogram")

    uniform_sino, _ = resample_views_to_uniform_s(
        sinogram2d,
        s_by_bin_view,
        output_bins=len(s_uniform))
    uniform_volume = uniform_sino[:, :, None].astype(np.float32)
    save_amide_raw(uniform_volume, uniform_raw_output)
    theta_spacing = voxel_spacing(np.rad2deg(theta))
    save_amide_notes(
        uniform_notes_output,
        uniform_volume.shape,
        (voxel_spacing(s_uniform), theta_spacing, 1.),
        "AMIDE raw import parameters for uniform-s sinogram")


def add_prefix_to_path(path, prefix):
    if not prefix:
        return path
    directory = os.path.dirname(path)
    basename = os.path.basename(path)
    return os.path.join(directory, prefix + basename)


def run_demo_cylinder(args):
    scanner = PETScanner(get_scanner_config(args.scanner))
    theta, s_by_bin_view = digitmi2d_line_geometry(scanner)
    if args.bounds is None:
        radius = scanner.get_radius()
        bounds = [-radius, radius, -radius, radius]
    else:
        bounds = parse_float4(args.bounds)

    if args.demo_cylinders is None:
        center = tuple(float(x) for x in str(args.demo_center).split(","))
        if len(center) != 2:
            raise ValueError("--demo_center expects x,y.")
        cylinder_specs = [(
            center[0],
            center[1],
            args.demo_radius,
            args.demo_value)]
    else:
        cylinder_specs = parse_cylinder_specs(args.demo_cylinders)
    attenuation_specs = (
        parse_cylinder_specs(args.attenuation_cylinders)
        if args.attenuation_cylinders is not None else [])

    activity_sino = project_cylinders(
        s_by_bin_view,
        theta,
        cylinder_specs)
    attenuation_factor = attenuation_factor_from_cylinders(
        s_by_bin_view,
        theta,
        attenuation_specs)
    measured_sino = activity_sino * attenuation_factor
    fbp_input_sino = measured_sino
    if args.attenuation_correction:
        fbp_input_sino = apply_attenuation_correction(
            measured_sino,
            attenuation_factor,
            eps=args.attenuation_eps)

    uniform_sino, s_uniform = resample_views_to_uniform_s(
        fbp_input_sino,
        s_by_bin_view,
        output_bins=args.detector_bins)
    ds = float(abs(s_uniform[1] - s_uniform[0]))
    filtered = ramp_filter(uniform_sino, ds, filter_name=args.filter)
    recon, xs, ys = filtered_backprojection(
        filtered,
        s_uniform,
        theta,
        args.image_size,
        bounds)

    sino_path = os.path.splitext(args.output_npz)[0] + "_sinogram.npz"
    ensure_parent_dir(sino_path)
    np.savez_compressed(
        sino_path,
        sinogram=measured_sino[:, :, None],
        activity_sinogram=activity_sino[:, :, None],
        attenuation_factor=attenuation_factor[:, :, None],
        fbp_input_sinogram=fbp_input_sino[:, :, None],
        uniform_sinogram=uniform_sino,
        s=s_uniform,
        theta=theta,
        cylinders=np.asarray(cylinder_specs, dtype=np.float32),
        attenuation_cylinders=np.asarray(attenuation_specs, dtype=np.float32),
    )
    print("saved", sino_path)
    save_sinogram_amide(measured_sino, s_by_bin_view, s_uniform, theta, args)
    if attenuation_specs:
        save_sinogram_amide(
            activity_sino,
            s_by_bin_view,
            s_uniform,
            theta,
            args,
            prefix="activity_")
        save_sinogram_amide(
            attenuation_factor,
            s_by_bin_view,
            s_uniform,
            theta,
            args,
            prefix="attenuation_factor_")

    print("demo: analytic cylinder")
    print("scanner:", args.scanner)
    print("cylinders:", cylinder_specs)
    print("attenuation_cylinders:", attenuation_specs)
    print("attenuation_correction:", args.attenuation_correction)
    print("activity_sinogram_range:", float(activity_sino.min()),
          float(activity_sino.max()))
    print("measured_sinogram_range:", float(measured_sino.min()),
          float(measured_sino.max()))
    print("attenuation_factor_range:", float(attenuation_factor.min()),
          float(attenuation_factor.max()))
    print("image_shape:", recon.shape)
    print("image_range:", float(recon.min()), float(recon.max()))
    save_outputs(recon, xs, ys, args.output_npz, args.output_png)


def run_fbp(args):
    scanner = PETScanner(get_scanner_config(args.scanner))
    scanner_shape = (
        scanner.get_bin_num(),
        scanner.get_view_num(),
        scanner.get_slice_num())
    shape = tuple(parse_int3(args.shape))
    if shape != scanner_shape:
        raise ValueError(
            "Input shape {} does not match scanner shape {}.".format(
                shape, scanner_shape))

    input_path = args.mich if args.mich is not None else args.input
    projection = load_projection(
        input_path,
        shape=shape,
        dtype=args.mich_dtype,
        header_bytes=args.mich_header_bytes)
    sinogram2d = projection[:, :, 0]
    if args.log_transform:
        sinogram2d = -np.log(
            np.maximum(sinogram2d, args.log_epsilon) / np.max(sinogram2d))

    theta, s_by_bin_view = digitmi2d_line_geometry(scanner)
    attenuation_specs = (
        parse_cylinder_specs(args.attenuation_cylinders)
        if args.attenuation_cylinders is not None else [])
    attenuation_factor = attenuation_factor_from_cylinders(
        s_by_bin_view,
        theta,
        attenuation_specs)
    fbp_input_sino = sinogram2d
    if args.attenuation_correction:
        fbp_input_sino = apply_attenuation_correction(
            sinogram2d,
            attenuation_factor,
            eps=args.attenuation_eps)

    uniform_sino, s_uniform = resample_views_to_uniform_s(
        fbp_input_sino,
        s_by_bin_view,
        output_bins=args.detector_bins)
    ds = float(abs(s_uniform[1] - s_uniform[0]))
    filtered = ramp_filter(uniform_sino, ds, filter_name=args.filter)

    if args.bounds is None:
        radius = scanner.get_radius()
        bounds = [-radius, radius, -radius, radius]
    else:
        bounds = parse_float4(args.bounds)

    recon, xs, ys = filtered_backprojection(
        filtered,
        s_uniform,
        theta,
        args.image_size,
        bounds)

    print("scanner:", args.scanner)
    print("scanner_shape:", scanner_shape)
    print("input:", input_path)
    print("sinogram_range:", float(sinogram2d.min()), float(sinogram2d.max()))
    print("attenuation_cylinders:", attenuation_specs)
    print("attenuation_correction:", args.attenuation_correction)
    print("fbp_input_range:", float(fbp_input_sino.min()),
          float(fbp_input_sino.max()))
    print("theta_range_deg:", float(np.rad2deg(theta.min())),
          float(np.rad2deg(theta.max())))
    print("s_range:", float(s_uniform[0]), float(s_uniform[-1]))
    print("image_shape:", recon.shape)
    print("image_range:", float(recon.min()), float(recon.max()))
    save_sinogram_amide(fbp_input_sino, s_by_bin_view, s_uniform, theta, args)
    if attenuation_specs:
        save_sinogram_amide(
            attenuation_factor,
            s_by_bin_view,
            s_uniform,
            theta,
            args,
            prefix="attenuation_factor_")
    save_outputs(recon, xs, ys, args.output_npz, args.output_png)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mich", default=None,
                        help="DigitMI2D .mich projection file")
    parser.add_argument("--input", "--sinogram", "--path", default=None,
                        help="projection file; supports .mich/raw, .npy, or .npz")
    parser.add_argument("--scanner", default=DEFAULT_SCANNER)
    parser.add_argument("--shape", default="287,144,1")
    parser.add_argument("--mich_dtype", "--dtype", dest="mich_dtype",
                        default="float32")
    parser.add_argument("--mich_header_bytes", "--header_bytes",
                        dest="mich_header_bytes", type=int, default=0)
    parser.add_argument("--image_size", type=int, default=320)
    parser.add_argument("--detector_bins", type=int, default=None)
    parser.add_argument("--bounds", default=None,
                        help="xmin,xmax,ymin,ymax; default uses scanner radius")
    parser.add_argument("--filter", choices=["ramp", "hann"], default="ramp")
    parser.add_argument("--demo_cylinder", action="store_true",
                        help="forward-project analytic cylinder(s) and FBP them")
    parser.add_argument("--demo_radius", type=float, default=60.)
    parser.add_argument("--demo_center", default="0,0")
    parser.add_argument("--demo_value", type=float, default=1.)
    parser.add_argument(
        "--demo_cylinders",
        default=None,
        help=("semicolon-separated x,y,r,value list; default None uses "
              "--demo_center/--demo_radius/--demo_value"))
    parser.add_argument(
        "--attenuation_cylinders",
        default=None,
        help=("semicolon-separated x,y,r,mu list in 1/mm. "
              "Example water cylinder: '0,0,180,0.0096'"))
    parser.add_argument(
        "--attenuation_correction",
        action="store_true",
        help="divide sinogram by exp(-integral mu dl) before FBP")
    parser.add_argument("--attenuation_eps", type=float, default=1e-6)
    parser.add_argument("--log_transform", action="store_true",
                        help="use -log(I/I0), for transmission CT data only")
    parser.add_argument("--log_epsilon", type=float, default=1e-6)
    parser.add_argument(
        "--output_npz",
        default=os.path.join(ROOT_DIR, "logs", DEFAULT_SCANNER, "fbp_image.npz"))
    parser.add_argument(
        "--output_png",
        default=os.path.join(ROOT_DIR, "logs", DEFAULT_SCANNER, "fbp_image.png"))
    parser.add_argument(
        "--sinogram_raw_output",
        default=os.path.join(
            ROOT_DIR, "logs", DEFAULT_SCANNER,
            "sinogram_float32_amide.raw"))
    parser.add_argument(
        "--sinogram_notes_output",
        default=os.path.join(
            ROOT_DIR, "logs", DEFAULT_SCANNER,
            "sinogram_amide_raw_import.txt"))
    parser.add_argument(
        "--uniform_sinogram_raw_output",
        default=os.path.join(
            ROOT_DIR, "logs", DEFAULT_SCANNER,
            "uniform_sinogram_float32_amide.raw"))
    parser.add_argument(
        "--uniform_sinogram_notes_output",
        default=os.path.join(
            ROOT_DIR, "logs", DEFAULT_SCANNER,
            "uniform_sinogram_amide_raw_import.txt"))
    args = parser.parse_args()
    if args.demo_cylinder:
        run_demo_cylinder(args)
        return
    if args.mich is None and args.input is None:
        raise ValueError("Pass --mich path/to/file.mich or --input path.")
    run_fbp(args)


if __name__ == "__main__":
    main()
