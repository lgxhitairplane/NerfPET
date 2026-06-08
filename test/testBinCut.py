import argparse
import os
import sys

import numpy as np


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from PETnerf.PETScanner import PETScanner
from PETnerf.config import get_scanner_config


DEFAULT_INPUT_MICH = "/home/notfound404/lgxGate/IQ_TrueCoin/mich/SCylinder80.mich"
DEFAULT_OUTPUT_MICH = os.path.join(ROOT_DIR, "test", "output", "testBinCut.mich")
DEFAULT_SCANNER = "bio4panel"


def load_mich_counts(path, scanner, dtype="float32", header_bytes=0):
    dtype = np.dtype(dtype)
    lor_num = scanner.get_lor_num()
    payload_bytes = os.path.getsize(path) - header_bytes
    expected_bytes = lor_num * dtype.itemsize

    if payload_bytes != expected_bytes:
        raise ValueError(
            f"{path} contains {payload_bytes} payload bytes, but "
            f"{scanner.__class__.__name__} expects {expected_bytes} bytes "
            f"({lor_num} LORs * {dtype.itemsize} bytes)."
        )

    return np.memmap(
        path,
        dtype=dtype,
        mode="r",
        offset=header_bytes,
        shape=(lor_num,),
    )


def write_bincut_mich(
    input_path,
    output_path,
    scanner,
    ratio,
    value=100,
    dtype="float32",
    header_bytes=0,
    zero_outside=False,
    chunk_size=1_000_000,
):
    if not 0 <= ratio < 0.5:
        raise ValueError("ratio must satisfy 0 <= ratio < 0.5")

    dtype = np.dtype(dtype)
    counts = load_mich_counts(
        input_path,
        scanner,
        dtype=dtype,
        header_bytes=header_bytes,
    )

    bin_num = scanner.get_bin_num()
    lor_num = scanner.get_lor_num()
    bin_start = int(np.floor(ratio * bin_num))
    bin_end = int(np.ceil(bin_num - ratio * bin_num))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(input_path, "rb") as src, open(output_path, "wb") as dst:
        if header_bytes:
            dst.write(src.read(header_bytes))
        dst.truncate(header_bytes + lor_num * dtype.itemsize)

    output_counts = np.memmap(
        output_path,
        dtype=dtype,
        mode="r+",
        offset=header_bytes,
        shape=(lor_num,),
    )

    for start in range(0, lor_num, chunk_size):
        stop = min(start + chunk_size, lor_num)
        if zero_outside:
            chunk = np.zeros(stop - start, dtype=dtype)
        else:
            chunk = np.asarray(counts[start:stop]).copy()

        bin_idx = np.arange(start, stop, dtype=np.int64) % bin_num
        mask = (bin_start < bin_idx) & (bin_idx < bin_end)
        chunk[mask] = value
        output_counts[start:stop] = chunk

    output_counts.flush()

    return {
        "input_path": input_path,
        "output_path": output_path,
        "lor_num": lor_num,
        "bin_num": bin_num,
        "bin_start_exclusive": bin_start,
        "bin_end_exclusive": bin_end,
        "changed_bins_per_view_slice": max(bin_end - bin_start - 1, 0),
        "value": dtype.type(value).item(),
        "zero_outside": zero_outside,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Set counts to a fixed value where "
            "ratio * bin_num < binIdx < bin_num - ratio * bin_num."
        )
    )
    parser.add_argument("--input", default=DEFAULT_INPUT_MICH,
                        help="input .mich path")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_MICH,
                        help="output .mich path")
    parser.add_argument("--scanner", default=DEFAULT_SCANNER,
                        help="scanner config name")
    parser.add_argument("--ratio", "--bincut_ratio", type=float, default=0.1,
                        help="bincut ratio, must be in [0, 0.5)")
    parser.add_argument("--value", type=float, default=100,
                        help="count value written to selected bins")
    parser.add_argument("--dtype", default="float32",
                        help="mich payload dtype")
    parser.add_argument("--header_bytes", type=int, default=0,
                        help="bytes to preserve before the count payload")
    parser.add_argument("--zero_outside", action="store_true",
                        help="write 0 outside selected bins instead of copying input counts")
    parser.add_argument("--chunk_size", type=int, default=1_000_000,
                        help="number of LOR counts processed per chunk")
    return parser.parse_args()


def main():
    args = parse_args()
    scanner = PETScanner(get_scanner_config(args.scanner))
    info = write_bincut_mich(
        input_path=args.input,
        output_path=args.output,
        scanner=scanner,
        ratio=args.ratio,
        value=args.value,
        dtype=args.dtype,
        header_bytes=args.header_bytes,
        zero_outside=args.zero_outside,
        chunk_size=args.chunk_size,
    )

    print("saved:", info["output_path"])
    print("input:", info["input_path"])
    print("lor_num:", info["lor_num"])
    print("bin_num:", info["bin_num"])
    print(
        "selected binIdx:",
        f'{info["bin_start_exclusive"]} < binIdx < '
        f'{info["bin_end_exclusive"]}',
    )
    print("changed_bins_per_view_slice:", info["changed_bins_per_view_slice"])
    print("value:", info["value"])
    print("zero_outside:", info["zero_outside"])


if __name__ == "__main__":
    main()
