"""tools/eval/hdr_ply.py — HDR Gaussian-splat PLYs read with nothing but numpy (PLAN.md M5, docs/hdr.md).

The file convention is the standard one: colour(w) = 0.5 + C0 * f_dc + sum_i Y_i(w) f_rest_i. An HDR PLY says in
its header what that colour IS:

    comment lichtfeld_hdr linear        linear radiance, unbounded above 1 (what other applications read)
    comment hdr_scale <s>               P99 luminance of the training images; natural display exposure 1/s
    comment hdr_color_max <cap>         absolute radiance cap the model was trained with
    comment hdr_display_exposure <1/s>  a hint for viewers (linear files only)
    comment lichtfeld_hdr log           LichtFeld's internal log SH (radiance = exp(min(colour, ln cap)))

Subcommands (from tools/: `uv run python eval/hdr_ply.py <cmd> …`):
    info <ply>                                   header, HDR comments, colour statistics
    stamp <in> <out> --mode log|linear --scale S [--color-max C]
                                                 add the HDR comments to a PLY (body untouched) — turns an M2 PLY
                                                 written before M5 (log values as f_dc) into a readable one
    compare <a> <b> [--directions 1024]          radiance of A against B on random directions (each file decoded
                                                 by its own header), relative-error quantiles
    range <ply>                                  min / max of the expansion on the sphere (baked LDR must be [0,1])
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

C0 = 0.28209479177387814
PLY_TYPES = {
    "float": "<f4", "float32": "<f4", "double": "<f8", "float64": "<f8",
    "uchar": "u1", "uint8": "u1", "char": "i1", "int8": "i1",
    "ushort": "<u2", "uint16": "<u2", "short": "<i2", "int16": "<i2",
    "uint": "<u4", "uint32": "<u4", "int": "<i4", "int32": "<i4",
}


# ── reading / writing ────────────────────────────────────────────────────────────────────────────────

def read_header(path: Path) -> tuple[list[str], int]:
    """Header lines (without 'ply' / 'end_header') and the byte offset of the body."""
    with open(path, "rb") as f:
        head = f.read(1 << 20)
    end = head.find(b"end_header\n")
    if end < 0:
        raise ValueError(f"{path}: no end_header in the first MiB")
    text = head[:end].decode("ascii", errors="replace").replace("\r\n", "\n")
    lines = [ln for ln in text.split("\n") if ln.strip()]
    if lines[0] != "ply":
        raise ValueError(f"{path}: not a PLY")
    return lines[1:], end + len(b"end_header\n")


def hdr_comments(lines: list[str]) -> dict:
    out: dict = {"mode": None, "scale": 0.0, "color_max": 0.0, "display_exposure": None}
    for ln in lines:
        if not ln.startswith("comment "):
            continue
        parts = ln.split()
        if len(parts) < 3:
            continue
        key, value = parts[1], parts[2]
        if key == "lichtfeld_hdr":
            out["mode"] = value
        elif key == "hdr_scale":
            out["scale"] = float(value)
        elif key == "hdr_color_max":
            out["color_max"] = float(value)
        elif key == "hdr_display_exposure":
            out["display_exposure"] = float(value)
    return out


def read_ply(path: Path) -> tuple[list[str], dict, np.ndarray]:
    """(header lines, hdr comments, structured vertex array) for a binary_little_endian splat PLY."""
    lines, offset = read_header(path)
    if "format binary_little_endian 1.0" not in lines:
        raise ValueError(f"{path}: only binary_little_endian is handled")
    count = None
    fields = []
    in_vertex = False
    for ln in lines:
        parts = ln.split()
        if parts[0] == "element":
            in_vertex = parts[1] == "vertex"
            if in_vertex:
                count = int(parts[2])
        elif parts[0] == "property" and in_vertex:
            if parts[1] == "list":
                raise ValueError("list properties are not handled")
            fields.append((parts[2], PLY_TYPES[parts[1]]))
    if count is None:
        raise ValueError(f"{path}: no vertex element")
    dtype = np.dtype(fields)
    data = np.fromfile(path, dtype=dtype, count=count, offset=offset)
    if data.shape[0] != count:
        raise ValueError(f"{path}: truncated body ({data.shape[0]} of {count})")
    return lines, hdr_comments(lines), data


def write_ply(path: Path, data: np.ndarray, comments: list[str] | None = None) -> None:
    """Binary little-endian PLY from a structured array (float32 columns)."""
    header = ["ply", "format binary_little_endian 1.0"]
    header += [f"comment {c}" for c in (comments or [])]
    header.append(f"element vertex {data.shape[0]}")
    inv = {v: k for k, v in PLY_TYPES.items()}
    for name in data.dtype.names:
        code = data.dtype[name].str.lstrip("<>|=")
        typ = {"f4": "float", "f8": "double", "u1": "uchar", "i1": "char", "u2": "ushort", "i2": "short",
               "u4": "uint", "i4": "int"}[code]
        header.append(f"property {typ} {name}")
    header.append("end_header")
    with open(path, "wb") as f:
        f.write(("\n".join(header) + "\n").encode("ascii"))
        f.write(np.ascontiguousarray(data).tobytes())


def stamp(src: Path, dst: Path, mode: str, scale: float, color_max: float) -> None:
    """Insert the HDR comments after the format line; the body is copied byte for byte."""
    lines, offset = read_header(src)
    out = ["ply"]
    inserted = False
    for ln in lines:
        if ln.startswith("comment lichtfeld_hdr") or ln.startswith("comment hdr_"):
            continue
        out.append(ln)
        if ln.startswith("format ") and not inserted:
            out += [f"comment lichtfeld_hdr {mode}", f"comment hdr_scale {scale!r}", f"comment hdr_color_max {color_max!r}"]
            if mode == "linear":
                out.append(f"comment hdr_display_exposure {1.0 / scale if scale > 0 else 1.0!r}")
            inserted = True
    out.append("end_header")
    with open(src, "rb") as f:
        f.seek(offset)
        body = f.read()
    with open(dst, "wb") as f:
        f.write(("\n".join(out) + "\n").encode("ascii"))
        f.write(body)


# ── the SH expansion (fastgs ordering, same as kernel_utils.cuh) ─────────────────────────────────────

def sh_columns(data: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """f_dc [N,3], f_rest as [N, K, 3] (PLY stores channel-major: K reds, K greens, K blues), degree."""
    names = data.dtype.names
    dc = np.stack([data[f"f_dc_{i}"] for i in range(3)], axis=-1).astype(np.float64)
    rest_names = [n for n in names if n.startswith("f_rest_")]
    rest_names.sort(key=lambda n: int(n.split("_")[-1]))
    if not rest_names:
        return dc, np.zeros((dc.shape[0], 0, 3)), 0
    K = len(rest_names) // 3
    flat = np.stack([data[n] for n in rest_names], axis=-1).astype(np.float64)  # [N, 3K]
    rest = flat.reshape(dc.shape[0], 3, K).transpose(0, 2, 1)  # [N, K, 3]
    degree = {3: 1, 8: 2, 15: 3}[K]
    return dc, rest, degree


def sh_basis(dirs: np.ndarray, K: int) -> np.ndarray:
    """[D, K] real SH (degree 1..3 bands) in the fastgs ordering and signs."""
    x, y, z = dirs[:, 0], dirs[:, 1], dirs[:, 2]
    cols = []
    if K >= 3:
        cols += [-0.48860251190291987 * y, 0.48860251190291987 * z, -0.48860251190291987 * x]
    if K >= 8:
        xx, yy, zz, xy, xz, yz = x * x, y * y, z * z, x * y, x * z, y * z
        cols += [1.0925484305920792 * xy, -1.0925484305920792 * yz, 0.94617469575755997 * zz - 0.31539156525251999,
                 -1.0925484305920792 * xz, 0.54627421529603959 * xx - 0.54627421529603959 * yy]
    if K >= 15:
        cols += [0.59004358992664352 * y * (-3.0 * xx + yy), 2.8906114426405538 * xy * z,
                 0.45704579946446572 * y * (1.0 - 5.0 * zz), 0.3731763325901154 * z * (5.0 * zz - 3.0),
                 0.45704579946446572 * x * (1.0 - 5.0 * zz), 1.4453057213202769 * z * (xx - yy),
                 0.59004358992664352 * x * (-xx + 3.0 * yy)]
    if not cols:
        return np.zeros((dirs.shape[0], 0))
    return np.stack(cols, axis=-1)


def fibonacci_directions(n: int) -> np.ndarray:
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    phi = np.pi * (3.0 - np.sqrt(5.0)) * np.arange(n)
    return np.stack([r * np.cos(phi), r * np.sin(phi), z], axis=-1)


def random_directions(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(n, 3))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def expansion(dc: np.ndarray, rest: np.ndarray, dirs: np.ndarray) -> np.ndarray:
    """colour(w) = 0.5 + C0 dc + Y(w) · rest  ->  [N, D, 3]."""
    out = (0.5 + C0 * dc)[:, None, :].repeat(dirs.shape[0], axis=1)
    if rest.shape[1] > 0:
        Y = sh_basis(dirs, rest.shape[1])  # [D, K]
        out = out + np.einsum("dk,nkc->ndc", Y, rest)
    return out


def radiance(data: np.ndarray, hdr: dict, dirs: np.ndarray) -> np.ndarray:
    """Linear radiance on `dirs` according to the file's own header ([N, D, 3])."""
    dc, rest, _ = sh_columns(data)
    c = expansion(dc, rest, dirs)
    if hdr["mode"] == "log":
        cap = hdr["color_max"]
        if cap > 0:
            c = np.minimum(c, np.log(cap))
        return np.exp(c)
    return c


def rel_error_quantiles(a: np.ndarray, b: np.ndarray, eps: float) -> dict:
    """Relative error |b - a| / max(|a|, eps), quantiles over every (splat, direction, channel)."""
    err = np.abs(b - a) / np.maximum(np.abs(a), eps)
    return {"median": float(np.median(err)), "p90": float(np.percentile(err, 90)),
            "p99": float(np.percentile(err, 99)), "max": float(err.max())}


# ── CLI ──────────────────────────────────────────────────────────────────────────────────────────────

def cmd_info(args) -> int:
    lines, hdr, data = read_ply(args.ply)
    dc, rest, degree = sh_columns(data)
    print(json.dumps({
        "path": str(args.ply), "vertices": int(data.shape[0]), "bytes": args.ply.stat().st_size,
        "sh_degree": degree, "hdr": hdr,
        "comments": [ln for ln in lines if ln.startswith("comment ")],
        "dc_colour": {"min": float((0.5 + C0 * dc).min()), "max": float((0.5 + C0 * dc).max()),
                      "median": float(np.median(0.5 + C0 * dc))},
        "rest_abs_max": float(np.abs(rest).max()) if rest.size else 0.0,
    }, indent=2))
    return 0


def cmd_stamp(args) -> int:
    stamp(args.src, args.dst, args.mode, args.scale, args.color_max if args.color_max > 0 else 1000.0 * args.scale)
    print(f"stamped {args.dst}: lichtfeld_hdr {args.mode}, hdr_scale {args.scale}")
    return 0


def cmd_compare(args) -> int:
    _, ha, da = read_ply(args.a)
    _, hb, db = read_ply(args.b)
    if da.shape[0] != db.shape[0]:
        print(f"vertex counts differ: {da.shape[0]} vs {db.shape[0]}")
        return 1
    dirs = random_directions(args.directions, args.seed)
    n = min(da.shape[0], args.max_splats)
    idx = np.random.default_rng(args.seed).choice(da.shape[0], size=n, replace=False) if n < da.shape[0] else slice(None)
    ra = radiance(da[idx], ha, dirs)
    rb = radiance(db[idx], hb, dirs)
    report = {
        "a": {"path": str(args.a), "mode": ha["mode"]}, "b": {"path": str(args.b), "mode": hb["mode"]},
        "splats": int(ra.shape[0]), "directions": int(dirs.shape[0]),
        "relative_error": rel_error_quantiles(ra, rb, args.eps),
        "dc_only_relative_error": rel_error_quantiles(ra.mean(axis=1), rb.mean(axis=1), args.eps),
        "negative_fraction_b": float((rb < 0).mean()),
    }
    print(json.dumps(report, indent=2))
    return 0


def cmd_range(args) -> int:
    _, hdr, data = read_ply(args.ply)
    dc, rest, degree = sh_columns(data)
    dirs = random_directions(args.directions, args.seed)
    n = min(data.shape[0], args.max_splats)
    idx = np.random.default_rng(args.seed).choice(data.shape[0], size=n, replace=False) if n < data.shape[0] else slice(None)
    c = expansion(dc[idx], rest[idx], dirs)
    dc_col = 0.5 + C0 * dc
    print(json.dumps({
        "path": str(args.ply), "hdr": hdr, "sh_degree": degree,
        "dc": {"min": float(dc_col.min()), "max": float(dc_col.max())},
        "expansion": {"min": float(c.min()), "max": float(c.max()),
                      "fraction_below_0": float((c < 0).mean()), "fraction_above_1": float((c > 1).mean())},
    }, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("info"); p.add_argument("ply", type=Path); p.set_defaults(fn=cmd_info)
    p = sub.add_parser("stamp"); p.add_argument("src", type=Path); p.add_argument("dst", type=Path)
    p.add_argument("--mode", choices=["log", "linear"], default="log"); p.add_argument("--scale", type=float, required=True)
    p.add_argument("--color-max", type=float, default=0.0); p.set_defaults(fn=cmd_stamp)
    p = sub.add_parser("compare"); p.add_argument("a", type=Path); p.add_argument("b", type=Path)
    p.add_argument("--directions", type=int, default=1024); p.add_argument("--max-splats", type=int, default=20000)
    p.add_argument("--eps", type=float, default=1e-3); p.add_argument("--seed", type=int, default=0); p.set_defaults(fn=cmd_compare)
    p = sub.add_parser("range"); p.add_argument("ply", type=Path); p.add_argument("--directions", type=int, default=512)
    p.add_argument("--max-splats", type=int, default=20000); p.add_argument("--seed", type=int, default=0); p.set_defaults(fn=cmd_range)
    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
