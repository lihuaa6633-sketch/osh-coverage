from __future__ import annotations

import io
from pathlib import Path

import numpy as np

from .models import PointCloud


def _ensure_table(data: np.ndarray, path: Path) -> np.ndarray:
    data = np.asarray(data, dtype=np.float64)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.ndim != 2 or data.shape[1] < 3:
        raise ValueError(f"{path}: expected a numeric table with at least three columns")
    return data


def _default_fields(column_count: int) -> tuple[str, ...]:
    return ("x", "y", "z") + tuple(f"field_{index}" for index in range(3, column_count))


def _read_delimited(path: Path, delimiter: str | None) -> PointCloud:
    lines = [
        line.strip()
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not lines:
        raise ValueError(f"{path}: point-cloud file is empty")
    tokens = lines[0].split(delimiter) if delimiter else lines[0].split()
    has_header = False
    try:
        [float(token) for token in tokens]
    except ValueError:
        has_header = True
    fields = tuple(token.strip() for token in tokens) if has_header else ()
    body = "\n".join(lines[1:] if has_header else lines)
    data = _ensure_table(np.loadtxt(io.StringIO(body), delimiter=delimiter), path)
    if not fields:
        fields = _default_fields(data.shape[1])
    return PointCloud(data, fields)


def _read_pcd(path: Path) -> PointCloud:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    header: dict[str, list[str]] = {}
    data_start = None
    for index, raw_line in enumerate(lines):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        key = parts[0].upper()
        header[key] = parts[1:]
        if key == "DATA":
            if not parts[1:] or parts[1].lower() != "ascii":
                raise ValueError(f"{path}: only ASCII PCD files are supported")
            data_start = index + 1
            break
    if data_start is None or "FIELDS" not in header:
        raise ValueError(f"{path}: invalid PCD header")
    base_fields = header["FIELDS"]
    counts = [int(value) for value in header.get("COUNT", ["1"] * len(base_fields))]
    if len(counts) != len(base_fields):
        raise ValueError(f"{path}: PCD FIELDS and COUNT lengths differ")
    fields: list[str] = []
    for name, count in zip(base_fields, counts):
        fields.extend([name] if count == 1 else [f"{name}_{index}" for index in range(count)])
    body = "\n".join(lines[data_start:]).strip()
    if not body:
        raise ValueError(f"{path}: PCD contains no points")
    data = _ensure_table(np.loadtxt(io.StringIO(body)), path)
    if data.shape[1] != len(fields):
        raise ValueError(f"{path}: PCD row width does not match FIELDS/COUNT")
    return PointCloud(data, tuple(fields))


def _read_ply(path: Path) -> PointCloud:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    if not lines or lines[0].strip() != "ply":
        raise ValueError(f"{path}: invalid PLY header")
    vertex_count = None
    fields: list[str] = []
    in_vertex_element = False
    data_start = None
    for index, raw_line in enumerate(lines[1:], start=1):
        parts = raw_line.strip().split()
        if not parts:
            continue
        if parts[0] == "format" and parts[1] != "ascii":
            raise ValueError(f"{path}: only ASCII PLY files are supported")
        if parts[0] == "element":
            in_vertex_element = parts[1] == "vertex"
            if in_vertex_element:
                vertex_count = int(parts[2])
        elif parts[0] == "property" and in_vertex_element:
            if parts[1] == "list":
                raise ValueError(f"{path}: list-valued vertex properties are unsupported")
            fields.append(parts[-1])
        elif parts[0] == "end_header":
            data_start = index + 1
            break
    if data_start is None or vertex_count is None or not fields:
        raise ValueError(f"{path}: incomplete PLY vertex header")
    body = "\n".join(lines[data_start : data_start + vertex_count])
    data = _ensure_table(np.loadtxt(io.StringIO(body)), path)
    if len(data) != vertex_count or data.shape[1] != len(fields):
        raise ValueError(f"{path}: PLY vertex data does not match its header")
    return PointCloud(data, tuple(fields))


def read_point_cloud(path: str | Path) -> PointCloud:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"point cloud not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".npy":
        data = _ensure_table(np.load(path, allow_pickle=False), path)
        return PointCloud(data, _default_fields(data.shape[1]))
    if suffix == ".pcd":
        return _read_pcd(path)
    if suffix == ".ply":
        return _read_ply(path)
    if suffix == ".csv":
        return _read_delimited(path, ",")
    if suffix in {".xyz", ".txt", ".pts"}:
        return _read_delimited(path, None)
    raise ValueError(f"unsupported point-cloud extension {suffix!r}: {path}")


def write_point_cloud(path: str | Path, cloud: PointCloud) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".npy":
        np.save(path, cloud.data, allow_pickle=False)
        return
    if suffix == ".pcd":
        header = "\n".join(
            [
                "# .PCD v0.7 - Point Cloud Data file format",
                "VERSION 0.7",
                f"FIELDS {' '.join(cloud.fields)}",
                f"SIZE {' '.join(['8'] * len(cloud.fields))}",
                f"TYPE {' '.join(['F'] * len(cloud.fields))}",
                f"COUNT {' '.join(['1'] * len(cloud.fields))}",
                f"WIDTH {len(cloud.data)}",
                "HEIGHT 1",
                "VIEWPOINT 0 0 0 1 0 0 0",
                f"POINTS {len(cloud.data)}",
                "DATA ascii",
            ]
        )
        with path.open("w", encoding="ascii", newline="\n") as stream:
            stream.write(header + "\n")
            np.savetxt(stream, cloud.data, fmt="%.10g")
        return
    if suffix == ".ply":
        header_lines = [
            "ply",
            "format ascii 1.0",
            f"element vertex {len(cloud.data)}",
            *[f"property double {field}" for field in cloud.fields],
            "end_header",
        ]
        with path.open("w", encoding="ascii", newline="\n") as stream:
            stream.write("\n".join(header_lines) + "\n")
            np.savetxt(stream, cloud.data, fmt="%.10g")
        return
    if suffix == ".csv":
        with path.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write(",".join(cloud.fields) + "\n")
            np.savetxt(stream, cloud.data, delimiter=",", fmt="%.10g")
        return
    if suffix in {".xyz", ".txt", ".pts"}:
        with path.open("w", encoding="utf-8", newline="\n") as stream:
            np.savetxt(stream, cloud.data, fmt="%.10g")
        return
    raise ValueError(f"unsupported output point-cloud extension {suffix!r}: {path}")
