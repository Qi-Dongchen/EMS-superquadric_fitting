#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import open3d as o3d
import plyfile

TARGET_EXTENT = 20.0  # match turtle.ply scale (~20 units max extent)


def convert_obj_to_ply(input_path: Path, output_path: Path, num_points: int = 6000) -> None:
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    if input_path.suffix.lower() != ".obj":
        raise ValueError(f"Input must be an .obj file: {input_path}")
    if output_path.suffix.lower() != ".ply":
        raise ValueError(f"Output must be a .ply file: {output_path}")

    mesh = o3d.io.read_triangle_mesh(str(input_path))
    if mesh.is_empty():
        raise ValueError(f"Failed to load a valid mesh from: {input_path}")

    pcd = mesh.sample_points_uniformly(number_of_points=num_points)

    # Normalize: center at origin and scale max extent to TARGET_EXTENT
    points = np.asarray(pcd.points)
    centroid = points.mean(axis=0)
    points = points - centroid
    max_extent = (points.max(axis=0) - points.min(axis=0)).max()
    if max_extent > 0:
        scale = TARGET_EXTENT / max_extent
        points = points * scale

    # Write PLY with xyz only (property double), matching turtle.ply format
    vertex_data = np.array(
        [(p[0], p[1], p[2]) for p in points],
        dtype=[('x', 'f8'), ('y', 'f8'), ('z', 'f8')]
    )
    el = plyfile.PlyElement.describe(vertex_data, 'vertex')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plyfile.PlyData([el], text=True).write(str(output_path))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert an OBJ mesh file to a PLY mesh file using Open3D."
    )
    parser.add_argument("input_obj", help="Path to input .obj file")
    parser.add_argument("output_ply", help="Path to output .ply file")
    parser.add_argument("--numPoints", type=int, default=6000,
                        help="Number of points to uniformly sample. Default is 6000.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    convert_obj_to_ply(Path(args.input_obj), Path(args.output_ply), args.numPoints)
    print(f"Converted: {args.input_obj} -> {args.output_ply}")


if __name__ == "__main__":
    main()
