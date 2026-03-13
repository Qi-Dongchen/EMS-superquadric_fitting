import argparse
import json
import sys
import numpy as np
import matplotlib.cm as cm
from EMS.superquadrics import superquadric
from EMS.utilities import visualize


def load_superquadrics(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    quadrics = []
    for entry in data:
        sq = superquadric(entry["shape"], entry["scale"], entry["euler_ZYX"], entry["translation"])
        quadrics.append(sq)
    return quadrics


def main(argv):
    parser = argparse.ArgumentParser(
        description="Visualize recovered superquadrics from a JSON file.")

    parser.add_argument(
        "json_path",
        help="Path to superquadrics.json file."
    )

    parser.add_argument(
        "--arcLength",
        type=float,
        default=0.2,
        help="Arclength (resolution) for rendering. Default is 0.2."
    )

    parser.add_argument(
        "--index",
        type=int,
        nargs="+",
        default=None,
        help="Indices of specific superquadrics to show (e.g. --index 0 2 5). Default: show all."
    )

    args = parser.parse_args(argv)

    quadrics = load_superquadrics(args.json_path)
    print(f"Loaded {len(quadrics)} superquadrics from {args.json_path}")

    indices = args.index if args.index is not None else range(len(quadrics))
    cmap = cm.get_cmap("tab10")

    geometries = []
    for i in indices:
        if i < 0 or i >= len(quadrics):
            print(f"Warning: index {i} out of range (0-{len(quadrics)-1}), skipping.")
            continue
        mesh = quadrics[i].showSuperquadric(arclength=args.arcLength)
        color = cmap(i % 10)[:3]
        mesh.paint_uniform_color(list(color))
        geometries.append(mesh)
        print(f"  [{i}] shape={quadrics[i].shape}, scale={quadrics[i].scale}")

    if geometries:
        visualize(geometries)
    else:
        print("No geometries to display.")


if __name__ == "__main__":
    main(sys.argv[1:])
