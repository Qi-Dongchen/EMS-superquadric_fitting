import argparse
import sys
import numpy as np
import matplotlib.cm as cm
from EMS.EMS_recovery import EMS_recovery
from EMS.utilities import read_point_cloud, showPoints, visualize, save_superquadrics
from sklearn.cluster import DBSCAN

def hierarchical_ems(
    point,
    OutlierRatio=0.9,           # prior outlier probability [0, 1) (default: 0.1)
    MaxIterationEM=20,           # maximum number of EM iterations (default: 20)
    ToleranceEM=1e-3,            # absolute tolerance of EM (default: 1e-3)
    RelativeToleranceEM=2e-1,    # relative tolerance of EM (default: 1e-1)
    MaxOptiIterations=2,         # maximum number of optimization iterations per M (default: 2)
    Sigma=0.3,                   # initial sigma^2 (default: 0 - auto generate)
    MaxiSwitch=2,                # maximum number of switches allowed (default: 2)
    AdaptiveUpperBound=True,    # Introduce adaptive upper bound to restrict the volume of SQ (default: false)
    Rescale=False,                # normalize the input point cloud (default: true)
    MaxLayer=5,                  # maximum depth
    Eps=1.7,                    # IMPORTANT: varies based on the size of the input pointcoud (DBScan parameter)
    MinPoints=60,               # DBScan parameter required minimum points
):
    # Auto-scale Eps based on point cloud size if not provided
    if Eps is None:
        extent = np.max(point, axis=0) - np.min(point, axis=0)
        Eps = 0.08 * np.max(extent)
        print(f"Auto Eps: {Eps:.4f} (max extent: {np.max(extent):.4f})")

    point_seg = {key: [] for key in list(range(0, MaxLayer+1))}
    point_outlier = {key: [] for key in list(range(0, MaxLayer+1))}
    point_seg[0] = [point]
    list_quadrics = []
    quadric_count = 1
    for h in range(MaxLayer):
        for c in range(len(point_seg[h])):
            # skip clusters that are too small for reliable fitting
            if len(point_seg[h][c]) < MinPoints:
                continue
            print(f"Counting number of generated quadrics: {quadric_count}")
            quadric_count += 1
            try:
                x_raw, p_raw = EMS_recovery(
                    point_seg[h][c],
                    OutlierRatio,
                    MaxIterationEM,
                    ToleranceEM,
                    RelativeToleranceEM,
                    MaxOptiIterations,
                    Sigma,
                    MaxiSwitch,
                    AdaptiveUpperBound,
                    Rescale,
                )
            except ValueError as e:
                print(f"  Skipping cluster (layer {h}, idx {c}): {e}")
                continue
            point_previous = point_seg[h][c]
            list_quadrics.append(x_raw)
            outlier = point_seg[h][c][p_raw < 0.1, :]
            point_seg[h][c] = point_seg[h][c][p_raw > 0.1, :]
            if np.sum(p_raw) < (0.8 * len(point_previous)):
                clustering = DBSCAN(eps=Eps, min_samples=MinPoints).fit(outlier)
                labels = list(set(clustering.labels_))
                labels = [item for item in labels if item >= 0]
                if len(labels) >= 1:
                    for i in range(len(labels)):
                        point_seg[h + 1].append(outlier[clustering.labels_ == i])
                point_outlier[h].append(outlier[clustering.labels_ == -1])
            else:
                point_outlier[h].append(outlier)
    return point_seg, point_outlier, list_quadrics


def main(argv):
    parser = argparse.ArgumentParser(
        description='Hierarchical EMS recovery of multiple superquadrics from a point cloud.')

    parser.add_argument(
        'path_to_data',
        help='Path to the point cloud file (*.ply, *.obj, or *.glb).'
    )

    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Visualize the recovered superquadrics and the input point cloud.'
    )

    parser.add_argument(
        '--arcLength',
        type=float,
        default=0.2,
        help='Arclength (resolution) for rendering superquadrics. Default is 0.2.'
    )

    parser.add_argument(
        '--export',
        action='store_true',
        help='Export recovered superquadrics to data/<ply_name>/superquadrics.json.'
    )

    parser.add_argument(
        '--numPoints',
        type=int,
        default=2000,
        help='Number of points to uniformly sample from mesh surfaces (obj/glb). Default is 2000.'
    )

    args = parser.parse_args(argv)

    point_cloud = read_point_cloud(args.path_to_data, num_points=args.numPoints)
    point_seg, point_outlier, list_quadrics = hierarchical_ems(point_cloud)

    if args.export:
        from pathlib import Path
        ply_stem = Path(args.path_to_data).stem
        output_dir = Path(__file__).resolve().parent.parent / "data" / ply_stem
        save_superquadrics(list_quadrics, str(output_dir))

    if args.visualize:
        geometries = []
        cmap = cm.get_cmap('tab10')
        for i, quadric in enumerate(list_quadrics):
            mesh = quadric.showSuperquadric(arclength=args.arcLength)
            color = cmap(i % 10)[:3]
            mesh.paint_uniform_color(list(color))
            geometries.append(mesh)
        geometries.append(showPoints(point_cloud))
        visualize(geometries)


if __name__ == "__main__":
    main(sys.argv[1:])