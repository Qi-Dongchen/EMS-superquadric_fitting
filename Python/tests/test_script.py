import argparse
import sys
import numpy as np

from EMS.utilities import read_point_cloud, showPoints, visualize
from EMS.EMS_recovery import EMS_recovery

import timeit

def main(argv):

    parser = argparse.ArgumentParser(
        description='Probabilistic Recovery of a superquadric surface from a point cloud file (*.ply, *.obj, or *.glb).')

    parser.add_argument(
        'path_to_data',
        help='Path to the point cloud file (*.ply, *.obj, or *.glb).'
    )

    parser.add_argument(
        '--visualize',
        action = 'store_true',
        help='Visualize the recoverd superquadric and the input point cloud.'
    )

    parser.add_argument(
        '--runtime',
        action = 'store_true',
        help='Show the runtime.'
    )

    parser.add_argument(
        '--result',
        action = 'store_true',       
        help='Print the recovered superquadric parameter.'
    )

    parser.add_argument(
        '--outlierRatio',
        type = float,
        default = 0.2,       
        help='Set the prior outlier ratio. Default is 0.2.'
    )

    parser.add_argument(
        '--adaptiveUpperBound',
        action = 'store_true',       
        help='Implemet addaptive upper bound to limit the volume of the superquadric.'
    )

    parser.add_argument(
        '--arcLength',
        type = float,
        default = 0.2,       
        help='Set the arclength (resolution) for rendering the superquadric. Default is 0.2.'
    )

    parser.add_argument(
        '--pointSize',
        type = float,
        default = 0.1,
        help='Set the point size for plotting the point cloud. Default is 0.2.'
    )

    parser.add_argument(
        '--numPoints',
        type = int,
        default = 2000,
        help='Number of points to uniformly sample from mesh surfaces (obj/glb). Default is 2000.'
    )

    args = parser.parse_args(argv)

    print('----------------------------------------------------')
    print('Loading point cloud from: ', args.path_to_data, '...')
    point = read_point_cloud(args.path_to_data, num_points=args.numPoints)
    print('Point cloud loaded.')
    print('----------------------------------------------------')

    # first run to eliminate jit compiling time
    sq_recovered, p = EMS_recovery(point)

    start = timeit.default_timer()
    sq_recovered, p = EMS_recovery(point, 
                                   OutlierRatio=args.outlierRatio, 
                                   AdaptiveUpperBound=args.adaptiveUpperBound
                      )
    stop = timeit.default_timer()
    print('Superquadric Recovered.')
    if args.runtime is True:
        print('Runtime: ', (stop - start) * 1000, 'ms')
    print('----------------------------------------------------')
    
    if args.result is True:
        print('shape =', sq_recovered.shape)
        print('scale =', sq_recovered.scale)
        print('euler =', sq_recovered.euler)
        print('translation =', sq_recovered.translation)
        print('----------------------------------------------------')
    
    if args.visualize is True:
        geometries = []
        geometries.append(sq_recovered.showSuperquadric(arclength=args.arcLength))
        geometries.append(showPoints(point))
        visualize(geometries)


if __name__ == "__main__":
    main(sys.argv[1:])