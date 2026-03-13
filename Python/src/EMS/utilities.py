import json
import os
import open3d as o3d
import numpy as np
import plyfile

def showSuperquadrics(x, threshold = 1e-2, num_limit = 10000, arclength = 0.02):
    # avoid numerical instability in sampling
    if x.shape[0] < 0.007:
        x.shape[0] = 0.007
    if x.shape[1] < 0.007:
        x.shape[1] = 0.007
    # sampling points in superellipse
    point_eta = uniformSampledSuperellipse(x.shape[0], [1, x.scale[2]], threshold, num_limit, arclength)
    point_omega = uniformSampledSuperellipse(x.shape[1], [x.scale[0], x.scale[1]], threshold, num_limit, arclength)

    # preallocate meshgrid
    M = np.shape(point_omega)[1]
    N = np.shape(point_eta)[1]
    x_mesh = np.ones((M, N))
    y_mesh = np.ones((M, N))
    z_mesh = np.ones((M, N))

    for m in range(M):
        for n in range(N):
            point_temp = np.zeros(3)
            point_temp[0 : 2] = point_omega[:, m] * point_eta[0, n]
            point_temp[2] = point_eta[1, n]
            point_temp = x.RotM @ point_temp + x.translation

            x_mesh[m, n] = point_temp[0]
            y_mesh[m, n] = point_temp[1]
            z_mesh[m, n] = point_temp[2]

    # Build triangle mesh from the grid
    vertices = np.stack([x_mesh.ravel(), y_mesh.ravel(), z_mesh.ravel()], axis=1)
    triangles = []
    for m in range(M - 1):
        for n in range(N - 1):
            idx = m * N + n
            triangles.append([idx, idx + 1, idx + N])
            triangles.append([idx + 1, idx + N + 1, idx + N])

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(np.array(triangles))
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color([0.0, 0.0, 1.0])
    return mesh



def uniformSampledSuperellipse(epsilon, scale, threshold = 1e-2, num_limit = 10000, arclength = 0.02):

    # initialize array storing sampled theta
    theta = np.zeros(num_limit)
    theta[0] = 0

    for i in range(num_limit):
        dt = dtheta(theta[i], arclength, threshold, scale, epsilon)
        theta_temp = theta[i] + dt

        if theta_temp > np.pi / 4:
            theta[i + 1] = np.pi / 4
            break
        else:
            if i + 1 < num_limit:
                theta[i + 1] = theta_temp
            else:
                raise Exception(
                'Number of the sampled points exceed the preset limit', \
                num_limit,
                'Please decrease the sampling arclength.'
                )
    critical = i + 1

    for j in range(critical + 1, num_limit):
        dt = dtheta(theta[j], arclength, threshold, np.flip(scale), epsilon)
        theta_temp = theta[j] + dt
        
        if theta_temp > np.pi / 4:
            break
        else:
            if j + 1 < num_limit:
                theta[j + 1] = theta_temp
            else:
                raise Exception(
                'Number of the sampled points exceed the preset limit', \
                num_limit,
                'Please decrease the sampling arclength.'
                )
    num_pt = j
    theta = theta[0 : num_pt + 1]

    point_fw = angle2points(theta[0 : critical + 1], scale, epsilon)
    point_bw = np.flip(angle2points(theta[critical + 1: num_pt + 1], np.flip(scale), epsilon), (0, 1))
    point = np.concatenate((point_fw, point_bw), 1)
    point = np.concatenate((point, np.flip(point[:, 0 : num_pt], 1) * np.array([[-1], [1]]), 
                           point[:, 1 : num_pt + 1] * np.array([[-1], [-1]]),
                           np.flip(point[:, 0 : num_pt], 1) * np.array([[1], [-1]])), 1)

    return point

def dtheta(theta, arclength, threshold, scale, epsilon):
    # calculation the sampling step size
    if theta < threshold:
        dt = np.abs(np.power(arclength / scale[1] +np.power(theta, epsilon), \
             (1 / epsilon)) - theta)
    else:
        dt = arclength / epsilon * ((np.cos(theta) ** 2 * np.sin(theta) ** 2) /
             (scale[0] ** 2 * np.cos(theta) ** (2 * epsilon) * np.sin(theta) ** 4 +
             scale[1] ** 2 * np.sin(theta) ** (2 * epsilon) * np.cos(theta) ** 4)) ** (1 / 2)
    
    return dt

def angle2points(theta, scale, epsilon):

    point = np.zeros((2, np.shape(theta)[0]))
    point[0] = scale[0] * np.sign(np.cos(theta)) * np.abs(np.cos(theta)) ** epsilon
    point[1] = scale[1] * np.sign(np.sin(theta)) * np.abs(np.sin(theta)) ** epsilon

    return point


def read_ply(path_to_file):
    # read points from a .ply file and store in an nparray
    plydata = plyfile.PlyData.read(path_to_file)
    pc = plydata['vertex']
    return np.column_stack((pc['x'], pc['y'], pc['z']))


def _sample_mesh_uniformly(mesh, num_points):
    # sample points uniformly from an Open3D triangle mesh surface
    pcd = mesh.sample_points_uniformly(number_of_points=num_points)
    return np.asarray(pcd.points)


def read_obj(path_to_file, num_points=2000):
    # read a .obj mesh and uniformly sample points from its surface
    mesh = o3d.io.read_triangle_mesh(path_to_file)
    if mesh.is_empty():
        raise ValueError(f"Failed to read OBJ file: {path_to_file}")
    return _sample_mesh_uniformly(mesh, num_points)


def read_glb(path_to_file, num_points=2000):
    # read a .glb/.gltf mesh and uniformly sample points from its surface
    import trimesh
    scene = trimesh.load(path_to_file)
    if isinstance(scene, trimesh.Scene):
        tm = scene.to_mesh()
    else:
        tm = scene
    if len(tm.vertices) == 0:
        raise ValueError(f"Failed to read GLB file or no vertices found: {path_to_file}")
    # convert to Open3D mesh for uniform sampling
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(np.asarray(tm.vertices))
    mesh.triangles = o3d.utility.Vector3iVector(np.asarray(tm.faces))
    return _sample_mesh_uniformly(mesh, num_points)


def read_point_cloud(path_to_file, num_points=2000):
    # auto-detect format by extension and read point cloud
    ext = os.path.splitext(path_to_file)[1].lower()
    if ext == '.ply':
        return read_ply(path_to_file)
    elif ext == '.obj':
        return read_obj(path_to_file, num_points)
    elif ext in ('.glb', '.gltf'):
        return read_glb(path_to_file, num_points)
    else:
        raise ValueError(f"Unsupported file format '{ext}'. Supported: .ply, .obj, .glb, .gltf")


def showPoints(point, color=[1.0, 0.0, 0.0]):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point[:, :3])
    pcd.paint_uniform_color(color)
    return pcd


def visualize(geometries):
    o3d.visualization.draw_geometries(geometries)


def save_superquadrics(list_quadrics, output_dir):
    from scipy.spatial.transform import Rotation as R
    os.makedirs(output_dir, exist_ok=True)
    data = []
    for i, sq in enumerate(list_quadrics):
        data.append({
            "id": i,
            "shape": sq.shape.tolist(),
            "scale": sq.scale.tolist(),
            "euler_ZYX": sq.euler.tolist(),
            "translation": sq.translation.tolist(),
            "quaternion_xyzw": sq.quat.tolist(),
        })
    # compute relative poses between adjacent superquadrics (i -> i+1)
    relative_poses = []
    for i in range(len(list_quadrics) - 1):
        sq_a = list_quadrics[i]
        sq_b = list_quadrics[i + 1]
        # relative rotation: R_rel = R_b^T @ R_a
        R_rel = sq_b.RotM.T @ sq_a.RotM
        # relative translation: t_rel = R_b^T @ (t_a - t_b)
        t_rel = sq_b.RotM.T @ (sq_a.translation - sq_b.translation)
        r_rel = R.from_matrix(R_rel)
        relative_poses.append({
            "from": i,
            "to": i + 1,
            "relative_euler_ZYX": r_rel.as_euler('ZYX').tolist(),
            "relative_quaternion_xyzw": r_rel.as_quat().tolist(),
            "relative_translation": t_rel.tolist(),
        })
    output = {
        "superquadrics": data,
        "relative_poses": relative_poses,
    }
    path = os.path.join(output_dir, "superquadrics.json")
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved {len(data)} superquadrics and {len(relative_poses)} relative poses to {path}")