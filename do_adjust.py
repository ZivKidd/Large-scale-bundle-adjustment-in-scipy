from __future__ import print_function
import time
from scipy.optimize import least_squares
import numpy as np
from scipy.sparse import lil_matrix
import matplotlib.pyplot as plt

FILE_NAME = "problem-49-7776-pre.txt"
# FILE_NAME = "prob.txt"

def read_bal_data(file_name):
    with open(file_name, "rt") as file:

        # 里面有49张影像 7776个点 共产生了31843次观测（从第i张影像观测到了第j个点则算一次）
        n_cameras, n_points, n_observations = map(
            int, file.readline().split())

        camera_indices = np.empty(n_observations, dtype=int)
        point_indices = np.empty(n_observations, dtype=int)
        points_2d = np.empty((n_observations, 2))

        # 读出观测的对应关系，从第camera_index张影像观测到了第point_index个点，其uv坐标为xy
        for i in range(n_observations):
            camera_index, point_index, x, y = file.readline().split()
            camera_indices[i] = int(camera_index)
            point_indices[i] = int(point_index)
            points_2d[i] = [float(x), float(y)]

        # 每张影像有9个参数,3个旋转，3个平移，1个焦距，2个畸变
        camera_params = np.empty(n_cameras * 9)
        for i in range(n_cameras * 9):
            camera_params[i] = float(file.readline())
        camera_params = camera_params.reshape((n_cameras, -1))

        # 每个点的xyz
        points_3d = np.empty(n_points * 3)
        for i in range(n_points * 3):
            points_3d[i] = float(file.readline())
        points_3d = points_3d.reshape((n_points, -1))

    return camera_params, points_3d, camera_indices, point_indices, points_2d


def rotate(points, rot_vecs):
    """Rotate points by given rotation vectors.

    Rodrigues' rotation formula is used.
    """
    theta = np.linalg.norm(rot_vecs, axis=1)[:, np.newaxis]
    with np.errstate(invalid='ignore'):
        v = rot_vecs / theta
        v = np.nan_to_num(v)
    dot = np.sum(points * v, axis=1)[:, np.newaxis]
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    return cos_theta * points + sin_theta * np.cross(v, points) + dot * (1 - cos_theta) * v


def project(points, camera_params):
    """Convert 3-D points to 2-D by projecting onto images."""
    points_proj = rotate(points, camera_params[:, :3])
    points_proj += camera_params[:, 3:6]
    points_proj = -points_proj[:, :2] / points_proj[:, 2, np.newaxis]
    f = camera_params[:, 6]
    k1 = camera_params[:, 7]
    k2 = camera_params[:, 8]
    n = np.sum(points_proj ** 2, axis=1)
    r = 1 + k1 * n + k2 * n ** 2
    points_proj *= (r * f)[:, np.newaxis]
    return points_proj


def fun(params, n_cameras, n_points, camera_indices, point_indices, points_2d):
    """Compute residuals.

    `params` contains camera parameters and 3-D coordinates.
    """
    camera_params = params[:n_cameras * 9].reshape((n_cameras, 9))
    points_3d = params[n_cameras * 9:].reshape((n_points, 3))
    points_proj = project(points_3d[point_indices], camera_params[camera_indices])
    return (points_proj - points_2d).ravel()


def bundle_adjustment_sparsity(n_cameras, n_points, camera_indices, point_indices):
    # 行，有多少个观测就有*2
    m = camera_indices.size * 2

    # 列，有多少个参数就有，因为参数是个n*1的向量
    n = n_cameras * 9 + n_points * 3

    # m*n的矩阵，n_cameras * 9列为相机参数，n_points * 3为三维点坐标
    A = lil_matrix((m, n), dtype=int)

    i = np.arange(camera_indices.size)
    for s in range(9):
        A[2 * i, camera_indices * 9 + s] = 1
        A[2 * i + 1, camera_indices * 9 + s] = 1

    for s in range(3):
        A[2 * i, n_cameras * 9 + point_indices * 3 + s] = 1
        A[2 * i + 1, n_cameras * 9 + point_indices * 3 + s] = 1

    return A

def prettylist(l):
    return '[%s]' % ', '.join("%4.1e" % f for f in l)


# 共31843个观测
# camera_indices为31843*1，每个代表这次观测对应的像片索引
# point_indices为31843*1，每个代表这次观测对应的三维点索引
# points_2d为31843*2，每个代表这次观测对应的二维点坐标
camera_params, points_3d, camera_indices, point_indices, points_2d = read_bal_data(FILE_NAME)

# Print information
n_cameras = camera_params.shape[0]
n_points = points_3d.shape[0]
# 这些都是参数，即相片的参数和三维点坐标
n = 9 * n_cameras + 3 * n_points
# 这些是残差，即重投影误差
m = 2 * points_2d.shape[0]

print("n_cameras: {}".format(n_cameras))
print("n_points: {}".format(n_points))
print("Total number of parameters: {}".format(n))
print("Total number of residuals: {}".format(m))

x0 = np.hstack((camera_params.ravel(), points_3d.ravel()))
f0 = fun(x0, n_cameras, n_points, camera_indices, point_indices, points_2d)

plt.figure(1)
plt.subplot(211)
plt.plot(f0)

# A代表雅克比矩阵的哪些部分不为0
A = bundle_adjustment_sparsity(n_cameras, n_points, camera_indices, point_indices)

t0 = time.time()
res = least_squares(fun, x0, jac_sparsity=A, verbose=2, x_scale='jac', ftol=1e-4, method='trf',
                    args=(n_cameras, n_points, camera_indices, point_indices, points_2d))
t1 = time.time()

print("Optimization took {0:.0f} seconds".format(t1 - t0))


print('Before:')
print('cam0: {}'.format(prettylist(x0[0:9])))
print('cam1: {}'.format(prettylist(x0[9:18])))

print('After:')
print('cam0: {}'.format(prettylist(res.x[0:9])))
print('cam1: {}'.format(prettylist(res.x[9:18])))


plt.subplot(212)
plt.plot(res.fun)
plt.show()