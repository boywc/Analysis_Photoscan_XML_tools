import numpy as np
from PhotoscanXMLAnalyse import *
import cv2

def xmlpos_to_mathpos(pos):
    """
    将 Photoscan XML 外参格式矩阵转换为数学常用坐标系下的外参格式。

    Args:
        pos (ndarray): 原始外参矩阵，shape=[3,4]。

    Returns:
        ndarray: 转换后外参矩阵，shape=[3,4]。
    """
    return_board = np.zeros_like(pos)
    return_board[:, :3] = pos[:, :3]
    x = pos[0, 3]
    y = pos[1, 3]
    z = pos[2, 3]
    return_board[0, 3] = y
    return_board[1, 3] = z
    return_board[2, 3] = x
    return return_board

def camera_calibration(cor_3D_A, cor_2D_A, ror3_zf=[1, 1]):
    """
    基于 2D-3D 同名点对，利用 DLT 算法估算相机内外参数。

    Args:
        cor_3D_A (ndarray): 三维点坐标，shape=[N,3]。
        cor_2D_A (ndarray): 二维像素坐标，shape=[N,2]。
        ror3_zf (list): [ro_zf, r3_zf]，方向选择因子，用于纠正可能的方向歧义。

    Returns:
        K (ndarray): 相机内参矩阵，shape=[3,3]。
        pose (ndarray): 外参（旋转+平移），shape=[3,4]。
    """
    ro_zf, r3_zf = ror3_zf

    pad_1 = np.ones([len(cor_3D_A), 1], dtype=np.float32)
    cor_3D_A = np.hstack((cor_3D_A, pad_1))

    zero_board = np.zeros_like(cor_3D_A)
    line_1 = np.hstack((cor_3D_A, zero_board))
    line_2 = np.hstack((zero_board, cor_3D_A))
    line_1_2 = np.hstack((line_1, line_2))
    line_1_2 = np.reshape(line_1_2, (len(cor_3D_A) * 2, 8))

    # 构造 DLT 方程右侧项
    line_1b = -1 * cor_2D_A[:, 0]
    line_1b = np.expand_dims(line_1b, axis=-1)
    line_1b = line_1b * cor_3D_A
    line_2b = -1 * cor_2D_A[:, 1]
    line_2b = np.expand_dims(line_2b, axis=-1)
    line_2b = line_2b * cor_3D_A
    line_1_2b = np.hstack((line_1b, line_2b))
    line_1_2b = np.reshape(line_1_2b, (len(cor_3D_A) * 2, 4))

    # 拼接最终的 DLT 矩阵
    P = np.hstack((line_1_2, line_1_2b))
    U, Sigma, VT = np.linalg.svd(P)
    V = VT.T
    m = V[:, -1]
    M = np.reshape(m, (3, 4))

    # 分解获得内外参
    A = M[:3, :3]
    b = M[:3, 3]
    ro = ro_zf / np.linalg.norm(A[2, :])
    cx = ro * ro * np.sum(A[0, :] * A[2, :])
    cy = ro * ro * np.sum(A[1, :] * A[2, :])

    cos_on = np.dot(np.cross(A[0, :], A[2, :]), np.cross(A[1, :], A[2, :]))
    cos_down = np.linalg.norm(np.cross(A[0, :], A[2, :])) * np.linalg.norm(np.cross(A[1, :], A[2, :]))
    cos = -1 * cos_on / cos_down
    sin = np.sin(np.arccos(cos))
    alpha = ro * ro * sin * np.linalg.norm(np.cross(A[0, :], A[2, :]))
    beta = ro * ro * sin * np.linalg.norm(np.cross(A[1, :], A[2, :]))

    K = np.array([[alpha,   -1 * alpha / np.tan(np.arccos(cos)), cx],
                  [0    ,   beta / sin                         , cy],
                  [0    ,   0                                  , 1]], dtype=np.float32)

    r1 = np.cross(A[1, :], A[2, :]) / np.linalg.norm(np.cross(A[1, :], A[2, :]))
    r3 = ro_zf * A[2, :] / np.linalg.norm(A[2, :])
    r2 = np.cross(r3, r1)

    R = np.array([r1, r2, r3], dtype=np.float32)
    t = np.dot(ro * np.linalg.inv(K), b)
    pose = np.hstack((R, np.expand_dims(t, -1)))

    return K, pose

def camera_calibration_with_verify(cor_3D_A_train, cor_2D_A_train):
    """
    对同名点进行多方向穷举，自动选取投影重投影误差最小的相机内外参。

    Args:
        cor_3D_A_train (ndarray): 三维点，shape=[N,3]。
        cor_2D_A_train (ndarray): 二维点，shape=[N,2]。

    Returns:
        min_loss (float): 最小重投影误差。
        min_k (ndarray): 最优内参矩阵。
        min_p (ndarray): 最优外参（旋转+平移）。
    """
    loss_list = []
    K_list = []
    P_list = []
    for cs in [[1, 1], [1, -1], [-1, 1], [-1, -1]]:
        K, P = camera_calibration(cor_3D_A_train, cor_2D_A_train, ror3_zf=cs)
        get_test_2D = point_3D_projection_to_camera(K, P, cor_3D_A_train)
        loss = np.sum(np.abs(get_test_2D - cor_2D_A_train))
        loss_list.append(loss)
        K_list.append(K)
        P_list.append(P)

    min_ind = np.argmin(loss_list)
    min_loss = loss_list[min_ind]
    min_p = P_list[min_ind]
    min_k = K_list[min_ind]
    return min_loss, min_k, min_p

def point_3D_projection_to_camera(K, pose, p2d):
    """
    给定内外参，将三维点投影到二维像素坐标（不含畸变）。

    Args:
        K (ndarray): 相机内参矩阵，shape=[3,3]。
        pose (ndarray): 相机外参，shape=[3,4]。
        p2d (ndarray): 三维点，shape=[N,3]。

    Returns:
        yc (ndarray): 投影后的二维点，shape=[N,2]。
    """
    pad_1_test = np.ones([len(p2d), 1], dtype=np.float32)
    cor_3D_A_test = np.hstack((p2d, pad_1_test)).T
    M = np.dot(K, pose)
    targ = np.dot(M, cor_3D_A_test)
    u = targ[0, :] / targ[2, :]
    v = targ[1, :] / targ[2, :]
    yc = np.array([u, v]).T
    return yc

def opencv_PNP(point3D, point_img, K, distortion=None):
    """
    调用 OpenCV 的 PnP 算法进行相机位姿估计（单张影像多点）。

    Args:
        point3D (ndarray): 三维点云，shape=[N,3]。
        point_img (ndarray): 对应二维像素点，shape=[N,2]。
        K (ndarray): 内参矩阵。
        distortion (ndarray or None): 畸变参数，默认为None。

    Returns:
        ndarray: 外参矩阵，shape=[3,4]。
    """
    ptin_img = np.array([(inf[0], inf[1]) for inf in point_img], dtype=np.float32)
    pt3D = np.array([(inf[0], inf[1], inf[2]) for inf in point3D], dtype=np.float32)
    retval, rvec, tvec = cv2.solvePnP(pt3D, ptin_img, K, distortion)
    M_R = cv2.Rodrigues(rvec)[0]
    return np.hstack((M_R, tvec))

def opencv_3D_projection_to_img(obj_points, K, pose, dist= np.zeros((1, 4), dtype=np.float32)):
    """
    使用 OpenCV projectPoints 工具，将三维点按内外参投影到二维像素坐标。

    Args:
        obj_points (ndarray): 三维点集，shape=[N,3]。
        K (ndarray): 相机内参矩阵。
        pose (ndarray): 外参（旋转+平移），shape=[3,4]。
        dist (ndarray): 畸变参数，默认全零。

    Returns:
        img_points (ndarray): 投影后的二维像素点，shape=[N,2]。
    """
    rvecs_in = pose[:, :3]
    tvecs_in = pose[:, 3]
    tvecs_in = np.expand_dims(tvecs_in, axis=-1)
    img_points, _ = cv2.projectPoints(obj_points, rvecs_in, tvecs_in, K, dist)
    img_points = np.array(img_points)
    img_points = np.squeeze(img_points)
    return img_points
def interpolation_dense_3D(img_A, point_2d, points_3d, show=0):
    w_list = np.arange(0, img_A.shape[1])
    h_list = np.arange(0, img_A.shape[0])
    w_mesh, h_mesh = np.meshgrid(w_list, h_list)
    w_mesh = np.reshape(w_mesh, -1)
    h_mesh = np.reshape(h_mesh, -1)
    mesh = np.array([w_mesh, h_mesh]).T
    sampling_x = griddata(point_2d, points_3d[:, 0], mesh, method='linear')
    sampling_y = griddata(point_2d, points_3d[:, 1], mesh, method='linear')
    sampling_z = griddata(point_2d, points_3d[:, 2], mesh, method='linear')
    point_3d_dense = np.array([sampling_x, sampling_y, sampling_z]).T
    color = np.reshape(img_A, (-1, 3))

    x = point_3d_dense[:, 0]
    y = point_3d_dense[:, 1]
    z = point_3d_dense[:, 2]
    x_bool = ~np.isnan(x)
    y_bool = ~np.isnan(y)
    z_bool = ~np.isnan(z)

    all_bool = x_bool * y_bool * z_bool
    mask_bool = all_bool
    mask_bool = np.reshape(mask_bool, (img_A.shape[0], img_A.shape[1]))
    point_3d_dense = point_3d_dense[all_bool]
    color = color[all_bool]
    if show:
        mask_bool_rt = np.zeros_like(mask_bool, dtype=np.uint8)
        mask_bool_rt[mask_bool] = 255
        mask_bool_rt = cv2.resize(mask_bool_rt, (400, 400))
        cv2.imshow('Image with Points', mask_bool_rt)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    # 返回稠密三维点云、颜色、以及照片中有映射像素和无映射像素的掩码
    return point_3d_dense, color, mask_bool


# 计算深度图
def depth_map(point_3d_dense, mask_bool, pos, show=0):
    pos = np.expand_dims(pos, axis=0)
    depth = (point_3d_dense - pos) ** 2
    depth = np.sqrt(np.sum(depth, axis=1))
    mask_bool_rt = np.zeros_like(mask_bool, dtype=np.float32)
    mask_bool_rt[mask_bool] = depth
    if show:
        im = plt.imshow(mask_bool_rt)
        cbar = plt.colorbar(im)
        cbar.set_label('Depth (m)')
        plt.show()
    return mask_bool_rt