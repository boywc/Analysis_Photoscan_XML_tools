import numpy as np
from PhotoscanXMLAnalyse import *

def xmlpos_to_mathpos(pos):
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

    ro_zf, r3_zf = ror3_zf

    pad_1 = np.ones([len(cor_3D_A), 1], dtype=np.float32)
    cor_3D_A = np.hstack((cor_3D_A, pad_1))

    zero_board = np.zeros_like(cor_3D_A)
    line_1 = np.hstack((cor_3D_A, zero_board))
    line_2 = np.hstack((zero_board, cor_3D_A))
    line_1_2 = np.hstack((line_1, line_2))
    line_1_2 = np.reshape(line_1_2, (len(cor_3D_A) * 2, 8))

    line_1b = -1 * cor_2D_A[:, 0]
    line_1b = np.expand_dims(line_1b, axis=-1)
    line_1b = line_1b * cor_3D_A
    line_2b = -1 * cor_2D_A[:, 1]
    line_2b = np.expand_dims(line_2b, axis=-1)
    line_2b = line_2b * cor_3D_A
    line_1_2b = np.hstack((line_1b, line_2b))
    line_1_2b = np.reshape(line_1_2b, (len(cor_3D_A) * 2, 4))

    P = np.hstack((line_1_2, line_1_2b))
    U, Sigma, VT = np.linalg.svd(P)
    V = VT.T
    m = V[:, -1]
    M = np.reshape(m, (3, 4))

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

   pad_1_test = np.ones([len(p2d), 1], dtype=np.float32)
   cor_3D_A_test = np.hstack((p2d, pad_1_test)).T
   M = np.dot(K, pose)
   targ = np.dot(M, cor_3D_A_test)
   u = targ[0, :] / targ[2, :]
   v = targ[1, :] / targ[2, :]
   yc = np.array([u, v]).T
   return yc

def opencv_PNP(point3D, point_img, K, distortion=None):
    ptin_img = np.array([(inf[0], inf[1]) for inf in point_img], dtype=np.float32)
    pt3D = np.array([(inf[0], inf[1], inf[2]) for inf in point3D], dtype=np.float32)
    retval, rvec, tvec = cv2.solvePnP(pt3D, ptin_img, K, distortion)
    M_R = cv2.Rodrigues(rvec)[0]
    return np.hstack((M_R, tvec))

def opencv_3D_projection_to_img(obj_points, K, pose, dist= np.zeros((1, 4), dtype=np.float32)):
    rvecs_in = pose[:, :3]
    tvecs_in = pose[:, 3]
    tvecs_in = np.expand_dims(tvecs_in, axis=-1)
    img_points, _ = cv2.projectPoints(obj_points, rvecs_in, tvecs_in, K, dist)
    img_points = np.array(img_points)
    img_points = np.squeeze(img_points)
    return img_points


######一：：：：：功能1，摄象机标定，计算位姿############
### 重新计算位姿矩阵，由于photoscan获得的位姿矩阵可能存在一定误差，这里提供了重新计算位姿矩阵的方法
obj_xml = ana_photoscan_xml("testdata/255.xml")
img_A_K, img_A_pose, num_A = obj_xml.get_cam_parameter_matrix("20220130181054")
points_3d, point_2d, point_color = obj_xml.get_img_to_pointcloud_corresponding_with_color(num_A)

####################################################
### 第一种方法，根据公式自行编写代码计算
# 输入三维点云和对应的二维点，计算位姿矩阵
# 这里的ror3_zf是计算参数有四种选项[1, 1]  [1, -1], [-1 ,1], [-1,-1]
# 对方程求解时会出现四个解，正确的解在这是其中之一，通过调整ror3_zf确定正确的解
# 通常[1, 1]就说正确的解，如果不对，再换其他的选项，试出正确的解
K, pose = camera_calibration(points_3d, point_2d, ror3_zf=[1, 1])
# 因此我们包装了一个函数，试了四种所有可能，重投计算最小误差自动分辨是哪个情况
min_loss, min_k, min_p = camera_calibration_with_verify(points_3d, point_2d)
print("Photoscan XML K: \n", img_A_K)
print("Photoscan XML pose: \n", img_A_pose)
print("Calculated K: \n", K)
print("Calculated pose: \n", pose)
print("Calculated_test K: \n", min_k)
print("Calculated_test pose: \n", min_p)
# 从计算结果可以看出，photoscan输出的pos矩阵和自己计算的矩阵存在一定差距
# 最后一列代表位置坐标的排列是不同的，的XML文件是x,y,z.自己计算的是y,z,x
# 这里提供了XMLpos和自己生成pos的格式转换方式  xmlpos_to_mathpos(pos)

####################################################
### 第二种方法，根据OpenCV PNP算法求解，需要输入摄象机内参，畸变（可选）。计算出外姿
# 案例1 不带畸变
pos_pnp = opencv_PNP(points_3d, point_2d, img_A_K, distortion=None)
print("Calculated pose by opencv: \n", pos_pnp)
# 案例2 不带畸变
distor = obj_xml.get_Distortion()
dist_coeffs = np.array([distor["K1"], distor["K2"], distor["P1"], distor["P2"], distor["K3"]])
pos_pnp = opencv_PNP(points_3d, point_2d, img_A_K, distortion=dist_coeffs)
print("Calculated pose by opencv with distortion: \n", pos_pnp)

######一：：：：：功能2，重投影############
# 将3D点投影回影像，有两个方法，
# 方法一：第一个是使用自己编写的方法
porject_2d = point_3D_projection_to_camera(K, pose, points_3d)
loss = np.mean(np.abs(porject_2d - point_2d))
print("Re Project loss:  ", loss)

# 方法二：使用opencv 自带的
# 案例1 带畸变
porject_2d = opencv_3D_projection_to_img(points_3d, K, pose, dist=dist_coeffs)
# 案例2 不带畸变
porject_2d = opencv_3D_projection_to_img(points_3d, K, pose)
loss = np.mean(np.abs(porject_2d - point_2d))
print("Re Project loss from opencv:  ", loss)

# 注意，重投影的时候使用的都是本系统生成的pos位姿矩阵，如果使用XML位姿矩阵需要先转换一下
# 案例3 使用XML文件的位姿矩阵
mathpos = xmlpos_to_mathpos(img_A_pose)
porject_2d = opencv_3D_projection_to_img(points_3d, img_A_K, mathpos)
loss = np.mean(np.abs(porject_2d - point_2d))
print("Re Project loss from xml pos:  ", loss)
