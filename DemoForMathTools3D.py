"""
DemoForMathTools3D.py

三维摄影测量数学工具库（MathTools3D）主要功能演示 Demo

本脚本演示如何结合 PhotoscanXMLAnalyse 的数据接口与 MathTools3D 中的标定、投影方法，
实现以下典型三维摄影测量数学处理流程：

- 摄像机位姿标定（包括基于公式推导法与OpenCV PNP法）
- 相机内外参/畸变对比分析
- 3D点重投影到影像平面（支持自定义和OpenCV方法）
- XML与数学推导位姿格式互转
- 重投影误差对比与输出
- 稠密三维重建与深度图生成（新功能）

本Demo假定 testdata/255.xml、影像及点云配准信息已准备好。

"""

from MathTools3D import *
from PhotoscanXMLAnalyse import ana_photoscan_xml
import numpy as np

# ========== 1. 摄像机标定与位姿重算 ==========

# 读取XML文件，获取目标相机的内外参与同名点
obj_xml = ana_photoscan_xml("testdata/255.xml")
img_A_K, img_A_pose, num_A = obj_xml.get_cam_parameter_matrix("20220130181054")  # 请根据实际影像关键字调整
points_3d, point_2d, point_color = obj_xml.get_img_to_pointcloud_corresponding_with_color(num_A)

# ========== 1.1 公式法求解相机位姿 ==========

# 摄像机标定：给定三维点云与二维像素点，依据投影模型求解内外参
# ror3_zf参数有四种选项，通常[1,1]正确，如不对可尝试其他组合
K, pose = camera_calibration(points_3d, point_2d, ror3_zf=[1, 1])

# 若不确定四解哪一组正确，可用自动试错函数选择最小重投影误差
min_loss, min_k, min_p = camera_calibration_with_verify(points_3d, point_2d)

print("Photoscan XML K: \n", img_A_K)
print("Photoscan XML pose: \n", img_A_pose)
print("Calculated K: \n", K)
print("Calculated pose: \n", pose)
print("Best-fit K (自动四解): \n", min_k)
print("Best-fit pose (自动四解): \n", min_p)
# 提示：XML输出pos矩阵最后一列顺序为x,y,z，自行计算pos为y,z,x
# 若需两者互转，可用 xmlpos_to_mathpos(pos)

# ========== 1.2 OpenCV PNP算法求解外参 ==========

# 方法1：无畸变参数
pos_pnp = opencv_PNP(points_3d, point_2d, img_A_K, distortion=None)
print("Calculated pose by OpenCV (无畸变): \n", pos_pnp)

# 方法2：加入畸变参数
distor = obj_xml.get_Distortion()
dist_coeffs = np.array([
    distor.get("K1", 0), distor.get("K2", 0),
    distor.get("P1", 0), distor.get("P2", 0), distor.get("K3", 0)
])
pos_pnp_dist = opencv_PNP(points_3d, point_2d, img_A_K, distortion=dist_coeffs)
print("Calculated pose by OpenCV (带畸变): \n", pos_pnp_dist)

# ========== 2. 3D点重投影回影像 ==========

# 方法一：自定义重投影方法
project_2d = point_3D_projection_to_camera(K, pose, points_3d)
loss = np.mean(np.abs(project_2d - point_2d))
print("Reproject loss (自定义): ", loss)

# 方法二：OpenCV重投影（带畸变）
project_2d_opencv_dist = opencv_3D_projection_to_img(points_3d, K, pose, dist=dist_coeffs)
# 方法三：OpenCV重投影（无畸变）
project_2d_opencv = opencv_3D_projection_to_img(points_3d, K, pose)
loss_opencv = np.mean(np.abs(project_2d_opencv - point_2d))
print("Reproject loss from OpenCV (无畸变): ", loss_opencv)

# ========== 3. XML输出位姿格式兼容转换 ==========

# 注意：本系统自算pose与XML输出pose排列不同，如需直接用XML外参矩阵须先格式转换
mathpos = xmlpos_to_mathpos(img_A_pose)
project_2d_xml = opencv_3D_projection_to_img(points_3d, img_A_K, mathpos)
loss_xml = np.mean(np.abs(project_2d_xml - point_2d))
print("Reproject loss from XML pos: ", loss_xml)

# ========== 4. 稠密三维重建与深度图生成（新功能） ==========

# 读取当前相机原始影像（假定 XML 影像名可直接拼路径）
img_name = obj_xml.camera_pose[num_A][0]
img_path = f"testdata/255/{img_name}"
img_A = cv2.imread(img_path)
if img_A is None:
    raise FileNotFoundError(f"未找到原始影像文件: {img_path}")

# --- 4.1 稠密三维点云插值映射 ---
point_3d_dense, color_dense, mask_bool = interpolation_dense_3D(img_A, point_2d, points_3d, show=1)
print(f"Dense 3D points count: {point_3d_dense.shape[0]}")
print(f"Valid mask shape: {mask_bool.shape}, color array shape: {color_dense.shape}")

# --- 4.2 深度图生成 ---
# 相机中心获取，格式转换（外参矩阵最后一列顺序 x, y, z）
pose_math = xmlpos_to_mathpos(img_A_pose)
cam_center = pose_math[:, 3]
depth_img = depth_map(point_3d_dense, mask_bool, cam_center, show=1)
print(f"Depth map shape: {depth_img.shape}, depth min: {np.min(depth_img):.3f}, max: {np.max(depth_img):.3f}")

# ========== 结论/提示 ==========

# 通过对比不同重投影loss，可以判断标定精度与XML输出参数精度是否一致。
# 若自定义、OpenCV与XML重投影loss均较小，说明三者计算基本一致。
# 若loss偏大，建议复查点选精度、点位编号或参数转换顺序。

