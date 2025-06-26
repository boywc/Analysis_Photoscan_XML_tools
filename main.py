"""
main.py

Photoscan 空中三角测量 XML 工具库的主要功能演示 Demo

本脚本演示如何调用 PhotoscanXMLAnalyse.py 提供的 ana_photoscan_xml 类，
进行如下常见航空摄影测量数据处理任务：

- XML文件读取与解析
- 相机位姿（外参）导出
- 相机五点射线与全部像素射线方向获取与保存
- 三维点云数据导出（SHP/MAT格式）
- 影像-点云配准对应关系（2D/3D）导出
- 畸变参数获取
- 关键参数查找与可视化
- 3D相机与点云可视化
- 原始影像上2D同名点高亮显示
- 三维点云投影到影像并可视化（新功能）

确保 testdata/255.xml 文件已存在。
"""

from PhotoscanXMLAnalyse import ana_photoscan_xml, draw_points_on_image, project_3d_to_2d_display
import numpy as np
import cv2

def main():
    # ========== 1. 设置XML相对路径 ==========
    xml_path = "testdata/255.xml"
    obj_xml = ana_photoscan_xml(xml_path)

    # ========== 2. 导出全部相机位姿为CSV ==========
    obj_xml.save_xml_pose("testdata/xml_information.csv")

    # ========== 3. 获取第一个相机的五点向量 ==========
    camera_num = 0  # 影像编号（从0起）
    img_name, rays_o, rays_d = obj_xml.get_rays_np_around_five_point(camera_num)
    print(f"[五点射线] 相机: {img_name}\n 原点: {rays_o}\n 方向: {rays_d}")

    # ========== 4. 批量保存所有相机的五点射线 ==========
    obj_xml.save_five_point_vector("testdata/five_point_vector.csv")

    # ========== 5. 绘制所有相机三维指向箭头图（3D） ==========
    obj_xml.draw_pose_vector(size=0.1)

    # ========== 6. 获取并可视化全部像素点的投影方向（可选） ==========
    # 若数据量大建议 show_key=False
    img_name, rays_o, rays_d_full = obj_xml.get_rays_np_all_pixel_directiont(camera_num, show_key=True)

    # ========== 7. 保存三维点云为SHP、MAT文件 ==========
    obj_xml.save_pointcloud_3d("testdata/pointcloud3d.shp")
    obj_xml.save_pointcloud_3d("testdata/pointcloud3d.mat")

    # ========== 8. 生成ArcGIS影像-点云配准TXT ==========
    obj_xml.get_img_to_pointcloud_corresponding_for_arcgis(camera_num)
    print(f"ArcGIS影像-点云配准TXT已输出到: {img_name}.txt")

    # ========== 9. 获取单幅影像所有2D-3D同名点 ==========
    cor_3D, cor_2D = obj_xml.get_img_to_pointcloud_corresponding(camera_num)
    print(f"[同名点3D] 形状: {cor_3D.shape}, [同名点2D] 形状: {cor_2D.shape}")

    # ========== 9+. 获取单幅影像所有2D-3D同名点及颜色信息 ==========
    cor_3D, cor_2D, cor_color = obj_xml.get_img_to_pointcloud_corresponding_with_color(camera_num)
    print(f"[同名点3D] 形状: {cor_3D.shape}, [同名点2D] 形状: {cor_2D.shape}, [颜色] 形状: {cor_color.shape}")
    print("前5个同名点（三维坐标 | 二维像素 | RGB颜色）：")
    for i in range(min(5, cor_3D.shape[0])):
        print(f"  3D: {cor_3D[i]}, 2D: {cor_2D[i]}, Color: {cor_color[i]}")

    # ========== 10. 获取两幅影像的同名点和三维点 ==========
    if len(obj_xml.camera_pose) > 1:
        img1_points, img2_points, cloud_points = obj_xml.get_img_to_pointcloud_corresponding_couple(0, 1)
        print(f"[两幅影像同名点] 影像1点数: {len(img1_points)}, 影像2点数: {len(img2_points)}, 三维点数: {len(cloud_points)}")

    # ========== 11. 查找相机内参/外参及索引（通过影像名关键字） ==========
    keyword = ".jpg"  # 你也可以用其他实际文件名的特征（如 "-0416"）
    result = obj_xml.get_cam_parameter_matrix(keyword)
    if result is not None:
        K, pose, idx = result
        print("[查找相机参数] 索引:", idx)
        print("K (内参矩阵):\n", K)
        print("Pose (外参矩阵):\n", pose)
    else:
        print(f"未找到包含关键字 {keyword} 的影像")

    # ========== 12. 获取畸变参数 ==========
    distortion_dict = obj_xml.get_Distortion()
    print("[畸变参数]", distortion_dict)

    # ========== 13. 查询三维点云的空间范围（XYZ轴最大最小值） ==========
    min_x, max_x, min_y, max_y, min_z, max_z = obj_xml.check_cloud_range()
    print(f"[点云空间范围] X: {min_x:.3f} ~ {max_x:.3f}，Y: {min_y:.3f} ~ {max_y:.3f}，Z: {min_z:.3f} ~ {max_z:.3f}")

    # ========== 14. 获取指定点的高程值（点云插值） ==========
    # 构造一组XY平面上的采样点，示例取点云X/Y范围中部
    test_points = np.array([
        [(min_x + max_x) / 2, (min_y + max_y) / 2],  # 中心点
        [min_x, min_y],  # 左下角
        [max_x, max_y],  # 右上角
    ])
    elevations = obj_xml.get_elevation(test_points)
    for i, (xy, z) in enumerate(zip(test_points, elevations)):
        print(f"[插值高程] 第{i + 1}个点 (x={xy[0]:.3f}, y={xy[1]:.3f}) -> z={z:.3f}")

    # ========== 15. 三维可视化：显示第一个相机的点云与姿态 ==========
    print("[三维可视化] 显示第0号相机的点云与姿态")
    obj_xml.point_and_camera_display([0], point_size=3, camera_box_size=3)
    # 如需显示多个相机：obj_xml.point_and_camera_display([0, 1, 2])

    # ========== 16. 在影像上绘制2D同名点并弹窗显示 ==========
    image_path = f"testdata/255/{img_name}"  # img_name 是当前相机影像名
    try:
        image = cv2.imread(image_path)
        if image is not None:
            draw_points_on_image(image, cor_2D, half_size=5, output_mode=None)  # output_mode=None为弹窗显示
            # 如需保存到文件：
            # draw_points_on_image(image, cor_2D, half_size=5, output_mode="testdata/255/with_points.jpg")
        else:
            print(f"[警告] 未找到影像文件：{image_path}，请检查路径和文件名。")
    except Exception as e:
        print(f"[错误] 图像读取或绘制失败：{e}")

    # ========== 17. 3D点云投影到影像并可视化（新功能） ==========
    # 获取当前影像的畸变参数、内外参、同名点
    distortion = obj_xml.get_Distortion()
    dist_coeffs = np.array([
        distortion.get("K1", 0), distortion.get("K2", 0),
        distortion.get("P1", 0), distortion.get("P2", 0), distortion.get("K3", 0)
    ])
    img_K, img_pose, img_idx = obj_xml.get_cam_parameter_matrix(img_name.split('.')[0])
    points_3d, points_2d, _ = obj_xml.get_img_to_pointcloud_corresponding_with_color(img_idx)
    image_proj_path = f"testdata/255/{img_name}"
    img_proj = cv2.imread(image_proj_path)
    project_3d_to_2d_display(points_3d, img_K, img_pose, dist_coeffs=dist_coeffs,
                      half_size=5)

if __name__ == "__main__":
    main()
