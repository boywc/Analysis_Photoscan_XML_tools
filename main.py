"""
main.py

Photoscan 空三角测量 XML 工具库的主要功能演示 Demo

本脚本演示如何调用 PhotoscanXMLAnalyse.py 提供的 ana_photoscan_xml 类，
进行如下常见航空摄影测量数据处理任务：

- XML文件读取与解析
- 相机位姿（外参）导出
- 相机五点射线与全部像素射线方向获取与保存
- 三维点云数据导出（SHP/MAT格式）
- 影像-点云配准对应关系（2D/3D）导出
- 畸变参数获取
- 关键参数查找与可视化

确保 testdata/255.xml 文件已存在。
"""

from PhotoscanXMLAnalyse import ana_photoscan_xml
import numpy as np

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
    obj_xml.draw_pose_vector(size=1.0)

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

if __name__ == "__main__":
    main()
