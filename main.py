
# 提供库文件调用示例
if __name__ == "__main__":
    # 读取左右两幅双目图像（预处理函数需自定义）
    img_A = cv2.imread(".//group1//416.png")
    img_B = cv2.imread(".//group1//417.png")
    img_A = preprocess(img_A)
    img_B = preprocess(img_B)
    h, w = img_A.shape[:2]

    # 载入xml文件并初始化分析对象
    obj_xml = ana_photoscan_xml("group1.xml")

    # 导出所有相机姿态到CSV
    obj_xml.save_xml_pose("xml_pos.csv")

    # 读取相机畸变参数
    distor = obj_xml.get_Distortion()
    dist_coeffs = np.array([distor["K1"], distor["K2"], distor["P1"], distor["P2"], distor["K3"]])

    # 按关键词获取内参/外参矩阵及编号
    img_A_K, img_A_pose, num_A = obj_xml.get_cam_parameter_matrix("-0416")
    img_B_K, img_B_pose, num_B = obj_xml.get_cam_parameter_matrix("-0417")

    # 查看某相机五点投影射线
    img_name, rays_o, rays_d = obj_xml.get_rays_np_around_five_point(num_A)
    print("Camera ID: %d  " % num_A, " Camera Name: %s  " % img_name)
    print("Camera Original Point: ", rays_o)
    print("Camera Vector Direction: ", rays_d)

    # 查看影像所有像素点射线方向并可视化
    img_name, rays_o, rays_d_mesh = obj_xml.get_rays_np_all_pixel_directiont(num_A, True)

    # 绘制所有相机投影向量三维图
    obj_xml.draw_pose_vector(10)

    # 保存三维点云（SHP和MAT两种格式）
    obj_xml.save_pointcloud_3d("Pointcloud3D.shp")
    obj_xml.save_pointcloud_3d("Pointcloud3D.mat")

    # 生成ArcGIS配准用TXT文件
    obj_xml.get_img_to_pointcloud_corresponding_for_arcgis(num_A)

    # 获取影像同名点与三维点对
    cor_3D_A, cor_2D_A = obj_xml.get_img_to_pointcloud_corresponding(num_A)

    # 用OpenCV PNP算法验证相机外参（外部函数需自定义）
    bace_opencv_pose_A = opencv_PNP(cor_3D_A, cor_2D_A, img_A_K, dist_coeffs)
    print("Base_Opencv_sovle_pose:  \n")
    print(bace_opencv_pose_A)
    print("XML_pose:  \n")
    print(img_A_pose)

    # 获取两幅影像同名点及三维点
    ptinA, ptinB, cor_3D_AB = obj_xml.get_img_to_pointcloud_corresponding_couple(num_A, num_B)
    bace_opencv_pose_B = opencv_PNP(cor_3D_AB, ptinB, img_A_K, dist_coeffs)
    print("Base_Opencv_sovle_pose:  \n")
    print(bace_opencv_pose_B)
    print("XML_pose:  \n")
    print(img_B_pose)

    # 绘制同名点匹配效果图（draw_Match为自定义工具函数）
    draw_Match(img_A, img_B, ptinA, ptinB, save_path="./match.jpg", draw_num=500)