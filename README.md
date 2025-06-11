类：class ana_photoscan_xml(object):
1.	创建类，读取XML文件：
obj_xml = ana_photoscan_xml("CE6PS.xml")

2.	保存XML文件中的位姿：
obj_xml.save_xml_pose(“文件名选填”)
默认文件名“xml_information.csv”。

3.	计算一幅相机的五点向量：
输入参数：num 影像编号，整形，例如：num=3
img_name, rays_o, rays_d = obj_xml.get_rays_np_around_five_point(num)
返回：  影像文件名img_name     影像向量原点rays_o    影像向量方向 rays_d

4.	保存所有相机的五点向量：
obj_xml.save_five_point_vector(“文件名选填”)
默认文件名：“five_point_vector.csv”

5.	绘制向量3D图：
obj_xml.draw_pose_vector()

6.	保存三维点云数据：
保存格式有两种，mat和shp。按文件后缀名自动区分。
案例1：obj_xml.save_pointcloud_3d("cloud.shp")
案例2：obj_xml.save_pointcloud_3d("cloud.mat")

7.	生成img到点云的对应点文件
输入num为影像编号：例如：num=3
obj_xml.get_img_to_pointcloud_corresponding(num)
生成txt文件：
第一列	第二列	第三列	第四列
同名点图像x	同名点图像y	同名点点云x	同名点点云y
这个文件主要用于arcgis配准。将影像贴图到点云的DEM上使用。

