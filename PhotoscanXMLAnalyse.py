import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import shapefile
import random
from scipy.io import savemat



def get_str(cxx):
    """
    从XML格式字符串提取中间的文本内容。

    Args:
        cxx (str): XML格式字符串，如"<Tag>value</Tag>"。
    Returns:
        str: 标签内的字符串值。
    """
    cxxStart = int(cxx.find(">")) + 1
    cxxEnd = int(cxx.find("</"))
    cxx_r = cxx[cxxStart:cxxEnd]
    return cxx_r


def get_float(cxx):
    """
    从XML格式字符串提取标签内的数值并转为float。

    Args:
        cxx (str): XML格式字符串。
    Returns:
        float: 标签内的浮点数值。
    """
    cxxStart = int(cxx.find(">")) + 1
    cxxEnd = int(cxx.find("</"))
    cxx_r = float(cxx[cxxStart:cxxEnd])
    return cxx_r


class ana_photoscan_xml(object):
    """
    Photoscan空三角测量XML文件分析主类。

    支持批量提取相机参数、点云、配准关系导出、可视化等多种常用操作。
    """

    def __init__(self, xml_name):
        """
        构造函数，解析并读取XML文件核心参数。

        Args:
            xml_name (str): Photoscan导出的空三角XML文件路径。
        """
        self.xml_name = xml_name
        f = open(xml_name, 'rb')
        fpList = f.readlines()  # 文件内容按行读入
        print("Open xml file: ", xml_name)
        self.camera_pose = []  # 存储所有相机姿态及参数
        self.pointcloud_3D = []  # 存储所有三维点云（TiePoints）
        print("Read xml file")
        for row, inf in enumerate(fpList):
            inf = str(inf)
            # 提取相机内参：像元、主点、焦距
            if inf.find('<ImageDimensions>') != -1:
                wwww = get_float(str(fpList[row + 1]))
                hhhh = get_float(str(fpList[row + 2]))
                ffff = get_float(str(fpList[row + 6]))
                self.hwf = [hhhh, wwww, ffff]
            # 提取主点坐标
            if inf.find('<PrincipalPoint>') != -1:
                self.w05 = get_float(str(fpList[row + 1]))
                self.h05 = get_float(str(fpList[row + 2]))
            # 提取相机姿态与位置信息
            if inf.find('<ImagePath>') != -1:
                id = int(get_float(str(fpList[row - 1])))
                save_data_list = []
                image_name = str(fpList[row])
                image_name = get_str(image_name).split("/")[-1]
                save_data_list.append(image_name)
                for add_ind in [4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 16, 17]:
                    in_data = get_float(str(fpList[row + add_ind]))
                    save_data_list.append(in_data)
                self.camera_pose.append(save_data_list)
                no_list = len(self.camera_pose)
                print("Image id: ", id, "   List_No: ", no_list - 1)
            # 提取三维点云（TiePoint）
            if inf.find('<TiePoint>') != -1:
                save_data_list_pc = []
                for add_ind in [2, 3, 4]:
                    in_data = get_float(str(fpList[row + add_ind]))
                    save_data_list_pc.append(in_data)
                self.pointcloud_3D.append(save_data_list_pc)
        print("Read xml file finish")
        self.K = []
        self.pose_matrix = []
        self.__make_matrix()

    def save_xml_pose(self, path="xml_information.csv"):
        """
        保存所有相机的姿态（外参矩阵、相机中心）到CSV文件。

        Args:
            path (str): 输出CSV文件名，默认为"xml_information.csv"。
        """
        csvF = open(path, 'w', newline="", encoding='utf-8')
        csv_writer = csv.writer(csvF)
        csv_writer.writerow(["NAME",
                             "M00", "M01", "M02",
                             "M10", "M11", "M12",
                             "M20", "M21", "M22",
                             "center_x", "center_y", "center_z"
                             ])
        for save_data_list in self.camera_pose:
            csv_writer.writerow(save_data_list)
        csvF.close()
        print("Save xml file pose finish")

    def __make_matrix(self):
        """
        解析XML内参数并生成相机内参矩阵K与所有相机的外参矩阵pose_matrix。
        """
        H, W, focal = self.hwf
        self.K = np.array([
            [focal, 0, self.w05],
            [0, focal, self.h05],
            [0, 0, 1]
        ])
        self.pose_matrix = []
        for row in self.camera_pose:
            M = np.array(row[1:10])
            M = np.reshape(M, (3, 3))
            T = np.array(row[10:13])
            T = np.reshape(T, (3, 1))
            pose = np.hstack([M, T])
            self.pose_matrix.append(pose)

    def get_rays_np_around_five_point(self, num):
        """
        获取指定影像的五个关键像素点（四角+中心）的投影射线（向量）。

        Args:
            num (int): 相机索引（从0开始）。
        Returns:
            img_name (str): 影像名
            rays_o (ndarray): 相机中心点
            rays_d (ndarray): 投影方向向量（五个点）
        """
        H, W, F = self.hwf
        i = np.array([0, W, 0, W, W / 2])
        i.astype(np.float32)
        j = np.array([0, 0, H, H, H / 2])
        j.astype(np.float32)
        K = self.K
        c2w = self.pose_matrix[num]
        img_name = self.camera_pose[num][0]

        dirs = np.stack([(i - K[0][2]) / K[0][0], (j - K[1][2]) / K[1][1], np.ones_like(i)], -1)
        c2w_r = c2w[:3, :3]
        rays_d = [dir.dot(c2w_r) for dir in dirs]
        rays_o = np.broadcast_to(c2w[:3, -1], np.shape(rays_d))
        return img_name, rays_o, rays_d

    def get_rays_np_all_pixel_directiont(self, num, show_key=False):
        """
        获取指定影像所有像素点的投影射线方向，并支持可视化RGB各分量。

        Args:
            num (int): 相机索引
            show_key (bool): 是否显示射线方向三通道可视化
        Returns:
            img_name (str): 影像名
            rays_o (ndarray): 相机中心
            rays_d (ndarray): 每个像素射线方向(H,W,3)
        """
        H, W, F = self.hwf
        h_list = np.arange(H)
        w_list = np.arange(W)
        w_mesh, h_mesh = np.meshgrid(w_list, h_list)
        x = np.reshape(w_mesh, (-1))
        y = np.reshape(h_mesh, (-1))
        K = self.K
        c2w = self.pose_matrix[num]
        img_name = self.camera_pose[num][0]
        dirs = np.stack([(x - K[0][2]) / K[0][0], (y - K[1][2]) / K[1][1], np.ones_like(x)], -1)
        c2w_r = c2w[:3, :3]
        rays_d = np.array([dir.dot(c2w_r) for dir in dirs])
        rays_o = c2w[:3, -1]
        rays_d = np.reshape(rays_d, (int(H), int(W), 3))
        if show_key:
            plt.subplot(131)
            plt.imshow(rays_d[:, :, 0])
            plt.colorbar()
            plt.subplot(132)
            plt.imshow(rays_d[:, :, 1])
            plt.colorbar()
            plt.subplot(133)
            plt.imshow(rays_d[:, :, 2])
            plt.colorbar()
            plt.show()
        return img_name, rays_o, rays_d

    def save_five_point_vector(self, path='five_point_vector.csv'):
        """
        批量保存所有相机的五点射线参数到CSV。

        Args:
            path (str): 输出文件名，默认为"five_point_vector.csv"。
        """
        csvF = open(path, 'w', newline="", encoding='utf-8')
        csv_writer = csv.writer(csvF)
        csv_writer.writerow(["NAME",
                             "base_x", "base_y", "base_z",
                             "upleft_x", "upleft_y", "upleft_z",
                             "upright_x", "upright_y", "upright_z",
                             "dounleft_x", "dounleft_y", "dounleft_z",
                             "dounright_x", "dounright_y", "dounright_z",
                             "center_x", "center_y", "center_z"
                             ])
        for inf in range(len(self.camera_pose)):
            img_name, rays_o, rays_d = self.get_rays_np_around_five_point(inf)
            save_row = [img_name]
            rays_o = rays_o[0, :].tolist()
            rays_d = np.reshape(rays_d, (-1,)).tolist()
            save_row.extend(rays_o)
            save_row.extend(rays_d)
            csv_writer.writerow(save_row)
        csvF.close()
        print("Save five point vector finish")

    def draw_pose_vector(self, size=0.2):
        """
        绘制所有相机的五点投影射线三维分布（箭头可缩放）。

        Args:
            size (float): 向量箭头缩放因子，默认为0.2。
        """
        draw_o_list = []
        draw_d_list = []
        for inf in range(len(self.camera_pose)):
            img_name, rays_o, rays_d = self.get_rays_np_around_five_point(inf)
            draw_o_list.append(rays_o)
            draw_d_list.append(rays_d)
        draw_o_list = np.vstack(draw_o_list)
        draw_d_list = np.vstack(draw_d_list)
        x = draw_o_list[:, 0]
        y = draw_o_list[:, 1]
        z = draw_o_list[:, 2]
        u = draw_d_list[:, 0] * size
        v = draw_d_list[:, 1] * size
        w = draw_d_list[:, 2] * size
        ax = plt.figure().add_subplot(projection='3d')
        ax.quiver(x, y, z, u, v, w, length=0.1, arrow_length_ratio=0.1)
        plt.show()

    def save_pointcloud_3d(self, save_path):
        """
        导出三维点云，支持SHP（矢量点）与MAT（Matlab）两种格式。

        Args:
            save_path (str): 文件路径，支持".shp"或".mat"后缀自动识别格式。
        """
        savetype = save_path.split(".")[-1]
        if savetype == "shp":
            file = shapefile.Writer(save_path, shapeType=1, autoBalance=1)
            file.field("x", "F", 50, 9)
            file.field("y", "F", 50, 9)
            file.field("z", "F", 50, 9)
            for inf in self.pointcloud_3D:
                file.point(inf[0], inf[1])
                file.record(inf[0], inf[1], inf[2])
            file.close()
            print("Save Shp File Success")
        if savetype == "mat":
            save_d = {}
            save_ar = np.array(self.pointcloud_3D)
            save_d["x"] = save_ar[:, 0]
            save_d["y"] = save_ar[:, 1]
            save_d["z"] = save_ar[:, 2]
            savemat(save_path, save_d)
            print("Save Mat File Success")

    def get_img_to_pointcloud_corresponding_for_arcgis(self, num):
        """
        生成单幅影像至点云的对应TXT文件，方便ArcGIS配准。

        Args:
            num (int): 相机索引
        """
        img_name = self.camera_pose[num][0]
        find_key_word = "<PhotoId>%d</PhotoId>" % num
        f = open(self.xml_name, 'r', encoding='utf-8')
        fpList = f.readlines()
        corresponding_list = []
        for row, inf in enumerate(fpList):
            inf = str(inf)
            if inf.find('<TiePoint>') != -1:
                x_point_cloud = get_float(str(fpList[row + 2]))
                y_point_cloud = get_float(str(fpList[row + 3]))
                start_tie_line = row
                son_i = 0
                while True:
                    row_inf = str(fpList[start_tie_line + son_i])
                    if row_inf.find('</TiePoint>') != -1:
                        end_tie_line = start_tie_line + son_i
                        break
                    son_i += 1
                for son_num, block_line in enumerate(fpList[start_tie_line:end_tie_line]):
                    if block_line.find(find_key_word) != -1:
                        x_img = get_float(str(fpList[row + son_num + 1]))
                        y_img = get_float(str(fpList[row + son_num + 2]))
                        corresponding_list.append([x_img, -1 * y_img, x_point_cloud, y_point_cloud])
        corresponding_list = np.array(corresponding_list)
        savename_txt = img_name + ".txt"
        np.savetxt(savename_txt, corresponding_list)
        print("Save image to point cloud corresponding file success")

    def get_img_to_pointcloud_corresponding(self, num):
        """
        获取单幅影像所有2D-3D同名点对应关系。

        Args:
            num (int): 相机索引
        Returns:
            tuple: (3D点坐标数组, 2D像素点坐标数组)
        """
        img_name = self.camera_pose[num][0]
        find_key_word = "<PhotoId>%d</PhotoId>" % num
        f = open(self.xml_name, 'r', encoding='utf-8')
        fpList = f.readlines()
        corresponding_list_3D = []
        corresponding_list_2D = []
        for row, inf in enumerate(fpList):
            inf = str(inf)
            if inf.find('<TiePoint>') != -1:
                x_point_cloud = get_float(str(fpList[row + 2]))
                y_point_cloud = get_float(str(fpList[row + 3]))
                z_point_cloud = get_float(str(fpList[row + 4]))
                start_tie_line = row
                son_i = 0
                while True:
                    row_inf = str(fpList[start_tie_line + son_i])
                    if row_inf.find('</TiePoint>') != -1:
                        end_tie_line = start_tie_line + son_i
                        break
                    son_i += 1
                for son_num, block_line in enumerate(fpList[start_tie_line:end_tie_line]):
                    if block_line.find(find_key_word) != -1:
                        x_img = get_float(str(fpList[row + son_num + 1]))
                        y_img = get_float(str(fpList[row + son_num + 2]))
                        corresponding_list_3D.append([x_point_cloud, y_point_cloud, z_point_cloud])
                        corresponding_list_2D.append([x_img, y_img])
        corresponding_list_3D = np.array(corresponding_list_3D)
        corresponding_list_2D = np.array(corresponding_list_2D)
        return corresponding_list_3D, corresponding_list_2D

    def get_img_to_pointcloud_corresponding_couple(self, num1, num2):
        """
        获取两幅影像的同名点及其三维点的配对，便于双目匹配或PNP验证。

        Args:
            num1 (int): 第一幅影像索引
            num2 (int): 第二幅影像索引
        Returns:
            tuple: (影像1像素点, 影像2像素点, 三维点)
        """
        img_name_1 = self.camera_pose[num1][0]
        find_key_word_1 = "<PhotoId>%d</PhotoId>" % num1
        img_name_2 = self.camera_pose[num2][0]
        find_key_word_2 = "<PhotoId>%d</PhotoId>" % num2
        f = open(self.xml_name, 'r', encoding='utf-8')
        fpList = f.readlines()
        img1_list = []
        img2_list = []
        pointcloud_list = []
        for row, inf in enumerate(fpList):
            inf = str(inf)
            if inf.find('<TiePoint>') != -1:
                x_point_cloud = get_float(str(fpList[row + 2]))
                y_point_cloud = get_float(str(fpList[row + 3]))
                z_point_cloud = get_float(str(fpList[row + 4]))
                point_1_kg = False
                point_2_kg = False
                start_tie_line = row
                son_i = 0
                while True:
                    row_inf = str(fpList[start_tie_line + son_i])
                    if row_inf.find('</TiePoint>') != -1:
                        end_tie_line = start_tie_line + son_i
                        break
                    son_i += 1
                for son_num, block_line in enumerate(fpList[start_tie_line:end_tie_line]):
                    if block_line.find(find_key_word_1) != -1:
                        x_img_1 = get_float(str(fpList[row + son_num + 1]))
                        y_img_1 = get_float(str(fpList[row + son_num + 2]))
                        point_1_kg = True
                    if block_line.find(find_key_word_2) != -1:
                        x_img_2 = get_float(str(fpList[row + son_num + 1]))
                        y_img_2 = get_float(str(fpList[row + son_num + 2]))
                        point_2_kg = True
                if point_1_kg and point_2_kg:
                    img1_list.append([x_img_1, y_img_1])
                    img2_list.append([x_img_2, y_img_2])
                    pointcloud_list.append([x_point_cloud, y_point_cloud, z_point_cloud])
        return img1_list, img2_list, pointcloud_list

    def get_cam_parameter_matrix(self, image_keyword):
        """
        按影像名关键字检索相机的内参矩阵、外参矩阵与索引编号。

        Args:
            image_keyword (str): 影像名的关键字（如文件名后缀）
        Returns:
            tuple: (内参矩阵K, 外参矩阵, 相机索引)
        """
        for sy, img_list in enumerate(self.camera_pose):
            if image_keyword in img_list[0]:
                this_pose = self.pose_matrix[sy]
                return self.K, this_pose, sy
        print("Not find ", image_keyword)
        return None

    def get_Distortion(self):
        """
        读取相机的畸变参数。

        Returns:
            dict: 相机畸变系数字典（如K1、K2、P1、P2、K3等）
        """
        f = open(self.xml_name, 'r', encoding='utf-8')
        fpList = f.readlines()
        corresponding_list = []
        for row, inf in enumerate(fpList):
            inf = str(inf)
            if inf.find('<Distortion>') != -1:
                start_tie_line = row + 1
                son_i = 0
                while True:
                    row_inf = str(fpList[start_tie_line + son_i])
                    if row_inf.find('</Distortion>') != -1:
                        end_tie_line = start_tie_line + son_i
                        break
                    son_i += 1
                break
        pere_list = fpList[start_tie_line: end_tie_line]
        pere = {}
        for li in pere_list:
            first_index = li.find('<')
            second_index = li.find('>')
            key = li[first_index + 1:second_index]
            val = get_float(li)
            pere[key] = val
        return pere

    def check_cloud_range(self):
        """
        获取点云的空间范围（XYZ轴的最小值与最大值）。

        Returns:
            tuple: (min_x, max_x, min_y, max_y, min_z, max_z)
                - min_x, max_x: 点云X轴最小、最大值
                - min_y, max_y: 点云Y轴最小、最大值
                - min_z, max_z: 点云Z轴最小、最大值
        """
        object_point = np.array(self.pointcloud_3D)
        return (np.min(object_point[:, 0]), np.max(object_point[:, 0]),
                np.min(object_point[:, 1]), np.max(object_point[:, 1]),
                np.min(object_point[:, 2]), np.max(object_point[:, 2]))

    def get_elevation(self, points):
        """
        获取指定平面点（x, y）在点云中的高程（Z值）。

        Args:
            points (ndarray): 输入二维点坐标数组，shape=[n, 2]，每行[x, y]

        Returns:
            ndarray: 对应输入点的高程（Z）数组，shape=[n,]
        """
        object_point = np.array(self.pointcloud_3D)
        sampling = griddata(object_point[:, :2], object_point[:, 2], points, method='linear')
        return sampling
