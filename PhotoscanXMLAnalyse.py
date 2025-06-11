import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import shapefile
import random
from scipy.io import savemat
# from Analyse3DTools import opencv_PNP, preprocess, draw_Match


def get_str(cxx):
    cxxStart = int(cxx.find(">"))+1
    cxxEnd = int(cxx.find("</"))
    cxx_r = cxx[cxxStart:cxxEnd]
    return cxx_r


def get_float(cxx):
    cxxStart = int(cxx.find(">"))+1
    cxxEnd = int(cxx.find("</"))
    cxx_r = float(cxx[cxxStart:cxxEnd])
    return cxx_r


class ana_photoscan_xml(object):

    def __init__(self, xml_name):
        self.xml_name = xml_name
        f = open(xml_name, 'rb') # 读取文件
        fpList = f.readlines()  # 内容生成列表

        print("Open xml file: ", xml_name)
        self.camera_pose = []
        self.pointcloud_3D = []
        print("Read xml file")
        for row, inf in enumerate(fpList):
            inf = str(inf)
            if inf.find('<ImageDimensions>') != -1:
                wwww = get_float(str(fpList[row + 1]))
                hhhh = get_float(str(fpList[row + 2]))
                ffff = get_float(str(fpList[row + 6]))
                self.hwf = [hhhh, wwww, ffff]

            if inf.find('<PrincipalPoint>') != -1:
                self.w05 = get_float(str(fpList[row + 1]))
                self.h05 = get_float(str(fpList[row + 2]))

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
                print("Image id: ", id, "   List_No: ", no_list-1)

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
        csvF = open(path, 'w', newline="")
        csv_writer = csv.writer(csvF)
        csv_writer.writerow(["NAME",
                             "M00","M01","M02",
                             "M10","M11","M12",
                             "M20","M21","M22",
                             "center_x", "center_y", "center_z"
                            ])
        for save_data_list in self.camera_pose:
            csv_writer.writerow(save_data_list)
        csvF.close()
        print("Save xml file pose finish")

    def __make_matrix(self):

        H, W, focal = self.hwf
        HWF = [H, W, focal]
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

        H, W, F = self.hwf
        i = np.array([0, W, 0, W, W/2])
        i.astype(np.float32)
        j = np.array([0, 0, H, H, H/2])
        j.astype(np.float32)
        K = self.K
        c2w = self.pose_matrix[num]
        img_name = self.camera_pose[num][0]

        dirs = np.stack([(i-K[0][2])/K[0][0], (j-K[1][2])/K[1][1], np.ones_like(i)], -1)
        c2w_r = c2w[:3,:3]
        rays_d = [dir.dot(c2w_r) for dir in dirs]
        rays_o = np.broadcast_to(c2w[:3, -1], np.shape(rays_d))

        return img_name, rays_o, rays_d

    def get_rays_np_all_pixel_directiont(self, num, show_key=False):

        H, W, F = self.hwf
        h_list = np.arange(H)
        w_list = np.arange(W)
        w_mesh, h_mesh = np.meshgrid(w_list, h_list)
        x = np.reshape(w_mesh, (-1))
        y = np.reshape(h_mesh, (-1))

        K = self.K
        c2w = self.pose_matrix[num]
        img_name = self.camera_pose[num][0]
        dirs = np.stack([(x-K[0][2])/K[0][0], (y-K[1][2])/K[1][1], np.ones_like(x)], -1)
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
        csvF = open(path, 'w', newline="")
        csv_writer = csv.writer(csvF)
        csv_writer.writerow(["NAME",
                             "base_x","base_y","base_z",
                             "upleft_x","upleft_y","upleft_z",
                             "upright_x","upright_y","upright_z",
                             "dounleft_x","dounleft_y","dounleft_z",
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
        u = draw_d_list[:, 0]*size
        v = draw_d_list[:, 1]*size
        w = draw_d_list[:, 2]*size

        ax = plt.figure().add_subplot(projection='3d')
        ax.quiver(x, y, z, u, v, w, length=0.1, arrow_length_ratio=0.1)

        plt.show()

    def save_pointcloud_3d(self, save_path):
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
        img_name = self.camera_pose[num][0]
        find_key_word = "<PhotoId>%d</PhotoId>" % num

        f = open(self.xml_name, 'r')  # 读取文件
        fpList = f.readlines()  # 内容生成列表

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
                        corresponding_list.append([x_img, -1*y_img, x_point_cloud, y_point_cloud])
        corresponding_list = np.array(corresponding_list)
        savename_txt = img_name + ".txt"
        np.savetxt(savename_txt, corresponding_list)
        print("Save image to point cloud corresponding file success")

    def get_img_to_pointcloud_corresponding(self, num):
        img_name = self.camera_pose[num][0]
        find_key_word = "<PhotoId>%d</PhotoId>" % num

        f = open(self.xml_name, 'r')  # 读取文件
        fpList = f.readlines()  # 内容生成列表

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
        img_name_1 = self.camera_pose[num1][0]
        find_key_word_1 = "<PhotoId>%d</PhotoId>" % num1
        img_name_2 = self.camera_pose[num2][0]
        find_key_word_2 = "<PhotoId>%d</PhotoId>" % num2

        f = open(self.xml_name, 'r')  # 读取文件
        fpList = f.readlines()  # 内容生成列表

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
        for sy, img_list in enumerate(self.camera_pose):
            if image_keyword in img_list[0]:
                this_pose = self.pose_matrix[sy]
                return self.K, this_pose, sy

        print("Not find ", image_keyword)
        return None

    def get_Distortion(self):
        f = open(self.xml_name, 'r')  # 读取文件
        fpList = f.readlines()  # 内容生成列表

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
        pere_list = fpList[start_tie_line : end_tie_line]
        pere = {}
        for li in pere_list:
            first_index = li.find('<')
            second_index = li.find('>')
            key = li[first_index+1:second_index]
            val = get_float(li)
            pere[key] = val
        return pere

    def.get_elevation(self, x_points, y_points):
        input_shape =
        object_point = np.array(self.pointcloud_3D)
        sampling = griddata(object_point[:, :2], object_point[:, 2], sampling_point[:, :2], method='linear')



if __name__ == "__main__":

    # 读取左右两幅双目图像
    img_A = cv2.imread(".//group1//416.png")
    img_B = cv2.imread(".//group1//417.png")
    img_A = preprocess(img_A)
    img_B = preprocess(img_B)
    h, w = img_A.shape[:2]

    # 载入xml文件
    obj_xml = ana_photoscan_xml("group1.xml") # LCAM_all_camera.xml   group1.xml

    # 将xml文件中的位姿保存至csv文件中
    obj_xml.save_xml_pose("xml_pos.csv")

    # 读取畸变参数
    distor = obj_xml.get_Distortion()
    dist_coeffs = np.array([distor["K1"], distor["K2"], distor["P1"], distor["P2"], distor["K3"]])

    # 按关键词检索，得到相机内参矩阵、外参矩阵、编号
    img_A_K, img_A_pose, num_A = obj_xml.get_cam_parameter_matrix("-0416")
    img_B_K, img_B_pose, num_B = obj_xml.get_cam_parameter_matrix("-0417")

    # 查看相机的向量五点，左上，右上，左下，右下，中间，相机中心
    img_name, rays_o, rays_d = obj_xml.get_rays_np_around_five_point(num_A)
    print("Camera ID: %d  " % num_A, " Camera Name: %s  " % img_name)
    print("Camera Original Point: ", rays_o)
    print("Camera Vector Direction: ", rays_d)

    # 查看一副影像所有像素点的向量的方向
    img_name, rays_o, rays_d_mesh = obj_xml.get_rays_np_all_pixel_directiont(num_A, True)

    # 绘制所有相机的指向图
    obj_xml.draw_pose_vector(10)

    # 保存三维点云数据
    obj_xml.save_pointcloud_3d("Pointcloud3D.shp")
    obj_xml.save_pointcloud_3d("Pointcloud3D.mat")

    # 生成img点至点云的对应txt文件，用于arcgis配准，将影像贴图到点云的DEM上使用。
        ########################################################
        ###    第一列        第二列       第三列       第四列    ###
        ###  同名点图像x   同名点图像y   同名点点云x   同名点点云y  ###
        ########################################################
    obj_xml.get_img_to_pointcloud_corresponding_for_arcgis(num_A)

    # 读取影像的同名点以及三维点
    cor_3D_A, cor_2D_A = obj_xml.get_img_to_pointcloud_corresponding(num_A)

    # 使用opencv_PNP验证位姿结果。输入三维点、像素点、摄象机内参、畸变系数。解算摄象机外参
    bace_opencv_pose_A = opencv_PNP(cor_3D_A, cor_2D_A, img_A_K, dist_coeffs)
    print("Base_Opencv_sovle_pose:  \n")
    print(bace_opencv_pose_A)
    print("XML_pose:  \n")
    print(img_A_pose)

    # 读取两幅影像之间的同名点以及对应的三位点
    ptinA, ptinB, cor_3D_AB = obj_xml.get_img_to_pointcloud_corresponding_couple(num_A, num_B)
    bace_opencv_pose_B = opencv_PNP(cor_3D_AB, ptinB, img_A_K, dist_coeffs)
    print("Base_Opencv_sovle_pose:  \n")
    print(bace_opencv_pose_B)
    print("XML_pose:  \n")
    print(img_B_pose)

    # 绘制同名点匹配图
    draw_Match(img_A, img_B, ptinA, ptinB, save_path="./match.jpg", draw_num=500)
