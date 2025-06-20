# Photoscan Aerial Triangulation XML Parser

**Photoscan 空三角测量 XML 文件解析与数据导出工具包**

本项目旨在为无人机航空摄影测量、三维重建等场景提供一套**高效、开源的 Photoscan 空三角测量 XML 结果解析工具**，支持批量读取、数据导出、相机参数提取、三维点云处理、配准文件生成和多种可视化分析，极大提升后处理自动化效率。

项目主页：[https://github.com/boywc/Analysis\_Photoscan\_XML\_tools](https://github.com/boywc/Analysis_Photoscan_XML_tools)

---

## 特性与功能

* 自动解析 Photoscan 导出的空三角测量 XML 文件
* 批量提取并导出相机位姿、内参、畸变参数、三维点云等核心数据
* 支持导出为 CSV、MAT、SHP 等常用格式
* 影像像素与三维点云空间关系一键查询、输出
* 快速生成 ArcGIS 影像-点云配准 TXT 文件
* 支持三维可视化分析（相机投影向量、点云等）
* 兼容外部工具（如 OpenCV PNP、ArcGIS DEM）

---

## 安装与环境依赖

建议 Python 3.7+ 环境，依赖如下：

```bash
pip install numpy matplotlib scipy pyshp opencv-python
```

---

## 安装本项目

你可以通过 Git 克隆获取源码：

```bash
git clone https://github.com/boywc/Analysis_Photoscan_XML_tools.git
cd Analysis_Photoscan_XML_tools
pip install -e .
```

> 或将 `photoscan_xml_parser.py` 直接拷贝到你的项目中也可使用。

---

## 快速上手

### 1. 创建类对象，读取 XML 文件

```python
from photoscan_xml_parser import ana_photoscan_xml

# 读取 Photoscan XML 文件
obj_xml = ana_photoscan_xml("CE6PS.xml")
```

### 2. 保存相机位姿信息

```python
# 保存为默认文件名 xml_information.csv
obj_xml.save_xml_pose()
# 或自定义文件名
obj_xml.save_xml_pose("camera_poses.csv")
```

### 3. 获取一幅相机的五点向量

```python
# num: 影像编号（整数，从0开始）
img_name, rays_o, rays_d = obj_xml.get_rays_np_around_five_point(num=3)
print(img_name, rays_o, rays_d)
```

### 4. 保存所有相机的五点向量

```python
obj_xml.save_five_point_vector("five_point_vector.csv")
```

### 5. 绘制三维相机向量

```python
obj_xml.draw_pose_vector()
```

### 6. 保存三维点云

```python
# 导出为 SHP
obj_xml.save_pointcloud_3d("cloud.shp")
# 导出为 MAT
obj_xml.save_pointcloud_3d("cloud.mat")
```

### 7. 生成 ArcGIS 影像到点云对应点文件

```python
# num: 影像编号
obj_xml.get_img_to_pointcloud_corresponding_for_arcgis(num=3)
# 自动生成 txt，字段为：图像x 图像y 点云x 点云y
```

### 8. 查询影像与点云的2D/3D同名点坐标

```python
cor_3D, cor_2D = obj_xml.get_img_to_pointcloud_corresponding(num=3)
# cor_3D: Nx3三维点
# cor_2D: Nx2像素坐标
```

---

## API 参考手册

| 方法名                                                      | 功能描述                      | 参数说明                     | 返回值               |
| -------------------------------------------------------- | ------------------------- | ------------------------ | ----------------- |
| `ana_photoscan_xml(xml_file)`                            | 读取 XML 文件，创建解析对象          | `xml_file`: XML文件名       | 类对象               |
| `save_xml_pose(filename=None)`                           | 保存相机位姿 CSV                | `filename`: 文件名（可选）      | 无                 |
| `get_rays_np_around_five_point(num)`                     | 获取相机5点（四角+中心）向量           | `num`: 影像编号              | img名, 原点, 方向      |
| `save_five_point_vector(filename=None)`                  | 批量保存所有相机五点向量              | `filename`: 文件名（可选）      | 无                 |
| `draw_pose_vector(size=0.2)`                             | 绘制所有相机姿态三维箭头图             | `size`: 向量缩放因子           | 无                 |
| `save_pointcloud_3d(save_path)`                          | 导出三维点云为 SHP 或 MAT         | `save_path`: 文件名（后缀自动判别） | 无                 |
| `get_img_to_pointcloud_corresponding_for_arcgis(num)`    | 导出 ArcGIS 影像点到点云对应 txt 文件 | `num`: 影像编号              | 无                 |
| `get_img_to_pointcloud_corresponding(num)`               | 获取某影像2D像素与3D点的全部配对        | `num`: 影像编号              | 3D点数组, 2D点数组      |
| `get_img_to_pointcloud_corresponding_couple(num1, num2)` | 获取两影像同名点与三维点配对            | `num1`, `num2`: 两影像编号    | 2D点1, 2D点2, 3D点数组 |
| `get_cam_parameter_matrix(keyword)`                      | 按影像名关键字查找内外参矩阵            | `keyword`: 字符串关键字        | 内参, 外参, 编号        |
| `get_Distortion()`                                       | 获取畸变参数                    | 无                        | 畸变参数字典            |

---

## 典型工作流示例

```python
obj_xml = ana_photoscan_xml("group1.xml")
obj_xml.save_xml_pose("xml_pos.csv")
distortion = obj_xml.get_Distortion()
img_K, img_pose, num = obj_xml.get_cam_parameter_matrix("-0416")
img_name, rays_o, rays_d = obj_xml.get_rays_np_around_five_point(num)
obj_xml.draw_pose_vector()
obj_xml.save_pointcloud_3d("Pointcloud3D.shp")
obj_xml.get_img_to_pointcloud_corresponding_for_arcgis(num)
cor_3D, cor_2D = obj_xml.get_img_to_pointcloud_corresponding(num)
```

---

## 贡献与交流

* 欢迎提交 issue、pull request 或通过邮件联系作者
* 如果你希望补充其他 XML 格式、支持新输出格式，欢迎参与开发！

---

## 许可协议

本项目基于 [MIT License](LICENSE) 开源。
请自由使用、修改和分发。

---

## 联系作者

* GitHub: [boywc](https://github.com/boywc)
* 邮箱: [liurenruis@gmail.com](mailto:liurenruis@gmail.com)
* 项目主页: [https://github.com/boywc/Analysis\_Photoscan\_XML\_tools](https://github.com/boywc/Analysis_Photoscan_XML_tools)
* Git 克隆地址: `https://github.com/boywc/Analysis_Photoscan_XML_tools.git`

---

**欢迎 Star、Fork 与使用本项目！如有问题或建议，欢迎通过 GitHub Issue 或邮件联系作者。**

---
