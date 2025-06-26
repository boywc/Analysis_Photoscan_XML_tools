# Photoscan Aerial Triangulation XML Parser

**Photoscan 空中三角测量 XML 文件解析与数据导出工具包**

本项目为论文 [**"High-Resolution and Spatial-Continuous 3-D Model Reconstruction of Martian Surface by Integrating Multisensor Data of Zhurong Rover "**](https://ieeexplore.ieee.org/abstract/document/10536102)（见IEEE Xplore）所配套的开源代码。旨在为三维重建与火星巡视器数据处理等场景提供一套**高效、开源的 Photoscan 空中三角测量 XML 结果解析工具**，支持批量读取、数据导出、相机参数提取、三维点云处理、配准文件生成和多种可视化分析，极大提升后处理自动化效率。

项目主页：[https://github.com/boywc/Analysis\_Photoscan\_XML\_tools](https://github.com/boywc/Analysis_Photoscan_XML_tools)

论文链接：[https://ieeexplore.ieee.org/abstract/document/10536102](https://ieeexplore.ieee.org/abstract/document/10536102)

---

## 数据与目录结构说明

示例数据以祝融号255号测站为例，项目目录结构如下：

```
项目根目录
│
├─ DemoForPhotoscanXMLAnalyse.py   # PhotoscanXML功能演示Demo（原main.py）
├─ DemoForMathTools3D.py           # 三维摄影测量数学工具演示Demo（新）
├─ PhotoscanXMLAnalyse.py          # Photoscan解析与导出主功能库
├─ MathTools3D.py                  # 摄影测量核心数学函数库（新）
├─ README.md                       # 项目说明文档
│
├─ testdata/                       # 测试数据目录
│   ├─ 255/                        # 存储255号站点原始影像（可选）
│   ├─ 255.xml                     # Photoscan导出的空三角测量XML
│   ├─ information_2CL.csv         # 影像与位置信息的对应表
```

**说明：**

* `testdata/255/` 为影像文件夹，存放255站点拍摄的原始影像（如有需要可自行添加影像）。
* `testdata/information_2CL.csv` 为影像的位置信息（如像控点、外参等辅助数据）。
* `testdata/255.xml` 是基于上述影像与位置信息，用Photoscan导出的空三角XML文件，本工具包所有解析、导出操作均基于该文件展开。

---

## 特性与功能

* 自动解析 Photoscan 导出的空三角测量 XML 文件
* 批量提取并导出相机位姿、内参、畸变参数、三维点云等核心数据
* 支持导出为 CSV、MAT、SHP 等常用格式
* 影像像素与三维点云空间关系一键查询、输出
* 快速生成 ArcGIS 影像-点云配准 TXT 文件
* 支持三维可视化分析（相机投影向量、点云等）
* 查询点云空间范围（XYZ轴最大最小值）
* 任意坐标点高程插值（点云地形断面获取）
* **支持提取三维点云与影像同名点的RGB颜色信息，实现彩色点云重建**
* **三维可视化显示指定相机点云与姿态（`point_and_camera_display`，新功能）**
* **在原始影像上高亮/保存同名点标记（`draw_points_on_image`，新功能）**
* **三维点云投影到影像并可视化（`project_3d_to_2d_display`，新功能）**

  * 任意三维点云可通过相机内外参和畸变参数准确投影至影像平面，实现成果可视化/误差分析/结构校验等场景
* 兼容外部工具（如 OpenCV PNP、ArcGIS DEM）
* **\[新] MathTools3D.py 提供三维摄影测量核心数学函数（自定义/OPENCV解算、重投影、位姿转换等）**

---

## 安装与环境依赖

建议 Python 3.8+ 环境，依赖如下：

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

> 或将 `PhotoscanXMLAnalyse.py`、`MathTools3D.py` 直接拷贝到你的项目中也可使用。

---

## 数据准备与快速上手

### 1. 数据准备

请确保 `testdata` 目录下存在如下文件：

* `255.xml`：Photoscan导出的空三角测量XML（主解析对象）
* `information_2CL.csv`：影像对应的位置信息（如像控点或航位推算结果）
* 可选：`255/` 文件夹存放原始影像文件

### 2. 演示样例

#### DemoForPhotoscanXMLAnalyse.py （推荐从此脚本入门 Photoscan XML 数据批量处理）

（见下方典型工作流示例）

#### DemoForMathTools3D.py （推荐从此脚本体验核心摄影测量数学功能）

（见下方典型工作流示例）

---

## 典型工作流示例

### 1. PhotoscanXMLAnalyse 典型工作流

```python
from PhotoscanXMLAnalyse import ana_photoscan_xml, draw_points_on_image, project_3d_to_2d_display
import numpy as np
import cv2

def main():
    xml_path = "testdata/255.xml"
    obj_xml = ana_photoscan_xml(xml_path)

    # XML 解析与数据导出
    obj_xml.save_xml_pose("testdata/xml_information.csv")

    # 提取相机参数
    img_name, rays_o, rays_d = obj_xml.get_rays_np_around_five_point(0)

    # 点云导出与可视化
    obj_xml.save_pointcloud_3d("testdata/pointcloud3d.shp")
    obj_xml.point_and_camera_display([0], point_size=3, camera_box_size=3)

    # 影像-点云配准及同名点获取
    cor_3D, cor_2D, _ = obj_xml.get_img_to_pointcloud_corresponding_with_color(0)
    image = cv2.imread(f"testdata/255/{img_name}")
    if image is not None:
        draw_points_on_image(image, cor_2D, half_size=5, output_mode=None)

    # 三维点云投影到影像
    distortion = obj_xml.get_Distortion()
    dist_coeffs = np.array([
        distortion.get("K1", 0), distortion.get("K2", 0),
        distortion.get("P1", 0), distortion.get("P2", 0), distortion.get("K3", 0)
    ])
    img_K, img_pose, img_idx = obj_xml.get_cam_parameter_matrix(img_name.split('.')[0])
    points_3d, points_2d, _ = obj_xml.get_img_to_pointcloud_corresponding_with_color(img_idx)
    img_proj = cv2.imread(f"testdata/255/{img_name}")
    project_3d_to_2d_display(points_3d, img_K, img_pose, dist_coeffs=dist_coeffs,
                             image=img_proj, output_mode=None, half_size=5)

if __name__ == "__main__":
    main()
```

---

### 2. MathTools3D 典型工作流

> 用于三维摄影测量的数学基础函数（重投影、标定、多方法对比）

```python
from MathTools3D import *
from PhotoscanXMLAnalyse import ana_photoscan_xml
import numpy as np

# 获取数据与同名点
obj_xml = ana_photoscan_xml("testdata/255.xml")
img_A_K, img_A_pose, num_A = obj_xml.get_cam_parameter_matrix("20220130181054")
points_3d, point_2d, _ = obj_xml.get_img_to_pointcloud_corresponding_with_color(num_A)

# 1. 公式法标定相机内外参
K, pose = camera_calibration(points_3d, point_2d, ror3_zf=[1, 1])
min_loss, min_k, min_p = camera_calibration_with_verify(points_3d, point_2d)

# 2. OpenCV PNP法（无畸变/有畸变）求解外参
pos_pnp = opencv_PNP(points_3d, point_2d, img_A_K)
distor = obj_xml.get_Distortion()
dist_coeffs = np.array([
    distor.get("K1", 0), distor.get("K2", 0),
    distor.get("P1", 0), distor.get("P2", 0), distor.get("K3", 0)
])
pos_pnp_dist = opencv_PNP(points_3d, point_2d, img_A_K, distortion=dist_coeffs)

# 3. 重投影误差分析（本地/Opencv两种方法）
project_2d = point_3D_projection_to_camera(K, pose, points_3d)
loss = np.mean(np.abs(project_2d - point_2d))
project_2d_cv = opencv_3D_projection_to_img(points_3d, K, pose, dist=dist_coeffs)
loss_cv = np.mean(np.abs(project_2d_cv - point_2d))

# 4. 外参格式互转（XML输出与数学解互通）
mathpos = xmlpos_to_mathpos(img_A_pose)
```

---

## API 参考手册

### 1. PhotoscanXMLAnalyse.py API 参考手册

| 方法名                                                                                                                       | 功能描述                                           | 参数说明                                                                                                                              | 返回值                                            |
| ------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------- |
| `ana_photoscan_xml(xml_file)`                                                                                             | 读取 XML 文件，创建解析对象                               | `xml_file`: XML文件名                                                                                                                | 类对象                                            |
| `save_xml_pose(filename=None)`                                                                                            | 保存相机位姿 CSV                                     | `filename`: 文件名（可选）                                                                                                               | 无                                              |
| `get_rays_np_around_five_point(num)`                                                                                      | 获取相机5点（四角+中心）向量                                | `num`: 影像编号                                                                                                                       | img名, 原点, 方向                                   |
| `save_five_point_vector(filename=None)`                                                                                   | 批量保存所有相机五点向量                                   | `filename`: 文件名（可选）                                                                                                               | 无                                              |
| `draw_pose_vector(size=0.2)`                                                                                              | 绘制所有相机姿态三维箭头图                                  | `size`: 向量缩放因子                                                                                                                    | 无                                              |
| `save_pointcloud_3d(save_path)`                                                                                           | 导出三维点云为 SHP 或 MAT                              | `save_path`: 文件名（后缀自动判别）                                                                                                          | 无                                              |
| `get_img_to_pointcloud_corresponding_for_arcgis(num)`                                                                     | 导出同名点像素坐标与对应的点云XY坐标生成配准txt文件（用于ArcGIS影像配准到DEM） | `num`: 影像编号                                                                                                                       | 无                                              |
| `get_img_to_pointcloud_corresponding(num)`                                                                                | 获取某影像2D像素与3D点的全部配对                             | `num`: 影像编号                                                                                                                       | 3D点数组, 2D点数组                                   |
| `get_img_to_pointcloud_corresponding_with_color(num)`                                                                     | 获取某影像2D像素、3D点及颜色(RGB)的全部配对（**新功能**）            | `num`: 影像编号                                                                                                                       | 3D点数组, 2D点数组, 颜色数组                             |
| `get_img_to_pointcloud_corresponding_couple(num1, num2)`                                                                  | 获取两影像同名点与三维点配对                                 | `num1`, `num2`: 两影像编号                                                                                                             | 2D点1, 2D点2, 3D点数组                              |
| `get_cam_parameter_matrix(keyword)`                                                                                       | 按影像名关键字查找内外参矩阵                                 | `keyword`: 字符串关键字                                                                                                                 | 内参, 外参, 编号                                     |
| `get_Distortion()`                                                                                                        | 获取畸变参数                                         | 无                                                                                                                                 | 畸变参数字典                                         |
| `check_cloud_range()`                                                                                                     | 查询点云的空间范围（XYZ轴最大最小值）                           | 无                                                                                                                                 | min\_x, max\_x, min\_y, max\_y, min\_z, max\_z |
| `get_elevation(points)`                                                                                                   | 点云插值获取指定点高程                                    | points: \[n,2]点坐标                                                                                                                 | n个点的高程数组                                       |
| `point_and_camera_display(num_list, ...)`                                                                                 | 三维可视化显示指定相机的同名点云、相机位置及视锥体等（**新功能**）            | `num_list`：相机索引列表                                                                                                                 | None                                           |
| `draw_points_on_image(image, points, ...)`                                                                                | 在影像上绘制点标记并弹窗/保存/返回（**新功能**，工具函数，类外）            | `image`，`points`，...                                                                                                              | 返回新图（或弹窗/保存）                                   |
| `project_3d_to_2d_display(points_3d, K, Rt, dist_coeffs=None, image=None, output_mode=None, WH=(2048,2048), half_size=5)` | 三维点云投影到影像平面，可视化/返回2D坐标/保存（**新功能**，工具函数，类外）     | `points_3d`：三维点；`K`：内参矩阵；`Rt`：外参矩阵；`dist_coeffs`：畸变参数；`image`：底图（可选）；`output_mode`：None弹窗/1返回2D/str保存；`WH`：输出分辨率；`half_size`：标记大小 | 返回2D像素点或弹窗显示/保存图像                              |

---

### 2. MathTools3D.py API 参考手册

| 方法名                                                      | 功能描述               | 参数说明                       | 返回值                            |
| -------------------------------------------------------- | ------------------ | -------------------------- | ------------------------------ |
| `camera_calibration(pts3d, pts2d, ror3_zf=[1,1])`        | 标定解算相机内外参（四解之一）    | 三维点，二维点，ror3\_zf参数（四种组合之一） | K, pose                        |
| `camera_calibration_with_verify(pts3d, pts2d)`           | 自动遍历四解，返回重投影误差最小的解 | 三维点，二维点                    | min\_loss, best\_K, best\_pose |
| `point_3D_projection_to_camera(K, pose, pts3d)`          | 用指定内外参将3D点投影回2D点   | K，外参，三维点                   | 投影后2D点                         |
| `opencv_PNP(pts3d, pts2d, K, distortion=None)`           | 调用OpenCV的PNP算法计算外参 | 3D点、2D点、内参K、畸变参数可选         | pose(3x4)                      |
| `opencv_3D_projection_to_img(pts3d, K, pose, dist=None)` | OpenCV方式3D->2D投影   | 3D点、K、外参、畸变                | 2D点                            |
| `xmlpos_to_mathpos(pos)`                                 | XML外参格式与数学解外参互转    | pos矩阵（3x4）                 | 转换后的pos                        |

---

> 更多详细参数说明和使用说明，请见各 .py 文件头部说明与函数 docstring。你也可以在 IDE 中使用函数提示查看参数和用法。

---

## 贡献与交流

* 欢迎提交 issue、pull request 或通过邮件联系作者
* 如果你希望补充其他 XML 格式、支持新输出格式，欢迎参与开发！

---

## 许可协议

本项目基于 [MIT License](LICENSE) 开源。
请自由使用、修改和分发。

---

## 代码维护与联系

**主要维护人员：**

* GitHub: [Henry-Baiheng](https://github.com/Henry-Baiheng)
* 邮箱: [1243244190@qq.com](mailto:1243244190@qq.com)

**作者与论文仓库：**

* GitHub: [boywc](https://github.com/boywc)
* 邮箱: [liurenruis@gmail.com](mailto:liurenruis@gmail.com)
* 项目主页: [https://github.com/boywc/Analysis\_Photoscan\_XML\_tools](https://github.com/boywc/Analysis_Photoscan_XML_tools)
* 论文链接：[https://ieeexplore.ieee.org/abstract/document/10536102](https://ieeexplore.ieee.org/abstract/document/10536102)
* Git 克隆地址: `https://github.com/boywc/Analysis_Photoscan_XML_tools.git`

---

\*\*欢迎 Star、Fork 与使用本项目！如有问题或建议，


欢迎通过 GitHub Issue 或邮件联系作者或维护者。\*\*

---
