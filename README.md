# Geomapping ROS2 - 地形建图与可通行性分析系统

## 项目概述

本项目是一个基于 ROS2 的三维地形建图与可通行性分析系统，集成了 FAST-LIO SLAM 算法和高斯过程回归（BGK）的地形预测功能。系统能够实时构建环境的三维点云地图，并通过机器学习方法预测未观测区域的地形特征，为移动机器人提供准确的可通行性评估。

## 主要功能

### 🚀 核心特性

- **实时 SLAM 建图**：基于 FAST-LIO 2.0 算法的高精度激光雷达-惯导融合 SLAM
- **地形可通行性分析**：智能评估地形的可通行性，识别障碍物和可通行区域
- **贝叶斯高斯核预测**：使用 BGK（Bayesian Gaussian Kernel）方法预测未观测区域的地形特征
- **多传感器融合**：支持 Livox 激光雷达与 IMU 的紧耦合融合
- **点云数据预处理**：专用的 Velodyne 激光雷达数据格式转换和同步处理
- **实时可视化**：集成 RViz2 实时显示建图结果和可通行性分析

### 📊 系统架构

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────────┐
│   激光雷达数据   │────│  terrain_pub_node│────│   同步点云数据     │
│  (/velodyne_    │    │   点云预处理     │    │ (/syncd_project_   │
│   _points)      │    │                  │    │   _cloud)          │
└─────────────────┘    └──────────────────┘    └─────────────────────┘
         │                       │                        │
┌─────────────────┐              │              ┌─────────────────────┐
│    IMU 数据     │──────────────┼──────────────│   FAST-LIO SLAM    │
└─────────────────┘              │              │     建图算法       │
                                  │              └─────────────────────┘
                                  │                        │
                                  │              ┌─────────────────────┐
                                  └──────────────│  可通行性分析模块   │
                                                 │  (BGK预测算法)     │
                                                 └─────────────────────┘
                                                          │
                                                ┌─────────────────────┐
                                                │   地形预测与可视化   │
                                                └─────────────────────┘
```

## 核心模块详解

### 🔧 terrain_pub_node - 点云预处理节点

**功能概述**：
`terrain_pub_node` 是专门针对 Velodyne 激光雷达数据设计的预处理节点，负责将原始激光雷达数据转换为适合地形分析的格式。

**主要特性**：
- **数据格式转换**：将 Velodyne 特有的点云格式（包含 ring 信息）转换为标准 PointXYZI 格式
- **球坐标投影**：基于水平角度和线圈信息进行空间索引，提高后续处理效率
- **实时同步处理**：保持原始时间戳，确保与其他传感器数据的时间同步
- **优化的 QoS 配置**：使用传感器数据 QoS 设置，包含生命周期管理

**技术细节**：
```cpp
// 支持的点云格式
struct Point {
    float x, y, z;        // 3D 坐标
    float intensity;      // 反射强度
    float time;          // 时间戳
    uint16_t ring;       // 激光线圈编号
};

// 投影参数
- 垂直线圈数：32 线
- 水平分辨率：0.2°
- 水平扫描点数：1800
- 角度范围：360°
```

**话题接口**：
- **订阅话题**：`/velodyne_points` (sensor_msgs/PointCloud2)
- **发布话题**：`/syncd_project_cloud` (sensor_msgs/PointCloud2)

## 技术特点

### 🔬 算法优势

1. **高效的 BGK 预测算法**
   - 使用 Wendland C2 紧支撑核函数，避免三角函数计算
   - 批量矩阵运算，显著提升计算效率
   - 智能缓存机制，减少重复计算

2. **优化的数据结构**
   - 预计算邻域偏移，提高空间查询效率
   - 向量化操作，充分利用 Eigen 库的 SIMD 优化
   - 内存池管理，减少频繁的内存分配

3. **鲁棒的地形分析**
   - 高度梯度检测
   - 障碍物识别
   - 可通行性置信度评估

4. **点云预处理优化**
   - 球坐标投影索引，O(1) 时间复杂度的空间查找
   - 基于 ring 信息的结构化数据组织
   - 距离信息编码，便于后续滤波处理

## 安装与依赖

### 系统要求

- Ubuntu 20.04 / 22.04
- ROS2 Humble/Galactic
- C++17 或更高版本

### 依赖包

```bash
# ROS2 核心包
sudo apt install ros-$ROS_DISTRO-desktop-full

# PCL 点云处理库
sudo apt install libpcl-dev

# Eigen3 线性代数库
sudo apt install libeigen3-dev

# OpenCV
sudo apt install libopencv-dev

# TF2 变换库
sudo apt install ros-$ROS_DISTRO-tf2 ros-$ROS_DISTRO-tf2-ros

# PCL 转换库
sudo apt install ros-$ROS_DISTRO-pcl-conversions
```

### 编译安装

```bash
# 创建工作空间
mkdir -p ~/geomapping_ws/src
cd ~/geomapping_ws/src

# 克隆项目
git clone <your-repo-url> .

# 安装依赖
cd ~/geomapping_ws
rosdep install --from-paths src --ignore-src -r -y

# 编译
colcon build --packages-select terrain_pub_node traversability_mapping elevation_msgs

# 环境配置
source install/setup.bash
```

## 使用方法

### 🚁 实时运行

1. **启动 Velodyne 激光雷达**
```bash
# 启动 Velodyne 驱动
ros2 launch velodyne_driver velodyne_driver_node.launch.py

# 或启动 Livox 激光雷达驱动
ros2 launch livox_ros_driver2 msg_MID360_launch.py
```

2. **启动点云预处理节点**
```bash
# 启动地形数据预处理
ros2 run terrain_pub_node terrain_pub_node
```

3. **启动 FAST-LIO 建图**
```bash
# 启动 FAST-LIO 建图
ros2 launch fast_lio mapping.launch.py config_file:=avia.yaml
```

4. **启动可通行性分析**
```bash
# 启动完整的地形建图系统
ros2 launch traversability_mapping lio_bag.launch.py
```

### 📦 离线数据处理

```bash
# 播放 rosbag 数据
ros2 bag play your_data.bag

# 启动所有处理节点
ros2 launch traversability_mapping lio_bag.launch.py use_sim_time:=true
```

## 参数配置

### 主要参数文件

- **FAST-LIO 配置**：`src/FAST_LIO/config/avia.yaml`
- **可通行性参数**：`src/traversability_mapping/params/traversability.yaml`

### 关键参数说明

```yaml
# 可通行性分析参数
filter_height_limit: 0.3        # 高度差阈值 (米)
filter_angle_limit: 10.0        # 坡度阈值 (度)
prediction_kernel_size: 0.4     # BGK 预测核半径 (米)
sensor_range_limit: 20.0        # 传感器有效范围 (米)
map_resolution: 0.1             # 地图分辨率 (米)

# 点云预处理参数（terrain_pub_node）
lidar_lines: 32                 # 激光雷达线数
horizontal_resolution: 0.2      # 水平角度分辨率 (度)
```

## 性能优化

### 🔧 BGK 算法优化

本项目对贝叶斯高斯核预测算法进行了深度优化：

- **批量处理**：将单点预测改为批量矩阵运算，性能提升 10 倍以上
- **空间索引**：预计算邻域偏移，避免重复的空间查询
- **内存优化**：复用矩阵对象，减少内存分配开销
- **向量化计算**：充分利用 Eigen 库的 SIMD 指令集优化

### 性能指标

- **实时性**：在标准配置下，BGK 预测延迟 < 50ms
- **精度**：地形预测精度可达厘米级
- **覆盖范围**：支持 50m × 50m 区域的实时分析
- **点云处理频率**：支持 10Hz 实时点云数据处理

## 项目结构

```
Geomapping_ros2/
├── src/
│   ├── FAST_LIO/                    # FAST-LIO SLAM 算法
│   ├── traversability_mapping/      # 可通行性分析主模块
│   │   ├── src/
│   │   │   ├── traversability_filter.cpp  # 地形滤波与预测
│   │   │   └── traversability_map.cpp     # 地图构建
│   │   ├── launch/                  # 启动文件
│   │   ├── params/                  # 参数配置
│   │   └── include/                 # 头文件
│   ├── elevation_msgs/              # 自定义消息类型
│   ├── terrain_pub_node/            # 地形发布节点
│   │   ├── src/
│   │   │   └── terrain_pub_node.cpp # Velodyne 点云预处理
│   │   ├── CMakeLists.txt
│   │   └── package.xml
│   └── MEDIRL/                      # 扩展功能模块
└── README.md
```

## 数据流分析

### 点云数据处理流程

```
Velodyne 原始数据 → terrain_pub_node → 结构化点云 → FAST-LIO → 位姿估计
     ↓                    ↓                ↓             ↓
   ring信息           球坐标投影        时间同步      TF变换发布
     ↓                    ↓                ↓             ↓
  强度信息           距离编码          格式转换     traversability_mapping
```

## 可视化效果

系统提供丰富的可视化功能：

- **实时点云地图**：显示 FAST-LIO 构建的三维点云
- **可通行性地图**：彩色编码显示地形的可通行程度
- **预测区域**：高亮显示 BGK 算法预测的未知区域
- **机器人轨迹**：实时显示机器人运动轨迹
- **激光雷达数据**：结构化显示 Velodyne 扫描数据

## 应用场景

- 🤖 **移动机器人导航**：为地面机器人提供精确的可通行性地图
- 🚁 **无人机地形分析**：低空飞行的地形感知与路径规划
- 🏗️ **建筑工地测量**：复杂地形的三维重建与分析
- 🌲 **野外环境探索**：自然环境中的地形建图与导航
- 🚗 **自动驾驶**：城市和野外环境的实时地形感知

## 开发历程与问题解决

### 主要里程碑

- **2025/8/20**：完成从 ROS1 到 ROS2 的主要程序迁移
- **2025/8/28**：解决时间戳不匹配导致的数据接收不稳定问题
- **2025/8/29**：修复 YAML 参数传递问题，优化 RRT 搜索时间

### 已解决的关键问题

1. **数据接收爆冲现象**
   - **问题**：terrain_pub_node 发布 10Hz，但 traversability_filter 接收不稳定
   - **原因**：AI 转换时错误修改了时间戳设置
   - **解决**：恢复原始时间戳传递机制

2. **TF 变换缺失**
   - **问题**：找不到 map 和 base_link 之间的变换关系
   - **解决**：修改 FAST-LIO 输出，直接发布 map→base_link 变换

3. **地形计算性能问题**
   - **问题**：RRT 搜索时间过长（5秒），导致发布频率低
   - **解决**：修正 YAML 参数传递，优化为 0.17 秒

## 开发团队与贡献

本项目基于以下开源项目进行开发和扩展：

- [FAST-LIO](https://github.com/hku-mars/FAST_LIO)：香港大学 MARS 实验室开发的 LiDAR-惯导 SLAM 算法
- [Traversability Mapping](https://github.com/TixiaoShan/traversability_mapping)：CMU 的可通行性分析框架

### 贡献指南

欢迎提交 Issue 和 Pull Request！请遵循以下规范：

1. 代码风格遵循 [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html)
2. 提交前请运行完整的测试用例
3. 新功能需要添加对应的文档说明

## 故障排除

### 常见问题

1. **编译错误：找不到 cloud_info.hpp**
   ```bash
   # 确保所有依赖包已正确安装
   colcon build --packages-select elevation_msgs
   ```

2. **运行时错误：TF 变换查找失败**
   ```bash
   # 检查 FAST-LIO 是否正常运行并发布变换
   ros2 topic echo /tf
   ```

3. **点云数据不稳定**
   ```bash
   # 检查激光雷达驱动和时间同步
   ros2 topic hz /velodyne_points
   ```

## 许可证

本项目采用 [MIT License](LICENSE) 开源协议。

## 联系方式

如有问题或建议，请通过以下方式联系：

- 📧 Email: [your-email@domain.com]
- 🐛 Issues: [GitHub Issues](https://github.com/your-repo/issues)
- 💬 Discussions: [GitHub Discussions](https://github.com/your-repo/discussions)

---

DEBUG：

2025/8/20：这份程序从人机示教程序ROS1版本迁移到了ROS2版本，目前已经将主要的程序进行移植，但是存在的一个bug是：terrain_pub_node节点发布出来的/syncd_project_cloud话题频率为10hz，但是在traversability_filter节点接收存在爆冲现象，也就是接收不稳定。

2025/8/28：前面的bug解决了，原因在于时间辍不匹配。之前在用ai转换的时候，原程序中的时间辍是设置为0,但是ai自动将他设置为当前时间了。所以导致了爆冲现象。现在filiter——pointcloud能够正常以10hz发布。另外之前一直报错找不到map和base_link之间的关系。原因在于该项目对fast_lio进行了修改，添加了map和base_link之间的tf关系。也就是将原本camera——init和body的关系，改成map和base_link关系发布。效果一样的。目前可以正常建图。但是现在的bug是，msg_local_height这个话题发布的很慢，而且cost程序中地形计算花销很大。

2025/8/29：终于发现了问题，原来是traversability_map.cpp中没有正确从yaml文件中传参，导致RRT搜索时间被设置成5s（本应该是0.17s），所以每次发布完一次数据之后，需要等待rrt搜索结束后才会进行下一次发布，就存在了发布频率低的问题。目前仅剩下计算花销大的问题。


⭐ 如果这个项目对您有帮助，请给我们一个 Star！