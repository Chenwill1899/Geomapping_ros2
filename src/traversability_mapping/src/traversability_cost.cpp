#include "utility.h"

#include <grid_map_ros/grid_map_ros.hpp>
#include <grid_map_msgs/msg/grid_map.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <tf2_eigen/tf2_eigen.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <vector>
#include <fstream>
#include <cmath>

using namespace grid_map;
using namespace std;

// 常量定义（需与配置文件保持一致）
#define MAP_SIZE 180         // 地图边长(栅格数)
#define localMapArrayLength 180  // 需与实际地图尺寸匹配
#define mapResolution 0.1      // 地图分辨率(m/栅格)

class TraversabilityCost : public rclcpp::Node {
private:
    // ROS 2订阅者和发布者
    rclcpp::Subscription<elevation_msgs::msg::OccupancyElevation>::SharedPtr sub_elevation_map;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_local_slope_cloud;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_local_uneven_cloud;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_local_step_cloud;
    rclcpp::Publisher<elevation_msgs::msg::OccupancyElevation>::SharedPtr pub_local_feature;

    // 消息和数据结构
    elevation_msgs::msg::OccupancyElevation elevation_map;  // 接收的高程地图
    elevation_msgs::msg::OccupancyElevation local_feature;   // 发布的特征地图
    pcl::PointCloud<PointType>::Ptr local_slope_map;
    pcl::PointCloud<PointType>::Ptr local_uneven_map;
    pcl::PointCloud<PointType>::Ptr local_step_map;
    Eigen::MatrixXf slope = Eigen::MatrixXf::Zero(MAP_SIZE, MAP_SIZE);
    Eigen::MatrixXf step = Eigen::MatrixXf::Zero(MAP_SIZE, MAP_SIZE);
    Eigen::Matrix<float, 3, 9> plane_fit_pinv_;

    // TF相关
    std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
    PointType robot_point;  // 机器人位置

public:
    TraversabilityCost() : Node("traversability_mapping") {
        // 初始化TF缓冲和监听器
        tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
        tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

        // 初始化订阅者和发布者
        sub_elevation_map = this->create_subscription<elevation_msgs::msg::OccupancyElevation>(
            "/msg_local_height", 5, 
            std::bind(&TraversabilityCost::elevation_map_handler, this, std::placeholders::_1)
        );

        pub_local_slope_cloud = this->create_publisher<sensor_msgs::msg::PointCloud2>(
            "/slope_pointcloud", 5
        );
        pub_local_uneven_cloud = this->create_publisher<sensor_msgs::msg::PointCloud2>(
            "/uneven_pointcloud", 5
        );
        pub_local_step_cloud = this->create_publisher<sensor_msgs::msg::PointCloud2>(
            "/step_pointcloud", 5
        );
        pub_local_feature = this->create_publisher<elevation_msgs::msg::OccupancyElevation>(
            "/msg_local_feature", 5
        );

        allocate_memory();
    }

    ~TraversabilityCost() {}

    void allocate_memory() {
        // 初始化点云指针
        local_slope_map.reset(new pcl::PointCloud<PointType>());
        local_uneven_map.reset(new pcl::PointCloud<PointType>());
        local_step_map.reset(new pcl::PointCloud<PointType>());

        // 初始化特征地图消息
        local_feature.header.frame_id = "map";
        local_feature.occupancy.info.width = localMapArrayLength;
        local_feature.occupancy.info.height = localMapArrayLength;
        local_feature.occupancy.info.resolution = mapResolution;
        
        local_feature.occupancy.info.origin.orientation.x = 0.0;
        local_feature.occupancy.info.origin.orientation.y = 0.0;
        local_feature.occupancy.info.origin.orientation.z = 0.0;
        local_feature.occupancy.info.origin.orientation.w = 1.0;

        // 预分配数组内存
        size_t map_size = localMapArrayLength * localMapArrayLength;
        local_feature.occupancy.data.resize(map_size, -1);
        local_feature.height.resize(map_size, 0.0f);
        local_feature.roughness.resize(map_size, 0.0f);
        local_feature.cost_map.resize(map_size, 0.0f);  // 注意：ROS 2消息字段名使用蛇形命名法

        Eigen::Matrix<float, 9, 3> left_m;
        left_m << 1,1,1, 1,2,1, 1,3,1,
                  2,1,1, 2,2,1, 2,3,1,
                  3,1,1, 3,2,1, 3,3,1;
        plane_fit_pinv_ = (left_m.transpose() * left_m).inverse() * left_m.transpose();
    }

   // 修改get_robot_position()函数中的TF查询部分
bool get_robot_position() {
    try {
        // 获取从base_link到map的变换（ROS 2 TF接口）
        geometry_msgs::msg::TransformStamped transform = tf_buffer_->lookupTransform(
            "map",                  // 目标坐标系
            "base_link",            // 源坐标系
            tf2::TimePointZero,     // 获取最新变换
            tf2::durationFromSec(1.0)  // 超时时间1秒（使用tf2的duration类型）
        );

        // 转换为机器人位置点
        robot_point.x = transform.transform.translation.x;
        robot_point.y = transform.transform.translation.y;
        robot_point.z = transform.transform.translation.z;
        return true;
    } catch (const tf2::TransformException& ex) {
        RCLCPP_ERROR(this->get_logger(), "TF变换失败: %s", ex.what());
        return false;
    }
}


    void elevation_map_handler(const elevation_msgs::msg::OccupancyElevation::ConstSharedPtr map_msg) {
        if (!get_robot_position()) {
            RCLCPP_WARN(this->get_logger(), "无法获取机器人位置");
            return;
        }

        elevation_map = *map_msg;
        slope.setZero();
        step.setZero();

        // 重置特征地图数据
        std::fill(local_feature.occupancy.data.begin(), local_feature.occupancy.data.end(), -1);
        std::fill(local_feature.height.begin(), local_feature.height.end(), 0.0f);
        std::fill(local_feature.roughness.begin(), local_feature.roughness.end(), 0.0f);
        std::fill(local_feature.cost_map.begin(), local_feature.cost_map.end(), 0.0f);

        // 更新时间戳和原点信息
        local_feature.header.stamp = this->now();
        local_feature.occupancy.header.stamp = local_feature.header.stamp;
        local_feature.occupancy.info.origin.position = map_msg->occupancy.info.origin.position;

        // 解析高程数据矩阵
        Eigen::MatrixXf dem_matrix_raw = Eigen::MatrixXf::Zero(MAP_SIZE, MAP_SIZE);
        int width = map_msg->occupancy.info.width;
        int height = map_msg->occupancy.info.height;

        for (int x = 0; x < width; ++x) {
            for (int y = 0; y < height; ++y) {
                size_t index = y * width + x;
                // 处理无效值（替换为0）
                if (map_msg->height[index] == -FLT_MAX) {
                    dem_matrix_raw(x, y) = 0.0f;
                } else {
                    dem_matrix_raw(x, y) = map_msg->height[index];
                }
            }
        }

        // 计算地形特征（坡度、台阶等）
        
        compute_terrain(dem_matrix_raw);
        

        // 填充点云和特征地图
        for (int i = 0; i < width; ++i) {
            for (int j = 0; j < height; ++j) {
                PointType slope_point;
                // 计算点云坐标（基于机器人位置）
                slope_point.x = (i - MAP_SIZE/2) * mapResolution + robot_point.x;
                slope_point.y = (j - MAP_SIZE/2) * mapResolution + robot_point.y;
                size_t index = i + j * width;

                // 仅处理有效点
                if (elevation_map.roughness[index] != -FLT_MAX) {
                    // 坡度点云
                    slope_point.z = slope(i, j);
                    local_slope_map->push_back(slope_point);

                    // 粗糙度点云（复用slope_point变量）
                    slope_point.z = elevation_map.roughness[index];
                    local_uneven_map->push_back(slope_point);

                    // 台阶点云
                    slope_point.z = step(i, j);
                    local_step_map->push_back(slope_point);

                    // 填充特征地图
                    local_feature.height[index] = step(i, j);
                    local_feature.roughness[index] = elevation_map.roughness[index];
                    local_feature.cost_map[index] = slope(i, j);  // 对应ROS 2消息的cost_map字段
                    local_feature.occupancy.data[index] = 0;
                }
            }
        }

        // 发布所有消息
        publish_slope_map();
        publish_uneven_map();
        publish_step_map();
        publish_feature_map();
    }

    void compute_terrain(const Eigen::MatrixXf& dem_matrix) {
        Eigen::Matrix3f dem_matrix_nn;  // 3x3邻域矩阵

        // 遍历所有非边缘点（避免越界）
        for (int n_row = 1; n_row < MAP_SIZE - 1; ++n_row) {
            for (int n_len = 1; n_len < MAP_SIZE - 1; ++n_len) {
                // 提取3x3邻域
                dem_matrix_nn = dem_matrix.block<3, 3>(n_row - 1, n_len - 1);

                // 跳过中心为空的点
                if (dem_matrix_nn(1, 1) == 0) continue;

                // 检查邻域是否有无效值（0）
                bool has_zero = (dem_matrix_nn.array() == 0).any();
                if (!has_zero) {
                    slope(n_row, n_len) = compute_theta(dem_matrix_nn);
                } else {
                    slope(n_row, n_len) = 0.0f;
                }

                // 计算台阶高度
                step(n_row, n_len) = compute_step(dem_matrix_nn);
            }
        }
    }

    float compute_step(const Eigen::Matrix3f& dem_nn) {
        float center_height = dem_nn(1, 1);
        float max_height_diff = 0.0f;

        // 遍历8个邻域点
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                if (i == 1 && j == 1) continue;  // 跳过中心
                if (dem_nn(i, j) == 0) continue;  // 跳过无效点
                max_height_diff = max(max_height_diff, abs(dem_nn(i, j) - center_height));
            }
        }
        return max_height_diff;
    }

    float compute_theta(const Eigen::Matrix3f& dem_nn) {
        // 构造最小二乘右边项（9个点的高程）
        Eigen::Matrix<float, 9, 1> right_m;
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                right_m[i*3 + j] = dem_nn(i, j);
            }
        }

        Eigen::Vector3f x = plane_fit_pinv_ * right_m;
        // 计算坡度（弧度）
        return acos(1.0f / sqrt(x[0]*x[0] + x[1]*x[1] + 1.0f));
    }

    // 发布坡度点云
    void publish_slope_map() {
        if (pub_local_slope_cloud->get_subscription_count() == 0) {
            local_slope_map->clear();
            return;
        }

        sensor_msgs::msg::PointCloud2 slope_msg;
        pcl::toROSMsg(*local_slope_map, slope_msg);
        slope_msg.header.frame_id = "map";
        slope_msg.header.stamp = this->now();
        pub_local_slope_cloud->publish(slope_msg);
        local_slope_map->clear();
    }

    // 发布粗糙度点云
    void publish_uneven_map() {
        if (pub_local_uneven_cloud->get_subscription_count() == 0) {
            local_uneven_map->clear();
            return;
        }

        sensor_msgs::msg::PointCloud2 uneven_msg;
        pcl::toROSMsg(*local_uneven_map, uneven_msg);
        uneven_msg.header.frame_id = "map";
        uneven_msg.header.stamp = this->now();
        pub_local_uneven_cloud->publish(uneven_msg);
        local_uneven_map->clear();
    }

    // 发布台阶点云
    void publish_step_map() {
        if (pub_local_step_cloud->get_subscription_count() == 0) {
            local_step_map->clear();
            return;
        }

        sensor_msgs::msg::PointCloud2 step_msg;
        pcl::toROSMsg(*local_step_map, step_msg);
        step_msg.header.frame_id = "map";
        step_msg.header.stamp = this->now();
        pub_local_step_cloud->publish(step_msg);
        local_step_map->clear();
    }

    // 发布特征地图
    void publish_feature_map() {
        if (pub_local_feature->get_subscription_count() == 0) return;
        pub_local_feature->publish(local_feature);
    }
};

int main(int argc, char**argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<TraversabilityCost>();
    RCLCPP_INFO(node->get_logger(), "Traversability mapping node started");
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
