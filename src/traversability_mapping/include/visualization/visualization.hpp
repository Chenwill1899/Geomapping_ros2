#pragma once

#include <Eigen/Geometry> // For Eigen::Vector3d, Eigen::Matrix3d, Eigen::Quaterniond
#include <iostream>
#include <unordered_map>
#include <vector>       // For std::vector
#include <memory>       // For std::shared_ptr

// ROS 2 消息头文件
#include <nav_msgs/msg/path.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <geometry_msgs/msg/point.hpp> // For geometry_msgs::Point

// ROS 2 和 PCL 转换
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

// ROS 2 核心库
#include "rclcpp/rclcpp.hpp"


namespace visualization
{
    struct BALL
    {
        Eigen::Vector3d center;
        double radius;
        BALL(const Eigen::Vector3d &c, double r) : center(c), radius(r){};
        BALL(){};
    };

    struct ELLIPSOID
    {
        Eigen::Vector3d c;
        double rx, ry, rz;
        Eigen::Matrix3d R;
        ELLIPSOID(const Eigen::Vector3d &center, const Eigen::Vector3d &r, const Eigen::Matrix3d &rot)
            : c(center), rx(r.x()), ry(r.y()), rz(r.z()), R(rot){};
        ELLIPSOID(){};
    };

    // ROS 2 Publisher 是模板，需要使用基类指针存储
    using PublisherMap = std::unordered_map<std::string, rclcpp::PublisherBase::SharedPtr>;

    enum Color
    {
        white,
        red,
        green,
        blue,
        yellow,
        chartreuse,
        black,
        gray,
        orange,
        purple,
        pink,
        steelblue
    };

    class Visualization
    {
    private:
        // ROS 2 中不再使用 ros::NodeHandle，而是 rclcpp::Node::SharedPtr
        rclcpp::Node::SharedPtr node_;
        PublisherMap publisher_map_;

        void setMarkerColor(visualization_msgs::msg::Marker &marker, // 使用 ROS 2 消息类型
                            Color color = blue,
                            double a = 1)
        {
            marker.color.a = a;
            switch (color)
            {
            case white:
                marker.color.r = 1.0f; marker.color.g = 1.0f; marker.color.b = 1.0f; break;
            case red:
                marker.color.r = 1.0f; marker.color.g = 0.0f; marker.color.b = 0.0f; break;
            case green:
                marker.color.r = 0.0f; marker.color.g = 1.0f; marker.color.b = 0.0f; break;
            case blue:
                marker.color.r = 0.0f; marker.color.g = 0.0f; marker.color.b = 1.0f; break;
            case yellow:
                marker.color.r = 1.0f; marker.color.g = 1.0f; marker.color.b = 0.0f; break;
            case chartreuse:
                marker.color.r = 0.5f; marker.color.g = 1.0f; marker.color.b = 0.0f; break;
            case black:
                marker.color.r = 0.0f; marker.color.g = 0.0f; marker.color.b = 0.0f; break;
            case gray:
                marker.color.r = 0.5f; marker.color.g = 0.5f; marker.color.b = 0.5f; break;
            case orange:
                marker.color.r = 1.0f; marker.color.g = 0.5f; marker.color.b = 0.0f; break;
            case purple:
                marker.color.r = 0.5f; marker.color.g = 0.0f; marker.color.b = 1.0f; break;
            case pink:
                marker.color.r = 1.0f; marker.color.g = 0.0f; marker.color.b = 0.6f; break;
            case steelblue:
                marker.color.r = 0.4f; marker.color.g = 0.7f; marker.color.b = 1.0f; break;
            }
        }

        void setMarkerColor(visualization_msgs::msg::Marker &marker, // 使用 ROS 2 消息类型
                            double a,
                            double r,
                            double g,
                            double b)
        {
            marker.color.a = a;
            marker.color.r = static_cast<float>(r); // 确保类型匹配
            marker.color.g = static_cast<float>(g);
            marker.color.b = static_cast<float>(b);
        }

        void setMarkerScale(visualization_msgs::msg::Marker &marker, // 使用 ROS 2 消息类型
                            const double &x,
                            const double &y,
                            const double &z)
        {
            marker.scale.x = x;
            marker.scale.y = y;
            marker.scale.z = z;
        }

        void setMarkerPose(visualization_msgs::msg::Marker &marker, // 使用 ROS 2 消息类型
                           const double &x,
                           const double &y,
                           const double &z)
        {
            marker.pose.position.x = x;
            marker.pose.position.y = y;
            marker.pose.position.z = z;
            marker.pose.orientation.w = 1;
            marker.pose.orientation.x = 0;
            marker.pose.orientation.y = 0;
            marker.pose.orientation.z = 0;
        }
        template <class ROTATION>
        void setMarkerPose(visualization_msgs::msg::Marker &marker, // 使用 ROS 2 消息类型
                           const double &x,
                           const double &y,
                           const double &z,
                           const ROTATION &R)
        {
            marker.pose.position.x = x;
            marker.pose.position.y = y;
            marker.pose.position.z = z;
            Eigen::Quaterniond r(R);
            marker.pose.orientation.w = r.w();
            marker.pose.orientation.x = r.x();
            marker.pose.orientation.y = r.y();
            marker.pose.orientation.z = r.z();
        }

    public:
        // 构造函数现在接收 rclcpp::Node::SharedPtr
        explicit Visualization(rclcpp::Node::SharedPtr node) : node_(node) {}

        template <class CENTER, class TOPIC>
        void visualize_a_ball(const CENTER &c,
                              const double &r,
                              const TOPIC &topic,
                              const Color color = blue,
                              const double a = 1)
        {
            auto got = publisher_map_.find(topic);
            if (got == publisher_map_.end())
            {
                // 创建 ROS 2 Publisher
                // 注意：这里需要明确模板参数类型
                rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr pub =
                    node_->create_publisher<visualization_msgs::msg::Marker>(topic, 10);
                publisher_map_[topic] = pub; // 存储基类指针
            }
            visualization_msgs::msg::Marker marker; // 使用 ROS 2 消息类型
            marker.header.frame_id = "map";
            marker.type = visualization_msgs::msg::Marker::SPHERE;
            marker.action = visualization_msgs::msg::Marker::ADD;
            setMarkerColor(marker, color, a);
            setMarkerScale(marker, 2 * r, 2 * r, 2 * r);
            setMarkerPose(marker, c[0], c[1], c[2]);
            marker.header.stamp = node_->get_clock()->now(); // 获取 ROS 2 时间
            // 发布时需要进行 static_pointer_cast
            std::static_pointer_cast<rclcpp::Publisher<visualization_msgs::msg::Marker>>(publisher_map_[topic])->publish(marker);
        }

        template <class PC, class TOPIC>
        void visualize_pointcloud(const PC &pc, const TOPIC &topic)
        {
            auto got = publisher_map_.find(topic);
            if (got == publisher_map_.end())
            {
                // 创建 ROS 2 Publisher
                rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub =
                    node_->create_publisher<sensor_msgs::msg::PointCloud2>(topic, 10);
                publisher_map_[topic] = pub; // 存储基类指针
            }
            pcl::PointCloud<pcl::PointXYZ> point_cloud;
            sensor_msgs::msg::PointCloud2 point_cloud_msg; // 使用 ROS 2 消息类型
            point_cloud.reserve(pc.size());
            for (const auto &pt : pc)
            {
                point_cloud.points.emplace_back(pt[0], pt[1], pt[2]);
            }
            pcl::toROSMsg(point_cloud, point_cloud_msg); // 使用 pcl_conversions
            point_cloud_msg.header.frame_id = "map";
            point_cloud_msg.header.stamp = node_->get_clock()->now(); // 获取 ROS 2 时间
            // 发布时需要进行 static_pointer_cast
            std::static_pointer_cast<rclcpp::Publisher<sensor_msgs::msg::PointCloud2>>(publisher_map_[topic])->publish(point_cloud_msg);
        }

        template <class PATH, class TOPIC>
        void visualize_path(const PATH &path, const TOPIC &topic)
        {
            auto got = publisher_map_.find(topic);
            if (got == publisher_map_.end())
            {
                // 创建 ROS 2 Publisher
                rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pub =
                    node_->create_publisher<nav_msgs::msg::Path>(topic, 10);
                publisher_map_[topic] = pub; // 存储基类指针
            }
            nav_msgs::msg::Path path_msg; // 使用 ROS 2 消息类型
            geometry_msgs::msg::PoseStamped tmpPose; // 使用 ROS 2 消息类型
            tmpPose.header.frame_id = "map";
            for (const auto &pt : path)
            {
                tmpPose.pose.position.x = pt[0];
                tmpPose.pose.position.y = pt[1];
                tmpPose.pose.position.z = pt[2];
                path_msg.poses.push_back(tmpPose);
            }
            path_msg.header.frame_id = "map";
            path_msg.header.stamp = node_->get_clock()->now(); // 获取 ROS 2 时间
            // 发布时需要进行 static_pointer_cast
            std::static_pointer_cast<rclcpp::Publisher<nav_msgs::msg::Path>>(publisher_map_[topic])->publish(path_msg);
        }

        template <class BALLS, class TOPIC>
        void visualize_balls(const BALLS &balls,
                             const TOPIC &topic,
                             const Color color = blue,
                             const double a = 0.2)
        {
            auto got = publisher_map_.find(topic);
            if (got == publisher_map_.end())
            {
                // 创建 ROS 2 Publisher
                rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub =
                    node_->create_publisher<visualization_msgs::msg::MarkerArray>(topic, 10);
                publisher_map_[topic] = pub; // 存储基类指针
            }
            visualization_msgs::msg::Marker marker; // 使用 ROS 2 消息类型
            marker.header.frame_id = "map";
            marker.type = visualization_msgs::msg::Marker::SPHERE;
            marker.action = visualization_msgs::msg::Marker::ADD;
            marker.id = 0;
            setMarkerColor(marker, color, a);
            visualization_msgs::msg::MarkerArray marker_array; // 使用 ROS 2 消息类型
            marker_array.markers.reserve(balls.size() + 1);
            marker.action = visualization_msgs::msg::Marker::DELETEALL;
            marker_array.markers.push_back(marker);
            marker.action = visualization_msgs::msg::Marker::ADD;
            for (const auto &ball : balls)
            {
                setMarkerPose(marker, ball.center[0], ball.center[1], ball.center[2]);
                auto d = 2 * ball.radius;
                setMarkerScale(marker, d, d, d);
                marker_array.markers.push_back(marker);
                marker.id++;
            }
            // 发布时需要进行 static_pointer_cast
            std::static_pointer_cast<rclcpp::Publisher<visualization_msgs::msg::MarkerArray>>(publisher_map_[topic])->publish(marker_array);
        }

        template <class ELLIPSOIDS, class TOPIC>
        void visualize_ellipsoids(const ELLIPSOIDS &ellipsoids,
                                  const TOPIC &topic,
                                  const Color color = blue,
                                  const double a = 0.2)
        {
            auto got = publisher_map_.find(topic);
            if (got == publisher_map_.end())
            {
                // 创建 ROS 2 Publisher
                rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub =
                    node_->create_publisher<visualization_msgs::msg::MarkerArray>(topic, 10);
                publisher_map_[topic] = pub; // 存储基类指针
            }
            visualization_msgs::msg::Marker marker; // 使用 ROS 2 消息类型
            marker.header.frame_id = "map";
            marker.type = visualization_msgs::msg::Marker::SPHERE;
            marker.action = visualization_msgs::msg::Marker::ADD;
            marker.id = 0;
            setMarkerColor(marker, color, a);
            visualization_msgs::msg::MarkerArray marker_array; // 使用 ROS 2 消息类型
            marker_array.markers.reserve(ellipsoids.size() + 1);
            marker.action = visualization_msgs::msg::Marker::DELETEALL;
            marker_array.markers.push_back(marker);
            marker.action = visualization_msgs::msg::Marker::ADD;
            for (const auto &e : ellipsoids)
            {
                setMarkerPose(marker, e.c[0], e.c[1], e.c[2], e.R);
                setMarkerScale(marker, 2 * e.rx, 2 * e.ry, 2 * e.rz);
                marker_array.markers.push_back(marker);
                marker.id++;
            }
            // 发布时需要进行 static_pointer_cast
            std::static_pointer_cast<rclcpp::Publisher<visualization_msgs::msg::MarkerArray>>(publisher_map_[topic])->publish(marker_array);
        }

        template <class PAIRLINE, class TOPIC>
        // eg for PAIRLINE: std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>>
        void visualize_pairline(const PAIRLINE &pairline, const TOPIC &topic, const Color &color = green, double scale = 0.1)
        {
            auto got = publisher_map_.find(topic);
            if (got == publisher_map_.end())
            {
                // 创建 ROS 2 Publisher
                rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr pub =
                    node_->create_publisher<visualization_msgs::msg::Marker>(topic, 10);
                publisher_map_[topic] = pub; // 存储基类指针
            }
            visualization_msgs::msg::Marker marker; // 使用 ROS 2 消息类型
            marker.header.frame_id = "map";
            marker.type = visualization_msgs::msg::Marker::LINE_LIST;
            marker.action = visualization_msgs::msg::Marker::ADD;
            setMarkerPose(marker, 0, 0, 0);
            setMarkerColor(marker, color, 1);
            setMarkerScale(marker, scale, scale, scale);
            marker.points.resize(2 * pairline.size());
            for (size_t i = 0; i < pairline.size(); ++i)
            {
                marker.points[2 * i + 0].x = pairline[i].first[0];
                marker.points[2 * i + 0].y = pairline[i].first[1];
                marker.points[2 * i + 0].z = pairline[i].first[2];
                marker.points[2 * i + 1].x = pairline[i].second[0];
                marker.points[2 * i + 1].y = pairline[i].second[1];
                marker.points[2 * i + 1].z = pairline[i].second[2];
            }
            // 发布时需要进行 static_pointer_cast
            std::static_pointer_cast<rclcpp::Publisher<visualization_msgs::msg::Marker>>(publisher_map_[topic])->publish(marker);
        }

        template <class ARROWS, class TOPIC>
        // ARROWS: pair<Vector3d, Vector3d>
        void visualize_arrows(const ARROWS &arrows, const TOPIC &topic, const Color &color)
        {
            auto got = publisher_map_.find(topic);
            if (got == publisher_map_.end())
            {
                // 创建 ROS 2 Publisher
                rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub =
                    node_->create_publisher<visualization_msgs::msg::MarkerArray>(topic, 10);
                publisher_map_[topic] = pub; // 存储基类指针
            }
            visualization_msgs::msg::Marker clear_previous_msg; // 使用 ROS 2 消息类型
            clear_previous_msg.action = visualization_msgs::msg::Marker::DELETEALL;
            visualization_msgs::msg::Marker arrow_msg; // 使用 ROS 2 消息类型
            arrow_msg.type = visualization_msgs::msg::Marker::ARROW;
            arrow_msg.action = visualization_msgs::msg::Marker::ADD;
            arrow_msg.header.frame_id = "map";
            arrow_msg.id = 0;
            arrow_msg.points.resize(2);
            setMarkerPose(arrow_msg, 0, 0, 0);
            setMarkerScale(arrow_msg, 0.4, 0.7, 0); // 注意：第三个参数 scale.z 设为 0 可能导致箭头不显示
            setMarkerColor(arrow_msg, color, 0.7);
            visualization_msgs::msg::MarkerArray arrow_list_msg; // 使用 ROS 2 消息类型
            arrow_list_msg.markers.reserve(1 + arrows.size());
            arrow_list_msg.markers.push_back(clear_previous_msg);
            for (const auto &arrow : arrows)
            {
                arrow_msg.points[0].x = arrow.first[0];
                arrow_msg.points[0].y = arrow.first[1];
                arrow_msg.points[0].z = arrow.first[2];
                arrow_msg.points[1].x = arrow.second[0];
                arrow_msg.points[1].y = arrow.second[1];
                arrow_msg.points[1].z = arrow.second[2];
                arrow_list_msg.markers.push_back(arrow_msg);
                arrow_msg.id += 1;
            }
            // 发布时需要进行 static_pointer_cast
            std::static_pointer_cast<rclcpp::Publisher<visualization_msgs::msg::MarkerArray>>(publisher_map_[topic])->publish(arrow_list_msg);
        }

        template <class TRAJ, class TOPIC>
        // TRAJ:
        void visualize_traj(const TRAJ &traj, const TOPIC &topic)
        {
            std::vector<Eigen::Vector3d> path;
            auto duration = traj.getTotalDuration();
            for (double t = 0; t < duration; t += 0.01)
            {
                path.push_back(traj.getPos(t));
            }
            visualize_path(path, topic);
            std::vector<Eigen::Vector3d> wayPts;
            for (const auto &piece : traj)
            {
                wayPts.push_back(piece.getPos(0));
            }
            // topic 是 TOPIC 类型，这里需要确保它能转换为 std::string
            // 如果 TOPIC 是 const char* 或 std::string，这没问题
            // 但如果 TOPIC 是其他类型，需要显式转换
            visualize_pointcloud(wayPts, static_cast<std::string>(topic) + "_wayPts");
        }

        template <class TRAJLIST, class TOPIC>
        // TRAJLIST: std::vector<TRAJ>
        void visualize_traj_list(const TRAJLIST &traj_list, const TOPIC &topic,
                                 const Color color = blue, double scale = 0.1)
        {
            auto got = publisher_map_.find(topic);
            if (got == publisher_map_.end())
            {
                // 创建 ROS 2 Publisher
                rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub =
                    node_->create_publisher<visualization_msgs::msg::MarkerArray>(topic, 10);
                publisher_map_[topic] = pub; // 存储基类指针
            }
            visualization_msgs::msg::Marker clear_previous_msg; // 使用 ROS 2 消息类型
            clear_previous_msg.action = visualization_msgs::msg::Marker::DELETEALL;
            visualization_msgs::msg::Marker path_msg; // 使用 ROS 2 消息类型
            path_msg.type = visualization_msgs::msg::Marker::LINE_STRIP;
            path_msg.action = visualization_msgs::msg::Marker::ADD;
            path_msg.header.frame_id = "map";
            path_msg.id = 0;
            setMarkerPose(path_msg, 0, 0, 0);
            setMarkerScale(path_msg, scale, scale, scale);
            visualization_msgs::msg::MarkerArray path_list_msg; // 使用 ROS 2 消息类型
            path_list_msg.markers.reserve(1 + traj_list.size());
            path_list_msg.markers.push_back(clear_previous_msg);
            double a_step = 0.8 / traj_list.size();
            double a = 1.0;
            geometry_msgs::msg::Point p_msg; // 使用 ROS 2 消息类型
            for (const auto &traj : traj_list)
            {
                setMarkerColor(path_msg, color, a);
                // a = a + a_step; // 原始代码注释，保留
                path_msg.points.clear();
                for (double t = 0; t < traj.getTotalDuration(); t += 0.01)
                {
                    auto p = traj.getPos(t);
                    p_msg.x = p.x();
                    p_msg.y = p.y();
                    p_msg.z = p.z();
                    path_msg.points.push_back(p_msg);
                }
                path_list_msg.markers.push_back(path_msg);
                path_msg.id += 1;
            }
            // 发布时需要进行 static_pointer_cast
            std::static_pointer_cast<rclcpp::Publisher<visualization_msgs::msg::MarkerArray>>(publisher_map_[topic])->publish(path_list_msg);
        }

        template <class PATHLIST, class TOPIC>
        // PATHLIST: std::vector<PATH>
        void visualize_path_list(const PATHLIST &path_list, const TOPIC &topic,
                                 const Color color = steelblue, double scale = 0.1)
        {
            auto got = publisher_map_.find(topic);
            if (got == publisher_map_.end())
            {
                // 创建 ROS 2 Publisher
                rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub =
                    node_->create_publisher<visualization_msgs::msg::MarkerArray>(topic, 10);
                publisher_map_[topic] = pub; // 存储基类指针
            }
            visualization_msgs::msg::Marker clear_previous_msg; // 使用 ROS 2 消息类型
            clear_previous_msg.action = visualization_msgs::msg::Marker::DELETEALL;
            visualization_msgs::msg::Marker path_msg; // 使用 ROS 2 消息类型
            path_msg.type = visualization_msgs::msg::Marker::LINE_STRIP;
            path_msg.action = visualization_msgs::msg::Marker::ADD;
            path_msg.header.frame_id = "map";
            path_msg.id = 0;
            setMarkerPose(path_msg, 0, 0, 0);
            setMarkerScale(path_msg, scale, scale, scale);
            visualization_msgs::msg::MarkerArray path_list_msg; // 使用 ROS 2 消息类型
            path_list_msg.markers.reserve(1 + path_list.size());
            path_list_msg.markers.push_back(clear_previous_msg);
            setMarkerColor(path_msg, color);
            for (const auto &path : path_list)
            {
                path_msg.points.resize(path.size());
                for (size_t i = 0; i < path.size(); ++i)
                {
                    path_msg.points[i].x = path[i].x();
                    path_msg.points[i].y = path[i].y();
                    path_msg.points[i].z = path[i].z();
                }
                path_list_msg.markers.push_back(path_msg);
                path_msg.id += 1;
            }
            // 发布时需要进行 static_pointer_cast
            std::static_pointer_cast<rclcpp::Publisher<visualization_msgs::msg::MarkerArray>>(publisher_map_[topic])->publish(path_list_msg);
        }

        template <class TOPIC_TYPE, class TOPIC>
        void registe(const TOPIC& topic) {
            auto got = publisher_map_.find(topic);
            if (got == publisher_map_.end()) {
                // 创建 ROS 2 Publisher
                // 注意：这里 TOPIC_TYPE 是消息类型，topic 是话题名
                typename rclcpp::Publisher<TOPIC_TYPE>::SharedPtr pub =
                    node_->create_publisher<TOPIC_TYPE>(topic, 10);
                publisher_map_[topic] = pub; // 存储基类指针
            }
        }
    };

} // namespace visualization