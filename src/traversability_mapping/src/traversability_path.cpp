#include "utility.h"

#include "elevation_msgs/msg/occupancy_elevation.hpp"
#include "traversability_mapping/msg/polynome.hpp"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <memory>
#include <mutex>
#include <vector>

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <geometry_msgs/msg/vector3.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <nav_msgs/msg/path.hpp>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl_conversions/pcl_conversions.h>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <tf2/LinearMath/Transform.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

class TraversabilityPath : public rclcpp::Node {
public:
    TraversabilityPath()
        : rclcpp::Node("traversability_path"),
          planning_flag_(false),
          state_list_size_(0),
          root_state_(new state_t),
          goal_state_(new state_t),
          tf_buffer_(std::make_shared<tf2_ros::Buffer>(this->get_clock())),
          tf_listener_(std::make_shared<tf2_ros::TransformListener>(*tf_buffer_)) {
        radius_ = this->declare_parameter<double>("local_plan.radius", 0.3);
        visitflag_ = this->declare_parameter<bool>("local_plan.visitflag", false);
        cost_max_ = this->declare_parameter<double>("planning.cost_max", 50.0);
        costmap_inflation_radius_ = this->declare_parameter<double>("local_plan.inflation_radius", costmapInflationRadius);
        omni_mode_ = this->declare_parameter<bool>("local_plan.omni_mode", true);
        omni_path_length_ = this->declare_parameter<double>("local_plan.omni_path_length", 3.0);
        omni_path_spacing_ = this->declare_parameter<double>("local_plan.omni_path_spacing", 0.25);
        map_frame_ = this->declare_parameter<std::string>("map_frame", "map");
        base_frame_ = this->declare_parameter<std::string>("base_frame", "base_link");
        goal_topic_ = this->declare_parameter<std::string>("goal_topic", "/RRT_goal");
        local_costmap_topic_ = this->declare_parameter<std::string>("local_costmap_topic", "/msg_local_reward");

        pub_lmpc_ = this->create_publisher<traversability_mapping::msg::Polynome>("/tltrajectory", 5);
        pub_global_path_ = this->create_publisher<nav_msgs::msg::Path>("/tllocal_path", 5);
        pub_path_cloud_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/path_trajectory", 5);
        pub_path_library_valid_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/path_library_valid", 5);
        pub_path_library_origin_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/path_library_origin", 5);
        pub_local_occupancy_ = this->create_publisher<nav_msgs::msg::OccupancyGrid>("/occupancy_costmap_local", 5);

        sub_goal_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
            goal_topic_,
            rclcpp::QoS(5).best_effort(),
            std::bind(&TraversabilityPath::goalPosHandler, this, std::placeholders::_1));
        sub_elevation_map_ = this->create_subscription<elevation_msgs::msg::OccupancyElevation>(
            local_costmap_topic_,
            5,
            std::bind(&TraversabilityPath::elevationMapHandler, this, std::placeholders::_1));

        path_cloud_.reset(new pcl::PointCloud<PointType>());
        path_cloud_local_.reset(new pcl::PointCloud<PointType>());
        path_cloud_global_.reset(new pcl::PointCloud<PointType>());
        path_cloud_valid_.reset(new pcl::PointCloud<PointType>());
        path_cloud_valid_with_obscost_.reset(new pcl::PointCloud<PointType>());
        kd_tree_from_cloud_.reset(new pcl::KdTreeFLANN<PointType>());

        createPathLibrary();
        RCLCPP_INFO(this->get_logger(), "Traversability path frontend started: %s -> /tltrajectory", goal_topic_.c_str());
    }

    ~TraversabilityPath() override {
        for (state_t *state : state_list_) {
            delete state;
        }
    }

private:
    static constexpr int kPathDepth = 3;
    static constexpr float kAngularVelocityMax = 12.0f / 180.0f * static_cast<float>(M_PI);
    static constexpr float kAngularVelocityRes = 1.0f / 180.0f * static_cast<float>(M_PI);
    static constexpr float kAngularVelocityMax2 = 6.0f / 180.0f * static_cast<float>(M_PI);
    static constexpr float kAngularVelocityRes2 = 1.5f / 180.0f * static_cast<float>(M_PI);
    static constexpr float kAngularVelocityMax3 = 4.0f / 180.0f * static_cast<float>(M_PI);
    static constexpr float kAngularVelocityRes3 = 1.0f / 180.0f * static_cast<float>(M_PI);
    static constexpr float kForwardVelocity = 0.1f;
    static constexpr float kDeltaTime = 1.0f;
    static constexpr int kSimTime = 10;
    static constexpr int kTrajectoryPointCount = 10;

    std::mutex mtx_;
    rclcpp::Subscription<elevation_msgs::msg::OccupancyElevation>::SharedPtr sub_elevation_map_;
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr sub_goal_;
    rclcpp::Publisher<traversability_mapping::msg::Polynome>::SharedPtr pub_lmpc_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pub_global_path_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_path_cloud_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_path_library_valid_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_path_library_origin_;
    rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr pub_local_occupancy_;

    elevation_msgs::msg::OccupancyElevation elevation_map_;
    float map_min_[3] = {0.0f, 0.0f, 0.0f};
    float map_max_[3] = {0.0f, 0.0f, 0.0f};
    bool planning_flag_;
    double radius_;
    double cost_max_;
    double costmap_inflation_radius_;
    bool omni_mode_;
    double omni_path_length_;
    double omni_path_spacing_;
    bool visitflag_;
    std::string map_frame_;
    std::string base_frame_;
    std::string goal_topic_;
    std::string local_costmap_topic_;

    int state_list_size_;
    std::vector<state_t *> state_list_;
    std::vector<state_t *> path_list_;
    std::vector<PointType> trajectory_points_;
    PointType goal_point_;
    PointType robot_point_;
    nav_msgs::msg::Path global_path_;
    pcl::PointCloud<PointType>::Ptr path_cloud_local_;
    pcl::PointCloud<PointType>::Ptr path_cloud_global_;
    pcl::PointCloud<PointType>::Ptr path_cloud_valid_;
    pcl::PointCloud<PointType>::Ptr path_cloud_valid_with_obscost_;
    pcl::PointCloud<PointType>::Ptr path_cloud_;
    pcl::KdTreeFLANN<PointType>::Ptr kd_tree_from_cloud_;
    state_t *root_state_;
    state_t *goal_state_;
    geometry_msgs::msg::TransformStamped latest_transform_;
    std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

    void createPathLibrary() {
        root_state_->x[0] = 0.0;
        root_state_->x[1] = 0.0;
        root_state_->x[2] = 0.0;
        root_state_->theta = 0.0f;
        root_state_->stateId = static_cast<int>(state_list_.size());
        root_state_->cost = 0.0f;
        root_state_->obscost = 0.0f;
        root_state_->validFlag = true;
        state_list_.push_back(root_state_);
        createPathLibrary(root_state_, 0);
        state_list_size_ = static_cast<int>(state_list_.size());

        for (int i = 0; i < state_list_size_; ++i) {
            PointType p;
            p.x = state_list_[i]->x[0];
            p.y = state_list_[i]->x[1];
            p.z = state_list_[i]->x[2];
            p.intensity = state_list_[i]->stateId;
            path_cloud_local_->push_back(p);
        }
    }

    void createPathLibrary(state_t *parent_state, int previous_depth) {
        const int this_depth = previous_depth + 1;
        if (this_depth > kPathDepth) {
            return;
        }

        float current_ang_vel_max = kAngularVelocityMax;
        float current_ang_vel_res = kAngularVelocityRes;
        if (this_depth == 2) {
            current_ang_vel_max = kAngularVelocityMax2;
            current_ang_vel_res = kAngularVelocityRes2;
        }
        if (this_depth >= 3) {
            current_ang_vel_max = kAngularVelocityMax3;
            current_ang_vel_res = kAngularVelocityRes3;
        }

        for (float v_theta = -current_ang_vel_max; v_theta <= current_ang_vel_max + 1e-6f; v_theta += current_ang_vel_res) {
            state_t *previous_state = parent_state;
            for (int i = 0; i < kSimTime; ++i) {
                state_t *new_state = new state_t;
                new_state->x[0] = previous_state->x[0] + (kForwardVelocity * std::cos(previous_state->theta)) * kDeltaTime;
                new_state->x[1] = previous_state->x[1] + (kForwardVelocity * std::sin(previous_state->theta)) * kDeltaTime;
                new_state->x[2] = previous_state->x[2];
                new_state->theta = previous_state->theta + v_theta * kDeltaTime;
                new_state->cost = parent_state->cost + distance(new_state->x, parent_state->x);
                new_state->obscost = 0.0f;
                new_state->validFlag = true;
                new_state->stateId = static_cast<int>(state_list_.size());
                new_state->parentState = previous_state;
                previous_state->childList.push_back(new_state);
                state_list_.push_back(new_state);
                previous_state = new_state;
            }
            createPathLibrary(previous_state, this_depth);
        }
    }

    void elevationMapHandler(const elevation_msgs::msg::OccupancyElevation::SharedPtr map_msg) {
        std::lock_guard<std::mutex> lock(mtx_);
        elevation_map_ = *map_msg;
        if (elevation_map_.occupancy.info.width == 0 || elevation_map_.occupancy.info.height == 0) {
            return;
        }
        if (elevation_map_.reward_cost.size() != elevation_map_.occupancy.data.size()) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000, "reward_cost size does not match occupancy grid");
            return;
        }
        updateCostMap();
        if (!omni_mode_) {
            updatePathLibrary();
        }
        if (!planning_flag_) {
            return;
        }
        updateTrajectory();
        publishTrajectory();
    }

    void goalPosHandler(const geometry_msgs::msg::PoseStamped::SharedPtr goal) {
        goal_point_.x = goal->pose.position.x;
        goal_point_.y = goal->pose.position.y;
        goal_point_.z = goal->pose.position.z;
        goal_state_->x[0] = goal_point_.x;
        goal_state_->x[1] = goal_point_.y;
        goal_state_->x[2] = goal_point_.z;

        if (!lookupRobotTransform()) {
            return;
        }
        robot_point_.x = latest_transform_.transform.translation.x;
        robot_point_.y = latest_transform_.transform.translation.y;
        robot_point_.z = latest_transform_.transform.translation.z;
        const double dist = std::hypot(goal_point_.x - robot_point_.x, goal_point_.y - robot_point_.y);
        if (dist > 1.0) {
            planning_flag_ = true;
        } else {
            trajectory_points_.clear();
            path_cloud_->clear();
            sensor_msgs::msg::PointCloud2 cloud_msg;
            pcl::toROSMsg(*path_cloud_, cloud_msg);
            cloud_msg.header.stamp = this->get_clock()->now();
            cloud_msg.header.frame_id = map_frame_;
            pub_path_cloud_->publish(cloud_msg);
        }
    }

    bool lookupRobotTransform() {
        try {
            latest_transform_ = tf_buffer_->lookupTransform(map_frame_, base_frame_, tf2::TimePointZero);
            return true;
        } catch (const tf2::TransformException &ex) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000, "Transform %s -> %s unavailable: %s", map_frame_.c_str(), base_frame_.c_str(), ex.what());
            return false;
        }
    }

    PointType transformLocalPoint(const PointType &local) const {
        tf2::Transform transform;
        tf2::fromMsg(latest_transform_.transform, transform);
        const tf2::Vector3 mapped = transform * tf2::Vector3(local.x, local.y, local.z);
        PointType point = local;
        point.x = mapped.x();
        point.y = mapped.y();
        point.z = mapped.z();
        return point;
    }

    void updateCostMap() {
        map_min_[0] = elevation_map_.occupancy.info.origin.position.x;
        map_min_[1] = elevation_map_.occupancy.info.origin.position.y;
        map_min_[2] = elevation_map_.occupancy.info.origin.position.z;
        map_max_[0] = map_min_[0] + elevation_map_.occupancy.info.resolution * elevation_map_.occupancy.info.width;
        map_max_[1] = map_min_[1] + elevation_map_.occupancy.info.resolution * elevation_map_.occupancy.info.height;
        map_max_[2] = map_min_[2];

        const auto initial_reward_cost = elevation_map_.reward_cost;
        const int size_map = static_cast<int>(elevation_map_.occupancy.data.size());
        const int inflation_size = std::max(0, static_cast<int>(costmap_inflation_radius_ / elevation_map_.occupancy.info.resolution));
        for (int i = 0; i < size_map; ++i) {
            const int id_x = i % static_cast<int>(elevation_map_.occupancy.info.width);
            const int id_y = i / static_cast<int>(elevation_map_.occupancy.info.width);
            if (initial_reward_cost[i] > cost_max_ - 0.1 || (elevation_map_.occupancy.data[i] < 0 && visitflag_)) {
                for (int dx = -inflation_size; dx <= inflation_size; ++dx) {
                    for (int dy = -inflation_size; dy <= inflation_size; ++dy) {
                        const int new_x = id_x + dx;
                        const int new_y = id_y + dy;
                        if (new_x < 0 || new_x >= static_cast<int>(elevation_map_.occupancy.info.width) ||
                            new_y < 0 || new_y >= static_cast<int>(elevation_map_.occupancy.info.height)) {
                            continue;
                        }
                        const int index = new_x + new_y * static_cast<int>(elevation_map_.occupancy.info.width);
                        elevation_map_.reward_cost[index] = static_cast<float>(cost_max_);
                    }
                }
            }
        }
        publishLocalOccupancy();
    }

    void publishLocalOccupancy() {
        if (pub_local_occupancy_->get_subscription_count() == 0) {
            return;
        }
        nav_msgs::msg::OccupancyGrid local_costmap;
        local_costmap.header.frame_id = elevation_map_.occupancy.header.frame_id;
        local_costmap.header.stamp = this->get_clock()->now();
        local_costmap.info = elevation_map_.occupancy.info;
        local_costmap.data.resize(local_costmap.info.width * local_costmap.info.height, -1);
        for (size_t i = 0; i < local_costmap.data.size() && i < elevation_map_.reward_cost.size(); ++i) {
            local_costmap.data[i] = static_cast<int8_t>(std::clamp(elevation_map_.reward_cost[i], 0.0f, 100.0f));
        }
        pub_local_occupancy_->publish(local_costmap);
    }

    void updatePathLibrary() {
        for (state_t *state : state_list_) {
            state->validFlag = true;
            state->obscost = 0.0f;
        }
        if (!lookupRobotTransform()) {
            return;
        }
        path_cloud_global_->clear();
        path_cloud_global_->reserve(path_cloud_local_->size());
        for (const PointType &local : path_cloud_local_->points) {
            path_cloud_global_->push_back(transformLocalPoint(local));
        }

        state_t probe;
        for (int i = 0; i < state_list_size_; ++i) {
            if (!state_list_[i]->validFlag) {
                continue;
            }
            probe.x[0] = path_cloud_global_->points[i].x;
            probe.x[1] = path_cloud_global_->points[i].y;
            probe.x[2] = path_cloud_global_->points[i].z;
            if (isInCollision(&probe)) {
                markInvalidState(state_list_[i]);
            }
        }

        path_cloud_valid_->clear();
        path_cloud_valid_with_obscost_->clear();
        for (int i = 0; i < state_list_size_; ++i) {
            if (!state_list_[i]->validFlag) {
                continue;
            }
            PointType p = path_cloud_global_->points[i];
            p.intensity = state_list_[i]->obscost;
            path_cloud_valid_with_obscost_->push_back(p);
            PointType indexed = path_cloud_global_->points[i];
            indexed.intensity = state_list_[i]->stateId;
            path_cloud_valid_->push_back(indexed);
        }
        publishPathLibraryClouds();
    }

    void publishPathLibraryClouds() {
        if (pub_path_library_valid_->get_subscription_count() != 0) {
            sensor_msgs::msg::PointCloud2 cloud_msg;
            pcl::toROSMsg(*path_cloud_valid_with_obscost_, cloud_msg);
            cloud_msg.header.stamp = this->get_clock()->now();
            cloud_msg.header.frame_id = map_frame_;
            pub_path_library_valid_->publish(cloud_msg);
        }
        if (pub_path_library_origin_->get_subscription_count() != 0) {
            sensor_msgs::msg::PointCloud2 cloud_msg;
            pcl::toROSMsg(*path_cloud_local_, cloud_msg);
            cloud_msg.header.stamp = this->get_clock()->now();
            cloud_msg.header.frame_id = base_frame_;
            pub_path_library_origin_->publish(cloud_msg);
        }
    }

    void updateTrajectory() {
        path_list_.clear();
        trajectory_points_.clear();
        if (omni_mode_) {
            updateOmniTrajectory();
            return;
        }
        if (path_cloud_valid_->empty()) {
            return;
        }
        std::vector<int> point_search_ind;
        std::vector<float> point_search_sq_dis;
        kd_tree_from_cloud_->setInputCloud(path_cloud_valid_);
        kd_tree_from_cloud_->radiusSearch(goal_point_, radius_, point_search_ind, point_search_sq_dis, 0);
        if (point_search_ind.empty()) {
            kd_tree_from_cloud_->nearestKSearch(goal_point_, 500, point_search_ind, point_search_sq_dis);
        }

        float min_cost = FLT_MAX;
        state_t *min_state = nullptr;
        for (size_t i = 0; i < point_search_ind.size(); ++i) {
            const int id = static_cast<int>(path_cloud_valid_->points[point_search_ind[i]].intensity);
            state_t *this_state = state_list_[id];
            const float dist_cost = point_search_sq_dis[i];
            float path_obs_cost = 0.0f;
            float path_occupancy_cost = 0.0f;
            int node_num = 0;
            while (this_state->parentState != nullptr) {
                path_occupancy_cost += getPathObsCost(this_state);
                path_obs_cost += this_state->obscost;
                node_num++;
                this_state = this_state->parentState;
            }
            if (node_num > 0) {
                path_occupancy_cost *= node_num;
                path_obs_cost /= node_num;
            }
            const float cost = dist_cost + path_obs_cost + path_occupancy_cost;
            if (cost < min_cost) {
                min_cost = cost;
                min_state = state_list_[id];
            }
        }
        if (min_state == nullptr) {
            return;
        }
        state_t *state = min_state;
        while (state->parentState != nullptr) {
            path_list_.insert(path_list_.begin(), state);
            state = state->parentState;
        }
        path_list_.insert(path_list_.begin(), state_list_[0]);
    }

    void updateOmniTrajectory() {
        if (!lookupRobotTransform()) {
            return;
        }

        robot_point_.x = latest_transform_.transform.translation.x;
        robot_point_.y = latest_transform_.transform.translation.y;
        robot_point_.z = latest_transform_.transform.translation.z;

        const double dx = goal_point_.x - robot_point_.x;
        const double dy = goal_point_.y - robot_point_.y;
        const double dist_to_goal = std::hypot(dx, dy);
        if (dist_to_goal < 1e-3) {
            return;
        }

        const double path_length = std::min(std::max(omni_path_spacing_, omni_path_length_), dist_to_goal);
        const double spacing = std::max(0.05, omni_path_spacing_);
        const int steps = std::max(1, static_cast<int>(std::ceil(path_length / spacing)));
        const double ux = dx / dist_to_goal;
        const double uy = dy / dist_to_goal;

        PointType start;
        start.x = robot_point_.x;
        start.y = robot_point_.y;
        start.z = robot_point_.z;
        start.intensity = 0.0f;
        trajectory_points_.push_back(start);

        for (int i = 1; i <= steps; ++i) {
            const double s = std::min(path_length, i * spacing);
            PointType p;
            p.x = static_cast<float>(robot_point_.x + ux * s);
            p.y = static_cast<float>(robot_point_.y + uy * s);
            p.z = static_cast<float>(robot_point_.z);
            p.intensity = static_cast<float>(s);

            if (isPointInCollision(p)) {
                break;
            }
            trajectory_points_.push_back(p);
        }

        if (trajectory_points_.size() < 2) {
            const int fallback_steps = std::max(1, std::min(steps, static_cast<int>(std::ceil(1.0 / spacing))));
            for (int i = 1; i <= fallback_steps; ++i) {
                const double s = std::min(path_length, i * spacing);
                PointType p;
                p.x = static_cast<float>(robot_point_.x + ux * s);
                p.y = static_cast<float>(robot_point_.y + uy * s);
                p.z = static_cast<float>(robot_point_.z);
                p.intensity = static_cast<float>(s);
                trajectory_points_.push_back(p);
            }
        }
    }

    void publishTrajectory() {
        if (omni_mode_) {
            publishOmniTrajectory();
            return;
        }

        if (pub_path_cloud_->get_subscription_count() != 0) {
            path_cloud_->clear();
            for (state_t *state : path_list_) {
                PointType p = path_cloud_global_->points[state->stateId];
                p.z += 0.1f;
                p.intensity = state->cost;
                path_cloud_->push_back(p);
            }
            sensor_msgs::msg::PointCloud2 cloud_msg;
            pcl::toROSMsg(*path_cloud_, cloud_msg);
            cloud_msg.header.stamp = this->get_clock()->now();
            cloud_msg.header.frame_id = map_frame_;
            pub_path_cloud_->publish(cloud_msg);
        }

        global_path_.poses.clear();
        for (state_t *state : path_list_) {
            geometry_msgs::msg::PoseStamped pose;
            pose.header.frame_id = base_frame_;
            pose.header.stamp = this->get_clock()->now();
            pose.pose.position.x = path_cloud_local_->points[state->stateId].x;
            pose.pose.position.y = path_cloud_local_->points[state->stateId].y;
            pose.pose.position.z = path_cloud_local_->points[state->stateId].z;
            pose.pose.orientation.w = 1.0;
            global_path_.poses.push_back(pose);
        }

        if (path_list_.size() >= 2) {
            traversability_mapping::msg::Polynome poly;
            for (int i = 0; i < kTrajectoryPointCount; ++i) {
                const int path_index = std::min<int>(i, static_cast<int>(path_list_.size()) - 1);
                const PointType &p = path_cloud_global_->points[path_list_[path_index]->stateId];
                geometry_msgs::msg::Point point;
                point.x = p.x;
                point.y = p.y;
                point.z = p.z;
                poly.pos_pts.push_back(point);
            }
            poly.t_pts.assign(kTrajectoryPointCount - 1, 0.05);
            poly.init_v = geometry_msgs::msg::Vector3();
            poly.init_a = geometry_msgs::msg::Vector3();
            poly.start_time = this->get_clock()->now();
            pub_lmpc_->publish(poly);
        }

        global_path_.header.frame_id = base_frame_;
        global_path_.header.stamp = this->get_clock()->now();
        pub_global_path_->publish(global_path_);
    }

    void publishOmniTrajectory() {
        if (trajectory_points_.size() < 2) {
            return;
        }

        const auto stamp = this->get_clock()->now();
        path_cloud_->clear();
        for (const PointType &point : trajectory_points_) {
            PointType p = point;
            p.z += 0.1f;
            path_cloud_->push_back(p);
        }

        if (pub_path_cloud_->get_subscription_count() != 0) {
            sensor_msgs::msg::PointCloud2 cloud_msg;
            pcl::toROSMsg(*path_cloud_, cloud_msg);
            cloud_msg.header.stamp = stamp;
            cloud_msg.header.frame_id = map_frame_;
            pub_path_cloud_->publish(cloud_msg);
        }

        global_path_.poses.clear();
        for (const PointType &point : trajectory_points_) {
            geometry_msgs::msg::PoseStamped pose;
            pose.header.frame_id = map_frame_;
            pose.header.stamp = stamp;
            pose.pose.position.x = point.x;
            pose.pose.position.y = point.y;
            pose.pose.position.z = point.z;
            pose.pose.orientation.w = 1.0;
            global_path_.poses.push_back(pose);
        }
        global_path_.header.frame_id = map_frame_;
        global_path_.header.stamp = stamp;
        pub_global_path_->publish(global_path_);

        traversability_mapping::msg::Polynome poly;
        const size_t count = std::min<size_t>(trajectory_points_.size(), kTrajectoryPointCount);
        for (size_t i = 0; i < count; ++i) {
            geometry_msgs::msg::Point point;
            point.x = trajectory_points_[i].x;
            point.y = trajectory_points_[i].y;
            point.z = trajectory_points_[i].z;
            poly.pos_pts.push_back(point);
        }
        poly.t_pts.assign(poly.pos_pts.size() - 1, 0.05);
        poly.init_v = geometry_msgs::msg::Vector3();
        poly.init_a = geometry_msgs::msg::Vector3();
        poly.start_time = stamp;
        pub_lmpc_->publish(poly);
    }

    void markInvalidState(state_t *state) {
        state->validFlag = false;
        for (state_t *child : state->childList) {
            markInvalidState(child);
        }
        if (state->parentState != nullptr && state->parentState->validFlag) {
            state_t *parent = state;
            const float lmax = kPathDepth * kForwardVelocity * kSimTime;
            while (parent->parentState != nullptr) {
                parent->obscost = std::max(lmax - state->cost, parent->obscost);
                parent = parent->parentState;
            }
        }
    }

    bool isInCollision(state_t *state) const {
        if (state->x[0] <= map_min_[0] || state->x[0] >= map_max_[0] ||
            state->x[1] <= map_min_[1] || state->x[1] >= map_max_[1]) {
            return false;
        }
        const int rounded_x = static_cast<int>((state->x[0] - map_min_[0]) / mapResolution);
        const int rounded_y = static_cast<int>((state->x[1] - map_min_[1]) / mapResolution);
        if (!inMap(rounded_x, rounded_y)) {
            return false;
        }
        const int index = rounded_x + rounded_y * static_cast<int>(elevation_map_.occupancy.info.width);
        return elevation_map_.reward_cost[index] > cost_max_ - 0.1 ||
               (elevation_map_.occupancy.data[index] < 0 && visitflag_);
    }

    bool isPointInCollision(const PointType &point) const {
        if (point.x <= map_min_[0] || point.x >= map_max_[0] ||
            point.y <= map_min_[1] || point.y >= map_max_[1]) {
            return false;
        }
        const int rounded_x = static_cast<int>((point.x - map_min_[0]) / elevation_map_.occupancy.info.resolution);
        const int rounded_y = static_cast<int>((point.y - map_min_[1]) / elevation_map_.occupancy.info.resolution);
        if (!inMap(rounded_x, rounded_y)) {
            return false;
        }
        const int index = rounded_x + rounded_y * static_cast<int>(elevation_map_.occupancy.info.width);
        return elevation_map_.reward_cost[index] > cost_max_ - 0.1 ||
               (elevation_map_.occupancy.data[index] < 0 && visitflag_);
    }

    float getPathObsCost(state_t *state) const {
        const PointType &p = path_cloud_global_->points[state->stateId];
        const int rounded_x = static_cast<int>((p.x - map_min_[0]) / mapResolution);
        const int rounded_y = static_cast<int>((p.y - map_min_[1]) / mapResolution);
        float path_obs_cost = 0.0f;
        constexpr int bound = 2;
        for (int x = rounded_x - bound; x <= rounded_x + bound; ++x) {
            for (int y = rounded_y - bound; y <= rounded_y + bound; ++y) {
                if (!inMap(x, y)) {
                    continue;
                }
                const int index = x + y * static_cast<int>(elevation_map_.occupancy.info.width);
                if (elevation_map_.reward_cost[index] > cost_max_ - 0.1 ||
                    (elevation_map_.occupancy.data[index] < 0 && visitflag_)) {
                    path_obs_cost += 1.0f;
                }
            }
        }
        return path_obs_cost;
    }

    bool inMap(int x, int y) const {
        return x >= 0 && y >= 0 &&
               x < static_cast<int>(elevation_map_.occupancy.info.width) &&
               y < static_cast<int>(elevation_map_.occupancy.info.height);
    }

    static float distance(const double from[3], const double to[3]) {
        return std::sqrt(
            (to[0] - from[0]) * (to[0] - from[0]) +
            (to[1] - from[1]) * (to[1] - from[1]) +
            (to[2] - from[2]) * (to[2] - from[2]));
    }
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<TraversabilityPath>());
    rclcpp::shutdown();
    return 0;
}
