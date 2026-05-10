#include "utility.h"
#include "planner/ikd_Tree.h"
#include "planner/Dstar.h"

//RRTstar 头文件
#include "planner/node.h"
#include "planner/kdtree.h"
#include "planner/sampler.h"
#include "planner/planningState.h"
#include "visualization/visualization.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <cfloat>

// ROS2 includes
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <nav_msgs/msg/path.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <memory> // 包含 shared_ptr 所需的头文件

// Assuming elevation_msgs is a custom message package, you'll need to create a ROS2 version of it
// For now, let's assume it's converted to a C++ struct or similar if no ROS2 msg definition is available
// If it's a custom ROS2 message, include its header:
// #include <elevation_msgs/msg/occupancy_elevation.hpp>

// Forward declaration of elevation_msgs::OccupancyElevation if it's a custom type not yet defined
// 重新声明 elevation_msgs::msg::OccupancyElevation，确保字段名称正确匹配 ROS 2 规范

class TraversabilityMapping : public rclcpp::Node {
public:
    using PointVector = KD_TREE<PointType>::PointVector;

private:
    // ROS2 Node Handle - 不再需要，直接使用 `this`
    // rclcpp::Node::SharedPtr nh_;
    // Mutex Memory Lock
    std::mutex mtx;
    // Transform Listener and Buffer
    std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
    geometry_msgs::msg::TransformStamped transform;

    // Subscriber
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr subFilteredGroundCloud;
    rclcpp::Subscription<elevation_msgs::msg::OccupancyElevation>::SharedPtr subReward;

    // Publisher
    rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr pubOccupancyMapGlobal;
    rclcpp::Publisher<elevation_msgs::msg::OccupancyElevation>::SharedPtr pubMsgLocalHeight;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubElevationCloud;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubCostCloud;
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pubLocalGoal;
    rclcpp::Publisher<elevation_msgs::msg::OccupancyElevation>::SharedPtr pubMsgGlobal;

    // Point Cloud Pointer
    pcl::PointCloud<PointType>::Ptr laserCloud; // save input filtered laser cloud for mapping
    pcl::PointCloud<PointType>::Ptr laserCloudElevation; // a cloud for publishing elevation map
    pcl::PointCloud<PointType>::Ptr laserCloudCost;      // a cloud for publishing cost map

    //ikd-tree
    KD_TREE<PointType>::Ptr kdtree_ptr;

    elevation_msgs::msg::OccupancyElevation msgLocalHeight; // 局部高程和rough
    elevation_msgs::msg::OccupancyElevation msgElevationGlobal; // 全局，存储全局高程、rough信息
    elevation_msgs::msg::OccupancyElevation rewardMap;  // 接受的局部代价，用于更新

    bool isSave = false; // true时保存全局地图

    int pubCount;

    // Map Arrays
    int mapArrayCount;
    int **mapArrayInd; // it saves the index of this submap in vector mapArray
    std::vector<childMap_t*> mapArray;

    PointType localMapOriginPoint;
    grid_t localMapOriginGrid;
    PointType localMapOriginPoint2;
    grid_t localMapOriginGrid2;

    // Global Variables for Traversability Calculation
    cv::Mat matCov, matEig, matVec;

    // Lists for New Scan
    std::vector<mapCell_t*> observingList1; // thread 1: save new observed cells
    std::vector<mapCell_t*> observingList2; // thread 2: calculate traversability of new observed cells

    bool haveRobotPoint;
    bool local_publish_every_scan_;
    bool publish_global_debug_;
    double visualization_hz_;
    int64_t last_visualization_pub_time_ns_;

    //用于horizen规划
    // PointType lastRobotPoint;
    // PointType lastGoalPoint;

    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pubSmoothPath;
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr subGoal;

    PointType robotPoint;
    PointType goalPoint;
    rclcpp::TimerBase::SharedPtr RRTstarPlanTimer;

    BiasSampler sampler_;

    // for informed sampling
    Eigen::Vector2d trans_, scale_;
    Eigen::Matrix2d rot_;

    // for GUILD sampling
    Eigen::Vector2d scale1_, scale2_;
    Eigen::Vector2d trans1_, trans2_;
    Eigen::Matrix2d rot1_, rot2_;
    Eigen::Vector2d start_;
    Eigen::Vector2d goal_;
    //视觉显示起点终点
    Eigen::Vector3d start3D;
    Eigen::Vector3d goal3D;

    int max_tree_node_nums_;
    int pool_max_nums_;
    int valid_tree_node_nums_;
    int curr_iter_;
    bool use_informed_sampling_;
    bool use_GUILD_sampling_;
    double steer_length_;
    double search_radius_;
    double goal_radius_;
    double search_time_;
    double first_path_use_time_;
    double final_path_use_time_;
    double cost_max_;
    double white_cost_;
    double unknown_cost_;
    double reward_obstacle_threshold_;
    double goal_bias_;
    double cost_bias_;
    double follow_dis_;
    double goal_local_dis_;
    PlanningState PlanningState_;
    bool filterflag;

    TreeNode *start_node_;
    TreeNode *goal_node_;
    std::vector<TreeNode *> nodes_pool_;     //没啥用，估计主要用于自动分配地址
    std::vector<Eigen::Vector3d> final_path_;
    std::vector<Eigen::Vector3d> sampleFinalPath;
    std::vector<std::vector<Eigen::Vector3d>> path_list_;
    std::vector<std::pair<double, double>> solution_cost_time_pair_list_;
    /* kd tree init */
    kdtree *kd_tree = kd_create(2);

    //RRTstar Visualization
    std::shared_ptr<visualization::Visualization> vis_ptr_;
    const double visHeight = 0.2; //rrt树的视觉高度
    const double localPathLen = 1.0; //每个 x m发布一个目标点 MPL要4m
    bool haveLastPos = false;
    Eigen::Vector3d lastRRTPos;
    int runTime = 0;
    int lastGoalIndex = 1;
    bool isSendGoal=false;
    bool has_valid_local_goal_=false;
    rclcpp::Time goal_start_time;
    double time_roll_;
    float inflate_r_;

    //*****************RRTstar planner  end**********************//

public:
    TraversabilityMapping():
        rclcpp::Node("traversability_mapping"),
        pubCount(1),
        mapArrayCount(0),
        last_visualization_pub_time_ns_(0)
    {
        // Initialize pointers to nullptr to prevent crashes
        mapArrayInd = nullptr;
        // 移除 nh_ = shared_from_this(); 这行，不再需要
        tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock()); // 直接使用 this
        tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

       
       this->declare_parameter<double>("planning.cost_max", 100.0);
        this->get_parameter("planning.cost_max", cost_max_);

        this->declare_parameter<double>("planning.search_time", 0.17);
        this->get_parameter("planning.search_time", search_time_);

        this->declare_parameter<double>("planning.steer_length", 1.0);
        this->get_parameter("planning.steer_length", steer_length_);

        this->declare_parameter<double>("planning.search_radius", 2.0);
        this->get_parameter("planning.search_radius", search_radius_);

        this->declare_parameter<double>("planning.goal_radius", 0.5);
        this->get_parameter("planning.goal_radius", goal_radius_);

        this->declare_parameter<int>("planning.max_tree_node_nums", 10000);
        this->get_parameter("planning.max_tree_node_nums", max_tree_node_nums_);

        this->declare_parameter<double>("planning.goal_bias", 0.05);
        this->get_parameter("planning.goal_bias", goal_bias_);

        this->declare_parameter<double>("planning.white_cost", 0.0);
        this->get_parameter("planning.white_cost", white_cost_);

        this->declare_parameter<double>("planning.unknown_cost", white_cost_);
        this->get_parameter("planning.unknown_cost", unknown_cost_);

        this->declare_parameter<double>("planning.reward_obstacle_threshold", 95.0);
        this->get_parameter("planning.reward_obstacle_threshold", reward_obstacle_threshold_);

        this->declare_parameter<double>("planning.cost_bias", 0.5);
        this->get_parameter("planning.cost_bias", cost_bias_);

        this->declare_parameter<double>("planning.time_roll", 10.0);
        this->get_parameter("planning.time_roll", time_roll_);

        // YAML 没有 float 类型，建议这里也声明为 double，再自行转换成 float
        this->declare_parameter<double>("planning.inflate_r", 0.3);
        double inflate_r_tmp;
        this->get_parameter("planning.inflate_r", inflate_r_tmp);
        inflate_r_ = static_cast<float>(inflate_r_tmp);

        this->declare_parameter<double>("planning.follow_dis", 2.5);
        this->get_parameter("planning.follow_dis", follow_dis_);

        this->declare_parameter<double>("planning.goal_local_dis", 15.0);
        this->get_parameter("planning.goal_local_dis", goal_local_dis_);

        this->declare_parameter<bool>("mapping.local_publish_every_scan", true);
        this->get_parameter("mapping.local_publish_every_scan", local_publish_every_scan_);

        this->declare_parameter<double>("mapping.visualization_hz", 2.0);
        this->get_parameter("mapping.visualization_hz", visualization_hz_);

        this->declare_parameter<bool>("mapping.publish_global_debug", false);
        this->get_parameter("mapping.publish_global_debug", publish_global_debug_);

        pool_max_nums_ = 2 * max_tree_node_nums_; //必须要大于搜索的最大节点数
        sampler_.setGoalBiased(goal_bias_);

        subFilteredGroundCloud = this->create_subscription<sensor_msgs::msg::PointCloud2>("/filtered_pointcloud", 5, std::bind(&TraversabilityMapping::cloudHandler, this, std::placeholders::_1));
        subReward = this->create_subscription<elevation_msgs::msg::OccupancyElevation>("/msg_local_reward", 5, std::bind(&TraversabilityMapping::rewardHandler, this, std::placeholders::_1));

        pubCostCloud = this->create_publisher<sensor_msgs::msg::PointCloud2>("/cost_pointcloud", 5);

        // publish local occupancy and elevation grid map
        pubMsgLocalHeight = this->create_publisher<elevation_msgs::msg::OccupancyElevation>("/msg_local_height", 5);
        pubMsgGlobal = this->create_publisher<elevation_msgs::msg::OccupancyElevation>("/msg_global_reward", 5);        //全局各种信息
        // publish elevation map for visualization
        pubElevationCloud = this->create_publisher<sensor_msgs::msg::PointCloud2>("/elevation_pointcloud", 5);
        //global Costmap
        pubOccupancyMapGlobal = this->create_publisher<nav_msgs::msg::OccupancyGrid>("/occupancy_costmap_global", 5);

        //get goal
        subGoal = this->create_subscription<geometry_msgs::msg::PoseStamped>("/move_base_simple/goal", 5, std::bind(&TraversabilityMapping::goalPosHandler, this, std::placeholders::_1));
        pubLocalGoal = this->create_publisher<geometry_msgs::msg::PoseStamped>("/RRT_goal", rclcpp::QoS(1).reliability(rclcpp::ReliabilityPolicy::BestEffort)); //局部跟随点
    //global path
    pubSmoothPath = this->create_publisher<nav_msgs::msg::Path>("/smooth_path", 5);

        // RRTstarPlanTimer = create_wall_timer(std::chrono::duration<double>(0.2), std::bind(&TraversabilityMapping::RRTstarHandler, this));
        RRTstarPlanTimer = this->create_wall_timer(std::chrono::duration<double>(0.2), std::bind(&TraversabilityMapping::RRTstarHandler, this));

        allocateMemory();
    }

    ~TraversabilityMapping(){
        // Clean up dynamically allocated memory
        if (mapArrayInd) {
            for (int i = 0; i < mapArrayLength; ++i) {
                delete[] mapArrayInd[i];
            }
            delete[] mapArrayInd;
        }
        
        // Clean up mapArray
        for (auto& childMap : mapArray) {
            delete childMap;
        }
        mapArray.clear();
    }

    void allocateMemory(){
        haveRobotPoint = false;
        PlanningState_ = WithoutGoal; // Changed == to = for assignment

        use_informed_sampling_ = true;
        use_GUILD_sampling_ = false;

        PlanningState_ = WithoutGoal;

        valid_tree_node_nums_ = 0;
        nodes_pool_.resize(pool_max_nums_); //必须要大于搜索的最大节点数
        for (int i = 0; i < pool_max_nums_; ++i)
        {
            nodes_pool_[i] = new TreeNode;
        }
        // rrt path visualization
        // 在构造函数中不能使用 shared_from_this()，因为节点还没有被 shared_ptr 管理
        // 我们将在构造函数完成后初始化 vis_ptr_
        vis_ptr_ = nullptr;

        // 暂时注释掉 vis_ptr_ 的使用，直到我们找到正确的初始化方式
        // vis_ptr_->registe<visualization_msgs::msg::Marker>("start");
        // vis_ptr_->registe<visualization_msgs::msg::Marker>("goal");
        // vis_ptr_->registe<visualization_msgs::msg::Marker>("waypoint");
        // vis_ptr_->registe<nav_msgs::msg::Path>("rrt_star_final_path");
        // vis_ptr_->registe<sensor_msgs::msg::PointCloud2>("rrt_star_final_wpts");
        // vis_ptr_->registe<visualization_msgs::msg::MarkerArray>("rrt_star_paths");

        //map
        laserCloud.reset(new pcl::PointCloud<PointType>());
        laserCloudElevation.reset(new pcl::PointCloud<PointType>());
        laserCloudCost.reset(new pcl::PointCloud<PointType>());

        //这是新添加的
        kdtree_ptr.reset(new KD_TREE<PointType>(0.3,0.6,0.2));
        if (!kdtree_ptr) {
            RCLCPP_FATAL(get_logger(), "Failed to initialize kdtree_ptr");
            return;
        }
        
        // initialize array for cmap
        RCLCPP_INFO(get_logger(), "Allocating mapArrayInd with size: %d x %d", mapArrayLength, mapArrayLength);
        if (!mapArrayInd) {
            mapArrayInd = new int*[mapArrayLength];
            if (!mapArrayInd) {
                RCLCPP_FATAL(get_logger(), "Failed to allocate memory for mapArrayInd");
                return;
            }
        }
        
        for (int i = 0; i < mapArrayLength; ++i) {
            mapArrayInd[i] = new int[mapArrayLength];
            if (!mapArrayInd[i]) {
                RCLCPP_ERROR(get_logger(), "Failed to allocate memory for mapArrayInd[%d]", i);
                return;
            }
        }

        for (int i = 0; i < mapArrayLength; ++i)
            for (int j = 0; j < mapArrayLength; ++j)
                mapArrayInd[i][j] = -1;
        
        RCLCPP_INFO(get_logger(), "Successfully allocated mapArrayInd");

        // Matrix Initialization
        matCov = cv::Mat (3, 3, CV_32F, cv::Scalar::all(0));
        matEig = cv::Mat (1, 3, CV_32F, cv::Scalar::all(0));
        matVec = cv::Mat (3, 3, CV_32F, cv::Scalar::all(0));


        initializeLocalOccupancyMap();
        initGlobalOccuEle_global();

    }

    bool Point2Gird(PointType point, ipoint2& resPoint){

        int cubeX = int((point.x + mapCubeLength/2.0) / mapCubeLength) + rootCubeIndex;  //cube:1x1m, cube位置  （0-90，0-90）
        int cubeY = int((point.y + mapCubeLength/2.0) / mapCubeLength) + rootCubeIndex;
        if (point.x + mapCubeLength/2.0 < 0)  --cubeX;
        if (point.y + mapCubeLength/2.0 < 0)  --cubeY;

        if (cubeX < 0 || cubeX >= mapArrayLength || cubeY < 0 || cubeY >= mapArrayLength){
            // Point is out of pre-allocated boundary, report error (you should increase map size)
            // RCLCPP_WARN(get_logger(), "Point cloud is out of elevation map boundary. Change params ->mapArrayLength<-. The program will crash!");
            return false;
        }
        float originX = (cubeX - rootCubeIndex) * mapCubeLength - mapCubeLength/2.0;    //点的实际位置，单位m
        float originY = (cubeY - rootCubeIndex) * mapCubeLength - mapCubeLength/2.0;

        // Find the index for this point in this sub-map (grid index)
        int gridX = (int)((point.x - originX) / mapResolution);
        int gridY = (int)((point.y - originY) / mapResolution);
        if (gridX < 0 || gridY < 0 || gridX >= mapCubeArrayLength+0.1 || gridY >= mapCubeArrayLength+0.1){
            RCLCPP_INFO(get_logger(), "maCube: %d", mapCubeArrayLength);
            RCLCPP_INFO(get_logger(), "point x: %f  y: %f  grid x: %d  y: %d", point.x, point.y, gridX, gridY);
            RCLCPP_WARN(get_logger(), "Cube is out of elevation map boundary. Change params ->mapArrayLength<-. The program will crash!");
            return false;
        }
        //The number of grid in cube is  mapCubeArrayLength
        int gx = cubeX * mapCubeArrayLength + gridX;
        int gy = cubeY * mapCubeArrayLength + gridY;
        resPoint.x = gx;
        resPoint.y = gy;
        return true;
    }
    //cwl
    void setCostInRadius(PointType center, double radius, float cost_val){
        int grid_radius = static_cast<int>(radius / mapResolution);  // 半径对应的栅格个数
        ipoint2 centerGrid;
        if(!Point2Gird(center, centerGrid)){
            RCLCPP_WARN(get_logger(), "Failed to convert center point to grid.");
            return;
        }

        for (int dx = -grid_radius; dx <= grid_radius; ++dx){
            for (int dy = -grid_radius; dy <= grid_radius; ++dy){
                int gx = centerGrid.x + dx;
                int gy = centerGrid.y + dy;

                // 检查是否在圆形范围内
                double dist = sqrt(static_cast<double>(dx * dx + dy * dy)) * mapResolution; // Cast to double for sqrt
                if(dist > radius) continue;

                // 获取对应grid的cell
                int cubeX = gx / mapCubeArrayLength;
                int cubeY = gy / mapCubeArrayLength;
                int gridX = gx % mapCubeArrayLength;
                int gridY = gy % mapCubeArrayLength;

                if (cubeX < 0 || cubeX >= mapArrayLength || cubeY < 0 || cubeY >= mapArrayLength) continue;
                if (mapArrayInd[cubeX][cubeY] == -1) continue;

                grid_t grid;
                grid.cubeX = cubeX;
                grid.cubeY = cubeY;
                grid.gridX = gridX;
                grid.gridY = gridY;
           
                
                mapCell_t* cell = grid2Cell(&grid);
                if (cell) {
                    float cost = cost_val;
                    float cost_inflate = cost_val;
                    updateCellCost2(cell, cost, cost_inflate);  // 调用已有函数
                }
            }
        }

        // RCLCPP_INFO(get_logger(), "Cost in radius %.2fm around (%.2f, %.2f) has been set to %.2f.", radius, center.x, center.y, cost_val);
    }

    //RRTstarplan
    Eigen::Vector2d steer(const Eigen::Vector2d &nearest_node_p, const Eigen::Vector2d &rand_node_p, double len)
    {
        Eigen::Vector2d diff_vec = rand_node_p - nearest_node_p;
        double dist = diff_vec.norm();
        if (diff_vec.norm() <= len)
            return rand_node_p;
        else
            return nearest_node_p + diff_vec * len / dist;
    }

    double calDist2D(const Eigen::Vector2d &p1, const Eigen::Vector2d &p2)
    {
        return (p1 - p2).norm();
    }

    RRTNode2DPtr addTreeNode(RRTNode2DPtr &parent, const Eigen::Vector2d &state,
                             const double &cost_from_start, const double &cost_from_parent,const double &dist_from_start, const double &dist_from_parent)
    {
        RRTNode2DPtr new_node_ptr = nodes_pool_[valid_tree_node_nums_];
        valid_tree_node_nums_++;
        new_node_ptr->parent = parent;
        parent->children.push_back(new_node_ptr);
        new_node_ptr->x = state;
        new_node_ptr->cost_from_start = cost_from_start;
        new_node_ptr->cost_from_parent = cost_from_parent;
        new_node_ptr->dist_from_start = dist_from_start;
        new_node_ptr->dist_from_parent = dist_from_parent;
        return new_node_ptr;
    }

    void fillPath(const RRTNode2DPtr &n, std::vector<Eigen::Vector3d> &path)
    {
        path.clear();
        RRTNode2DPtr node_ptr = n;
        while (node_ptr->parent)
        {
            Eigen::Vector3d tmpNode(node_ptr->x[0],node_ptr->x[1],visHeight);
            path.push_back(tmpNode);
            node_ptr = node_ptr->parent;
        }
        Eigen::Vector3d startNode(start_node_->x[0],start_node_->x[1],visHeight);
        path.push_back(startNode);
        std::reverse(std::begin(path), std::end(path));
    }

    void changeNodeParent(RRTNode2DPtr &node, RRTNode2DPtr &parent, const double &cost_from_parent, const double &dist_from_parent)
    {
        if (node->parent)
            node->parent->children.remove(node); // DON'T FORGET THIS, remove it form its parent's children list
        node->parent = parent;
        node->cost_from_parent = cost_from_parent;
        node->dist_from_parent = dist_from_parent;
        node->cost_from_start = parent->cost_from_start + cost_from_parent;
        node->dist_from_start = parent->dist_from_start + dist_from_parent;
        parent->children.push_back(node);

        // for all its descedants, change the cost_from_start and tau_from_start;
        RRTNode2DPtr descendant(node);
        std::queue<RRTNode2DPtr> Q;
        Q.push(descendant);
        while (!Q.empty())
        {
            descendant = Q.front();
            Q.pop();
            for (const auto &leafptr : descendant->children)
            {
                leafptr->cost_from_start = leafptr->cost_from_parent + descendant->cost_from_start;
                leafptr->dist_from_start = leafptr->dist_from_parent + descendant->dist_from_start;
                Q.push(leafptr);
            }
        }
    }

    void sampleWholeTree(const RRTNode2DPtr &root, std::vector<Eigen::Vector3d> &vertice, std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> &edges)
    {
        if (root == nullptr)
            return;

        // whatever dfs or bfs
        RRTNode2DPtr node = root;
        std::queue<RRTNode2DPtr> Q;
        Q.push(node);
        while (!Q.empty())
        {
            node = Q.front();
            Q.pop();
            for (const auto &leafptr : node->children)
            {
                Eigen::Vector3d tmpleafptr(leafptr->x[0],leafptr->x[1],visHeight);
                vertice.push_back(tmpleafptr);
                Eigen::Vector3d tmpnodex(node->x[0],node->x[1],visHeight);
                edges.emplace_back(std::make_pair(tmpnodex, tmpleafptr));
                Q.push(leafptr);
            }
        }
    }

    bool rrt_star(const int &max_iter,const double &max_time)
    {
        rclcpp::Time rrt_start_time = this->get_clock()->now();
        double rrt_time = (this->get_clock()->now() - rrt_start_time).seconds();
        if(PlanningState_==WithoutGoal){

        }
        else if(PlanningState_ == Global){

        }
        bool goal_found = false;

        // Add start and goal nodes to kd tree
        kd_insert2(kd_tree, start_node_->x[0], start_node_->x[1],  start_node_);
        //  cout<<"222"<<endl;

        /* main loop */
        while (rrt_time < max_time && curr_iter_ < max_iter)
        {
            curr_iter_++;
            rrt_time =(this->get_clock()->now() - rrt_start_time).seconds();
            Eigen::Vector2d x_rand;

            //采样
            sampler_.samplingOnce(x_rand);

            if (isStateValid(x_rand))
            {
                continue;
            }

            //找到位置上距离该点最近的点，朝这个点伸展一个点
            struct kdres *p_nearest = kd_nearest2(kd_tree, x_rand[0], x_rand[1]);
            if (p_nearest == nullptr)
            {
                RCLCPP_ERROR(get_logger(), "nearest query error");
                continue;
            }
            RRTNode2DPtr nearest_node = (RRTNode2DPtr)kd_res_item_data(p_nearest);
            kd_res_free(p_nearest);

            // int v = cost_from_costMap(nearest_node->x); //自适应步长
            // Eigen::Vector2d x_new = steer(nearest_node->x, x_rand, steer_length_/(1+v));
            Eigen::Vector2d x_new = steer(nearest_node->x, x_rand, steer_length_);
            if (!isSegmentValid(nearest_node->x, x_new))
            {
                continue;
            }

            /* 1. 依据总代价找到x_new的父节点*/
            /* kd_tree bounds search for parent */
            Neighbour neighbour_nodes;
            neighbour_nodes.nearing_nodes.reserve(50);
            neighbour_nodes.center = x_new;
            struct kdres *nbr_set;
            nbr_set = kd_nearest_range2(kd_tree, x_new[0], x_new[1],  search_radius_);
            if (nbr_set == nullptr)
            {
                RCLCPP_ERROR(get_logger(), "bkwd kd range query error");
                break;
            }
            while (!kd_res_end(nbr_set))
            {
                RRTNode2DPtr curr_node = (RRTNode2DPtr)kd_res_item_data(nbr_set);
                neighbour_nodes.nearing_nodes.emplace_back(curr_node, false, false);
                // store range query result so that we dont need to query again for rewire;
                kd_res_next(nbr_set); // go to next in kd tree range query result
            }
            kd_res_free(nbr_set); // reset kd tree range query

            /* choose parent from kd tree range query result*/
            double dist2nearest = calDist2D(nearest_node->x, x_new);
            double cost2nearest = featurecost_between(nearest_node->x, x_new)*dist2nearest;

            double min_dist_from_start(nearest_node->dist_from_start + dist2nearest);
            double min_cost_from_start(nearest_node->cost_from_start + cost2nearest);
            double cost_from_p(cost2nearest);
            double dist_from_p(dist2nearest);
            RRTNode2DPtr min_node(nearest_node); // set the nearest_node as the default parent
            // cout<<"3333"<<endl;

            // TODO sort by potential cost-from-start
            for (auto &curr_node : neighbour_nodes.nearing_nodes)
            {
                if (curr_node.node_ptr == nearest_node) // the nearest_node already calculated and checked collision free
                {
                    continue;
                }

                // check potential first, then check edge collision
                double curr_dist = calDist2D(curr_node.node_ptr->x, x_new);
                double curr_cost = featurecost_between(curr_node.node_ptr->x, x_new)*curr_dist;
                double potential_dist_from_start = curr_node.node_ptr->dist_from_start + curr_dist;
                double potential_cost_from_start = curr_node.node_ptr->cost_from_start + curr_cost;
                if (min_cost_from_start > potential_cost_from_start)
                {
                    bool connected = isSegmentValid(curr_node.node_ptr->x, x_new);
                    curr_node.is_checked = true;
                    if (connected)
                    {
                        curr_node.is_valid = true;
                        cost_from_p = curr_cost;
                        dist_from_p = curr_dist;
                        min_dist_from_start = potential_dist_from_start;
                        min_cost_from_start = potential_cost_from_start;
                        min_node = curr_node.node_ptr;
                    }
                }
            }
            // cout<<"444444"<<endl;

            /* parent found within radius, then add a node to rrt and kd_tree */
            // sample-rejection
            /* 1.1 add the randomly sampled node to rrt_tree */
            RRTNode2DPtr new_node(nullptr);
            new_node = addTreeNode(min_node, x_new, min_cost_from_start, cost_from_p, min_dist_from_start, dist_from_p);

            /* 1.2 add the randomly sampled node to kd_tree */
            kd_insert2(kd_tree, x_new[0], x_new[1],  new_node);
            // end of find parent

            /* 3.rewire */
            for (auto &curr_node : neighbour_nodes.nearing_nodes)
            {
                double dist_to_potential_child = calDist2D(new_node->x, curr_node.node_ptr->x);
                double cost_to_potential_child = featurecost_between(new_node->x, curr_node.node_ptr->x)*dist_to_potential_child;
                bool not_consistent = new_node->cost_from_start + cost_to_potential_child < curr_node.node_ptr->cost_from_start ? 1 : 0;
                if(not_consistent)
                {
                    bool connected(false);
                    if (curr_node.is_checked)
                        connected = curr_node.is_valid;
                    else
                        connected = isSegmentValid(new_node->x, curr_node.node_ptr->x);
                    if (connected)
                    {
                        changeNodeParent(curr_node.node_ptr, new_node, cost_to_potential_child, dist_to_potential_child);
                    }
                }
            }
            /* end of rewire */
            // cout<<"cost from start:"<<new_node->cost_from_start<<"  cost from parent:"<<new_node->cost_from_parent<<endl;

            //检查是否离终点近
            if(PlanningState_ == Global){
                // RCLCPP_WARN(get_logger(), "valid_nodenums: %d", valid_tree_node_nums_);
                double dist_to_goal = calDist2D(x_new, goal_node_->x);
                double cost_to_goal = featurecost_between(goal_node_->x, x_new)*dist_to_goal;
                // cout<<"new_node: x"<< x_new[0]<<"  y"<<x_new[1] <<" dist_to_goal:"<< dist_to_goal<<endl;
                /* 2. try to connect to goal if possible */
                if (dist_to_goal <= goal_radius_)
                {
                    bool is_connected2goal = isSegmentValid(x_new, goal_node_->x);
                    // this test can be omitted if sample-rejction is applied
                    bool is_better_path = goal_node_->cost_from_start > cost_to_goal + new_node->cost_from_start;
                    // RCLCPP_WARN(get_logger(), "goal: %lf, new: %lf", (goal_node_->cost_from_start), (cost_to_goal + new_node->cost_from_start));
                    //    RCLCPP_WARN(get_logger(), "-----------------4cost------------%f, id:%d", goal_node_->cost_from_start, idx);
                    if (is_connected2goal && is_better_path)
                    {
                        if (!goal_found)
                        {
                            first_path_use_time_ = (this->get_clock()->now() - rrt_start_time).seconds();
                        }
                        goal_found = true;
                        changeNodeParent(goal_node_, new_node, cost_to_goal, dist_to_goal);
                        std::vector<Eigen::Vector3d> curr_best_path;
                        // RCLCPP_WARN(get_logger(), "----------all cost--------------cost:%f,dist:%f", goal_node_->cost_from_start, goal_node_->dist_from_start);
                        fillPath(goal_node_, curr_best_path);
                        path_list_.emplace_back(curr_best_path);
                        solution_cost_time_pair_list_.emplace_back(goal_node_->cost_from_start, (this->get_clock()->now() - rrt_start_time).seconds());
                        // vis_ptr_->visualize_path(curr_best_path, "rrt_star_final_path");
                        // vis_ptr_->visualize_pointcloud(curr_best_path, "rrt_star_final_wpts");
                        if (use_informed_sampling_){
                            double c_square = (goal_ - start_).squaredNorm() / 4.0;
                            scale_[0] = goal_node_->dist_from_start / 2.0;
                            scale_[1] = sqrt(scale_[0] * scale_[0] - c_square);
                            sampler_.setInformedSacling(scale_);  //此处会将采样标志位置为椭圆
                        }
                    }
                }
            }

        }
        //  RCLCPP_WARN(get_logger(), "valid_nodenums: %d", valid_tree_node_nums_);
        //  RCLCPP_INFO_STREAM(get_logger(), "spend time: " << rrt_time << ", iter times:" << curr_iter_);
        //  cout<<"-----"<<endl;
        /*rrt搜索视觉部分*/
        std::vector<Eigen::Vector3d> vertice;  //点
        std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> edges;    //边
        sampleWholeTree(start_node_, vertice, edges);
        std::vector<visualization::BALL> balls;
        balls.reserve(vertice.size());
        visualization::BALL node_p;
        node_p.radius = 0.1;
        for (size_t i = 0; i < vertice.size(); ++i)
        {
            node_p.center = vertice[i];
            balls.push_back(node_p);
        }
        // vis_ptr_->visualize_balls(balls, "tree_vertice", visualization::Color::blue, 0.1);
        // vis_ptr_->visualize_pairline(edges, "tree_edges", visualization::Color::green, 0.01); //话题是tree_edges

        if (goal_found && PlanningState_==Global)
        {
            final_path_use_time_ = (this->get_clock()->now() - rrt_start_time).seconds();
            // RCLCPP_INFO_STREAM(get_logger(), "[RRT*]: cost from start: " << goal_node_->cost_from_start);
            fillPath(goal_node_, final_path_);
            // RCLCPP_INFO_STREAM(get_logger(), "[RRT*]: first_path_use_time: " << first_path_use_time_ << ", first cost: " << solution_cost_time_pair_list_.front().first);
            //清理节点数据
            for (int i = 0; i < valid_tree_node_nums_; i++)
            {
                nodes_pool_[i]->parent = nullptr;
                nodes_pool_[i]->children.clear();
            }
            kd_clear(kd_tree);
            valid_tree_node_nums_ = 2;

            return goal_found;
        }
        else if (PlanningState_==WithoutGoal)
        {
            return true;
        }
        else
        {
            return false;
        }
    }

    bool plan()
    {
        reset();
        sampler_.reset(); // !important

        if((PlanningState_ == WithoutGoal) || (PlanningState_ == Roll)){
            for (int i = 0; i < valid_tree_node_nums_; i++)
            {
                nodes_pool_[i]->parent = nullptr;
                nodes_pool_[i]->children.clear();
            }
            kd_clear(kd_tree);
            start_node_ = nodes_pool_[1];
            start_node_->x = start_;
            start_node_->cost_from_start = 0.0;
            start_node_->dist_from_start = 0.0;
            valid_tree_node_nums_ = 2;
            int max_iter = 30000;
            double max_time = search_time_;
            curr_iter_ = 0;
            return rrt_star(max_iter, max_time);
        }
        else if(PlanningState_ == Global){
            // if (isStateValid(start_))
            // {
            //     RCLCPP_ERROR(get_logger(), "[RRT*]: start pos collide or out of bound");
            //     return false;
            // }
            if (isStateValid(goal_))
            {
                RCLCPP_ERROR(get_logger(), "[RRT*]: Goal pos collide or out of bound");
                return false;
            }

            // cout<<"inital valid_tree_nums:"<< valid_tree_node_nums_<<endl;
            curr_iter_ = 0;
            goal_node_ = nodes_pool_[0];
            goal_node_->x = goal_;
            goal_node_->cost_from_start = DBL_MAX; // important
            goal_node_->dist_from_start = DBL_MAX;
            // valid_tree_node_nums_ = 2;              // put start and goal in tree
            int max_iter = 30000;
            double max_time = search_time_;
            //初始椭圆的参数
            calInformedSet(10000000000.0, start_, goal_, scale_, trans_, rot_);
            sampler_.setInformedTransRot(trans_, rot_);
            // cout<<"111"<<endl;

            return rrt_star(max_iter, max_time);
        }
        return false;
    }

    void reset()
    {
        final_path_.clear();
        path_list_.clear();
        solution_cost_time_pair_list_.clear();
    }

    /*a2：椭球的短半轴长度的平方（通常是Cbest的值的一半）。
    foci1和foci2：椭球的两个焦点（或者在二维平面上，可以认为是两个焦点所在的点）。
    scale：计算得到的椭球的半轴长度，它是一个包含三个值的Eigen::Vector3d。
    trans：椭球的中心点，也是一个Eigen::Vector3d。
    rot：椭球的旋转矩阵，一个3x3的Eigen::Matrix3d。*/
    void calInformedSet(double a2, const Eigen::Vector2d &foci1, const Eigen::Vector2d &foci2,
                        Eigen::Vector2d &scale, Eigen::Vector2d &trans, Eigen::Matrix2d &rot)
    {
        trans = (foci1 + foci2) / 2.0;
        scale[0] = a2 / 2.0; // 初始路径一半
        Eigen::Vector2d diff(foci2 - foci1);
        double c_square = diff.squaredNorm() / 4.0;
        double theta = atan2(diff[1],diff[0]);
        rot(0,0) = cos(theta); rot(0,1) = -sin(theta);
        rot(1,0) = sin(theta);rot(1,1) = cos(theta);
    }

    bool isStateValid(const Eigen::Vector2d &pos)
    {
        bool in_map = false;
        const double cost = inflatedCostAt(pos, &in_map);
        if(!in_map)
            return true;
        if(cost > cost_max_-0.1){
            // RCLCPP_WARN(get_logger(), "This point is occ");
            return true;}
        return false;
    }

    double inflatedCostAt(const Eigen::Vector2d &point, bool *in_map = nullptr)
    {
        PointType tmpPos;
        tmpPos.x = point[0];
        tmpPos.y = point[1];
        tmpPos.z = -1;
        ipoint2 resPoint;
        if(!Point2Gird(tmpPos, resPoint)){
            if (in_map) *in_map = false;
            return cost_max_;
        }
        int gridIndex = resPoint.x + resPoint.y * mapLength*(int)mapInvResolution;
        if (gridIndex < 0 || gridIndex >= static_cast<int>(msgElevationGlobal.occupancy.data.size())) {
            if (in_map) *in_map = false;
            return cost_max_;
        }
        if (in_map) *in_map = true;
        return static_cast<double>(msgElevationGlobal.occupancy.data[gridIndex]);
    }

    double cost_from_costMap(const Eigen::Vector2d &point)
    {
        bool in_map = false;
        const double cost = inflatedCostAt(point, &in_map);
        if (!in_map)
            return 1.0;
        return 0.01 * std::clamp(cost, 0.0, 100.0);
        // RCLCPP_WARN(get_logger(), "This point is occ");
    }

    double featurecost_between(const Eigen::Vector2d &point1, const Eigen::Vector2d &point2)
    {
        double m1 = cost_from_costMap(point1);
        double m2 = cost_from_costMap(point2);
        return (1+cost_bias_*(1/(1.0001f-m1)+1/(1.0001f-m2)-2));
        // return 1;
    }

    /*检测两节点间线段是否有效
    1.创建一个 RayCaster 对象,用于执行光线投射算法。
    2.将 p0 和 p1 的二维坐标转换为三维坐标 p03d 和 p13d。
    3.调用 raycaster.setInput(p03d / mapResolution, p13d / mapResolution) 设置光线的起点和终点坐标。如果不需要进行光线投射,则直接返回 true。
    4.定义一个 half 向量,表示单个网格单元的半长度。
    5.通过 raycaster.step(ray_pt) 函数逐步沿着光线前进,获取每个网格单元的坐标 ray_pt。
    6.对于每个 ray_pt 对应的二维坐标 tmp2d(乘以 mapResolution 得到实际坐标),检查该点是否为有效状态,如果不是,则返回 false。
    7.如果所有经过的网格单元都是有效状态,则返回 true。*/
    bool isSegmentValid(const Eigen::Vector2d &p0, const Eigen::Vector2d &p1, double max_dist = DBL_MAX)
    {
        Eigen::Vector2d dp_ = p1 - p0;
        Eigen::Vector3d dp(dp_[0],dp_[1],0.0);
        double dist = dp.norm();
        if (dist > max_dist)
        {
            return false;
        }
        RayCaster raycaster;
        Eigen::Vector3d p03d(p0[0],p0[1],0.0);
        Eigen::Vector3d p13d(p1[0],p1[1],0.0);
        bool need_ray = raycaster.setInput(p03d / mapResolution, p13d / mapResolution); //(ray start, ray end)
        if (!need_ray)
            return true;
        Eigen::Vector3d half = Eigen::Vector3d(0.5, 0.5, 0.5);
        Eigen::Vector3d ray_pt;
        if (!raycaster.step(ray_pt)) // skip the ray start point
            return true;
        while (raycaster.step(ray_pt))
        {
            Eigen::Vector3d tmp = (ray_pt + half) * mapResolution;
            Eigen::Vector2d tmp2d(tmp[0],tmp[1]);
            if (isStateValid(tmp2d))
            {
                return false;
            }
        }
        return true;
    }

    Eigen::Vector2d selectLocalPlanningGoal(const Eigen::Vector2d &global_goal)
    {
        has_valid_local_goal_ = false;
        Eigen::Vector2d to_goal = global_goal - start_;
        const double dist_global = to_goal.norm();
        if (dist_global < 1e-3){
            has_valid_local_goal_ = !isStateValid(global_goal);
            return global_goal;
        }

        const double horizon = std::min(goal_local_dis_, localMapLength * 0.45);
        const Eigen::Vector2d direction = to_goal / dist_global;
        const Eigen::Vector2d lateral(-direction[1], direction[0]);
        const double target_distance = std::min(dist_global, horizon);
        const bool final_goal_in_local_range = dist_global <= horizon;

        std::vector<double> forward_scales = {1.0, 0.85, 0.7, 0.55};
        std::vector<double> lateral_offsets = {0.0, 0.8, -0.8, 1.6, -1.6, 2.4, -2.4};

        Eigen::Vector2d best_goal = start_ + direction * target_distance;
        double best_score = -DBL_MAX;
        bool found = false;
        for (double scale : forward_scales){
            const double forward = std::max(1.0, target_distance * scale);
            for (double offset : lateral_offsets){
                Eigen::Vector2d candidate = start_ + direction * forward + lateral * offset;
                if (final_goal_in_local_range && std::abs(offset) < 1e-6 && scale == 1.0){
                    candidate = global_goal;
                }
                if (isStateValid(candidate)){
                    continue;
                }
                if (!isSegmentValid(start_, candidate, horizon + 0.5)){
                    continue;
                }

                const double progress = (candidate - start_).dot(direction);
                const double lateral_penalty = std::abs((candidate - start_).dot(lateral));
                const double inflated_cost = inflatedCostAt(candidate);
                const double goal_distance = (global_goal - candidate).norm();
                const double score = progress - 0.25 * lateral_penalty - 0.03 * inflated_cost - 0.05 * goal_distance;
                if (score > best_score){
                    best_score = score;
                    best_goal = candidate;
                    found = true;
                }
            }
        }

        if (!found){
            RCLCPP_WARN(get_logger(), "No collision-free local subgoal found; waiting for a new RViz goal");
            return start_;
        }
        has_valid_local_goal_ = true;
        return best_goal;
    }

    void clearActiveGoalPlanning()
    {
        final_path_.clear();
        sampleFinalPath.clear();
        path_list_.clear();
        solution_cost_time_pair_list_.clear();
        PlanningState_ = WithoutGoal;
        lastGoalIndex = 1;
        isSendGoal = false;
        has_valid_local_goal_ = false;
        nav_msgs::msg::Path empty_path;
        empty_path.header.frame_id = "map";
        empty_path.header.stamp = this->get_clock()->now();
        pubSmoothPath->publish(empty_path);
    }

    void goalPosHandler(const geometry_msgs::msg::PoseStamped::ConstSharedPtr goal){
        std::lock_guard<std::mutex> lock(mtx);

        // For ROS2, file operations are not typically tied to callbacks this way.
        // If file saving is required, consider a separate service or a dedicated thread/function.
        // std::lock_guard<std::mutex> lock(mtx);
        goalPoint.x = goal->pose.position.x;
        goalPoint.y = goal->pose.position.y;
        goalPoint.z = goal->pose.position.z;
        Eigen::Vector3d goal_global;
        goal_global<<goalPoint.x, goalPoint.y,(visHeight+2);
        // vis_ptr_->visualize_a_ball(goal_global, 0.5, "goal", visualization::Color::yellow);

        RCLCPP_WARN(get_logger(), "get goal point=(%f,%f)",goalPoint.x,goalPoint.y);
        // RCLCPP_WARN(get_logger(), "get start point=(%f,%f)",robotPoint.x,robotPoint.y);

        PlanningState_ = Global;
        lastGoalIndex = 1;
        isSendGoal = false; //直接发送终点标志位
        has_valid_local_goal_ = false;
        final_path_.clear();
        sampleFinalPath.clear();
        path_list_.clear();
        solution_cost_time_pair_list_.clear();

        return;
    }


    void RRTstarHandler() { // 移除了 TimerEvent 参数
        std::lock_guard<std::mutex> lock(mtx);
        try{
            transform = tf_buffer_->lookupTransform("map","base_link", tf2::TimePointZero);
            robotPoint.x = transform.transform.translation.x;
            robotPoint.y = transform.transform.translation.y;
            robotPoint.z = transform.transform.translation.z;
            start_<< robotPoint.x, robotPoint.y;
            RCLCPP_WARN(get_logger(), "get start point=(%f,%f)",robotPoint.x,robotPoint.y);
            haveRobotPoint = true;
        }
        catch (tf2::TransformException &ex){
            RCLCPP_ERROR(get_logger(), "Transfrom Failure1: %s", ex.what());
            // return;
        }
        if(!haveRobotPoint){
            RCLCPP_WARN(get_logger(), "No recive robot position!");
            return;
        }

        Eigen::Vector2d gridMapOrigin; //二维向量
        Eigen::Vector2d gridMapRange;

        gridMapOrigin<< start_[0]-localMapLength/2, start_[1]-localMapLength/2;
        gridMapRange<<localMapLength-0.4, localMapLength-0.4; //减去0.5是因为抖动老是超出边界
        if(PlanningState_ == WithoutGoal){
            RCLCPP_WARN(get_logger(), "No recive goal point, expand tree");
            PlanningState_ = WithoutGoal;
            sampler_.initWithoutGoal(start_, gridMapOrigin, gridMapRange);  //传入地图参数用于后面采样，地图左下角为起点
            nav_msgs::msg::Path NonePath;
            NonePath.header.frame_id = "map";
            NonePath.header.stamp = this->get_clock()->now();
            pubSmoothPath->publish(NonePath); //路径视觉化清零
        }
        //1.有goal且第一次规划  2. 有goal且roll
        else if(PlanningState_== Global){
            RCLCPP_WARN(get_logger(), "Recive goal point, start_planning");
            goal_start_time = this->get_clock()->now();

            Eigen::Vector2d goal_inital;
            goal_inital<< goalPoint.x, goalPoint.y;
            goal_ = selectLocalPlanningGoal(goal_inital);
            if(!has_valid_local_goal_ || isStateValid(goal_)){
                RCLCPP_WARN(
                    get_logger(),
                    "RViz goal/local subgoal is in collision or out of bound; clearing planner state and waiting for next goal"
                );
                clearActiveGoalPlanning();
                haveRobotPoint = false;
                return;
            }

            double distDiffNorm = (goal_-start_).norm() ;

            if(distDiffNorm>localMapLength/2){
                gridMapOrigin<< start_[0]-(distDiffNorm+localMapLength/6), start_[1]-(distDiffNorm+localMapLength/6); //采样范围比两点间再大一点
                gridMapRange<< 2*(distDiffNorm+localMapLength/6), 2*(distDiffNorm+localMapLength/6);
            }
            sampler_.initWithGoal(start_, goal_, gridMapOrigin, gridMapRange);

            //视觉化
            start3D<<start_[0], start_[1],(visHeight+0.2);
            goal3D<<goal_[0], goal_[1],(visHeight+0.2);
            // vis_ptr_->visualize_a_ball(start3D, 0.1, "start", visualization::Color::pink);
            // vis_ptr_->visualize_a_ball(goal3D, 1, "goal_local", visualization::Color::steelblue);
            //起点与终点距离小于0.6则认为到达目标 fastlio:0.35 odom:0.65
            // cout<<"dist:"<<distDiffNorm<<endl;
            if(distDiffNorm<= 0.65){
                runTime = 0;
                RCLCPP_WARN(get_logger(), "UGV is closed to goal!! plan finished");
                PlanningState_= WithoutGoal;
                return;
            }
            else if(distDiffNorm<= 1.1){    //起点与终点距离小于1.1则直接发送goal并返回
                geometry_msgs::msg::PoseStamped RRTgoal;
                RRTgoal.header.stamp = this->get_clock()->now();
                RRTgoal.header.frame_id = "map";
                RRTgoal.pose.position.x = goal3D[0];
                RRTgoal.pose.position.y = goal3D[1];
                RRTgoal.pose.position.z = goal3D[2];
                pubLocalGoal->publish(RRTgoal);
                haveRobotPoint = false;
                isSendGoal = true;
                return;
            }
            else{
                // RCLCPP_WARN(get_logger(), "distance from start(%.2f,%.2f) to goa;(%.2f,%.2f)   is %.2f",
                //            start3D[0],start3D[1],goal3D[0],goal3D[1],distDiffNorm);
            }

        }
        else if(PlanningState_ == Roll){
            sampler_.initWithoutGoal(start_, gridMapOrigin, gridMapRange);  //传入地图参数用于后面采样，地图左下角为起点
            double sum_time = (this->get_clock()->now() - goal_start_time).seconds();
            if(sum_time < time_roll_ ){    //时间未到，发点
                //    runTime = 0;
                std::vector<Eigen::Vector3d> final_path = sampleFinalPath;
                if(final_path.size()>=1){    //当路径点个数大于1时，
                    geometry_msgs::msg::PoseStamped RRTgoal;
                    RRTgoal.header.stamp = this->get_clock()->now();
                    RRTgoal.header.frame_id = "map";
                    bool haveEnoughDiff = false; // 修正此变量未被使用的问题
                    double tmpLen = 0;
                    Eigen::Vector3d tmpLastStart3D(start_[0], start_[1],0);
                    Eigen::Vector3d curPoint;
                    if(!isSendGoal){    //如果不直接发goal，直接发跟踪点
                        for(size_t k = lastGoalIndex; k < final_path.size(); k++){ // 使用 size_t

                            Eigen::Vector3d tmpDistDiff = tmpLastStart3D-final_path[k];
                            tmpLastStart3D = final_path[k];
                            tmpLen += tmpDistDiff.norm();

                            if(tmpLen  >= (localPathLen+follow_dis_)){
                                RRTgoal.pose.position.x = final_path[k][0];
                                RRTgoal.pose.position.y = final_path[k][1];
                                RRTgoal.pose.position.z = final_path[k][2];
                                curPoint = final_path[k];
                                haveEnoughDiff = true;
                                lastGoalIndex = static_cast<int>(k);      //记录跟踪点在路径中的索引，并转换为int
                                // RCLCPP_WARN(get_logger(), " lastGoalIndex = %d, ",lastGoalIndex); //跟踪点在路径点中的索引
                                break;
                            }
                        }
                    }
                    if(!haveEnoughDiff){ //当没有中间点发布时，就直接发布终点
                        isSendGoal = true;
                        RRTgoal.pose.position.x = goal3D[0];
                        RRTgoal.pose.position.y = goal3D[1];
                        RRTgoal.pose.position.z = goal3D[2];
                        curPoint = goal3D;
                    }
                    pubLocalGoal->publish(RRTgoal);
                    // cout<<"111111111111111"<<endl;
                    // vis_ptr_->visualize_a_ball(curPoint, 0.1, "waypoint", visualization::Color::green); //跟踪点，距离2.5m
                }
                haveRobotPoint = false;
                return;
            }
            else{    //时间到了，重规划
                goal_start_time = this->get_clock()->now();
                PlanningState_ = Global;
            }
        }

        // RCLCPP_WARN(get_logger(), "--------------------start RRT* plan--------------");
        bool rrt_star_res = plan();

        //接收到目标点后仅执行一次，后续不再执行
        if (rrt_star_res && PlanningState_==Global){
            runTime++;
            std::vector<std::vector<Eigen::Vector3d>> routes = path_list_;
            // RCLCPP_INFO_STREAM(get_logger(), "path_num:" << path_list_.size());
            // vis_ptr_->visualize_path_list(routes, "rrt_star_paths", visualization::Color::blue); //多个路径
            std::vector<Eigen::Vector3d> final_path = final_path_;
            // vis_ptr_->visualize_path(final_path, "rrt_star_final_path");    //最终最优的路径

            smooth_path(final_path);

            // vis_ptr_->visualize_pointcloud(final_path, "rrt_star_final_wpts");
            std::vector<std::pair<double, double>> slns = solution_cost_time_pair_list_;
            // RCLCPP_INFO_STREAM(get_logger(), "[RRT*] final path cost: " << slns.back().first);

            std::vector<Eigen::Vector3d> sampleFinalPath_;
            //对final_path进行采样，间距小于1m
            for(size_t k = 0; k < final_path.size()-1; k++){ // 使用 size_t
                Eigen::Vector3d firstPoint = final_path[k];
                Eigen::Vector3d twoPoint = final_path[k+1];
                Eigen::Vector3d tmpDistDiff = twoPoint-firstPoint;
                double twoPointDiff = tmpDistDiff.norm();
                Eigen::Vector3d unitVector = tmpDistDiff/twoPointDiff;

                if(twoPointDiff  >= localPathLen){
                    int segmentNum = static_cast<int>(twoPointDiff/localPathLen); // Cast to int

                    for(int i = 0; i <= segmentNum;i++){
                       Eigen::Vector3d tmpPoint =  firstPoint+static_cast<double>(i)*localPathLen*unitVector; // Cast i to double
                       sampleFinalPath_.push_back(tmpPoint);
                    }
                }
                else{
                    sampleFinalPath_.push_back(firstPoint);
                }
            }

            // RCLCPP_INFO_STREAM(get_logger(), "[RRT*] final sample path size: " << sampleFinalPath_.size());  //最终路径上的采样点数

            if(sampleFinalPath_.size()>1){
                sampleFinalPath_.push_back(final_path[final_path.size()-1]); //添加最终路径的终点
                sampleFinalPath.clear();
                sampleFinalPath.assign(sampleFinalPath_.begin(), sampleFinalPath_.end());

                geometry_msgs::msg::PoseStamped RRTgoal;
                RRTgoal.header.stamp = this->get_clock()->now();
                RRTgoal.header.frame_id = "map";
                // bool haveEnoughDiff = false; // 移除未使用的变量

                //得到路径后就发第一个点，之后当前面所有点与起点的累计距离大于3就发布下一个点
                Eigen::Vector3d tmpLastStart3D(start_[0], start_[1],0);
                double tmpLen = 0;
                RRTgoal.pose.position.x = sampleFinalPath[1][0];
                RRTgoal.pose.position.y = sampleFinalPath[1][1];
                RRTgoal.pose.position.z = sampleFinalPath[1][2];
                Eigen::Vector3d pushlishPoint;
                pushlishPoint = sampleFinalPath[1];
                bool isSendGoal2 = true;
                for(size_t k = 1; k < sampleFinalPath.size(); k++){ // 使用 size_t

                    Eigen::Vector3d tmpDistDiff = tmpLastStart3D-sampleFinalPath[k];
                    tmpLastStart3D = sampleFinalPath[k];
                    tmpLen += tmpDistDiff.norm();     //将路径上前面所有点与起点的的距离进行累加，直到大于3

                    if(tmpLen  >= localPathLen+follow_dis_+0.5){
                        RRTgoal.pose.position.x = sampleFinalPath[k][0];
                        RRTgoal.pose.position.y = sampleFinalPath[k][1];
                        RRTgoal.pose.position.z = sampleFinalPath[k][2];
                        pushlishPoint = sampleFinalPath[k];
                        isSendGoal2 = false;
                        break;
                    }
                }

                if(isSendGoal2){     //累计距离小于3就直接发布终点
                    RRTgoal.pose.position.x = goal3D[0];
                    RRTgoal.pose.position.y = goal3D[1];
                    RRTgoal.pose.position.z = goal3D[2];
                    pushlishPoint = goal3D;
                }

                pubLocalGoal->publish(RRTgoal);
                lastGoalIndex = 1;
                // vis_ptr_->visualize_a_ball(pushlishPoint, 0.1, "waypoint", visualization::Color::green);
            }
            PlanningState_ = Roll;
        }
        haveRobotPoint = false;
    }

    void smooth_path(std::vector<Eigen::Vector3d> &path){
        nav_msgs::msg::Path originPath;
        nav_msgs::msg::Path splinePath;
        geometry_msgs::msg::PoseStamped tmpPose;
        tmpPose.header.frame_id = "map";
        std::vector<Eigen::Vector3d> currentPath = filterSmoothPath(pathFromCurrent(path));
        for (const auto &pt : currentPath)
        {
            tmpPose.pose.position.x = pt[0];
            tmpPose.pose.position.y = pt[1];
            tmpPose.pose.position.z = pt[2];
            originPath.poses.push_back(tmpPose);
        }
        originPath.header.frame_id = "map";
        originPath.header.stamp = this->get_clock()->now();
        // Assuming path_smoothing::CubicSplineInterpolator is a separate library compatible with ROS2 msgs
        // You might need to adapt this part if the library is ROS1 specific.
        // path_smoothing::CubicSplineInterpolator csi("smooth");
        // csi.interpolatePath(originPath, splinePath);
        // pubSmoothPath->publish(splinePath);

        // NOTE: For the sake of direct conversion, I'm commenting out the spline part.
        // You would need a ROS2 equivalent or reimplement `CubicSplineInterpolator`
        // if this is a custom class.
        pubSmoothPath->publish(originPath);
        splinePath = originPath;

        // Extract Path
        path.clear();
        for (size_t i = 0; i < splinePath.poses.size(); ++i){ // Use size_t for loop counter
            Eigen::Vector3d tmpNode(splinePath.poses[i].pose.position.x, splinePath.poses[i].pose.position.y,visHeight);
            path.push_back(tmpNode);
        }
    }

    std::vector<Eigen::Vector3d> pathFromCurrent(const std::vector<Eigen::Vector3d> &path) const {
        std::vector<Eigen::Vector3d> trimmed;
        Eigen::Vector3d current(robotPoint.x, robotPoint.y, visHeight);
        trimmed.push_back(current);

        if (path.empty()){
            return trimmed;
        }

        if (path.size() == 1){
            if ((path.front().head<2>() - current.head<2>()).norm() > 0.2){
                trimmed.push_back(path.front());
            }
            return trimmed;
        }

        size_t best_segment = 0;
        double best_dist_sq = std::numeric_limits<double>::max();
        for (size_t i = 0; i + 1 < path.size(); ++i){
            const Eigen::Vector2d a = path[i].head<2>();
            const Eigen::Vector2d b = path[i + 1].head<2>();
            const Eigen::Vector2d ab = b - a;
            const double ab_sq = ab.squaredNorm();
            double t = 0.0;
            if (ab_sq > 1e-6){
                t = std::clamp((current.head<2>() - a).dot(ab) / ab_sq, 0.0, 1.0);
            }
            const Eigen::Vector2d projection = a + t * ab;
            const double dist_sq = (current.head<2>() - projection).squaredNorm();
            if (dist_sq < best_dist_sq){
                best_dist_sq = dist_sq;
                best_segment = i;
            }
        }

        for (size_t i = best_segment + 1; i < path.size(); ++i){
            if ((path[i].head<2>() - trimmed.back().head<2>()).norm() < 0.2){
                continue;
            }
            trimmed.push_back(path[i]);
        }

        if (trimmed.size() == 1 &&
            (path.back().head<2>() - trimmed.back().head<2>()).norm() > 0.05){
            trimmed.push_back(path.back());
        }
        return trimmed;
    }

    std::vector<Eigen::Vector3d> filterSmoothPath(const std::vector<Eigen::Vector3d> &path) {
        if (path.size() < 3){
            return path;
        }

        std::vector<Eigen::Vector3d> filtered;
        filtered.reserve(path.size());
        filtered.push_back(path.front());

        for (size_t i = 1; i + 1 < path.size(); ++i){
            Eigen::Vector3d candidate = 0.25 * path[i - 1] + 0.5 * path[i] + 0.25 * path[i + 1];
            candidate[2] = visHeight;
            const Eigen::Vector2d candidate2d(candidate[0], candidate[1]);
            const Eigen::Vector2d last2d(filtered.back()[0], filtered.back()[1]);
            const Eigen::Vector2d next2d(path[i + 1][0], path[i + 1][1]);

            if (!isStateValid(candidate2d) &&
                isSegmentValid(last2d, candidate2d) &&
                isSegmentValid(candidate2d, next2d)){
                filtered.push_back(candidate);
            }else{
                filtered.push_back(path[i]);
            }
        }

        if ((path.back().head<2>() - filtered.back().head<2>()).norm() >= 0.05){
            filtered.push_back(path.back());
        }
        return filtered;
    }

    void generateBox(BoxPointType &boxpoint, const PointType *robotcenter, std::vector<float> boxlengths){
        float &x_dist = boxlengths[0];
        float &y_dist = boxlengths[1];
        float &z_dist = boxlengths[2];

        boxpoint.vertex_min[0] = robotcenter->x - x_dist;
        boxpoint.vertex_max[0] = robotcenter->x + x_dist;
        boxpoint.vertex_min[1] = robotcenter->y - y_dist;
        boxpoint.vertex_max[1] = robotcenter->y + y_dist;
        boxpoint.vertex_min[2] = robotcenter->z - z_dist;
        boxpoint.vertex_max[2] = robotcenter->z + z_dist;
    }

    void rewardHandler(const elevation_msgs::msg::OccupancyElevation::ConstSharedPtr rewardMsg){
        std::lock_guard<std::mutex> lock(mtx);
        rewardMap = *rewardMsg;
        std::vector<float> inflated_reward_cost = rewardMap.reward_cost;
        //膨胀代价地图
        int sizeMap = static_cast<int>(rewardMap.occupancy.data.size()); // Cast to int
        int inflationSize = static_cast<int>(inflate_r_ / rewardMap.occupancy.info.resolution);
        for (int i = 0; i < sizeMap; ++i) {
            int idX = static_cast<int>(i % rewardMap.occupancy.info.width);
            int idY = static_cast<int>(i / rewardMap.occupancy.info.width);
            if (rewardMap.reward_cost[i] > reward_obstacle_threshold_){
                for (int m = -inflationSize; m <= inflationSize; ++m) {
                    for (int n = -inflationSize; n <= inflationSize; ++n) {
                        int newIdX = idX + m;
                        int newIdY = idY + n;
                        if (newIdX < 0 || newIdX >= static_cast<int>(rewardMap.occupancy.info.width) || newIdY < 0 || newIdY >= static_cast<int>(rewardMap.occupancy.info.height)) // Cast for comparison
                            continue;
                        int index = newIdX + newIdY * static_cast<int>(rewardMap.occupancy.info.width); // Cast for multiplication
                        inflated_reward_cost[index] = cost_max_;
                    }
                }
            }
        }

        // local map origin x and y
        // localMapOriginPoint2.x = robotPoint.x - localMapLength / 2;
        // localMapOriginPoint2.y = robotPoint.y - localMapLength / 2;
        // localMapOriginPoint2.z = robotPoint.z;
        localMapOriginPoint2.x = rewardMap.occupancy.info.origin.position.x;
        localMapOriginPoint2.y = rewardMap.occupancy.info.origin.position.y;
        localMapOriginPoint2.z = rewardMap.occupancy.info.origin.position.z;
        // local map origin cube id (in global map)
        localMapOriginGrid2.cubeX = static_cast<int>((localMapOriginPoint2.x + mapCubeLength/2.0f) / mapCubeLength) + rootCubeIndex; // Cast to int, add f suffix
        localMapOriginGrid2.cubeY = static_cast<int>((localMapOriginPoint2.y + mapCubeLength/2.0f) / mapCubeLength) + rootCubeIndex; // Cast to int, add f suffix
        if (localMapOriginPoint2.x + mapCubeLength/2.0f < 0)  --localMapOriginGrid2.cubeX; // Add f suffix
        if (localMapOriginPoint2.y + mapCubeLength/2.0f < 0)  --localMapOriginGrid2.cubeY; // Add f suffix
        // local map origin grid id (in sub-map)
        float originCubeOriginX2, originCubeOriginY2; // the orign of submap that the local map origin belongs to (note the submap may not be created yet, cannot use originX and originY)
        originCubeOriginX2 = (localMapOriginGrid2.cubeX - rootCubeIndex) * mapCubeLength - mapCubeLength/2.0f; // Add f suffix
        originCubeOriginY2 = (localMapOriginGrid2.cubeY - rootCubeIndex) * mapCubeLength - mapCubeLength/2.0f; // Add f suffix
        localMapOriginGrid2.gridX = static_cast<int>((localMapOriginPoint2.x - originCubeOriginX2) / mapResolution); // Cast to int
        localMapOriginGrid2.gridY = static_cast<int>((localMapOriginPoint2.y - originCubeOriginY2) / mapResolution); // Cast to int

        for(int i = 2;i < localMapArrayLength-2; ++i){
            for (int j = 2; j < localMapArrayLength-2; ++j){
                int indX = localMapOriginGrid2.gridX + i;
                int indY = localMapOriginGrid2.gridY + j;

                grid_t thisGrid2;

                thisGrid2.cubeX = localMapOriginGrid2.cubeX + indX / mapCubeArrayLength;
                thisGrid2.cubeY = localMapOriginGrid2.cubeY + indY / mapCubeArrayLength;

                // RCLCPP_WARN(get_logger(), "thisGrid2 position, x:%d m, y:%d m", thisGrid2.cubeX, thisGrid2.cubeY);
                if (mapArrayInd[thisGrid2.cubeX][thisGrid2.cubeY] == -1){
                    continue;
                }
                thisGrid2.gridX = indX % mapCubeArrayLength;
                thisGrid2.gridY = indY % mapCubeArrayLength;
          
                
                mapCell_t *thisCell2 = grid2Cell(&thisGrid2);
                if (thisCell2) {
                    int index = i + j * localMapArrayLength;
                    updateCellCost2(thisCell2, rewardMap.reward_cost[index], inflated_reward_cost[index]);
                }
            }
        }
    }

    void updateCellCost2(mapCell_t *thisCell, const float &cost, const float &costInflate){
        // std::cout<<"cost will be updated:"<<cost<<endl;
        thisCell->updateCost(cost, costInflate);
    }

    void cloudHandler(const sensor_msgs::msg::PointCloud2::ConstSharedPtr laserCloudMsg){
        
        std::lock_guard<std::mutex> lock(mtx);
        // Get Robot Position
        if (getRobotPosition() == false)
            return;
        if(haveRobotPoint == false)
            haveRobotPoint = true;
        // Convert Point Cloud
        PointType center;
        center.x = 5.0f; // Add f suffix
        center.y = 0.0f; // Add f suffix
        center.z = 0.0f; // Add f suffix
        
        // setCostInRadius(center, 1.0, 100.0);
        Eigen::Vector3d robotOrigin = Eigen::Vector3d(robotPoint.x, robotPoint.y, robotPoint.z);
        // caster_->setRrigin(robotOrigin);
        pcl::fromROSMsg(*laserCloudMsg, *laserCloud);
        kdtree_ptr->Build((*laserCloud).points);
        // Register New Scan
        updateElevationMap();

        
        // Additional file logging part needs to be adapted for ROS2 (e.g., using `rclcpp::Log`) or removed.
        // std::ofstream file("/home/zle/Desktop/figure/毕业论文/time_record/ele_time.txt", std::ios::app);
        // if (file.is_open()) {
        //     file << elapsed_time_ms << std::endl;
        //     file.close();
        // } else {
        //     std::cerr << "Failed to open file for writing" << std::endl;
        // }

        //初始化占据栅格并更新
        localMapProcess();

        const bool publish_debug = shouldPublishVisualization();
        if (publish_debug && hasDebugCloudSubscribers()){
            elevationMapProcess();
        }

        publishMap(publish_debug);
    }

    //更新点云高度z
    void updateElevationMap(){
        int cloudSize = static_cast<int>(laserCloud->points.size()); // Cast to int
        for (int i = 0; i < cloudSize; ++i){ //更新每个点云
            // laserCloud->points[i].z -= 0.2; // for visualization用于显示laser扫描到的地方
            updateElevationMap(&laserCloud->points[i]);
        }
    }

    void updateElevationMap(PointType *point){
        // Find point index in global map
        grid_t thisGrid;
        if (findPointGridInMap(&thisGrid, point) == false) return;
        // Get current cell pointer
    
        
        mapCell_t *thisCell = grid2Cell(&thisGrid);  //cell指针
        if (thisCell) {
            // update elevation  使用卡尔曼滤波更新网格的高程信息
            updateCellElevation(thisCell, point);
            // update occupancy  更新网格单元的占用信息
            updateCellOccupancy(thisCell, point);
            // update roughness
            updateCellRoughness(thisCell, point);
            // update observation time更新网格单元的被观测次数
            updateCellObservationTime(thisCell);
        }
    }

    void updateCellRoughness(mapCell_t *thisCell, PointType *point){

        float hVar;
        int  obsCount;
        float r = calcElevationRoughness(point,hVar,obsCount);
        if (thisCell->roughness == -FLT_MAX){

            thisCell->roughness = r;
            thisCell->roughnessVar =  1.0f/(obsCount); // Add f suffix
            return;
        }
        // if(thisCell->observeTimes > 30) return;
        // RCLCPP_WARN(get_logger(), "(%f,%d)",hVar ,obsCount);

        // Predict:
        float x_pred = thisCell->roughness; // x = F * x + B * u
        float P_pred = thisCell->roughnessVar; // P = F*P*F + Q
        // Update:
        float R_factor = (thisCell->observeTimes > 20) ? 10.0f : 1.0f; // Add f suffix
        float R = pointDistance(robotPoint, *point) * R_factor; // measurement noise: R, scale it with dist and observed times
        float K = P_pred / (P_pred + R);// Gain: K  = P * H^T * (HPH + R)^-1
        float y = point->z; // measurement: y
        float x_final = x_pred + K * (y - x_pred); // x_final = x_pred + K * (y - H * x_pred)
        float P_final = (1.0f - K) * P_pred; // P_final = (I - K * H) * P_pred // Add f suffix
        // Update cell
        thisCell->updateRoughness(x_final, P_final);

    }

    void updateCellObservationTime(mapCell_t *thisCell){
        ++thisCell->observeTimes;
        if (thisCell->observeTimes >= traversabilityObserveTimeTh)
            observingList1.push_back(thisCell);
    }

    void updateCellOccupancy(mapCell_t *thisCell, PointType *point){
        // Update log_odds
        float p;  // Probability of being occupied knowing current measurement.
        if (point->intensity == 100)
            p = p_occupied_when_laser;
        else
            p = p_occupied_when_no_laser;
        thisCell->log_odds += std::log(p / (1.0f - p)); // Add f suffix

        if (thisCell->log_odds < -large_log_odds)
            thisCell->log_odds = -large_log_odds;
        else if (thisCell->log_odds > large_log_odds)
            thisCell->log_odds = large_log_odds;
        // Update occupancy
        float occupancy;
        if (thisCell->log_odds < -max_log_odds_for_belief)
            occupancy = 0.0f; // Use float literal
        else if (thisCell->log_odds > max_log_odds_for_belief)
            occupancy = 100.0f; // Use float literal
        else{
            occupancy = 0.0f; // Use float literal
            // occupancy = (int)(lround((1.0f - 1.0f / (1.0f + std::exp(thisCell->log_odds))) * 100.0f)); //zle sim // Use float literals
        }
        // update cell
        thisCell->updateOccupancy(occupancy);
    }

    void updateCellElevation(mapCell_t *thisCell, PointType *point){
        // Kalman Filter: update cell elevation using Kalman filter
        // https://www.cs.cornell.edu/courses/cs4758/2012sp/materials/MI63slides.pdf

        // cell is observed for the first time, no need to use Kalman filter
        if (thisCell->elevation == -FLT_MAX){
            thisCell->elevation = point->z;              //点云高度赋值给高程
            thisCell->elevationVar = pointDistance(robotPoint, *point);
            return;
        }

        //Kalman  Predict:
        float x_pred = thisCell->elevation; // x = F * x + B * u
        float P_pred = thisCell->elevationVar + 0.01f; // P = F*P*F + Q // Add f suffix
        // Update:
        float R_factor = (thisCell->observeTimes > 20) ? 10.0f : 1.0f; // Add f suffix
        float R = pointDistance(robotPoint, *point) * R_factor; // measurement noise: R, scale it with dist and observed times
        float K = P_pred / (P_pred + R);// Gain: K  = P * H^T * (HPH + R)^-1
        float y = point->z; // measurement: y
        float x_final = x_pred + K * (y - x_pred); // x_final = x_pred + K * (y - H * x_pred)
        float P_final = (1.0f - K) * P_pred; // P_final = (I - K * H) * P_pred // Add f suffix
        // Update cell
        thisCell->updateElevation(x_final, P_final);
    }

    //根据grid指针返回cell（每个cell就是一个grid）
 
    
    //整个地图分为多个submap，存在mapArray(一维向量)中，每个mapArray又被分为多个cell，存在cellArray(二维向量),返回该位置cell(mapCell_t格式）
    mapCell_t* grid2Cell(grid_t *thisGrid){      //cellArray：二维向量，每个都是mapCell_t格式
        // Check for null pointer
        if (!thisGrid) {
            RCLCPP_ERROR(get_logger(), "Null pointer passed to grid2Cell");
            return nullptr;
        }
        
        // Check array bounds
        if (thisGrid->cubeX < 0 || thisGrid->cubeX >= mapArrayLength || 
            thisGrid->cubeY < 0 || thisGrid->cubeY >= mapArrayLength) {
            RCLCPP_ERROR(get_logger(), "Array index out of bounds in grid2Cell: cubeX=%d, cubeY=%d", 
                        thisGrid->cubeX, thisGrid->cubeY);
            return nullptr;
        }
        
        // Check if mapArrayInd is valid
        if (mapArrayInd[thisGrid->cubeX][thisGrid->cubeY] == -1) {
            RCLCPP_ERROR(get_logger(), "Invalid mapArrayInd at (%d, %d)", thisGrid->cubeX, thisGrid->cubeY);
            return nullptr;
        }
        
        int mapID = mapArrayInd[thisGrid->cubeX][thisGrid->cubeY];
        
        // Check if mapArray index is valid
        if (mapID < 0 || mapID >= static_cast<int>(mapArray.size())) {
            RCLCPP_ERROR(get_logger(), "Invalid mapArray index: %d", mapID);
            return nullptr;
        }
        
        // Check if mapArray element is valid
        if (!mapArray[mapID]) {
            RCLCPP_ERROR(get_logger(), "Null pointer in mapArray at index: %d", mapID);
            return nullptr;
        }
        
        // Check grid bounds
        if (thisGrid->gridX < 0 || thisGrid->gridX >= mapCubeArrayLength ||
            thisGrid->gridY < 0 || thisGrid->gridY >= mapCubeArrayLength) {
            RCLCPP_ERROR(get_logger(), "Grid index out of bounds: gridX=%d, gridY=%d", 
                        thisGrid->gridX, thisGrid->gridY);
            return nullptr;
        }
        
        return mapArray[mapID]->cellArray[thisGrid->gridX][thisGrid->gridY]; //返回该位置cell(mapCell_t格式）
    }

    bool findPointGridInMap(grid_t *gridOut, PointType *point){
        // Check for null pointers
        if (!gridOut || !point) {
            RCLCPP_ERROR(get_logger(), "Null pointer passed to findPointGridInMap");
            return false;
        }
        
        // Calculate the cube index that this point belongs to. (Array dimension: mapArrayLength * mapArrayLength)
        grid_t thisGrid;
        getPointCubeIndex(&thisGrid.cubeX, &thisGrid.cubeY, point);
        // Decide whether a point is out of pre-allocated map
        if (thisGrid.cubeX >= 0 && thisGrid.cubeX < mapArrayLength &&
            thisGrid.cubeY >= 0 && thisGrid.cubeY < mapArrayLength){
            // Point is in the boundary, but this sub-map is not allocated before
            // Allocate new memory for this sub-map and save it to mapArray
            if (mapArrayInd[thisGrid.cubeX][thisGrid.cubeY] == -1){
                childMap_t *thisChildMap = new childMap_t(mapArrayCount, thisGrid.cubeX, thisGrid.cubeY);
                mapArray.push_back(thisChildMap);
                mapArrayInd[thisGrid.cubeX][thisGrid.cubeY] = mapArrayCount;
                ++mapArrayCount;
            }
        }else{
            // Point is out of pre-allocated boundary, report error (you should increase map size)
            RCLCPP_ERROR(get_logger(), "Point cloud is out of elevation map boundary. Change params ->mapArrayLength<-. The program will crash!");
            return false;
        }
        // sub-map id
        thisGrid.mapID = mapArrayInd[thisGrid.cubeX][thisGrid.cubeY];
        
        // Check if mapArray index is valid
        if (thisGrid.mapID < 0 || thisGrid.mapID >= static_cast<int>(mapArray.size())) {
            RCLCPP_ERROR(get_logger(), "Invalid mapArray index: %d", thisGrid.mapID);
            return false;
        }
        
        // Check if mapArray element is valid
        if (!mapArray[thisGrid.mapID]) {
            RCLCPP_ERROR(get_logger(), "Null pointer in mapArray at index: %d", thisGrid.mapID);
            return false;
        }
        
        // Find the index for this point in this sub-map (grid index)
        thisGrid.gridX = static_cast<int>((point->x - mapArray[thisGrid.mapID]->originX) / mapResolution); // Cast to int
        thisGrid.gridY = static_cast<int>((point->y - mapArray[thisGrid.mapID]->originY) / mapResolution); // Cast to int
        if (thisGrid.gridX < 0 || thisGrid.gridY < 0 || thisGrid.gridX >= mapCubeArrayLength || thisGrid.gridY >= mapCubeArrayLength)
            return false;

        *gridOut = thisGrid;
        return true;
    }

    void getPointCubeIndex(int *cubeX, int *cubeY, PointType *point){
        *cubeX = static_cast<int>((point->x + mapCubeLength/2.0f) / mapCubeLength) + rootCubeIndex; // Add f suffix
        *cubeY = static_cast<int>((point->y + mapCubeLength/2.0f) / mapCubeLength) + rootCubeIndex; // Add f suffix

        if (point->x + mapCubeLength/2.0f < 0)  --*cubeX; // Add f suffix
        if (point->y + mapCubeLength/2.0f < 0)  --*cubeY; // Add f suffix
    }


    void StateBGK(){
        // (Commented out original code, as it's not a direct conversion for ROS2 and might require a significant re-evaluation of its purpose and dependencies)
    }

    void dist(const Eigen::MatrixXf &xStar, const Eigen::MatrixXf &xTrain, Eigen::MatrixXf &d) const {
        d = Eigen::MatrixXf::Zero(xStar.rows(), xTrain.rows());
        for (int i = 0; i < xStar.rows(); ++i) {
            d.row(i) = (xTrain.rowwise() - xStar.row(i)).rowwise().norm();
        }
    }

    void covSparse(const Eigen::MatrixXf &xStar, const Eigen::MatrixXf &xTrain, Eigen::MatrixXf &Kxz) const {
        dist(xStar/(predictionKernalSize+0.1f), xTrain/(predictionKernalSize+0.1f), Kxz); // Add f suffix

        Kxz = (((2.0f + (Kxz * 2.0f * 3.1415926f).array().cos()) * (1.0f - Kxz.array()) / 3.0f) + // Add f suffix
               (Kxz * 2.0f * 3.1415926f).array().sin() / (2.0f * 3.1415926f)).matrix() * 1.0f;    // Add f suffix
        // Clean up for values with distance outside length scale, possible because Kxz <= 0 when dist >= predictionKernalSize

        // RCLCPP_WARN(get_logger(), "xStar size = %d, xTrain size = %d",xStar.rows(),xTrain.rows());
        // RCLCPP_WARN(get_logger(), "Kxz rows = %d, Kxz cols = %d",Kxz.rows(),Kxz.cols());
        // std::cout<<"xTrain.rowwise()  = " << xTrain.rowwise() <<std::endl;
        for (int i = 0; i < Kxz.rows(); ++i)
            for (int j = 0; j < Kxz.cols(); ++j)
                if (Kxz(i,j) < 0.0f) Kxz(i,j) = 0.0f; // Use float literal
    }

    bool shouldPublishVisualization(){
        if (visualization_hz_ <= 0.0){
            return true;
        }

        const auto now = this->get_clock()->now();
        const int64_t now_ns = now.nanoseconds();
        const int64_t period_ns = static_cast<int64_t>(1.0e9 / visualization_hz_);
        if (last_visualization_pub_time_ns_ == 0 ||
            now_ns - last_visualization_pub_time_ns_ >= period_ns){
            last_visualization_pub_time_ns_ = now_ns;
            return true;
        }
        return false;
    }

    bool hasDebugCloudSubscribers() const {
        return pubElevationCloud->get_subscription_count() != 0 ||
               pubCostCloud->get_subscription_count() != 0;
    }

    void publishMap(bool publish_debug){
        if (local_publish_every_scan_ || publish_debug){
            publishLocalMap(); //发布给局部特征
        }

        if (!publish_debug){
            return;
        }

        publishTraversabilityMap(); //可视化高层图
        publishCostMap();  //发布代价点云

        if (publish_global_debug_){
            publishOccupancyGlobalMap();    //全局代价地图，占据栅格格式
            publishOccupancyElevationGlobalMap();  //全局地图特征
        }
    }

    void publishOccupancyGlobalMap(){

        if (pubOccupancyMapGlobal->get_subscription_count() == 0) // Check for subscribers
            return;
        pubOccupancyMapGlobal->publish(msgElevationGlobal.occupancy);
    }

    void publishOccupancyElevationGlobalMap(){

        if (pubMsgGlobal->get_subscription_count() == 0) // Check for subscribers
            return;
        pubMsgGlobal->publish(msgElevationGlobal);
    }

    //初始化占据栅格并更新
    void localMapProcess(){

        // 1.3 Initialize local occupancy grid map to unknown, height to -FLT_MAX
        std::fill(msgLocalHeight.occupancy.data.begin(), msgLocalHeight.occupancy.data.end(), -1);
        std::fill(msgLocalHeight.height.begin(), msgLocalHeight.height.end(), -FLT_MAX);
        std::fill(msgLocalHeight.cost_map.begin(), msgLocalHeight.cost_map.end(), 0); // 使用 cost_map
        std::fill(msgLocalHeight.roughness.begin(), msgLocalHeight.roughness.end(), -FLT_MAX);
        std::fill(msgLocalHeight.reward_cost.begin(), msgLocalHeight.reward_cost.end(), unknown_cost_);  // 使用 reward_cost //不要设置FLT_MAX，否则会导致MPL第一次规划不出来

        // local map origin x and y
        localMapOriginPoint.x = robotPoint.x - localMapLength / 2.0f; // Add f suffix
        localMapOriginPoint.y = robotPoint.y - localMapLength / 2.0f; // Add f suffix
        localMapOriginPoint.z = robotPoint.z;
        // local map origin cube id (in global map)
        localMapOriginGrid.cubeX = static_cast<int>((localMapOriginPoint.x + mapCubeLength/2.0f) / mapCubeLength) + rootCubeIndex; // Cast to int, add f suffix
        localMapOriginGrid.cubeY = static_cast<int>((localMapOriginPoint.y + mapCubeLength/2.0f) / mapCubeLength) + rootCubeIndex; // Cast to int, add f suffix
        if (localMapOriginPoint.x + mapCubeLength/2.0f < 0)  --localMapOriginGrid.cubeX; // Add f suffix
        if (localMapOriginPoint.y + mapCubeLength/2.0f < 0)  --localMapOriginGrid.cubeY; // Add f suffix
        // local map origin grid id (in sub-map)
        float originCubeOriginX, originCubeOriginY; // the orign of submap that the local map origin belongs to (note the submap may not be created yet, cannot use originX and originY)
        originCubeOriginX = (localMapOriginGrid.cubeX - rootCubeIndex) * mapCubeLength - mapCubeLength/2.0f; // Add f suffix
        originCubeOriginY = (localMapOriginGrid.cubeY - rootCubeIndex) * mapCubeLength - mapCubeLength/2.0f; // Add f suffix
        localMapOriginGrid.gridX = static_cast<int>((localMapOriginPoint.x - originCubeOriginX) / mapResolution); // Cast to int
        localMapOriginGrid.gridY = static_cast<int>((localMapOriginPoint.y - originCubeOriginY) / mapResolution); // Cast to int

        // 2 Calculate local occupancy grid map root position
        msgLocalHeight.header.stamp = this->get_clock()->now();
        msgLocalHeight.occupancy.header.stamp = msgLocalHeight.header.stamp;
        msgLocalHeight.occupancy.info.origin.position.x = localMapOriginPoint.x;
        msgLocalHeight.occupancy.info.origin.position.y = localMapOriginPoint.y;
        msgLocalHeight.occupancy.info.origin.position.z = localMapOriginPoint.z +0.2f; // add 10, just for visualization, add f suffix

        msgElevationGlobal.header.stamp = msgLocalHeight.header.stamp;
        msgElevationGlobal.occupancy.header.stamp = msgLocalHeight.header.stamp;

        // extract all info
        for (int i = 0; i < localMapArrayLength; ++i){
            for (int j = 0; j < localMapArrayLength; ++j){

                int indX = localMapOriginGrid.gridX + i;
                int indY = localMapOriginGrid.gridY + j;

                grid_t thisGrid;

                thisGrid.cubeX = localMapOriginGrid.cubeX + indX / mapCubeArrayLength;
                thisGrid.cubeY = localMapOriginGrid.cubeY + indY / mapCubeArrayLength;

                thisGrid.gridX = indX % mapCubeArrayLength;
                thisGrid.gridY = indY % mapCubeArrayLength;

                // int gx = thisGrid.cubeX * (int)mapInvResolution + thisGrid.gridX;
                // int gy = thisGrid.cubeY * (int)mapInvResolution + thisGrid.gridY; //mapInvResolution 是最小grid  resolution

                int gx = thisGrid.cubeX * mapCubeArrayLength + thisGrid.gridX;
                int gy = thisGrid.cubeY * mapCubeArrayLength + thisGrid.gridY;

                // if sub-map is not created yet
                if (mapArrayInd[thisGrid.cubeX][thisGrid.cubeY] == -1) {
                    continue;
                }

                
                mapCell_t *thisCell = grid2Cell(&thisGrid);    //存储着各种信息

                // skip unknown grid
                if (thisCell && thisCell->elevation != -FLT_MAX){
                    int index = i + j * localMapArrayLength; // index of the 1-D array
                    int globalIndex = gx + gy * (mapLength * static_cast<int>(mapInvResolution)); // Cast for multiplication
                    // RCLCPP_WARN(get_logger(), "gridX= %d, gridY= %d",  thisGrid.gridX , thisGrid.gridY);
                    if (globalIndex < 0 || globalIndex >= msgElevationGlobal.occupancy.data.size()) {
                        RCLCPP_ERROR(get_logger(), "globalIndex %d out of bounds (size: %zu)", 
                                    globalIndex, msgElevationGlobal.occupancy.data.size());
                        continue;
                    }
                    
                    msgLocalHeight.height[index] = thisCell->elevation;
                    if(filterflag){
                        if(thisCell->occupancy > 99 && thisCell->roughness<0.3f)    //借助直接从点云提取的信息，赋给roughness // Add f suffix
                            msgLocalHeight.roughness[index] = 1.0f; // Use float literal
                        else
                            msgLocalHeight.roughness[index] = thisCell->roughness;}
                    else
                        msgLocalHeight.roughness[index] = thisCell->roughness;

                    msgElevationGlobal.height[globalIndex] = msgLocalHeight.height[index];
                    msgElevationGlobal.roughness[globalIndex] = msgLocalHeight.roughness[index];
                    msgElevationGlobal.occupancy.data[globalIndex] = static_cast<int8_t>(thisCell->costInflate); // Cast to int8_t for occupancy data
                    // msgElevationGlobal.occupancy.data[globalIndex] = thisCell->occupancy > 80 ? 100 : 0; //zle sim

                    //terrain_1 max:1.7 min:-1.7    box: 地面-0.8  障碍物：0
                    // RCLCPP_WARN(get_logger(), "cost elevation:%f,roughness:%f,all:%d",thisCell->elevation ,thisCell->roughness,msgElevationGlobal.occupancy.data[globalIndex]);
                }
            }
        }
    }

    void publishLocalMap(){
        if (pubMsgLocalHeight->get_subscription_count() == 0)
            return;

        pubMsgLocalHeight->publish(msgLocalHeight);
    }

    void initializeLocalOccupancyMap(){

        // initialization of customized map message
        msgLocalHeight.header.frame_id = "map";
        msgLocalHeight.occupancy.info.width = localMapArrayLength;
        msgLocalHeight.occupancy.info.height = localMapArrayLength;
        msgLocalHeight.occupancy.info.resolution = mapResolution;

        msgLocalHeight.occupancy.info.origin.orientation.x = 0.0;
        msgLocalHeight.occupancy.info.origin.orientation.y = 0.0;
        msgLocalHeight.occupancy.info.origin.orientation.z = 0.0;
        msgLocalHeight.occupancy.info.origin.orientation.w = 1.0;

        msgLocalHeight.occupancy.data.resize(msgLocalHeight.occupancy.info.width * msgLocalHeight.occupancy.info.height);
        msgLocalHeight.height.resize(msgLocalHeight.occupancy.info.width * msgLocalHeight.occupancy.info.height);
        msgLocalHeight.reward_cost.resize(msgLocalHeight.occupancy.info.width * msgLocalHeight.occupancy.info.height); // reward_cost
        msgLocalHeight.cost_map.resize(msgLocalHeight.occupancy.info.width * msgLocalHeight.occupancy.info.height); // cost_map
        msgLocalHeight.roughness.resize(msgLocalHeight.occupancy.info.width * msgLocalHeight.occupancy.info.height);

    }

    void initGlobalOccuEle_global(){
        msgElevationGlobal.header.frame_id = "map";
        msgElevationGlobal.occupancy.info.width = mapLength * static_cast<int>(mapInvResolution); // Cast
        msgElevationGlobal.occupancy.info.height = mapLength * static_cast<int>(mapInvResolution); // Cast
        msgElevationGlobal.occupancy.info.resolution = mapResolution;

        msgElevationGlobal.occupancy.info.origin.orientation.x = 0.0;
        msgElevationGlobal.occupancy.info.origin.orientation.y = 0.0;
        msgElevationGlobal.occupancy.info.origin.orientation.z = 0.0;
        msgElevationGlobal.occupancy.info.origin.orientation.w = 1.0;

        msgElevationGlobal.occupancy.info.origin.position.x = -0.5f - rootCubeIndex; // Add f suffix
        msgElevationGlobal.occupancy.info.origin.position.y = -0.5f - rootCubeIndex; // Add f suffix
        msgElevationGlobal.occupancy.info.origin.position.z = -0.5f; // Add f suffix

        msgElevationGlobal.occupancy.data.resize(msgElevationGlobal.occupancy.info.width * msgElevationGlobal.occupancy.info.height);
        msgElevationGlobal.height.resize(msgElevationGlobal.occupancy.info.width * msgElevationGlobal.occupancy.info.height);
        msgElevationGlobal.roughness.resize(msgElevationGlobal.occupancy.info.width * msgElevationGlobal.occupancy.info.height);
        msgElevationGlobal.cost_map.resize(msgElevationGlobal.occupancy.info.width * msgElevationGlobal.occupancy.info.height); // cost_map
        std::fill(msgElevationGlobal.occupancy.data.begin(), msgElevationGlobal.occupancy.data.end(), static_cast<int8_t>(unknown_cost_)); // Cast to int8_t
        std::fill(msgElevationGlobal.height.begin(), msgElevationGlobal.height.end(), -FLT_MAX);
        std::fill(msgElevationGlobal.roughness.begin(), msgElevationGlobal.roughness.end(), -FLT_MAX);
        std::fill(msgElevationGlobal.cost_map.begin(), msgElevationGlobal.cost_map.end(), 0); // cost_map

    }

    bool getRobotPosition(){
        try{
            transform = tf_buffer_->lookupTransform("map","base_link", tf2::TimePointZero);
        }
        catch (tf2::TransformException &ex){
            RCLCPP_ERROR(get_logger(), "Transform Failure2: %s", ex.what());
            return false;
        }

        robotPoint.x = transform.transform.translation.x;
        robotPoint.y = transform.transform.translation.y;
        robotPoint.z = transform.transform.translation.z;
        // RCLCPP_WARN(get_logger(), "z = %f",robotPoint.z);
        return true;
    }

    float calcElevationRoughness(const PointType *centerPoint,float &hVar,int  &obsCount){
        float roughness = 0.0f; // Use float literal
        hVar = 1000.0f; // Use float literal
        obsCount = 1;
        // set a box region
        BoxPointType boxpoint;
        generateBox(boxpoint,centerPoint, {0.3f,0.3f,2.0f} ); //半径0.3
        PointVector searchPoints;
        kdtree_ptr->Box_Search(boxpoint,searchPoints);
        // std::vector<float> ps;
        size_t pointSize = searchPoints.size(); // Use size_t for vector size
        int count = 0;
        float z_sum = 0.0f; // Use float literal
        float minh = FLT_MAX;
        float maxh = -FLT_MAX;
        // float dCrit = 0.424f; // This variable is declared but not used. // Add f suffix
        int minmaxCunt = 0; // Corrected typo: minmaxCount

        for(size_t i = 0 ;i < pointSize; i++){ // Use size_t for loop counter
            if(std::isnan(searchPoints[i].z)) continue; // Use std::isnan

            float diffh = searchPoints[i].z-centerPoint->z;    //点与中心点的高度差
            float h = std::fabs(diffh); // Use std::fabs
            float diffx = searchPoints[i].x-centerPoint->x;
            float diffy = searchPoints[i].y-centerPoint->y;
            float dist = sqrt(diffx*diffx+diffy*diffy);

            if(dist <= 0.25f){ // Add f suffix
                minh = minh>searchPoints[i].z ? searchPoints[i].z: minh;
                maxh = maxh<searchPoints[i].z ? searchPoints[i].z: maxh;
            }
            if(h >= 0.03f){ // Add f suffix
                minmaxCunt ++;
            }
            z_sum+= h;
            count++;
        }

        if(count< 2)
            return roughness;
        else{
            float means = z_sum / static_cast<float>(count); // Cast to float
            hVar= maxh - minh;
            obsCount = count;
            // roughness =  means ;
            float ratio = static_cast<float>(minmaxCunt) / static_cast<float>(count); // Cast to float
            if(minh != FLT_MAX && maxh != -FLT_MAX && maxh > minh)
                roughness =  (1.0f-ratio)*means + ratio*(maxh - minh); // Use f suffix for float literals
            // roughness =  means + ratio*(maxh - minh);
            else
                roughness =  means ;
            //  roughness = means * (maxh - minh);
            return roughness;
        }

    }
    //获取机器人周围30cube（30m）范围内的点云,并都赋值给laserCloudElevation
    void elevationMapProcess(){
        // 1. Find robot current cube index
        int currentCubeX, currentCubeY;        //机器人的cube位置，每个cube：1x1m，所以全局地图变为200x200个cube
        // pcl::PointCloud<PointType>::Ptr localElevation;
        // localElevation.reset(new pcl::PointCloud<PointType>());
        getPointCubeIndex(&currentCubeX, &currentCubeY, &robotPoint);
        // 2. Loop through all the sub-maps that are nearby    mapCubelength:submap地图长度（1m）
        int visualLength = static_cast<int>(100.0f / mapCubeLength); // Cast to int, add f suffix
        for (int i = -visualLength; i <= visualLength; ++i){
            for (int j = -visualLength; j <= visualLength; ++j){

                if (sqrt(static_cast<float>(i*i+j*j)) >= visualLength) continue; // Use static_cast for float conversion

                int idx = i + currentCubeX;
                int idy = j + currentCubeY;

                if (idx < 0 || idx >= mapArrayLength ||  idy < 0 || idy >= mapArrayLength) continue;

                if (mapArrayInd[idx][idy] == -1) continue;
                *laserCloudElevation += mapArray[mapArrayInd[idx][idy]]->cloud; // PCL += operator for smart pointers works on dereferenced objects
                *laserCloudCost += mapArray[mapArrayInd[idx][idy]]->cloudCost; // PCL += operator for smart pointers works on dereferenced objects
            }
        }

        // kdtree_ptr->Build((*laserCloudElevation).points);

    }

    void publishTraversabilityMap(){

        if (pubElevationCloud->get_subscription_count() == 0)
        {
            laserCloudElevation->clear();
            return;
        }

        // Publish elevation point cloud
        sensor_msgs::msg::PointCloud2 laserCloudTemp;
        pcl::toROSMsg(*laserCloudElevation, laserCloudTemp);
        laserCloudTemp.header.frame_id = "map";
        laserCloudTemp.header.stamp = this->get_clock()->now();
        pubElevationCloud->publish(laserCloudTemp);
        // free memory
        laserCloudElevation->clear();
    }

    void publishCostMap(){

        if (pubCostCloud->get_subscription_count() == 0)
        {
            laserCloudCost->clear();
            return;
        }
        // 修改 z 值为 0
        for (auto& point : laserCloudCost->points) {
            point.z = 0.0f; // Use float literal
        }

        sensor_msgs::msg::PointCloud2 laserCloudTemp2;
        pcl::toROSMsg(*laserCloudCost, laserCloudTemp2);
        laserCloudTemp2.header.frame_id = "map";
        laserCloudTemp2.header.stamp = this->get_clock()->now();
        pubCostCloud->publish(laserCloudTemp2);

        // free memory
        laserCloudCost->clear();
    }

};




int main(int argc, char** argv){

    rclcpp::init(argc, argv);

    // 使用 std::make_shared 创建节点实例，这是 ROS 2 的推荐方式
    // 它确保了节点被 shared_ptr 管理，从而使得 shared_from_this() 可以安全使用
    auto tMapping = std::make_shared<TraversabilityMapping>();

    rclcpp::spin(tMapping); // 让节点持续运行，处理回调

    rclcpp::shutdown(); // 清理ROS资源

    return 0;
}
