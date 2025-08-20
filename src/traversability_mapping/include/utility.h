#ifndef _UTILITY_TM_H_ROS2
#define _UTILITY_TM_H_ROS2

// ROS 2 核心库
#include <rclcpp/rclcpp.hpp>

// ROS 2 标准消息头文件 (注意路径变化)
#include <std_msgs/msg/header.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <nav_msgs/msg/path.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <geometry_msgs/msg/pose_array.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>

// ROS 2 交互标记 (interactive_markers)
#include <interactive_markers/interactive_marker_server.hpp> // ROS 2版本通常是 interactive_markers/interactive_marker_server.hpp

// Navigation 2 (Nav2) 核心插件接口
// 注意：nav_core/base_global_planner.h 在 ROS 2 中不再直接存在
// 如果是 Nav2 插件，你需要包含 Nav2 的插件接口，例如：
// #include <nav2_core/global_planner.hpp>
// #include <nav2_core/controller.hpp>
// # costmap_2d 在 ROS 2 中通常是 nav2_costmap_2d
// #include <nav2_costmap_2d/costmap_2d_ros.hpp>

// Eigen 库
#include <Eigen/Core>
#include <Eigen/SVD>
#include <vector> 
using std::vector; 
// OpenCV 库 (ROS 2 中通常直接使用 OpenCV 库，不再通过 cv_bridge/cv_bridge.h 来包含核心部分)
#include <opencv2/core.hpp> // 替换 opencv2/core/core.hpp
#include <opencv2/core/eigen.hpp> // 保持不变
#include <opencv2/imgproc.hpp> // 例如，如果需要图像处理
#include <opencv2/highgui.hpp> // 例如，如果需要图像显示
#include <opencv2/opencv.hpp> // 这个包含可能覆盖上面的，但具体取决于你的 OpenCV 安装

// cv_bridge 和 image_transport (ROS 2 版本)
#include <cv_bridge/cv_bridge.h> // ROS 2 版本头文件路径不变
#include <image_transport/image_transport.hpp> // ROS 2 版本头文件路径改变

// PCL 库
#include <pcl/common/common.h>
#include <pcl/point_types.h>
// #include <pcl_ros/point_cloud.h> // ROS 2 中不再直接使用这个，PCL 点云和 ROS 消息转换通过 pcl_conversions
#include <pcl_conversions/pcl_conversions.h> // ROS 2 版本头文件路径不变
#include <pcl/range_image/range_image.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/io/pcd_io.h>
#include <pcl_ros/transforms.hpp>  
// TF2 库 (ROS 2 替代 tf)
#include <tf2_eigen/tf2_eigen.hpp>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp> // 用于几何消息的转换
#include "tf2_ros/transform_listener.h" // 包含 TransformListener 和 Buffer 的定义
#include "tf2_ros/buffer.h"             // 直接包含 tf2_ros::Buffer 的定义（推荐）
// #include <pcl_ros/transforms.h> // ROS 2 中不再直接使用这个，而是 tf2 相关的 pcl 转换函数

// C++ 标准库
#include <vector>
#include <cmath>
#include <algorithm>
#include <queue>
#include <iostream>
#include <fstream>
#include <chrono>   // 替代 ctime，ROS 2 推荐使用 std::chrono
#include <cfloat>
#include <iterator>
#include <sstream>
#include <string>
#include <array>    // C++11
#include <thread>   // C++11
#include <mutex>    // C++11
#include <limits>   // 用于 std::numeric_limits

// 自定义消息 (假设已迁移到 ROS 2，并放在对应的消息包中)
#include <visualization_msgs/msg/marker.hpp>      // 替换 marker/Marker.h
#include <visualization_msgs/msg/marker_array.hpp>// 替换 marker/MarkerArray.h// 替换 elevation_msgs/OccupancyElevation.h

// 自定义结构体和函数 (保持不变，但路径可能需要调整)
// #include "planner/kdtree.h"
// #include "planner/cubic_spline_interpolator.h"
// #include  "planner/raycast.h"
// 这些路径是相对的，取决于你的实际文件结构，可能需要调整为 <package_name>/include/<sub_dir>/<file.h>
// 例如：
#include "planner/kdtree.h"
#include "planner/cubic_spline_interpolator.h"
#include "planner/raycast.h"
#include "elevation_msgs/msg/occupancy_elevation.hpp"

// using namespace std; // 在头文件中不推荐使用 using namespace std; 污染全局命名空间

// 类型定义
typedef pcl::PointXYZI PointType;
// kdtree_t 和 kdres_t 的定义应在 kdtree.h 中提供，这里只是声明
typedef struct kdtree kdtree_t;
typedef struct kdres kdres_t;

// Environment
extern const bool urbanMapping = true;

// VLP-16
extern const int N_SCAN = 32;   //线数
extern const int Horizon_SCAN = 1800;   //线上点数

// Map Params
extern const float mapResolution = 0.1; // map resolution
extern const float mapInvResolution = 1.0/mapResolution; // map inverse resolution
extern const float mapCubeLength = 1; // the length of a sub-map (meters)   cube
extern const int mapCubeArrayLength = mapCubeLength / mapResolution; // 10 the grid dimension of a sub-map (mapCubeLength / mapResolution) 一个cube中每行grid个数,1x1m中有100个grid
extern const int mapLength = 200; //m       200m
extern const int mapArrayLength = mapLength / mapCubeLength; // the sub-map dimension of global map (2000m x 2000m)   200x200
extern const int rootCubeIndex = mapArrayLength / 2; // by default, robot is at the center of global map at the beginning
extern const int rootGridIndex = rootCubeIndex*(int)mapInvResolution;

// Filter Ring Params
extern const int scanNumCurbFilter = 15;
extern const int scanNumSlopeFilter = 15;
extern const int scanNumMax = 15;

// Filter Threshold Params
extern const float sensorRangeLimit = 12; // only keep points with in ...  只考虑10m以内点
extern const int filterHeightMapArrayLength = sensorRangeLimit*2 / mapResolution;

// BGK Prediction Params
extern const bool predictionEnableFlag = true;
extern const float predictionKernalSize = 0.2; // predict elevation within x meters 0.2 训练数据来源的范围  0.4

// Occupancy Params
extern const float p_occupied_when_laser = 0.9;
extern const float p_occupied_when_no_laser = 0.2;
extern const float large_log_odds = 100;
extern const float max_log_odds_for_belief = 20;

// 2D Map Publish Params
extern const int localMapLength = 18; // length of the local occupancy grid map (meter)
extern const int localMapArrayLength = localMapLength / mapResolution;

// Visualization Params
extern const float visualizationRadius = (float)localMapLength;
extern const float visualizationFrequency = 2; // n, skip n scans then publish, n=0, visualize at each scan

// Robot Params
extern const float robotRadius = 0.4;
extern const float sensorHeight = 0.5;

// Traversability Params
extern const int traversabilityObserveTimeTh = 10;
extern const float traversabilityCalculatingDistance = 8.0;

// Planning Cost Params
extern const int NUM_COSTS = 3;
// c++11 initializer list can be used directly for std::vector
extern const std::vector<int> costHierarchy = {2};

// PRM Planner Settings
extern const bool planningUnknown = true;
extern const float costmapInflationRadius = 0.2; // 膨胀大小，按车子的体积来算
extern const float neighborSampleRadius  = 0.5;
extern const float neighborConnectHeight = 1.0;
extern const float neighborConnectRadius = 2.0;
extern const float neighborSearchRadius = localMapLength / 2;

// 前向声明结构体 (通常在头文件顶部或使用前)
struct grid_t;
struct mapCell_t;
struct childMap_t;
struct state_t;
struct neighbor_t;


/*
    This struct is used to send map from mapping package to prm package
    定义了每个sub_map在map中的索引，每个cell在sub_map中的索引
    */
struct grid_t{
    int mapID;  //submap（1x1m）在全局地图中的一维索引
    int cubeX;  //submap在全局地图中的二维索引,单位是1m
    int cubeY;
    int gridX;  //cell（0.1x0.1m）在submap中的索引
    int gridY;
    int gridIndex;
};

/*
    Cell Definition:
    a cell is a member of a grid in a sub-map
    a grid can have several cells in it.
    a cell represent one height information
    */

struct mapCell_t{

    PointType *xyz; // it's a pointer to the corresponding point in the point cloud of submap
    PointType *xyzc;    //x,y,z,cost,zle
    grid_t grid;

    float log_odds;

    int observeTimes;

    float roughness,roughnessVar;

    float occupancy, occupancyVar;
    float elevation, elevationVar;  //var:机器人与每个雷达点云的距离
    float cost, costInflate;

    mapCell_t(){
        log_odds = 0.5;
        observeTimes = 0;

        elevation = -std::numeric_limits<float>::max(); // 使用 std::numeric_limits<float>::max() 替代 FLT_MAX
        elevationVar = 1e3;

        occupancy = 0; // initialized as unkown
        occupancyVar = 1e3;

        roughness = -std::numeric_limits<float>::max();
        roughnessVar = 1e3;
        cost = 0;      //每个网格的通行代价
        costInflate = 0;
    }

    void updatePoint(){
        xyz->z = elevation;
        xyz->intensity = occupancy;
    }
    void updateElevation(float elevIn, float varIn){
        elevation = elevIn;
        elevationVar = varIn;
        updatePoint();
    }
    void updateOccupancy(float occupIn){
        occupancy = occupIn;
        updatePoint();
    }

    void updateRoughness(float rIn, float varIn){
        roughness = rIn;
        roughnessVar = varIn;
    }
    //用于更新每个cell的代价值
    void updatePointCost(){
        xyzc->z = elevation;
        xyzc->intensity = cost;
    }
    void updateCost(float costln, float costFlate){
        cost = costln;
        costInflate = costFlate;
        updatePointCost();
    }

};


/*
    Sub-map Definition:sub_map定义，submap是二维的正方形网格，由很多cell构成
    childMap_t is a small square. We call it "cellArray".
    It composes the whole map
    */
struct childMap_t{

    std::vector<std::vector<mapCell_t*> > cellArray; // 使用 std::vector
    int subInd; //sub-map's index in 1d mapArray
    int indX; // sub-map's x index in 2d array mapArrayInd
    int indY; // sub-map's y index in 2d array mapArrayInd
    float originX; // sub-map's x root coordinate
    float originY; // sub-map's y root coordinate
    pcl::PointCloud<PointType> cloud;
    pcl::PointCloud<PointType> cloudCost;

    childMap_t(int id, int indx, int indy){

        subInd = id;
        indX = indx;
        indY = indy;
        originX = (indX - rootCubeIndex) * mapCubeLength - mapCubeLength/2.0;
        originY = (indY - rootCubeIndex) * mapCubeLength - mapCubeLength/2.0;

        // allocate and initialize each cell
        cellArray.resize(mapCubeArrayLength);
        for (int i = 0; i < mapCubeArrayLength; ++i)
            cellArray[i].resize(mapCubeArrayLength);

        for (int i = 0; i < mapCubeArrayLength; ++i)
            for (int j = 0; j < mapCubeArrayLength; ++j)
                cellArray[i][j] = new mapCell_t;
        // allocate point cloud for visualization
        cloud.points.resize(mapCubeArrayLength*mapCubeArrayLength); //100
        cloudCost.points.resize(mapCubeArrayLength*mapCubeArrayLength);

        // initialize cell pointer to cloud point
        for (int i = 0; i < mapCubeArrayLength; ++i)
            for (int j = 0; j < mapCubeArrayLength; ++j){
                cellArray[i][j]->xyz = &cloud.points[i + j*mapCubeArrayLength];
                cellArray[i][j]->xyzc = &cloudCost.points[i + j*mapCubeArrayLength];
            }
        // initialize each point in the point cloud, also each cell
        for (int i = 0; i < mapCubeArrayLength; ++i){
            for (int j = 0; j < mapCubeArrayLength; ++j){

                // point cloud initialization
                int index = i + j * mapCubeArrayLength;
                cloud.points[index].x = originX + i * mapResolution;
                cloud.points[index].y = originY + j * mapResolution;
                cloud.points[index].z = std::numeric_limits<float>::quiet_NaN();
                cloud.points[index].intensity = cellArray[i][j]->occupancy;

                // //初始化代价点云
                cloudCost.points[index].x = originX + i * mapResolution;
                cloudCost.points[index].y = originY + j * mapResolution;
                cloudCost.points[index].z = std::numeric_limits<float>::quiet_NaN();
                cloudCost.points[index].intensity = cellArray[i][j]->cost;

                // cell position in the array of submap
                cellArray[i][j]->grid.mapID = subInd;
                cellArray[i][j]->grid.cubeX = indX;
                cellArray[i][j]->grid.cubeY = indY;
                cellArray[i][j]->grid.gridX = i;
                cellArray[i][j]->grid.gridY = j;
                cellArray[i][j]->grid.gridIndex = index;
            }
        }
    }
};



/*
    Robot State Defination
    */


struct state_t{
    double x[3]; //  1 - x, 2 - y, 3 - z
    float theta;
    int stateId;
    float cost;
    float obscost;
    bool validFlag;
    // # Cost types
    // # 0. obstacle cost
    // # 1. elevation cost
    // # 2. distance cost
    float costsToRoot[NUM_COSTS];
    float costsToParent[NUM_COSTS]; // used in RRT*
    float costsToGo[NUM_COSTS];

    state_t* parentState; // parent for this state in PRM and RRT*
    std::vector<neighbor_t> neighborList; // PRM adjencency list with edge costs
    std::vector<state_t*> childList; // RRT*

    // default initialization
    state_t(){
        parentState = nullptr; // 替代 NULL
        for (int i = 0; i < NUM_COSTS; ++i){
            costsToRoot[i] = std::numeric_limits<float>::max(); // 替代 FLT_MAX
            costsToParent[i] = std::numeric_limits<float>::max();
            costsToGo[i] = std::numeric_limits<float>::max();
        }
    }
    // use a state input to initialize new state

    state_t(state_t* stateIn){
        // pose initialization
        for (int i = 0; i < 3; ++i)
            x[i] = stateIn->x[i];
        theta = stateIn->theta;
        // regular initialization
        parentState = nullptr;
        for (int i = 0; i < NUM_COSTS; ++i){
            costsToRoot[i] = std::numeric_limits<float>::max();
            costsToParent[i] = stateIn->costsToParent[i];
        }
    }
};


struct neighbor_t{
    state_t* neighbor;
    float edgeCosts[NUM_COSTS]; // the cost from this state to neighbor
    neighbor_t(){
        neighbor = nullptr;
        for (int i = 0; i < NUM_COSTS; ++i)
            edgeCosts[i] = std::numeric_limits<float>::max();
    }
};


////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////      Some Functions    ////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////
// 确保 compareState 在一个可访问的全局或类的静态作用域中声明
// 或者将其作为参数传递
state_t *compareState; // 如果它仍然是全局变量，保持 extern 声明

bool isStateExsiting(neighbor_t neighborIn){
    return neighborIn.neighbor == compareState; // 更简洁的布尔表达式
}

float pointDistance(PointType p1, PointType p2){
    return sqrt(pow(p1.x-p2.x, 2) + pow(p1.y-p2.y, 2) + pow(p1.z-p2.z, 2)); // 使用 pow() 替代 * 操作
}

#endif