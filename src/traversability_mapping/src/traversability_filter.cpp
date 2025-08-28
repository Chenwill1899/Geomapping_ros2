#include "utility.h"
#include <atomic>
#include "traversability_mapping/msg/cloud_info.hpp"
class TraversabilityFilter : public rclcpp::Node 
{ // 继承 rclcpp::Node
    
private:
    // ROS 2 订阅者和发布者
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_cloud_;
    // rclcpp::Subscription<自定义消息类型>::SharedPtr sub_laser_cloud_info_; // 如果有对应的ROS 2自定义消息
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_cloud_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_cloud_visual_hi_res_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_cloud_visual_low_res_;
    // rclcpp::Publisher<sensor_msgs::msg::LaserScan>::SharedPtr pub_laser_scan_; // 如果需要

    // 额外的发布者（如果它们在ROS 2中有对应的话题类型）
    // rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_slope_cloud_;
    // rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_roughness_cloud_;
    // rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_step_cloud_;

    // Point Cloud
    pcl::PointCloud<PointType>::Ptr laserCloudIn;
    pcl::PointCloud<PointType>::Ptr laserCloudOut;
    pcl::PointCloud<PointType>::Ptr laserCloudObstacles;

    // TF2 Listener 和 Buffer
    std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
    geometry_msgs::msg::TransformStamped transform_; // ROS 2 中的 transform 类型

    // A few points
    PointType robotPoint;
    PointType localMapOrigin;

    // point cloud saved as N_SCAN * Horizon_SCAN form
    std::vector<std::vector<PointType>> laserCloudMatrix;

    // Matrice
    cv::Mat obstacleMatrix;
    cv::Mat rangeMatrix;

    // For downsample
    float **minHeight;
    float **maxHeight;
    bool **obstFlag;
    bool **initFlag;
    std::string frameID; // 使用 std::string

    double filterHeightLimit;
    double filterAngleLimit;
    double positiveParam;
    double negativeHeight;
    bool filterflag;

public:
    // 构造函数
    // rclcpp::Node 的构造函数需要传递节点名称
    TraversabilityFilter() : Node("traversability_filter_node") { // "traversability_filter_node" 是您的节点名称
        
        // 获取参数 (ROS 2 使用 declare_parameter 和 get_parameter)
        // 注意：nh.getParam 在 ROS 2 中不再使用
        // 声明参数并获取值，如果没有提供参数，可以使用默认值
        this->declare_parameter<std::string>("frameID", "map");
        this->get_parameter("frameID", frameID);

        this->declare_parameter<double>("/filter/filterHeightLimit", 0.1); // 提供默认值
        this->get_parameter("/filter/filterHeightLimit", filterHeightLimit);

        this->declare_parameter<double>("/filter/filterAngleLimit", 20.0); // 提供默认值
        this->get_parameter("/filter/filterAngleLimit", filterAngleLimit);

        this->declare_parameter<double>("/filter/positiveParam", 0.1); // 提供默认值
        this->get_parameter("/filter/positiveParam", positiveParam);

        this->declare_parameter<double>("/filter/negativeHeight", 0.1); // 之前写错了，这里假设是 negativeHeight
        this->get_parameter("/filter/negativeHeight", negativeHeight);

        this->declare_parameter<bool>("/filter/filterflag", true); // 提供默认值
        this->get_parameter("/filter/filterflag", filterflag);
        std::atomic_flag processing_{ATOMIC_FLAG_INIT};
        // ROS 2 订阅者
        auto sub_qos = rclcpp::SensorDataQoS()
                 .keep_last(1)        // 只留最新
                 .best_effort();      // 不要可靠重传，防止背压
        rclcpp::SubscriptionOptions opts;
        auto cbg = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
        opts.callback_group = cbg;

        sub_cloud_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/syncd_project_cloud", sub_qos,
            std::bind(&TraversabilityFilter::cloudHandler, this, std::placeholders::_1),
            opts);

        // this->create_subscription<消息类型>(话题名称, QoS设置, 回调函数)
        // sub_cloud_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
        //     "/syncd_project_cloud", 
        //     rclcpp::QoS(rclcpp::SystemDefaultsQoS()), // 或 rclcpp::SensorDataQoS()
        //     std::bind(&TraversabilityFilter::cloudHandler, this, std::placeholders::_1));

        // ROS 2 发布者
        // this->create_publisher<消息类型>(话题名称, QoS设置)
        pub_cloud_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/filtered_pointcloud", rclcpp::SystemDefaultsQoS());
        pub_cloud_visual_hi_res_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/filtered_pointcloud_visual_high_res", rclcpp::SystemDefaultsQoS());
        pub_cloud_visual_low_res_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/filtered_pointcloud_visual_low_res", rclcpp::SystemDefaultsQoS());
        // pub_laser_scan_ = this->create_publisher<sensor_msgs::msg::LaserScan>("/pointcloud_2_laserscan", rclcpp::SystemDefaultsQoS());  

        // TF2 Listener 和 Buffer 初始化
        // TF2 的 listener 需要一个 buffer 来存储变换，并且 buffer 最好由 shared_ptr 管理
        tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
        tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

        allocateMemory();
    }

    void allocateMemory(){
        laserCloudIn.reset(new pcl::PointCloud<PointType>());
        laserCloudOut.reset(new pcl::PointCloud<PointType>());
        laserCloudObstacles.reset(new pcl::PointCloud<PointType>());

        // cv::Mat 初始化保持不变
        obstacleMatrix = cv::Mat(N_SCAN, Horizon_SCAN, CV_32S, cv::Scalar::all(-1));
        rangeMatrix =  cv::Mat(N_SCAN, Horizon_SCAN, CV_32F, cv::Scalar::all(-1));

        // std::vector 初始化保持不变
        laserCloudMatrix.resize(N_SCAN);
        for (int i = 0; i < N_SCAN; ++i)
            laserCloudMatrix[i].resize(Horizon_SCAN);

        // 动态二维数组的初始化保持不变
        initFlag = new bool*[filterHeightMapArrayLength];
        for (int i = 0; i < filterHeightMapArrayLength; ++i)
            initFlag[i] = new bool[filterHeightMapArrayLength];

        obstFlag = new bool*[filterHeightMapArrayLength];
        for (int i = 0; i < filterHeightMapArrayLength; ++i)
            obstFlag[i] = new bool[filterHeightMapArrayLength];

        minHeight = new float*[filterHeightMapArrayLength];
        for (int i = 0; i < filterHeightMapArrayLength; ++i)
            minHeight[i] = new float[filterHeightMapArrayLength];

        maxHeight = new float*[filterHeightMapArrayLength];
        for (int i = 0; i < filterHeightMapArrayLength; ++i)
            maxHeight[i] = new float[filterHeightMapArrayLength];

        resetParameters();
    }

    void resetParameters(){
        laserCloudIn->clear();
        laserCloudOut->clear();
        laserCloudObstacles->clear();
        // RCLCPP_WARN(this->get_logger(), "Clear SUCCESS"); // 替换 ROS_WARN

        obstacleMatrix = cv::Mat(N_SCAN, Horizon_SCAN, CV_32S, cv::Scalar::all(-1));
        rangeMatrix =  cv::Mat(N_SCAN, Horizon_SCAN, CV_32F, cv::Scalar::all(-1));
        // RCLCPP_WARN(this->get_logger(), "Matrix SUCCESS"); // 替换 ROS_WARN

        for (int i = 0; i < filterHeightMapArrayLength; ++i){
            for (int j = 0; j < filterHeightMapArrayLength; ++j){
                initFlag[i][j] = false;
                obstFlag[i][j] = false;
            }
        }
    }

    // // ROS 2 中的回调函数签名
    // void cloudHandler(const sensor_msgs::msg::PointCloud2::SharedPtr cloud_msg){
    //     // 回调函数内部的逻辑需要将 ROS 1 的 API 替换为 ROS 2/TF2 的 API
    //     // 例如：tf::TransformListener listener; -> tf2_ros::Buffer/TransformListener
    //     // msg->header.stamp -> cloud_msg->header.stamp
    //     // ROS_INFO -> RCLCPP_INFO
    //     // std_msgs::Header -> std_msgs::msg::Header
    // }

    // 析构函数，需要释放动态分配的内存
    ~TraversabilityFilter(){
        for (int i = 0; i < filterHeightMapArrayLength; ++i) {
            delete[] initFlag[i];
            delete[] obstFlag[i];
            delete[] minHeight[i];
            delete[] maxHeight[i];
        }
        delete[] initFlag;
        delete[] obstFlag;
        delete[] minHeight;
        delete[] maxHeight;
    }




    // void laserCloudInfoHandler(const traversability_mapping::cloud_infoConstPtr &msgIn)
    // {
    //     // extractRawCloud(laserCloudMsg);
    //    extractRawCloud(msgIn);

    //     if (transformCloud() == false) { 
    //         return;
    //     }
        
    //     cloud2Matrix(msgIn);

    //     applyFilter();

    //     extractFilteredCloud();

    //     downsampleCloud();

    //     predictCloudBGK();

    //     publishCloud();

    //     // publishLaserScan();

    //     resetParameters();
    // }

    void cloudHandler(const sensor_msgs::msg::PointCloud2::SharedPtr laserCloudMsg){
        
        extractRawCloud(laserCloudMsg);

        if (!transformCloud()) {
            return;
        }

        cloud2Matrix();

        applyFilter();

        extractFilteredCloud();

        downsampleCloud();

        predictCloudBGK();

        publishCloud();

        // publishLaserScan();

        resetParameters();
        
    }

  void extractRawCloud(const sensor_msgs::msg::PointCloud2::ConstSharedPtr& laserCloudMsg)
{
    // 确保 laserCloudIn 指针已初始化
    if (!laserCloudIn) {
        // 如果未初始化，这里应该根据您的PointType进行初始化
        laserCloudIn.reset(new pcl::PointCloud<PointType>());
    }

    // ROS 消息 -> PCL 点云
    // 注意：pcl::fromROSMsg 接收 const &，所以可以直接传入 *laserCloudMsg
    pcl::fromROSMsg(*laserCloudMsg, *laserCloudIn);

    // 提取范围信息
    for (int i = 0; i < N_SCAN; ++i) {
        for (int j = 0; j < Horizon_SCAN; ++j) {
            int index = j + i * Horizon_SCAN;

            // 确保索引在点云范围内，防止越界访问
            if (index >= laserCloudIn->points.size()) {
                // 如果点云大小不足，跳过或处理错误
                // 可以添加日志警告：RCLCPP_WARN(this->get_logger(), "Point cloud index out of bounds: %d", index);
                continue;
            }

            // 检查 intensity 是否为 NaN
            // `== std::numeric_limits<float>::quiet_NaN()` 这种写法对于NaN是错误的，
            // NaN与任何值（包括它自己）的比较结果都是false。
            // 应该使用 std::isnan() 函数来检查 NaN。
            if (std::isnan(laserCloudIn->points[index].intensity)) {
                continue; // 跳过 NaN 点
            }

            // 保存范围信息
            // 确保 rangeMatrix 的类型和大小匹配
            if (rangeMatrix.rows > i && rangeMatrix.cols > j && rangeMatrix.type() == CV_32F) {
                rangeMatrix.at<float>(i, j) = laserCloudIn->points[index].intensity;
            } else {
                 // 可以在这里添加日志或错误处理，如果矩阵访问越界或类型不匹配
                 // RCLCPP_ERROR(this->get_logger(), "rangeMatrix access error at (%d, %d)", i, j);
            }

            // 重置障碍物状态为 0 - free
            // 确保 obstacleMatrix 的类型和大小匹配
            if (obstacleMatrix.rows > i && obstacleMatrix.cols > j && obstacleMatrix.type() == CV_32S) {
                obstacleMatrix.at<int>(i, j) = 0;
            } else {
                // 可以在这里添加日志或错误处理
                // RCLCPP_ERROR(this->get_logger(), "obstacleMatrix access error at (%d, %d)", i, j);
            }
        }
    }
}
    


// 假设您的类中已经有以下成员变量：
// std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
// // std::shared_ptr<tf2_ros::TransformListener> tf_listener_; // ROS 2 中通常由 Buffer 内部管理，或者在使用时创建
// std::string frameID; // 例如 "map" 或 "odom"
// PointType robotPoint; // 您的自定义点类型
// pcl::PointCloud<PointType>::Ptr laserCloudIn; // 假设 laserCloudIn 已经初始化
// rclcpp::Node::SharedPtr node_ptr_; // 用于获取 logger 和 clock

bool transformCloud()
{
    try {
        // 等价于 ROS1 的 ros::Time(0)：取“最新可用”变换
        // 保留你原代码的 frameID <- base_link 的方向
        transform_ = tf_buffer_->lookupTransform(
            "map",                 // 目标坐标系（例如 "map"/"odom"）
            "base_link",             // 源坐标系
            rclcpp::Time(0));        // 最新变换（保持你原来的语义）

    } catch (const tf2::TransformException &ex) {
        // 等价于原来的 catch 分支：失败返回 false
        // RCLCPP_WARN(this->get_logger(), "Transform failure: %s", ex.what());
        return false;
    }

    // 记录机器人位置（保持你原逻辑）
    robotPoint.x = transform_.transform.translation.x;
    robotPoint.y = transform_.transform.translation.y;
    robotPoint.z = transform_.transform.translation.z;

    // 与原来一致：把输入点云视作来自 "base_link"
    // （注意：这是给你自己的处理逻辑看的，不是 ROS2 消息头）
    laserCloudIn->header.frame_id = "base_link";
    laserCloudIn->header.stamp    = 0;  // 与原代码相同的占位写法

    // 执行点云坐标变换（保留“先到临时，再覆盖”的结构）
    pcl::PointCloud<PointType> laserCloudTemp;

    // 将 geometry_msgs 的 TransformStamped 转为 Eigen 仿射矩阵（float）
    Eigen::Affine3f tf_eigen = tf2::transformToEigen(transform_).cast<float>();

    // 用 PCL 完成点云坐标变换
    pcl::transformPointCloud(*laserCloudIn, laserCloudTemp, tf_eigen);

    // 覆盖回输入
    *laserCloudIn = std::move(laserCloudTemp);

    return true;
}


    /*
    输入：laserCloudIn
    功能：将点云laserCloudIn信息赋值给laserCloudMatrix
    */
    void cloud2Matrix(){

        for (int i = 0; i < N_SCAN; ++i){
            for (int j = 0; j < Horizon_SCAN; ++j){
                int index = j  + i * Horizon_SCAN;
                PointType p = laserCloudIn->points[index];
                laserCloudMatrix[i][j] = p;
            }
        }
    }

    //将点云存储到二维矩阵laserCloudMatrix中(lio与lego输出的点云访问索引不一样，)
    void cloud2Matrix(const traversability_mapping::msg::CloudInfo::SharedPtr msgIn){

        int cloudSize = laserCloudIn->points.size();
        int i = 0;//行序号，就是线数
        int last_j = 0;
        int j = 0;//列序号，每条线上的点数
        for (int index = 0; index < cloudSize; ++index){
                j = msgIn->point_col_ind[index];
                if (last_j > j ) i++;   
                last_j = j;
                if(i > N_SCAN)
                {
                   RCLCPP_WARN(this->get_logger(),"i= %d",i);
                    // i = 0;
                }
                PointType p = laserCloudIn->points[index];  //访问单个点
                // p.z += 1; //sim时修改，使车在平面上
                laserCloudMatrix[i][j] = p;  
        }
    }

    /*
    input：   rangeMatrix(laserCloud的intensity)
    function：根据相邻点的instensity判断该点是否为障碍物，存储到obstacleMatrix
    */
    void applyFilter(){

        if(filterflag){
            positiveCurbFilter();
            negativeCurbFilter();
            slopeFilter();
        }
    }

    //水平方向距离查对点云判断：第0～scanNumcurb的每条线上遍历每一点，计算相邻两个点的距离差(range difference),如果同时满足下面四个条件，将其标记为障碍物，存储到obstacleMatrix
    void positiveCurbFilter(){
        int rangeCompareNeighborNum = 3;
        float diff[Horizon_SCAN - 1];

        for (int i = 0; i < scanNumCurbFilter; ++i){ //i:线数
            // calculate range difference
            for (int j = 0; j < Horizon_SCAN - 1; ++j)  //j:点数
                diff[j] = rangeMatrix.at<float>(i, j) - rangeMatrix.at<float>(i, j+1); //左右两点距离差值

            for (int j = rangeCompareNeighborNum; j < Horizon_SCAN - rangeCompareNeighborNum; ++j){

                // Point that has been verified by other filters
                if (obstacleMatrix.at<int>(i, j) == 1)
                    continue;

                bool breakFlag = false;
                // point is too far away, skip comparison since it can be inaccurate
                if (rangeMatrix.at<float>(i, j) > sensorRangeLimit)
                    continue;
                // 所有点在有效距离内
                for (int k = -rangeCompareNeighborNum; k <= rangeCompareNeighborNum; ++k)
                    if (rangeMatrix.at<float>(i, j+k) == -1){
                        breakFlag = true;
                        break;
                    }
                if (breakFlag == true) continue;
                // 左右两点差值符号相反记为障碍物
                for (int k = -rangeCompareNeighborNum; k < rangeCompareNeighborNum-1; ++k)
                    if (diff[j+k] * diff[j+k+1] <= 0){
                        breakFlag = true;
                        break;
                    }
                if (breakFlag == true) continue;
                // 该点左右两点距离差/到该点距离<0.03
                if (abs(rangeMatrix.at<float>(i, j-rangeCompareNeighborNum) - rangeMatrix.at<float>(i, j+rangeCompareNeighborNum)) /rangeMatrix.at<float>(i, j) < positiveParam)
                    continue;
                // if "continue" is not used at this point, it is very likely to be an obstacle point
                obstacleMatrix.at<int>(i, j) = 1;
            }
        }
    }

    //水平方向上点云高度查进行判断：同理，对第0~第scanNumCurbFilter(8)每一线，遍历该线上的每一点，如果该点和其前后rangeCompareNeighborNum(3)个邻居点相比，满足两点的高度差大于0.15，并且两点间的距离小于等于1.0，则将该点标记为障碍物点。
    void negativeCurbFilter(){
        int rangeCompareNeighborNum = 3;

        for (int i = 0; i < scanNumCurbFilter; ++i){
            for (int j = 0; j < Horizon_SCAN; ++j){
                // Point that has been verified by other filters
                if (obstacleMatrix.at<int>(i, j) == 1)
                    continue;
                // point without range value cannot be verified
                if (rangeMatrix.at<float>(i, j) == -1)
                    continue;
                // point is too far away, skip comparison since it can be inaccurate
                if (rangeMatrix.at<float>(i, j) > sensorRangeLimit)
                    continue;
                // check neighbors
                for (int m = -rangeCompareNeighborNum; m <= rangeCompareNeighborNum; ++m){
                    int k = j + m;
                    if (k < 0 || k >= Horizon_SCAN)
                        continue;
                    if (rangeMatrix.at<float>(i, k) == -1)
                        continue;
                    // height diff greater than threashold, might be a negative curb
                    if (laserCloudMatrix[i][j].z - laserCloudMatrix[i][k].z > negativeHeight
                        && pointDistance(laserCloudMatrix[i][j], laserCloudMatrix[i][k]) <= 1.0){
                        obstacleMatrix.at<int>(i, j) = 1;
                        break;
                    }
                }
            }
        }
    }

    //根据同一水平位置上下两线对应点的俯仰角作为倾斜角(slope angle)，根据倾斜角大小判断该点是否为障碍物点
    void slopeFilter(){
        
        for (int i = 0; i < scanNumSlopeFilter; ++i){
            for (int j = 0; j < Horizon_SCAN; ++j){
                // Point that has been verified by other filters
                if (obstacleMatrix.at<int>(i, j) == 1)
                    continue;
                // point without range value cannot be verified
                if (rangeMatrix.at<float>(i, j) == -1 || rangeMatrix.at<float>(i+1, j) == -1)
                    continue;
                // point is too far away, skip comparison since it can be inaccurate
                if (rangeMatrix.at<float>(i, j) > sensorRangeLimit)
                    continue;
                // Two range filters here:
                // 1. if a point's range is larger than scanNumSlopeFilter th ring point's range
                // 2. if a point's range is larger than the upper point's range
                // then this point is very likely on obstacle. i.e. a point under the car or on a pole
                // if (  (rangeMatrix.at<float>(scanNumSlopeFilter, j) != -1 && rangeMatrix.at<float>(i, j) > rangeMatrix.at<float>(scanNumSlopeFilter, j))
                //     || (rangeMatrix.at<float>(i, j) > rangeMatrix.at<float>(i+1, j)) ){
                //     obstacleMatrix.at<int>(i, j) = 1;
                //     continue;
                // }
                // Calculate slope angle
                float diffX = laserCloudMatrix[i+1][j].x - laserCloudMatrix[i][j].x;
                float diffY = laserCloudMatrix[i+1][j].y - laserCloudMatrix[i][j].y;
                float diffZ = laserCloudMatrix[i+1][j].z - laserCloudMatrix[i][j].z;
                float angle = atan2(diffZ, sqrt(diffX*diffX + diffY*diffY)) * 180 / M_PI;
                // Slope angle is larger than threashold, mark as obstacle point
                if (angle < -filterAngleLimit || angle > filterAngleLimit){
                    obstacleMatrix.at<int>(i, j) = 1;
                    continue;
                }
            }
        }
    }
   
    /*
    input: laserCloudMatrix(laserCloudIn的二维矩阵格式),obstacleMatrix
    output: laserCloudOut,发布/filtered_pointcloud_visual_high_res
    function：对点云进行提取，更新点云intensity(100即为障碍物)，跳过无效点和距离较远点云，保存到laserCloudOut并发布到/filtered_pointcloud_visual_high_res，并将标记的障碍物点云保存到laserCloudObstacles
    */
   // ROS 2 版本的 extractFilteredCloud (ROS API 替换)
    void extractFilteredCloud(){
    for (int i = 0; i < scanNumMax; ++i){
        for (int j = 0; j < Horizon_SCAN; ++j){
            if (rangeMatrix.at<float>(i, j) > sensorRangeLimit || rangeMatrix.at<float>(i, j) < 0.2 ||
                rangeMatrix.at<float>(i, j) == -1)
                continue;
            PointType p = laserCloudMatrix[i][j];
            p.intensity = obstacleMatrix.at<int>(i,j) == 1 ? 100 : 0;
            // RCLCPP_INFO_STREAM(this->get_logger(), "high_res_z" << p.intensity); // 替换 ROS_INFO_STREAM
            laserCloudOut->push_back(p);
            if (p.intensity == 100)
                laserCloudObstacles->push_back(p);
        }
    }

    // Publish laserCloudOut for visualization (before downsample and BGK prediction)
    // ROS 2 中不再使用 getNumSubscribers()
    // 你可以通过 pub_cloud_visual_hi_res_->get_subscription_count() 来检查是否有订阅者
    if (pub_cloud_visual_hi_res_->get_subscription_count() != 0){
        sensor_msgs::msg::PointCloud2 laserCloudTemp; // ROS 2 消息类型
        pcl::toROSMsg(*laserCloudOut, laserCloudTemp);
        // ros::Time::now() 替换为 this->now()
        laserCloudTemp.header.stamp = this->now(); // 获取节点当前时间
        laserCloudTemp.header.frame_id = frameID;
        pub_cloud_visual_hi_res_->publish(laserCloudTemp); // 使用 ROS 2 发布者
    }
    }

    /*输入：laserCloudOut
      输出：laserCloudOut，发布/filtered_point_visual_low_res
      功能:网格化降采样点云，每个网格cell只存储一个点，更新记录这个点处的最大高度和最小高度，根据二者差值大小
           更新instensity，100（障碍物）或0（free），更新z为该点处最大高度并发布
    */
    void downsampleCloud(){
    float roundedX = float(int(robotPoint.x * 10.0f)) / 10.0f;
    float roundedY = float(int(robotPoint.y * 10.0f)) / 10.0f;
    localMapOrigin.x = roundedX - sensorRangeLimit;
    localMapOrigin.y = roundedY - sensorRangeLimit;
    
    int cloudSize = laserCloudOut->points.size();
    for (int i = 0; i < cloudSize; ++i){
        int idx = (laserCloudOut->points[i].x - localMapOrigin.x) / mapResolution;
        int idy = (laserCloudOut->points[i].y - localMapOrigin.y) / mapResolution;
        if (idx < 0 || idy < 0 || idx >= filterHeightMapArrayLength || idy >= filterHeightMapArrayLength)
            continue;
        if (laserCloudOut->points[i].intensity == 100)
            obstFlag[idx][idy] = true;
        if (initFlag[idx][idy] == false){
            minHeight[idx][idy] = laserCloudOut->points[i].z;
            maxHeight[idx][idy] = laserCloudOut->points[i].z;
            initFlag[idx][idy] = true;
        } else {
            minHeight[idx][idy] = std::min(minHeight[idx][idy], laserCloudOut->points[i].z);
            maxHeight[idx][idy] = std::max(maxHeight[idx][idy], laserCloudOut->points[i].z);
            // RCLCPP_INFO_STREAM(this->get_logger(), "laserCloudOut_z" << laserCloudOut->points[i].z);
        }
    }
    pcl::PointCloud<PointType>::Ptr laserCloudTemp(new pcl::PointCloud<PointType>());
    for (int i = 0; i < filterHeightMapArrayLength; ++i){
        for (int j = 0; j < filterHeightMapArrayLength; ++j){
            if (initFlag[i][j] == false)
                continue;
            PointType thisPoint;
            thisPoint.x = localMapOrigin.x + i * mapResolution + mapResolution / 2.0;
            thisPoint.y = localMapOrigin.y + j * mapResolution + mapResolution / 2.0;
            thisPoint.z = maxHeight[i][j];
            // RCLCPP_INFO_STREAM(this->get_logger(), "maxHeight" << maxHeight[i][j]);

            if (obstFlag[i][j] == true || maxHeight[i][j] - minHeight[i][j] > filterHeightLimit){
                obstFlag[i][j] = true;
                thisPoint.intensity = 100; // obstacle
                laserCloudTemp->push_back(thisPoint);
            }else{
                thisPoint.intensity = 0; // free
                laserCloudTemp->push_back(thisPoint);
            }
        }
    }

    *laserCloudOut = *laserCloudTemp;

    // Publish laserCloudOut for visualization (after downsample but beforeBGK prediction)
    // 替换 getNumSubscribers()
    if (pub_cloud_visual_low_res_->get_subscription_count() != 0){
        sensor_msgs::msg::PointCloud2 laserCloudTemp_msg; // ROS 2 消息类型
        pcl::toROSMsg(*laserCloudOut, laserCloudTemp_msg);
        // ros::Time::now() 替换为 this->now()
        laserCloudTemp_msg.header.stamp = this->now(); // 获取节点当前时间
        laserCloudTemp_msg.header.frame_id = frameID;
        pub_cloud_visual_low_res_->publish(laserCloudTemp_msg); // 使用 ROS 2 发布者
    }
}

    /*
    input：maxHeight[][],obstFlag[][],initFlag[][]
    output: 更新laserCloudOut
    function：在当前局部地图范围内利用已经扫描到的栅格去更新未知栅格的高度和occupied（是否为障碍物）   只预测sensorRangeLimit以内的
    更新方式：利用该未知栅格周围a米范围内的已知栅格去预测    a=kernelGridLength的开方
    */
    // void predictCloudBGK(){

    //     if (predictionEnableFlag == false)
    //         return;

    //     int kernelGridLength = int(predictionKernalSize / mapResolution);  //    0.4/0.1=4

    //     //遍历局部地图的每一点
    //     for (int i = 0; i < filterHeightMapArrayLength; ++i){
    //         for (int j = 0; j < filterHeightMapArrayLength; ++j){
    //             // skip observed point
    //             if (initFlag[i][j] == true)
    //                 continue;
    //             PointType testPoint;   //要估计的点
    //             testPoint.x = localMapOrigin.x + i * mapResolution + mapResolution / 2.0;   //localMapOrigin.x：局部地图的起点位置
    //             testPoint.y = localMapOrigin.y + j * mapResolution + mapResolution / 2.0;
    //             testPoint.z = robotPoint.z; // this value is not used except for computing distance with robotPoint
    //             // skip grids too far
    //             if (pointDistance(testPoint, robotPoint) > sensorRangeLimit)
    //                 continue;
    //             // Training data
    //             vector<float> xTrainVec; // training data x and y coordinates
    //             vector<float> yTrainVecElev; // training data elevation
    //             vector<float> yTrainVecOccu; // training data occupancy
    //             // Fill trainig data (vector)
    //             for (int m = -kernelGridLength; m <= kernelGridLength; ++m){
    //                 for (int n = -kernelGridLength; n <= kernelGridLength; ++n){
    //                     // skip grids too far
    //                     if (std::sqrt(float(m*m + n*n)) * mapResolution > predictionKernalSize)
    //                         continue;
    //                     int idx = i + m;
    //                     int idy = j + n;
    //                     // index out of boundry
    //                     if (idx < 0 || idy < 0 || idx >= filterHeightMapArrayLength || idy >= filterHeightMapArrayLength)
    //                         continue;
    //                     // save only observed grid in this scan
    //                     if (initFlag[idx][idy] == true){
    //                         xTrainVec.push_back(localMapOrigin.x + idx * mapResolution + mapResolution / 2.0);
    //                         xTrainVec.push_back(localMapOrigin.y + idy * mapResolution + mapResolution / 2.0);
    //                         yTrainVecElev.push_back(maxHeight[idx][idy]);
    //                         yTrainVecOccu.push_back(obstFlag[idx][idy] == true ? 1 : 0);
    //                     }
    //                 }
    //             }
    //             // no training data available, continue
    //             if (xTrainVec.size() == 0)
    //                 continue;
    //             // convert from vector to eigen
    //             Eigen::MatrixXf xTrain = Eigen::Map<const Eigen::Matrix<float, -1, -1, Eigen::RowMajor>>(xTrainVec.data(), xTrainVec.size() / 2, 2);
    //             Eigen::MatrixXf yTrainElev = Eigen::Map<const Eigen::Matrix<float, -1, -1, Eigen::RowMajor>>(yTrainVecElev.data(), yTrainVecElev.size(), 1);
    //             Eigen::MatrixXf yTrainOccu = Eigen::Map<const Eigen::Matrix<float, -1, -1, Eigen::RowMajor>>(yTrainVecOccu.data(), yTrainVecOccu.size(), 1);
    //             // Test data (current grid)
    //             vector<float> xTestVec;
    //             xTestVec.push_back(testPoint.x);
    //             xTestVec.push_back(testPoint.y);
    //             Eigen::MatrixXf xTest = Eigen::Map<const Eigen::Matrix<float, -1, -1, Eigen::RowMajor>>(xTestVec.data(), xTestVec.size() / 2, 2);
    //             // 预测当前点的elevation和occupancy
    //             Eigen::MatrixXf Ks; // covariance matrix
    //             covSparse(xTest, xTrain, Ks); // sparse kernel  Ks:1xn，n为xtrain个数

    //             Eigen::MatrixXf ybarElev = (Ks * yTrainElev).array();    //1x1
    //             Eigen::MatrixXf ybarOccu = (Ks * yTrainOccu).array();
    //             Eigen::MatrixXf kbar = Ks.rowwise().sum().array();      //计算每行的和

    //             // Update Elevation with Prediction
    //             if (std::isnan(ybarElev(0,0)) || std::isnan(ybarOccu(0,0)) || std::isnan(kbar(0,0)))
    //                 continue;

    //             if (kbar(0,0) == 0)
    //                 continue;

    //             float elevation = ybarElev(0,0) / kbar(0,0);
    //             float occupancy = ybarOccu(0,0) / kbar(0,0);

    //             PointType p;
    //             p.x = xTestVec[0];
    //             p.y = xTestVec[1];
    //             p.z = elevation;
    //             p.intensity = (occupancy > 0.5) ? 100 : 0;

    //             laserCloudOut->push_back(p);
    //         }
    //     }
    // }
    void predictCloudBGK() 
    {
    if (predictionEnableFlag == false) return;

    // ------- 常量与缓存准备 -------
    const float radius = predictionKernalSize;      // 预测核半径（米）
    const float res    = mapResolution;             // 栅格分辨率（米）
    const int   N      = filterHeightMapArrayLength;
    const float halfRes = 0.5f * res;
    const float radius2 = radius * radius;
    const float maxRange2 = sensorRangeLimit * sensorRangeLimit;

    // 预计算 & 缓存：圆形邻接偏移（含距离）
    struct Off { int di, dj; float dist; };
    static int   cachedN   = -1;
    static float cachedRes = -1.f, cachedRadius = -1.f;
    static std::vector<Off> OFFS;

    const int L = static_cast<int>(std::ceil(radius / res)); // 半径对应的栅格步长

    if (cachedN != N || cachedRes != res || cachedRadius != radius) {
        OFFS.clear();
        OFFS.reserve((2*L+1) * (2*L+1));
        for (int di = -L; di <= L; ++di) {
            for (int dj = -L; dj <= L; ++dj) {
                if (di == 0 && dj == 0) continue;
                float dx = di * res, dy = dj * res;
                float d2 = dx*dx + dy*dy;
                if (d2 <= radius2) {
                    // sqrt 只在这里做一次，后续重用
                    OFFS.push_back({di, dj, std::sqrt(d2)});
                }
            }
        }
        cachedN = N; cachedRes = res; cachedRadius = radius;
    }

    // -------- 生成候选未知栅格（仅对这些做预测）--------
    // 从每个已观测栅格向外扩一圈（OFFS），收集未知格，且在传感器半径内
    static std::vector<std::pair<int,int>> candidates;
    candidates.clear();
    candidates.reserve(OFFS.size() * 1024);

    static std::vector<uint8_t> mark; // 标记数组，避免重复加入同一候选
    mark.assign(static_cast<size_t>(N) * static_cast<size_t>(N), 0);

    const float originX = localMapOrigin.x;
    const float originY = localMapOrigin.y;

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            if (!initFlag[i][j]) continue; // 只从已观测格扩散
            for (const auto& o : OFFS) {
                int ii = i + o.di;
                int jj = j + o.dj;
                if ((unsigned)ii >= (unsigned)N || (unsigned)jj >= (unsigned)N) continue;
                if (initFlag[ii][jj]) continue; // 已观测的不需要预测
                size_t idx1d = static_cast<size_t>(ii) * N + jj;
                if (mark[idx1d]) continue;

                // 传感器量测范围内才考虑
                float cx = originX + ii * res + halfRes;
                float cy = originY + jj * res + halfRes;
                float dx = cx - robotPoint.x, dy = cy - robotPoint.y;
                if (dx*dx + dy*dy > maxRange2) continue;

                mark[idx1d] = 1;
                candidates.emplace_back(ii, jj);
            }
        }
    }

    if (candidates.empty()) return;

    // 估计 push_back 数量，减少扩容
    laserCloudOut->points.reserve(laserCloudOut->points.size() + candidates.size());

    // ------- 紧支撑核（Wendland C2）：无三角函数、仅多项式 -------
    auto weight = [radius](float dist) -> float {
        // r in [0,1]
        float r = dist / (radius + 1e-3f);
        if (r >= 1.0f) return 0.0f;
        // w(r) = (1 - r)^4 * (4r + 1)
        float t = 1.0f - r;
        float t2 = t * t;
        float t4 = t2 * t2;
        return t4 * (4.0f * r + 1.0f);
    };

    // -------- 对候选未知格进行预测 --------
    for (const auto& c : candidates) {
        const int i = c.first;
        const int j = c.second;

        float cx = originX + i * res + halfRes;
        float cy = originY + j * res + halfRes;

        float sumW = 0.0f;
        float sumElev = 0.0f;
        float sumOcc  = 0.0f;

        // 在预计算邻接偏移上累加
        for (const auto& o : OFFS) {
            int ii = i + o.di;
            int jj = j + o.dj;
            if ((unsigned)ii >= (unsigned)N || (unsigned)jj >= (unsigned)N) continue;
            if (!initFlag[ii][jj]) continue; // 只用已观测邻域作为训练样本

            float w = weight(o.dist);
            if (w <= 0.0f) continue;

            sumW    += w;
            sumElev += w * maxHeight[ii][jj];
            sumOcc  += w * (obstFlag[ii][jj] ? 1.0f : 0.0f);
        }

        if (sumW <= 0.0f) continue;

        PointType p;
        p.x = cx;
        p.y = cy;
        p.z = sumElev / sumW;
        p.intensity = (sumOcc / sumW > 0.5f) ? 100 : 0;

        laserCloudOut->push_back(p);
    }
}

   

    void dist(const Eigen::MatrixXf &xStar, const Eigen::MatrixXf &xTrain, Eigen::MatrixXf &d) const {
        d = Eigen::MatrixXf::Zero(xStar.rows(), xTrain.rows());
        for (int i = 0; i < xStar.rows(); ++i) {
            d.row(i) = (xTrain.rowwise() - xStar.row(i)).rowwise().norm();
        }
    }

    //x_test:1个，x_train:n个   Kxz：1xn
    void covSparse(const Eigen::MatrixXf &xStar, const Eigen::MatrixXf &xTrain, Eigen::MatrixXf &Kxz) const {
        dist(xStar/(predictionKernalSize+0.1), xTrain/(predictionKernalSize+0.1), Kxz);
        
        Kxz = (((2.0f + (Kxz * 2.0f * 3.1415926f).array().cos()) * (1.0f - Kxz.array()) / 3.0f) +
              (Kxz * 2.0f * 3.1415926f).array().sin() / (2.0f * 3.1415926f)).matrix() * 1.0f;
        // Clean up for values with distance outside length scale, possible because Kxz <= 0 when dist >= predictionKernalSize

        // ROS_WARN("xStar size = %d, xTrain size = %d",xStar.rows(),xTrain.rows());
        // ROS_WARN("Kxz rows = %d, Kxz cols = %d",Kxz.rows(),Kxz.cols());
        // std::cout<<"xTrain.rowwise()  = " << xTrain.rowwise() <<std::endl;
        for (int i = 0; i < Kxz.rows(); ++i)
            for (int j = 0; j < Kxz.cols(); ++j)
                if (Kxz(i,j) < 0) Kxz(i,j) = 0;
    }

    //将laserCloudOut发布   /filtered_pointcloud
    void publishCloud(){
    sensor_msgs::msg::PointCloud2 laserCloudTemp; // ROS 2 消息类型
    pcl::toROSMsg(*laserCloudOut, laserCloudTemp);
    // ros::Time::now() 替换为 this->now()
    laserCloudTemp.header.stamp = this->now(); // 获取节点当前时间
    laserCloudTemp.header.frame_id = frameID;
    pub_cloud_->publish(laserCloudTemp); // 使用 ROS 2 发布者
}

};


int main(int argc, char** argv){
    // ros::init(argc, argv, "traversability_mapping"); 替换为 rclcpp::init
    rclcpp::init(argc, argv);
    
    // 创建节点实例，rclcpp::Node的智能指针
    auto node = std::make_shared<TraversabilityFilter>();

    // ROS_INFO 替换为 RCLCPP_INFO
    RCLCPP_INFO(node->get_logger(), "-----> Traversability Filter Started.");

    // ros::spin() 替换为 rclcpp::spin()
    rclcpp::spin(node);

    // ros::shutdown() 替换为 rclcpp::shutdown()
    rclcpp::shutdown();

    return 0;
}