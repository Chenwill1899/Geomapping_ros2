#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

#include <algorithm>
#include <cmath>
#include <string>

class TerrainPublisherNode : public rclcpp::Node {
public:
    // 替换构造函数中的 QoS 配置
    TerrainPublisherNode() : Node("terrain_pub_node") {
        this->declare_parameter<int>("scan_lines", 16);
        this->declare_parameter<int>("target_scan_lines", 16);
        this->declare_parameter<int>("horizon_scan", 900);
        this->declare_parameter<double>("vertical_fov_deg", 30.0);
        this->declare_parameter<std::string>("output_frame", "/base_link");

        this->get_parameter("scan_lines", scan_lines_);
        this->get_parameter("target_scan_lines", target_scan_lines_);
        this->get_parameter("horizon_scan", horizon_scan_);
        this->get_parameter("vertical_fov_deg", vertical_fov_deg_);
        this->get_parameter("output_frame", output_frame_);

        scan_lines_ = std::max(1, scan_lines_);
        target_scan_lines_ = std::max(scan_lines_, target_scan_lines_);
        horizon_scan_ = std::max(1, horizon_scan_);
        vertical_fov_deg_ = std::max(1e-3, vertical_fov_deg_);

        auto sensor_qos = rclcpp::SensorDataQoS();          // = BEST_EFFORT + KEEP_LAST(1)
        sensor_qos.lifespan(std::chrono::milliseconds(100)); // 过期即丢，防止积旧

        pubLaserCloudFull_terrain = this->create_publisher<sensor_msgs::msg::PointCloud2>(
            "/syncd_project_cloud", sensor_qos);
        

        sub_pcl = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/velodyne_points", sensor_qos,
            std::bind(&TerrainPublisherNode::standard_pcl_cbk, this, std::placeholders::_1));
    }


private:
    void standard_pcl_cbk(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
        float horizonAngle, range;
        pcl::PointCloud<pcl::PointXYZI> pl_terrain;
        pl_terrain.clear();
        pl_terrain.points.resize(target_scan_lines_ * horizon_scan_);
        pcl::PointXYZI thisPoint;
        int rowIdn, columnIdn;
        size_t index, cloudSize;

        pcl::PointCloud<pcl::PointXYZ> pl_orig;
        pcl::fromROSMsg(*msg, pl_orig);
        cloudSize = pl_orig.points.size();

        if (cloudSize == 0) return;

        for (size_t i = 0; i < cloudSize; ++i) {
            thisPoint.x = pl_orig.points[i].x;
            thisPoint.y = pl_orig.points[i].y;
            thisPoint.z = pl_orig.points[i].z;
            range = sqrt(thisPoint.x * thisPoint.x + thisPoint.y * thisPoint.y + thisPoint.z * thisPoint.z);
            if (!std::isfinite(range) || range <= 0.0f)
                continue;

            rowIdn = project_ring(thisPoint, range);
            if (rowIdn < 0 || rowIdn >= target_scan_lines_)
                continue;

            horizonAngle = atan2(thisPoint.x, thisPoint.y) * 180 / M_PI;

            const double horizontal_resolution_deg = 360.0 / static_cast<double>(horizon_scan_);
            columnIdn = -round((horizonAngle - 90.0) / horizontal_resolution_deg) + horizon_scan_ / 2;
            if (columnIdn >= horizon_scan_)
                columnIdn -= horizon_scan_;

            if (columnIdn < 0 || columnIdn >= horizon_scan_)
                continue;

            thisPoint.intensity = (float)rowIdn + (float)columnIdn / 10000.0;

            index = static_cast<size_t>(columnIdn + rowIdn * horizon_scan_);
            pl_terrain.points[index] = thisPoint;
            pl_terrain.points[index].intensity = range;
        }

        sensor_msgs::msg::PointCloud2 laserCloudmsg;
        pcl::toROSMsg(pl_terrain, laserCloudmsg);
        //需要保证用原消息时间戳
        laserCloudmsg.header.stamp = msg->header.stamp;
        laserCloudmsg.header.frame_id = output_frame_;

        pubLaserCloudFull_terrain->publish(laserCloudmsg);
    }

    int project_ring(const pcl::PointXYZI& point, float range) const {
        const double elevation_deg = std::asin(point.z / range) * 180.0 / M_PI;
        if (scan_lines_ <= 1) {
            return 0;
        }
        const double step = vertical_fov_deg_ / static_cast<double>(scan_lines_ - 1);
        return static_cast<int>(std::lround((vertical_fov_deg_ * 0.5 - elevation_deg) / step));
    }

    int scan_lines_ = 16;
    int target_scan_lines_ = 16;
    int horizon_scan_ = 900;
    double vertical_fov_deg_ = 30.0;
    std::string output_frame_ = "/base_link";
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudFull_terrain;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_pcl;
};

int main(int argc, char *argv[]) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<TerrainPublisherNode>());
    rclcpp::shutdown();
    return 0;
}
