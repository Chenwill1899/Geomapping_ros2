#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

namespace velodyne_ros {
  struct EIGEN_ALIGN16 Point {
      PCL_ADD_POINT4D;
      float intensity;
      float time;
      uint16_t ring;
      EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  };
}  // namespace velodyne_ros

POINT_CLOUD_REGISTER_POINT_STRUCT(velodyne_ros::Point,
    (float, x, x)
    (float, y, y)
    (float, z, z)
    (float, intensity, intensity)
    (float, time, time)
    (uint16_t, ring, ring)
)

class TerrainPublisherNode : public rclcpp::Node {
public:
    // 替换构造函数中的 QoS 配置
    TerrainPublisherNode() : Node("terrain_pub_node") {
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
        pl_terrain.points.resize(32 * 1800);
        pcl::PointXYZI thisPoint;
        size_t rowIdn, columnIdn, index, cloudSize; 

        pcl::PointCloud<velodyne_ros::Point> pl_orig;
        pcl::fromROSMsg(*msg, pl_orig);
        cloudSize = pl_orig.points.size();

        if (cloudSize == 0) return;

        for (size_t i = 0; i < cloudSize; ++i) {
            thisPoint.x = pl_orig.points[i].x;
            thisPoint.y = pl_orig.points[i].y;
            thisPoint.z = pl_orig.points[i].z;

            rowIdn = pl_orig.points[i].ring;
            if (rowIdn < 0 || rowIdn >= 32)
                continue;

            horizonAngle = atan2(thisPoint.x, thisPoint.y) * 180 / M_PI;

            columnIdn = -round((horizonAngle - 90.0) / 0.2) + 1800 / 2;
            if (columnIdn >= 1800)
                columnIdn -= 1800;

            if (columnIdn < 0 || columnIdn >= 1800)
                continue;

            range = sqrt(thisPoint.x * thisPoint.x + thisPoint.y * thisPoint.y + thisPoint.z * thisPoint.z);
            thisPoint.intensity = (float)rowIdn + (float)columnIdn / 10000.0;

            index = columnIdn + rowIdn * 1800;
            pl_terrain.points[index] = thisPoint;
            pl_terrain.points[index].intensity = range;
        }

        sensor_msgs::msg::PointCloud2 laserCloudmsg;
        pcl::toROSMsg(pl_terrain, laserCloudmsg);
        //需要保证用原消息时间戳
        laserCloudmsg.header.stamp = msg->header.stamp;
        laserCloudmsg.header.frame_id = "/base_link";

        pubLaserCloudFull_terrain->publish(laserCloudmsg);
    }

    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudFull_terrain;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_pcl;
};

int main(int argc, char *argv[]) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<TerrainPublisherNode>());
    rclcpp::shutdown();
    return 0;
}
