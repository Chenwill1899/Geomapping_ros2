#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
import time


class TopicListener(Node):

    def __init__(self):
        super().__init__('topic_listener')
        self.subscription = self.create_subscription(
            PointCloud2,
            '/filtered_pointcloud',
            # '/syncd_project_cloud',
            self.listener_callback,
            10)
        self.last_received_time = None
        self.msg_count = 0

    def listener_callback(self, msg):
        current_time = time.time()

        if self.last_received_time is not None:
            # 计算接收的时间间隔
            time_diff = current_time - self.last_received_time
            frequency = 1 / time_diff if time_diff > 0 else 0
            self.get_logger().info(f"Frequency: {frequency:.2f} Hz")

        self.last_received_time = current_time
        self.msg_count += 1


def main(args=None):
    rclpy.init(args=args)

    listener = TopicListener()

    rclpy.spin(listener)

    # Destroy the node explicitly
    listener.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
