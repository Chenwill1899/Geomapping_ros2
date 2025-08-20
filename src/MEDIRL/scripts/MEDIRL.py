#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pytorch=1.6.0 torchvision cudatoolkit=10.1
"""
MEDIRL--zle--2024/4/8
ROS2版本
"""
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from nav_msgs.msg import OccupancyGrid
from elevation_msgs.msg import OccupancyElevation
import numpy as np
import math
import torch
from torch.autograd import Variable
from os.path import join
import time
from env_dilated import OnlyEnvDilated

class MEDIRL_env(Node):
    def __init__(self):
        super().__init__('MEDIRL')
        
        self.grid_size = 180
        self.net = OnlyEnvDilated(feat_in_size=4, feat_out_size=25)
        self.net.init_weights()
        # 加载模型时添加map_location参数，避免设备不匹配问题
        self.net.load_state_dict(torch.load(
            join('/home/mexxiie/prj/Geo_Semantic_fusion_nav_ws', 'step16-loss0.pth'),
            map_location=torch.device('cpu')
        )['net_state'])
        self.net.eval()
        
        self.have_sem = True
        self.have_gem = False
        
        # 初始化消息
        self.reward_msg = OccupancyElevation()
        self.reward_msg.header.frame_id = 'map'
        grid_total = self.grid_size * self.grid_size
        
        # 初始化数组时确保类型正确
        self.reward_msg.reward_cost = [0.0 for _ in range(grid_total)]
        self.reward_msg.occupancy.data = [-1 for _ in range(grid_total)]
        self.reward_msg.occupancy.info.width = self.grid_size
        self.reward_msg.occupancy.info.height = self.grid_size
        self.reward_msg.occupancy.info.resolution = 0.1

        self.feat = np.zeros((4, self.grid_size, self.grid_size), dtype=np.float32)
        
        # 创建发布者和订阅者
        self.pub_reward = self.create_publisher(OccupancyElevation, '/msg_local_reward', 10)
        self.sub_gem = self.create_subscription(
            OccupancyElevation, '/msg_local_feature', self.gem_callback, 10)
        
    def gem_callback(self, gem_msg):
        self.get_logger().info(f"enter gem_callback")
        self.have_gem = True
        self.reward_msg.occupancy.info.origin.position = gem_msg.occupancy.info.origin.position
        self.reward_msg.occupancy.data = gem_msg.occupancy.data[:]
        
        # 确保特征数据类型正确且范围合理
        self.feat[0] = np.clip(
            np.array(gem_msg.cost_map, dtype=np.float32).reshape(self.grid_size, self.grid_size),
            -1e38, 1e38
        )  # slope
        self.feat[1] = np.clip(
            np.array(gem_msg.height, dtype=np.float32).reshape(self.grid_size, self.grid_size),
            -1e38, 1e38
        )  # step
        self.feat[2] = np.clip(
            np.array(gem_msg.roughness, dtype=np.float32).reshape(self.grid_size, self.grid_size),
            -1e38, 1e38
        )  # rough
        
        if self.have_sem and self.have_gem:
            self.reward_msg.header.stamp = self.get_clock().now().to_msg()
            self.reward_msg.occupancy.header.stamp = self.reward_msg.header.stamp
            self.feat_input(self.feat)

    def feat_input(self, feat):
        # 特征预处理，确保数值范围合理
        for i in range(3):
            feat[i] = np.clip(10 * feat[i], -1e38, 1e38)
        
        # 转换为PyTorch变量
        feat_var = Variable(torch.from_numpy(np.expand_dims(feat, axis=0)).float())

        t1 = time.time()
        r_var = self.net(feat_var)  # （1，1，90，90）
        t2 = time.time()
        self.get_logger().info(f"推理时间: {t2-t1:.6f}秒")

        # 处理输出结果
        r = -r_var[0].data.numpy().squeeze()  # 转换为（90，90）的numpy
        
        # 归一化并裁剪到合理范围
        r = np.clip(r, 0, 8)  # 草地场景参数
        r = 100 * (r / 8)  # 归一化到0-100
        
        # 确保最终结果在float32范围内
        r = np.clip(r, -3.4e38, 3.4e38)
        
        # 转换为Python列表并赋值
        self.reward_msg.reward_cost = r.flatten().tolist()
        
        # 发布消息
        self.pub_reward.publish(self.reward_msg)


def main(args=None):
    rclpy.init(args=args)
    medirl_node = MEDIRL_env()
    rclpy.spin(medirl_node)
    medirl_node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
