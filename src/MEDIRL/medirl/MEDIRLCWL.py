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
from .env_dilated import OnlyEnvDilated
from torch.autograd import Variable
import torch
from os.path import join
import matplotlib.pyplot as plt
import time


class MEDIRL_env(Node):
    def __init__(self):
        super().__init__('MEDIRLCWL')
        
        self.grid_size = 180
        self.net = OnlyEnvDilated(feat_in_size=4,feat_out_size=25)
        self.net.init_weights()
        self.net.load_state_dict(torch.load(join('/home/zle/Documents/MEDIRL/picture_process_keji/exp/train-4.9-canghai-sem5', 'step16-loss0.pth'))['net_state'])
        self.net.eval()
        
        # 加载全局代价地图
        self.global_costmap = self.load_global_costmap('/path/to/your/global_costmap.npy')  # 修改为您的路径
        self.global_map_resolution = 0.1  # 与局部地图相同的分辨率
        
        # 全局地图以机器人初始位置为原点(0,0)
        # 需要知道全局地图的尺寸来计算中心点
        if self.global_costmap is not None:
            self.global_map_height, self.global_map_width = self.global_costmap.shape
            # 计算全局地图的世界坐标范围（以机器人初始位置为原点）
            self.global_map_min_x = -(self.global_map_width // 2) * self.global_map_resolution
            self.global_map_max_x = (self.global_map_width - self.global_map_width // 2) * self.global_map_resolution
            self.global_map_min_y = -(self.global_map_height // 2) * self.global_map_resolution
            self.global_map_max_y = (self.global_map_height - self.global_map_height // 2) * self.global_map_resolution
            
            self.get_logger().info(f"Global map world bounds: x[{self.global_map_min_x:.1f}, {self.global_map_max_x:.1f}], "
                         f"y[{self.global_map_min_y:.1f}, {self.global_map_max_y:.1f}]")
        
        # 融合权重参数
        self.local_weight = 0.7   # 局部网络输出权重
        self.global_weight = 0.3  # 全局代价地图权重
        
        self.have_sem = True
        self.have_gem = False
        self.reward_msg = OccupancyElevation()
        self.reward_msg.header.frame_id = 'map'
        self.reward_msg.rewardCost = [0]*(self.grid_size*self.grid_size)
        self.reward_msg.occupancy.data = [-1]*(self.grid_size*self.grid_size)
        self.reward_msg.occupancy.info.width = self.grid_size
        self.reward_msg.occupancy.info.height = self.grid_size
        self.reward_msg.occupancy.info.resolution = 0.1

        self.feat = np.zeros((4, self.grid_size, self.grid_size))
        self.pub_reward = self.create_publisher(OccupancyElevation, '/msg_local_reward', 10)
        self.sub_gem = self.create_subscription(OccupancyElevation, '/msg_local_feature', self.gem_callback, 10)
    
    def load_global_costmap(self, filepath):
        """加载全局代价地图"""
        try:
            if filepath.endswith('.npy'):
                global_map = np.load(filepath)
            elif filepath.endswith('.txt'):
                global_map = np.loadtxt(filepath)
            else:
                self.get_logger().error("Unsupported file format. Use .npy or .txt")
                return None
            
            self.get_logger().info(f"Loaded global costmap with shape: {global_map.shape}")
            return global_map
        except Exception as e:
            self.get_logger().error(f"Failed to load global costmap: {e}")
            return None
    
    def world_to_global_grid(self, world_x, world_y):
        """将世界坐标转换为全局地图的网格坐标"""
        # 全局地图以机器人初始位置(0,0)为中心
        grid_x = int((world_x - self.global_map_min_x) / self.global_map_resolution)
        grid_y = int((world_y - self.global_map_min_y) / self.global_map_resolution)
        return grid_x, grid_y
    
    def extract_global_cost_patch(self, local_origin_x, local_origin_y):
        """从全局代价地图中提取对应局部区域的代价值"""
        if self.global_costmap is None:
            self.get_logger().warn("Global costmap not loaded, using zeros")
            return np.zeros((self.grid_size, self.grid_size))
        
        # 将局部地图左下角原点转换为全局地图的网格坐标
        global_start_x, global_start_y = self.world_to_global_grid(local_origin_x, local_origin_y)
        
        # 初始化局部代价patch
        global_patch = np.zeros((self.grid_size, self.grid_size))
        
        # 提取对应区域
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                global_x = global_start_x + i
                global_y = global_start_y + j
                
                # 检查边界
                if (0 <= global_x < self.global_map_width and 
                    0 <= global_y < self.global_map_height):
                    global_patch[i, j] = self.global_costmap[global_x, global_y]
        
        return global_patch
    
    def fuse_costmaps(self, local_cost, global_cost):
        """融合局部和全局代价地图"""
        # 简单的加权融合
        fused_cost = (self.local_weight * local_cost + 
                     self.global_weight * global_cost)
        return fused_cost
    
    def gem_callback(self, gem_msg):
        self.have_gem = True
        self.reward_msg.occupancy.info.origin.position = gem_msg.occupancy.info.origin.position
        self.reward_msg.occupancy.data = gem_msg.occupancy.data[:]  #赋值，用于局部规划判断未访问区域,未访问区域为-1
        self.feat[0] = np.array(gem_msg.costMap).reshape(self.grid_size, self.grid_size) #slope
        self.feat[1] = np.array(gem_msg.height).reshape(self.grid_size, self.grid_size) #step
        self.feat[2] = np.array(gem_msg.roughness).reshape(self.grid_size, self.grid_size) #rough
        if self.have_sem and self.have_gem:
            self.reward_msg.header.stamp = self.get_clock().now().to_msg()
            self.reward_msg.occupancy.header.stamp = self.reward_msg.header.stamp
            self.feat_input(self.feat)

    def feat_input(self, feat):
            for i in range(3):
                feat[i] = 10*feat[i]
            feat_var = Variable(torch.from_numpy(np.expand_dims(feat, axis=0)).float())

            r_var = self.net(feat_var) #（1，1，90，90）
            r = -r_var[0].data.numpy().squeeze() #转换为（90，90）的numpy
            
            # 获取局部地图原点在世界坐标系中的位置
            local_origin_x = self.reward_msg.occupancy.info.origin.position.x
            local_origin_y = self.reward_msg.occupancy.info.origin.position.y
            
            # 从全局代价地图中提取对应区域
            global_cost_patch = self.extract_global_cost_patch(local_origin_x, local_origin_y)
            
            # 融合局部和全局代价
            r = self.fuse_costmaps(r, global_cost_patch)
            
            r[r<0]=0    #草地
            r[r>8]=8
            r = 100*(r-0)/(8-0)

            self.reward_msg.header.stamp = self.get_clock().now().to_msg()
            self.reward_msg.rewardCost = r.flatten()
            self.pub_reward.publish(self.reward_msg)


def main(args=None):
    rclpy.init(args=args)
    MEDIRL = MEDIRL_env()
    rclpy.spin(MEDIRL)
    MEDIRL.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main() 