#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MEDIRL--zle--2024/4/8
ROS2版本 - 精确匹配reward_cost的float32[]类型
"""
import rclpy
from rclpy.node import Node
import numpy as np
import torch
from os.path import join
from nav_msgs.msg import OccupancyGrid
from elevation_msgs.msg import OccupancyElevation
from .env_dilated import OnlyEnvDilated


def filter_ground_artifact_reward(
    reward,
    step,
    roughness,
    slope,
    occupancy,
    *,
    enabled=True,
    min_cost=20.0,
    max_step_abs=0.08,
    max_roughness=0.08,
    max_slope=0.08,
    occupied_threshold=50,
    clear_value=0.0,
):
    reward_arr = np.asarray(reward, dtype=np.float32).copy()
    if not enabled:
        return reward_arr, 0

    step_arr = np.asarray(step, dtype=np.float32)
    roughness_arr = np.asarray(roughness, dtype=np.float32)
    slope_arr = np.asarray(slope, dtype=np.float32)
    occupancy_arr = np.asarray(occupancy, dtype=np.int16)
    if (
        reward_arr.shape != step_arr.shape
        or reward_arr.shape != roughness_arr.shape
        or reward_arr.shape != slope_arr.shape
        or reward_arr.shape != occupancy_arr.shape
    ):
        return reward_arr, 0

    flat_ground_artifact = (
        (reward_arr >= float(min_cost))
        & np.isfinite(step_arr)
        & np.isfinite(roughness_arr)
        & np.isfinite(slope_arr)
        & (occupancy_arr >= 0)
        & (occupancy_arr < int(occupied_threshold))
        & (np.abs(step_arr) <= float(max_step_abs))
        & (roughness_arr <= float(max_roughness))
        & (slope_arr <= float(max_slope))
    )
    if not np.any(flat_ground_artifact):
        return reward_arr, 0
    reward_arr[flat_ground_artifact] = np.float32(clear_value)
    return reward_arr, int(np.count_nonzero(flat_ground_artifact))


class MEDIRL_env(Node):
    def __init__(self):
        super().__init__('MEDIRL')

        self.declare_parameter('device', 'cuda')
        self.declare_parameter('debug_timing', False)
        self.declare_parameter('ground_artifact_filter.enabled', True)
        self.declare_parameter('ground_artifact_filter.min_cost', 20.0)
        self.declare_parameter('ground_artifact_filter.max_step_abs', 0.08)
        self.declare_parameter('ground_artifact_filter.max_roughness', 0.08)
        self.declare_parameter('ground_artifact_filter.max_slope', 0.08)
        self.declare_parameter('ground_artifact_filter.occupied_threshold', 50)
        self.declare_parameter('ground_artifact_filter.clear_value', 0.0)
        requested_device = self.get_parameter('device').value
        self.debug_timing = bool(self.get_parameter('debug_timing').value)
        self.ground_artifact_filter = {
            'enabled': bool(self.get_parameter('ground_artifact_filter.enabled').value),
            'min_cost': float(self.get_parameter('ground_artifact_filter.min_cost').value),
            'max_step_abs': float(self.get_parameter('ground_artifact_filter.max_step_abs').value),
            'max_roughness': float(self.get_parameter('ground_artifact_filter.max_roughness').value),
            'max_slope': float(self.get_parameter('ground_artifact_filter.max_slope').value),
            'occupied_threshold': int(self.get_parameter('ground_artifact_filter.occupied_threshold').value),
            'clear_value': float(self.get_parameter('ground_artifact_filter.clear_value').value),
        }
        if requested_device == 'cuda' and not torch.cuda.is_available():
            self.get_logger().warn("CUDA requested for MEDIRL but unavailable; falling back to CPU")
            requested_device = 'cpu'
        self.device = torch.device(requested_device)
        if self.device.type == 'cpu':
            torch.set_num_threads(4)
        
        self.grid_size = 180
        self.total_cells = self.grid_size * self.grid_size  # 计算总单元格数
        
        # 初始化模型
        self.net = OnlyEnvDilated(feat_in_size=4, feat_out_size=25)
        self.net.init_weights()
        self.net.load_state_dict(torch.load(join(
            '/home/mexxiie/prj/Geo_Semantic_fusion_nav_ws', 
            'step16-loss0.pth'
        ), map_location=self.device)['net_state'])
        self.net.to(self.device)
        self.net.eval()
        
        self.have_sem = True
        self.have_gem = False
        
        # 初始化奖励消息（严格匹配消息定义）
        self.reward_msg = OccupancyElevation()
        self.reward_msg.header.frame_id = 'map'
        self.reward_msg.occupancy.info.width = self.grid_size
        self.reward_msg.occupancy.info.height = self.grid_size
        self.reward_msg.occupancy.info.resolution = 0.1
        self.reward_msg.occupancy.data = [-1] * self.total_cells
        
        # 初始化所有float32数组（严格匹配消息定义的类型）
        self.reward_msg.height = [0.0] * self.total_cells
        self.reward_msg.roughness = [0.0] * self.total_cells
        self.reward_msg.cost_map = [0.0] * self.total_cells
        self.reward_msg.reward_cost = [0.0] * self.total_cells  # 重点修复
        
        self.feat = np.zeros((4, self.grid_size, self.grid_size), dtype=np.float32)
        
        # 创建发布者和订阅者
        self.pub_reward = self.create_publisher(OccupancyElevation, '/msg_local_reward', 10)
        self.sub_gem = self.create_subscription(
            OccupancyElevation, 
            '/msg_local_feature', 
            self.gem_callback, 
            10
        )
        
        self.get_logger().info("MEDIRL node initialized with correct message types")

    def gem_callback(self, gem_msg):
        self.have_gem = True
        
        # 复制元数据
        self.reward_msg.header = gem_msg.header
        self.reward_msg.occupancy = gem_msg.occupancy
        
        try:
            # 验证输入数据长度是否匹配（关键检查）
            if len(gem_msg.height) != self.total_cells:
                self.get_logger().error(f"Height data length mismatch: {len(gem_msg.height)} vs {self.total_cells}")
                return
                
            if len(gem_msg.roughness) != self.total_cells:
                self.get_logger().error(f"Roughness data length mismatch: {len(gem_msg.roughness)} vs {self.total_cells}")
                return
                
            if len(gem_msg.cost_map) != self.total_cells:
                self.get_logger().error(f"Cost map data length mismatch: {len(gem_msg.cost_map)} vs {self.total_cells}")
                return

            # 正确转换输入特征为float32
            self.feat[0] = np.array(gem_msg.cost_map, dtype=np.float32).reshape(self.grid_size, self.grid_size)
            self.feat[1] = np.array(gem_msg.height, dtype=np.float32).reshape(self.grid_size, self.grid_size)
            self.feat[2] = np.array(gem_msg.roughness, dtype=np.float32).reshape(self.grid_size, self.grid_size)
            occupancy = np.array(gem_msg.occupancy.data, dtype=np.int16).reshape(self.grid_size, self.grid_size)
            
            if self.have_sem and self.have_gem:
                self.feat_input(self.feat, occupancy)
                
        except Exception as e:
            self.get_logger().error(f"gem_callback failed: {str(e)}")

    def feat_input(self, feat, occupancy):
        try:
            slope_feature = feat[0].copy()
            step_feature = feat[1].copy()
            roughness_feature = feat[2].copy()

            # 特征预处理
            for i in range(3):
                feat[i] = 10 * feat[i]
            
            # 模型推理
            if self.debug_timing and self.device.type == 'cuda':
                torch.cuda.synchronize()
            with torch.inference_mode():
                # 转换为float32张量（与消息类型匹配）
                feat_tensor = torch.from_numpy(np.expand_dims(feat, axis=0)).to(
                    device=self.device, dtype=torch.float32)
                r_tensor = self.net(feat_tensor)
            if self.debug_timing and self.device.type == 'cuda':
                torch.cuda.synchronize()
            
            # 转换为numpy float32数组（关键步骤）
            r = -r_tensor[0].cpu().numpy().squeeze().astype(np.float32)
            
            # 验证输出维度（关键检查）
            if r.shape != (self.grid_size, self.grid_size):
                self.get_logger().error(f"Output shape mismatch: {r.shape} vs ({self.grid_size}, {self.grid_size})")
                # 尝试调整大小（应急处理）
                r = np.resize(r, (self.grid_size, self.grid_size))
            
            # 应用归一化并再次确保类型
            r = np.clip(r, 0, 8).astype(np.float32)
            r = (100 * (r / 8)).astype(np.float32)  # 简化计算，确保类型
            r, cleared_cells = filter_ground_artifact_reward(
                r,
                step_feature,
                roughness_feature,
                slope_feature,
                occupancy,
                **self.ground_artifact_filter,
            )
            if cleared_cells and self.debug_timing:
                self.get_logger().info(f"Filtered {cleared_cells} flat-ground reward artifacts")
            
            # 转换为Python原生float列表（严格匹配ROS要求）
            reward_cost_list = r.flatten().tolist()
            
            # 最终验证
            if len(reward_cost_list) != self.total_cells:
                self.get_logger().error(f"Reward cost length mismatch: {len(reward_cost_list)} vs {self.total_cells}")
                return

            # 检查所有元素类型
            if not all(isinstance(x, float) for x in reward_cost_list):
                self.get_logger().error(f"reward_cost中存在非float类型！前10类型示例: {[type(x) for x in reward_cost_list[:10]]}")
                return

            # 检查所有元素是否为有限数且在float32范围内
            if not all(np.isfinite(x) and -3.4e38 < x < 3.4e38 for x in reward_cost_list):
                self.get_logger().error(f"reward_cost中存在无穷大、NaN或超出float32范围的数值！前10数值: {reward_cost_list[:10]}")
                return
            
            # 赋值并发布
            self.reward_msg.reward_cost = reward_cost_list
            self.pub_reward.publish(self.reward_msg)
            
        except Exception as e:
            self.get_logger().error(f"feat_input failed: {str(e)}")


def main(args=None):
    rclpy.init(args=args)
    medirl_node = MEDIRL_env()
    rclpy.spin(medirl_node)
    medirl_node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
