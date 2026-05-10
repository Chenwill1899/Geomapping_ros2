#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: Ihab S. Mohamed, Vehicle Autonomy and Intelligence Lab - Indiana University, Bloomington, USA
"""
import yaml
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import csv

from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.animation import FuncAnimation, PillowWriter
import matplotlib.patches as patches
from pathlib import Path
import matplotlib.cm as cm
import seaborn as sns
from casadi import *


def map_axis_limits():
    return (0.0, 20.0), (-10.0, 10.0)


def map_axis_limits_from_config(config):
    simulation = (config or {}).get("simulation", {})
    if "map_size" not in simulation:
        return map_axis_limits()
    size_x, size_y = (float(value) for value in simulation["map_size"][:2])
    origin = simulation.get("map_origin", [0.0, 0.0])
    origin_x, origin_y = (float(value) for value in origin[:2])
    return (origin_x, origin_x + size_x), (origin_y, origin_y + size_y)


def animation_axis_limits(targets):
    return map_axis_limits()


#动图
def animate_simulation(dt ,ob_num_max, safety_dist, robot_r, atau, targets ,results_rootpath, sampled_us=[], optimal_us=None, cbf_type=0):

    # 读取CSV文件
    robot = pd.read_csv(results_rootpath + '/results.csv')
    ob_ = pd.read_csv(results_rootpath + '/obs_results.csv')

    states = []
    robot_size = len(robot)
    ob_size = len(ob_)
    if robot_size >= ob_size:
        size = ob_size
    else:
        size = robot_size

    print(f"size:= {size}")

    print(f"sampled_us: {len(sampled_us)}")
    print(f"optimal_us: {len(optimal_us)}")
    max_num = 60
    # 如果数据长度大于 max_num，适当跳过一些数据
    if size > max_num:
        indices = np.linspace(0, robot_size - 1, max_num).astype(int)
        reset_sampled_us = []
        reset_optimal_us = []
        for i in indices:
            states.append([robot['x'][i], robot['y'][i], robot['theta'][i], robot['dx'][i], robot['dy'][i]])
            reset_sampled_us.append(sampled_us[i])
            reset_optimal_us.append(optimal_us[i])
            size = max_num 

    # 为 ob 数据选择均匀的 max_num 个点
        ob_indices = np.linspace(0, ob_size - 1, max_num).astype(int)
        ob = ob_.iloc[ob_indices].reset_index(drop=True)  # 重采样后的障碍物数据

        sampled_us = np.copy(reset_sampled_us)
        optimal_us = np.copy(reset_optimal_us)
    else:
        for i in range(size):
            states.append([robot['x'][i], robot['y'][i], robot['theta'][i], robot['dx'][i], robot['dy'][i]])
        ob = ob_

    def kinematics(s, u, dt=0.1):
        # Set the control input: linear and angular velocities
        v, w = u
        # Update x, y, yaw of the vehicle

        s[0] += np.cos(s[2]) * v * dt
        s[1] += np.sin(s[2]) * v * dt
        s[2] += w * dt

        return s
    
    def get_predicted_trajecotories(current_state, control_sequence):
        predicted_state = [current_state]

        for ctrl in control_sequence:
            next_state = kinematics(predicted_state[-1], ctrl, dt)
            predicted_state.append(next_state.copy())
        return np.array(predicted_state) 
    
    def distance(p1, p2):
        """计算两点之间的欧几里得距离"""
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    def exp_ellipse(_s, _ob, NUMS=3, atau=1.0):
        """
        计算椭圆的中心、半长轴和半短轴。
        
        :param s: 机器人状态，包含 [x, y, theta, vx, vy]，其中 (x, y) 为位置，(vx, vy) 为速度
        :param ob: 障碍物状态，包含 [x, y, r, a, b, fx, fy]，其中 (x, y) 为障碍物位置，r 为半径，(fx, fy) 为焦点位置
        :param NUMS: 用于生成点集的数量，默认 100
        :param atau: 计算焦点位置的比例，默认 1.0
        :return: 更新后的障碍物参数
        """

        # 计算焦点 F1 和 F2
        F1 = np.array([_ob[0], _ob[1]])  # 焦点 1
        F2 = np.array([_ob[0] + atau * (_ob[5] - _s[3]), _ob[1] + atau * (_ob[6] - _s[4])])  # 焦点 2

        if np.array_equal(F1, F2):
            # 如果焦点重合，则无法计算角度
            _ob[3] = _ob[2]  # 更新半径
            _ob[4] = 0  # 更新角度
            a = _ob[2]
            b = _ob[2]
            return a,b
        else:
            deltaX = F2[0] - F1[0]
            deltaY = F2[1] - F1[1]
            angle_rad = np.arctan2(deltaY, deltaX)  # 计算旋转角度

        # 生成点集
        points1 = np.zeros((NUMS, 2))
        points2 = np.zeros((NUMS, 2))
        for i in range(NUMS):
            theta = angle_rad + np.pi * 0.7 + i * (np.pi * 0.6 / (NUMS-1))
            points1[i] = F1 + _ob[2] * np.array([np.cos(theta), np.sin(theta)])

            theta = angle_rad - np.pi * 0.3 + i * (np.pi * 0.6 / (NUMS-1))
            points2[i] = F2 + _ob[2] * np.array([np.cos(theta), np.sin(theta)])

        # 计算半长轴 a 和半短轴 b
        sum_distances = 0.0
        for i in range(NUMS):
            dist_F1 = distance(points1[i], F1)
            dist_F2 = distance(points1[i], F2)
            sum_distances += dist_F1 + dist_F2
        
        for i in range(NUMS):
            dist_F1_ = distance(points2[i], F1)
            dist_F2_ = distance(points2[i], F2)
            sum_distances += dist_F1_ + dist_F2_

        average_2a = sum_distances / (NUMS * 2)
        a = average_2a / 2.0  # 半长轴
        c = distance(F1, F2) / 2.0  # 焦距的一半
        err = _ob[2] - a + c  # 修正误差
        a += err
        a2 = a * a
        b = sqrt(a2 - c * c)  # 半短轴
        center_x = (F1[0]+F2[0])/2
        center_y = (F1[1]+F2[1])/2
        return center_x,center_y,a,b,angle_rad
    def check_danger(_s, _ob):
        """
        检查机器人是否与障碍物接近，判断是否需要扩展椭圆。
        
        参数:
            s (list): 机器人状态 [x, y, ..., vx, vy]，其中 x, y 是位置，vx, vy 是速度。
            ob (list): 障碍物状态 [x, y, r, ..., vx, vy]，其中 x, y 是位置，vx, vy 是速度。
        
        返回:
            bool: 如果机器人与障碍物接近，返回 True；否则返回 False。
        """
        # 计算障碍物与机器人在 x, y 方向的相对位置
        dx = _ob[0] - _s[0]
        dy = _ob[1] - _s[1]
        
        # 计算障碍物与机器人在 x, y 方向的相对速度
        dvx = _ob[5] - _s[3]
        dvy = _ob[6] - _s[4]
        
        # 计算位置和速度的点积
        dot_product = dx * dvx + dy * dvy
        
        # 如果点积小于0，说明两者正在靠近
        if dot_product < 0:
            return True
        else:
            return False

    def cal_cos_v(_s, _ob):
        """
        检查机器人是否与障碍物接近，判断是否需要扩展椭圆。
        
        参数:
            s (list): 机器人状态 [x, y, ..., vx, vy]，其中 x, y 是位置，vx, vy 是速度。
            ob (list): 障碍物状态 [x, y, r, ..., vx, vy]，其中 x, y 是位置，vx, vy 是速度。
        
        返回:
            bool: 如果机器人与障碍物接近，返回 True；否则返回 False。
        """
        # 计算障碍物与机器人在 x, y 方向的相对位置
        dx = _ob[0] - _s[0]
        dy = _ob[1] - _s[1]
        
        # 计算障碍物与机器人在 x, y 方向的相对速度
        dvx = _ob[5] - _s[3]
        dvy = _ob[6] - _s[4]
        
        dist = sqrt(dx*dx+dy*dy)
        dv = sqrt(dvx*dvx+dvy*dvy)
        # 计算位置和速度的点积
        dot_product = dx * dvx + dy * dvy
        
        cos_v = dot_product/dist

        cos_vx = cos_v*dx/dist
        cos_vy = cos_v*dy/dist

        tau = abs(0.05*dist/(cos_v+0.001))
        if(abs(tau)>atau):
            tau = atau


        return cos_vx,cos_vy,tau



    def update(frame, states):

        plt.gca().cla()  # Clear the current axes

        # Set axis limits
        # plt.xlim(-1, targets[0]+1)
        # plt.ylim(targets[1]-5, targets[1]+5)

        xlim, ylim = animation_axis_limits(targets)
        plt.xlim(*xlim)
        plt.ylim(*ylim)
        
        # Set aspect ratio to be equal, so each cell will be square-shaped
        plt.gca().set_aspect('equal', adjustable='box')

        # # Generate circle for CBF
        # if self.controller.obs.static_obstacles_on:
        #     for obs_i in self.controller.obs.static_obs:
        #         circle = plt.Circle(obs_i[:2], obs_i[2], color='grey', fill=True, linestyle='--', linewidth=2, alpha=0.5)
        #         circle_ = plt.Circle(obs_i[:2], obs_i[2] + self.controller.r + self.controller.safety_dist, 
        #                         edgecolor='red',  # 边界颜色为红色
        #                         facecolor='none',  # 内部无颜色
        #                         linestyle='--',  # 虚线
        #                         linewidth=1,  # 边界宽度
        #                         alpha=0.5)  # 透明度                    
        #         plt.gca().add_artist(circle)
        #         plt.gca().add_artist(circle_)
        print(f"frame={frame}")
        # num_columns = len(ob_.columns)
        # obs_num = int(num_columns/6) #/6是因为一个障碍物有六个数据 x y theta dx dy slack


        
        for id in range(ob_num_max):
            # _s = (robot['x'][frame], robot['y'][frame], robot['theta'][frame], robot['dx'][frame], robot['dy'][frame])
            _s = states[frame]
            _ob = (ob[f'x{id}'][frame], ob[f'y{id}'][frame], ob[f'r{id}'][frame], ob[f'r{id}'][frame], 0,ob[f'dx{id}'][frame], ob[f'dy{id}'][frame])
            if cbf_type == 4:
                if ((check_danger(_s, _ob)) & (atau != 0.0)):      
                    # 计算圆形障碍物的位置和半径
                    circle = plt.Circle([ob[f'x{id}'][frame], ob[f'y{id}'][frame]], 
                                        ob[f'r{id}'][frame], 
                                        color='grey',  # 颜色为灰色
                                        fill=True,  # 填充颜色
                                        linestyle='--',  # 虚线
                                        linewidth=2,  # 边界宽度
                                        alpha=0.5)  # 透明度
                    plt.gca().add_artist(circle)
                    center_x,center_y,a,b,theta = exp_ellipse(_s, _ob, atau=atau)
                    # print(f"_s:{_s}")
                    # print(f"_ob:{_ob}")
                    # print(f"a:{a},b:{b}")
                    # 计算椭圆区域的位置和大小
                    ellipse = patches.Ellipse([center_x, center_y], 
                                    width=a*2,  # 椭圆宽度
                                    height=b*2,  # 椭圆高度
                                    angle=theta,  # 旋转角度，如果需要的话，可以调整
                                    edgecolor='red',  # 边界颜色为红色
                                    facecolor='none',  # 内部无颜色
                                    linestyle='--',  # 虚线
                                    linewidth=1,  # 边界宽度
                                    alpha=0.5)  # 透明度
                    plt.gca().add_artist(ellipse)
                    # 计算扩展的安全区域椭圆
                    ellipse_safety = patches.Ellipse([center_x, center_y], 
                                            width=(a + safety_dist + robot_r) * 2,  # 扩展后的椭圆宽度
                                            height=(b + safety_dist + robot_r) * 2,  # 扩展后的椭圆高度
                                            angle=theta,  # 旋转角度
                                            edgecolor='blue',  # 边界颜色为蓝色
                                            facecolor='none',  # 内部无颜色
                                            linestyle='--',  # 虚线
                                            linewidth=1,  # 边界宽度
                                            alpha=0.5)  # 透明度
                    plt.gca().add_artist(ellipse_safety)                
                else:
                    circle = plt.Circle([ob[f'x{id}'][frame],ob[f'y{id}'][frame]], 
                                        ob[f'r{id}'][frame], 
                                        color='grey', # 颜色为灰色
                                        fill=True,  # 填充颜色
                                        linestyle='--', # 虚线
                                        linewidth=2, # 边界宽度
                                        alpha=0.5) # 透明度
                    plt.gca().add_artist(circle)
                    circle_ = plt.Circle([ob[f'x{id}'][frame],ob[f'y{id}'][frame]],
                                    ob[f'r{id}'][frame] + safety_dist + robot_r, 
                                    edgecolor='red',  # 边界颜色为红色
                                    facecolor='none',  # 内部无颜色
                                    linestyle='--',  # 虚线
                                    linewidth=1,  # 边界宽度
                                    alpha=0.5)  # 透明度
                    plt.gca().add_artist(circle_)
                    
            elif cbf_type == 2:    
                if ((check_danger(_s, _ob)) & (atau != 0.0)):
                    circle = plt.Circle([ob[f'x{id}'][frame],ob[f'y{id}'][frame]], 
                                        ob[f'r{id}'][frame], 
                                        color='grey', # 颜色为灰色
                                        fill=True,  # 填充颜色
                                        linestyle='--', # 虚线
                                        linewidth=2, # 边界宽度
                                        alpha=0.5) # 透明度
                    plt.gca().add_artist(circle)

                    circle_ = plt.Circle([ob[f'x{id}'][frame],ob[f'y{id}'][frame]],
                                    ob[f'r{id}'][frame] + safety_dist + robot_r, 
                                    edgecolor='red',  # 边界颜色为红色
                                    facecolor='none',  # 内部无颜色
                                    linestyle='--',  # 虚线
                                    linewidth=1,  # 边界宽度
                                    alpha=0.5)  # 透明度
                    plt.gca().add_artist(circle_)       

                    circle__ = plt.Circle([ob[f'x{id}'][frame] + (ob[f'dx{id}'][frame]-states[frame][3])*atau ,ob[f'y{id}'][frame] + (ob[f'dy{id}'][frame]-states[frame][4])*atau],
                                    ob[f'r{id}'][frame] + safety_dist + robot_r, 
                                    edgecolor='blue',  # 边界颜色为蓝色
                                    facecolor='none',  # 内部无颜色
                                    linestyle='--',  # 虚线
                                    linewidth=1,  # 边界宽度
                                    alpha=0.5)  # 透明度
                    plt.gca().add_artist(circle__)   

                else:    
                    circle = plt.Circle([ob[f'x{id}'][frame],ob[f'y{id}'][frame]], 
                                        ob[f'r{id}'][frame], 
                                        color='grey', # 颜色为灰色
                                        fill=True,  # 填充颜色
                                        linestyle='--', # 虚线
                                        linewidth=2, # 边界宽度
                                        alpha=0.5) # 透明度
                    plt.gca().add_artist(circle)
                    circle_ = plt.Circle([ob[f'x{id}'][frame],ob[f'y{id}'][frame]],
                                    ob[f'r{id}'][frame] + safety_dist + robot_r, 
                                    edgecolor='red',  # 边界颜色为红色
                                    facecolor='none',  # 内部无颜色
                                    linestyle='--',  # 虚线
                                    linewidth=1,  # 边界宽度
                                    alpha=0.5)  # 透明度
                    plt.gca().add_artist(circle_)

            elif cbf_type == 1:    
                circle = plt.Circle([ob[f'x{id}'][frame],ob[f'y{id}'][frame]], 
                                    ob[f'r{id}'][frame], 
                                    color='grey', # 颜色为灰色
                                    fill=True,  # 填充颜色
                                    linestyle='--', # 虚线
                                    linewidth=2, # 边界宽度
                                    alpha=0.5) # 透明度
                plt.gca().add_artist(circle)

                circle_ = plt.Circle([ob[f'x{id}'][frame],ob[f'y{id}'][frame]],
                                ob[f'r{id}'][frame] + safety_dist + robot_r, 
                                edgecolor='red',  # 边界颜色为红色
                                facecolor='none',  # 内部无颜色
                                linestyle='--',  # 虚线
                                linewidth=1,  # 边界宽度
                                alpha=0.5)  # 透明度
                plt.gca().add_artist(circle_)       
                cos_vx,cos_vy,tau = cal_cos_v(_s, _ob)
                circle__ = plt.Circle([ob[f'x{id}'][frame] + cos_vx*tau ,ob[f'y{id}'][frame] + cos_vy*atau],
                                ob[f'r{id}'][frame] + safety_dist + robot_r, 
                                edgecolor='blue',  # 边界颜色为蓝色
                                facecolor='none',  # 内部无颜色
                                linestyle='--',  # 虚线
                                linewidth=1,  # 边界宽度
                                alpha=0.5)  # 透明度
                plt.gca().add_artist(circle__)   

            else:
                circle = plt.Circle([ob[f'x{id}'][frame],ob[f'y{id}'][frame]], 
                                    ob[f'r{id}'][frame], 
                                    color='grey', # 颜色为灰色
                                    fill=True,  # 填充颜色
                                    linestyle='--', # 虚线
                                    linewidth=2, # 边界宽度
                                    alpha=0.5) # 透明度
                plt.gca().add_artist(circle)

                circle_ = plt.Circle([ob[f'x{id}'][frame],ob[f'y{id}'][frame]],
                                ob[f'r{id}'][frame] + safety_dist + robot_r, 
                                edgecolor='red',  # 边界颜色为红色
                                facecolor='none',  # 内部无颜色
                                linestyle='--',  # 虚线
                                linewidth=1,  # 边界宽度
                                alpha=0.5)  # 透明度
                plt.gca().add_artist(circle_)       
        
        # Plot MPPI trajectories
        if len(sampled_us) > 0:
            # print(f"(sampled_us):{sampled_us[frame].shape}")
            # pred_trajs = [get_predicted_trajecotories(states[frame], sampled_us[frame][i]) for i in range(len(sampled_us))]
            num_trajs_plotted = np.minimum(50,  sampled_us[frame].shape[0])  # plot maximum 50 trajs per step
            for idx in range(num_trajs_plotted):
                # print(sampled_us[frame][idx])
                cur_state = np.copy(states[frame])
                pred_traj = get_predicted_trajecotories(cur_state, sampled_us[frame][idx])
                x_pos, y_pos = pred_traj[:, 0], pred_traj[:, 1]
                plt.plot(x_pos, y_pos, color="k", alpha=0.1)

        # plot optimal predicted trajectory
        if optimal_us is not None:
            # print(optimal_us[frame])
            cur_state = np.copy(states[frame])
            opt_pred_traj = get_predicted_trajecotories(cur_state, optimal_us[frame])
            x_pos, y_pos = opt_pred_traj[:, 0], opt_pred_traj[:, 1]
            plt.plot(x_pos, y_pos, color="orange", alpha=0.4, label="optimized traj.")
            plt.legend(fontsize=4)
        x, y, theta, dx, dy = states[frame]
        dx = 0.1 * np.cos(theta)
        dy = 0.1 * np.sin(theta)

        # Plot the trajectory up to the current frame
        plt.plot([state[0] for state in states[:frame+1]], [state[1] for state in states[:frame+1]], '-o', markersize=4, alpha=0.5)

        # Plot the orientation at the current frame
        plt.arrow(x, y, dx, dy, head_width=0.3, head_length=0.3, fc='red', ec='red')

        plt.scatter(0.0, 0.0, s=100, color="green", alpha=0.4, label="start")
        plt.scatter(targets[0], targets[1], s=100, color="purple", alpha=0.4, label="target")

        plt.title('Simulation Result with Car Orientation')
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.grid(True)
        plt.legend(loc="upper left")

    fig = plt.figure(figsize=(6, 6))
    anim = FuncAnimation(fig, update, frames=size, fargs=(states, ), interval=100, blit=False)
    # plt.show()
            # 指定要创建的文件夹路径
    folder_path = Path(results_rootpath)

    # 使用 pathlib.Path 的 mkdir 方法创建文件夹
    # exist_ok=True 参数表示如果文件夹已存在，则不抛出异常
    try:
        folder_path.mkdir(parents=True, exist_ok=True)
        print(f"文件夹 {folder_path} 创建成功或已存在。")
    except OSError as error:
        print(f"创建文件夹 {folder_path} 时出错：{error}")
    # plt.show()
    save_path = results_rootpath + '/animation.gif'
    anim.save(save_path, writer=PillowWriter(fps=5))
    print("-----------------------------")
    print("---------GIF保存完毕-----------")
    print("-----------------------------")


"""  绘制 路径图 """
def pathPlotting(ob_num_max, robot_r, targets ,results_rootpath):

    # 读取CSV文件
    robot_ = pd.read_csv(results_rootpath + '/results.csv')
    ob_ = pd.read_csv(results_rootpath + '/obs_results.csv')

    states = []
    robot_size = len(robot_)
    ob_size = len(ob_)
    if robot_size >= ob_size:
        size = ob_size
    else:
        size = robot_size
    # 如果数据长度大于 max_num，适当跳过一些数据
    max_num = 300
    if size > max_num:
        robot_indices = np.linspace(0, robot_size - 1, max_num).astype(int)
        robot = robot_.iloc[robot_indices].reset_index(drop=True)  # 重采样后的障碍物数据


    # 为 ob 数据选择均匀的 max_num 个点
        ob_indices = np.linspace(0, ob_size - 1, max_num).astype(int)
        ob = ob_.iloc[ob_indices].reset_index(drop=True)  # 重采样后的障碍物数据
    else:
        robot = robot_
        ob = ob_

    """Plots the robot path in the x-y plane."""
    # 全局设置（背景保持白色，网格参数调整）
    plt.rcParams.update({
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'savefig.facecolor': 'white',
        'axes.grid': True,               # 开启网格线（<a target="_blank" href="https://www.cnblogs.com/ivanlee717/p/17483909.html" class="hitref" data-title="matplot画网格线分层级的用法- ivanlee717 - 博客园" data-snippet='Matplotlib 的 grid() 函数可以用于绘制网格线。 该函数的常用参数如下： - b 设置是否显示网格线。 可以取值为True 或False。' data-url="https://www.cnblogs.com/ivanlee717/p/17483909.html">1</a><a target="_blank" href="http://www.runoob.com/matplotlib/matplotlib-grid.html" class="hitref" data-title="Matplotlib 网格线 - 菜鸟教程" data-snippet='Matplotlib 网格线我们可以使用pyplot 中的grid() 方法来设置图表中的网格线。 grid() 方法语法格式如下： matplotlib.pyplot.grid(b=None, which=&#39;major&#39;, axis=&#39;both&#39; ...' data-url="http://www.runoob.com/matplotlib/matplotlib-grid.html">7</a>）
        'grid.color': '#e0e0e0',         # 浅灰色（HEX格式，<a target="_blank" href="https://wenku.csdn.net/answer/63f85veryr" class="hitref" data-title="matplotlib 浅灰色 - CSDN文库" data-snippet='要设置matplotlib绘图的浅灰色，可以使用RGB颜色代码表示。以下是一个示例代码，用于设置浅灰色背景： import matplotlib.pyplot as plt # 设置浅灰色RGB ...' data-url="https://wenku.csdn.net/answer/63f85veryr">4</a><a target="_blank" href="http://www.runoob.com/matplotlib/matplotlib-grid.html" class="hitref" data-title="Matplotlib 网格线 - 菜鸟教程" data-snippet='Matplotlib 网格线我们可以使用pyplot 中的grid() 方法来设置图表中的网格线。 grid() 方法语法格式如下： matplotlib.pyplot.grid(b=None, which=&#39;major&#39;, axis=&#39;both&#39; ...' data-url="http://www.runoob.com/matplotlib/matplotlib-grid.html">7</a>）
        'grid.alpha': 0.3,               # 透明度（<a target="_blank" href="https://www.python91.com/python_Matplotlib/187.html" class="hitref" data-title="matplotlib中的grid()方法如何设置网格线外观 - python编程" data-snippet='设置透明度：可以使用alpha参数来设置网格线的透明度，例如grid(alpha=0.5)可以将网格线的透明度设置为0.5。 综合使用这些参数可以实现各种网格线的外观设置 ...' data-url="https://www.python91.com/python_Matplotlib/187.html">10</a>）
        'grid.linestyle': '--',          # 虚线样式（可选）
        'axes.axisbelow': True           # 网格线在数据下方（<a target="_blank" href="https://www.osgeo.cn/matplotlib/users/dflt_style_changes.html" class="hitref" data-title="更改为默认样式— Matplotlib 3.3.3 文档" data-snippet='记号和网格现在绘制在实体元素（如填充轮廓）的上方，但绘制在线的下方。若要返回到上一个绘制线上方刻度和网格的行为，请设置 rcParams[&#39;axes.axisbelow&#39;] = False .' data-url="https://www.osgeo.cn/matplotlib/users/dflt_style_changes.html">6</a>）
    })

    # 保持seaborn主题但覆盖网格颜色（可选）
    sns.set_theme(style="whitegrid", rc={'grid.color': '#e0e0e0', 'grid.alpha': 0.3})

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(robot_['x'][:], robot_['y'][:], color='c', label="C-RCBF")
    
    ax.set_xlabel('x position [m]', fontsize=10, labelpad=0)
    ax.set_ylabel('y position [m]', fontsize=10, labelpad=0)
    ax.tick_params(axis='both', labelsize=10, colors='#444444')
    # plt.title("Robot path")
    plt.tight_layout()


    xlim, ylim = map_axis_limits()
    plt.xlim(*xlim)
    plt.ylim(*ylim)
    ax.set_aspect('equal', adjustable='box')  # 保持比例同时锁定坐标范
   # 机器人初始位置
    ax.plot(0.0, 0.0, 'r.', label="Initial position")
    ax.plot(targets[0], targets[1], 'g*', 
            markersize=15,  # 调整此处数值控制大小
            markeredgewidth=1.2,  # 边缘线宽（可选）
            label="Goal")
    # 障碍物
    for id in range(ob_num_max):
        # 绘制初始障碍物位置
        circle = plt.Circle([ob[f'x{id}'][0],ob[f'y{id}'][0]], 
                                    ob[f'r{id}'][0], 
                                        color='#2E4053', # 颜色为灰色
                                            fill=True,  # 填充颜色
                                                # linestyle='--', # 虚线
                                                    linewidth=2, # 边界宽度
                                                        alpha=0.2) # 透明度
        plt.gca().add_artist(circle)

        circle_ = plt.Circle([ob[f'x{id}'][0],ob[f'y{id}'][0]],
                                ob[f'r{id}'][0] + 0.3 + robot_r, 
                                    edgecolor='#6C3483',  # 边界颜色为红色
                                        facecolor='none',  # 内部无颜色
                                            linestyle='--',  # 虚线
                                                linewidth=1,  # 边界宽度
                                                    alpha=0.2)  # 透明度
        plt.gca().add_artist(circle_)    
                # 计算每个时间步的距离
        distances = np.sqrt(
            (robot_['x'] - ob[f'x{id}'])**2 + 
            (robot_['y'] - ob[f'y{id}'])**2
        )
        # 找到最小距离的索引
        t_min = distances.idxmin()

        # 在轨迹绘制部分修改为：
        # 计算终止索引（不超过数据长度）
        end_idx = min(t_min + 20, len(ob[f'x{id}'])-1)

        # 创建渐变映射（基于实际绘制长度）
        gray_cmap = LinearSegmentedColormap.from_list(
            'gray_gradient', ['#AEB6BF', '#000000'], N=end_idx
        )

        # 绘制渐变轨迹（仅到end_idx）
        for i in range(1, end_idx+1):
            t = i / end_idx  # 归一化到实际绘制长度
            ax.plot(ob[f'x{id}'][i-1:i+1], 
                    ob[f'y{id}'][i-1:i+1],
                    color=gray_cmap(t), 
                    linestyle=':', 
                    linewidth=2.2,
                    alpha=0.5,
                    zorder=1)

        # 在轨迹末端添加方向箭头
        if end_idx > 1:
            # 计算最后两个点的方向向量
            dx = ob[f'x{id}'][end_idx] - ob[f'x{id}'][end_idx-1]
            dy = ob[f'y{id}'][end_idx] - ob[f'y{id}'][end_idx-1]
            
            # 绘制箭头（缩放系数0.8使箭头更紧凑）
            ax.arrow(ob[f'x{id}'][end_idx-1], 
                    ob[f'y{id}'][end_idx-1],
                    # dx*0.8, dy*0.8,
                    -0.3, 0.0,
                    head_width=0.3,
                    head_length=0.5,
                    fc=gray_cmap(1.0),  # 使用最深颜色
                    ec=gray_cmap(1.0),
                    linewidth=1.8,
                    alpha=0.7,
                    zorder=2)



        # 获取障碍物属性
        x_ob = ob[f'x{id}'][t_min]
        y_ob = ob[f'y{id}'][t_min]
        r_ob = ob[f'r{id}'][t_min]

        # 绘制 最小距离 的 障碍物位置
        circle = plt.Circle([x_ob,y_ob], 
                                    r_ob, 
                                        color='#2E4053', # 颜色为灰色
                                            fill=True,  # 填充颜色
                                                # linestyle='--', # 虚线
                                                    linewidth=1, # 边界宽度
                                                        alpha=0.6) # 透明度
        plt.gca().add_artist(circle)

        circle_ = plt.Circle([x_ob,y_ob],
                                r_ob + 0.3 + robot_r, 
                                    edgecolor='#6C3483',  # 边界颜色为红色
                                        facecolor='none',  # 内部无颜色
                                            linestyle='--',  # 虚线
                                                linewidth=1,  # 边界宽度
                                                    alpha=0.6)  # 透明度
        plt.gca().add_artist(circle_)    

        # 计算速度方向（基于位置变化）
        if t_min < len(ob) - 1:
            vx = ob[f'x{id}'][t_min+1] - ob[f'x{id}'][t_min]
            vy = ob[f'y{id}'][t_min+1] - ob[f'y{id}'][t_min]
        else:
            vx = ob[f'x{id}'][t_min] - ob[f'x{id}'][t_min-1]
            vy = ob[f'y{id}'][t_min] - ob[f'y{id}'][t_min-1]

        # 归一化速度方向（保持箭头长度一致）
        norm = np.hypot(vx, vy)
        if norm > 0:
            vx /= norm
            vy /= norm

        # 修改速度箭头部分为：
        ax.arrow(
            x_ob, y_ob, 
            # vx*0.5, vy*0.5,
            -0.2, 0.0,
            head_width=0.25, 
            color='#CD5C5C',     # 橙色箭头
            linewidth=1.5,      # 加粗线宽
            alpha=0.9,          # 提高透明度
            zorder=4,
            head_starts_at_zero=True  # 箭头从起点开始
        )

    legend_elements = [
        Line2D([0], [0], 
            color='#6C3483', 
            lw=1.5,
            linestyle='--',
            markersize=10,      # 标记大小
            markeredgewidth=0,
            label='Valid radius'),             
        # 渐变轨迹图示
        Line2D([0], [0],
            color='#808080',  # 中间灰色
            linestyle=':',
            linewidth=2.2,
            marker='>',       # 末端箭头
            markersize=10,
            markeredgecolor='#404040',
            markerfacecolor='#202020',
            alpha=0.7,
            label='Path'),              
        Line2D([0], [0], 
            color='#CD5C5C', 
            lw=1.5,
            marker='>',         # 添加箭头标记
            markersize=10,      # 标记大小
            markeredgewidth=0,
            label='Acceleration'),]   
    
    # 创建第一个图例（原有元素，右下角）    
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))  
    original_legend = ax.legend(
        handles=by_label.values(),
        labels=by_label.keys(),
        loc='upper left',
        framealpha=0.9,
        title_fontsize=10,
        borderpad=0.3         # 图例内边距
    )

    ax.add_artist(original_legend)  # 必须保留原有图例对象
    ax.legend(
        handles=legend_elements,
        loc='lower right',
        framealpha=0.9,
        title='Obstacles',
        title_fontsize=10,
        borderpad=0.3         # 图例内边距
    )

    ax.set_facecolor('white')  # 设置坐标轴背景为白色

    save_path = results_rootpath + '/path.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight', transparent=False)
    plt.close()




def plot_cbf(dt, ob_num_max, robot_r, safety_dist, results_rootpath, static_obs_list = [], IF_static_obstacle = False):
    """Plots the CBF values."""

    # 读取CSV文件
    robot_ = pd.read_csv(results_rootpath + '/results.csv')
    ob_ = pd.read_csv(results_rootpath + '/obs_results.csv')

    robot_size = len(robot_)
    ob_size = len(ob_)
    if robot_size >= ob_size:
        size = ob_size
    else:
        size = robot_size
    # 如果数据长度大于 max_num，适当跳过一些数据
    max_num = 300
    if size > max_num:
        robot_indices = np.linspace(0, robot_size - 1, max_num).astype(int)
        robot = robot_.iloc[robot_indices].reset_index(drop=True)  # 重采样后的障碍物数据
        size = max_num

    # 为 ob 数据选择均匀的 max_num 个点
        ob_indices = np.linspace(0, ob_size - 1, max_num).astype(int)
        ob = ob_.iloc[ob_indices].reset_index(drop=True)  # 重采样后的障碍物数据
    else:
        robot = robot_
        ob = ob_

    def cbf_h(s, obs, robot_r, safety_dist):
        h = sqrt((s[0]-obs[0])**2 + (s[1]-obs[1])**2) - (robot_r + obs[2])
        return h

    cbfs = []
    if IF_static_obstacle:
        for i in range(len(static_obs_list)):
            h = []
            for j in range(size):
                s = [robot['x'][j], robot['y'][j]]
                # static_obs_list: x y r 
                h.append(cbf_h(s, static_obs_list[i], robot_r, safety_dist))
            cbfs.append(h)

    # 动态障碍
    cbfs_mov = []
    d_cbfs_mov = []
    distence_mov = []

    for id in range(ob_num_max):
        h = []
        dh = []
        dist = []
        # for x in self.mpc.data['_x']:
        for j in range(size):
            obs = (ob[f'x{id}'][j], ob[f'y{id}'][j], ob[f'r{id}'][j])
            s = [robot['x'][j], robot['y'][j]]
            h.append(cbf_h(s, obs, robot_r, safety_dist))
            h1 = h[1:]
            h1.append(h[-1])
            dh = (np.array(h1)-np.array(h))/dt
            distance_p = sqrt((s[0] - obs[0])**2 + (s[1] - obs[1])**2) - (robot_r + obs[2])
            dist.append(distance_p)
            if (distance_p - safety_dist)<= 0:
                print('------ clash!!! ------') 
                print(f"distance_p:{distance_p}")
            # acc_x =  ((self.mpc.data['_x'][j][3] - self.mpc.data['_x'][j-1][3])/config.Ts)**2
            # acc_y =  ((self.mpc.data['_x'][j][4] - self.mpc.data['_x'][j-1][4])/config.Ts)**2
            # acc = sqrt(acc_x+acc_y) 
            # print('acc: ', acc)
        cbfs_mov.append(h)
        d_cbfs_mov.append(dh)
        distence_mov.append(dist)  
            
            
    sns.set_theme()
    fig, ax = plt.subplots(figsize=(9, 5))
    for i in range(len(cbfs)):
        ax.plot(cbfs[i], label="h_obs"+str(i))
    for i in range(len(cbfs_mov)):
        ax.plot(cbfs_mov[i], label="h_mov_obs"+str(i))
    #for i in range(len(d_cbfs_mov)):
    #   ax.plot(d_cbfs_mov[i], label="dh_mov_obs"+str(i))
    #实际距离
    # for i in range(len(distence_mov)):
    #     ax.plot(distence_mov[i], label="dist_mov_obs"+str(i)) 
        
    plt.axhline(y=0, color='k', linestyle='--')
    plt.axhline(y=safety_dist, color='r', linestyle='--')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('h [m]')
    plt.title("CBF Values")
    plt.tight_layout()
    plt.legend()

    # Save the figure as an image
    plt.savefig(results_rootpath + '/cbf_dist.png')
    plt.close()

def cla_mindist(robot_r, ob_num_max, results_rootpath):
    """Claculate the MinDistance values."""

    # 读取CSV文件
    robot = pd.read_csv(results_rootpath + '/results.csv')
    ob = pd.read_csv(results_rootpath + '/obs_results.csv')

    robot_size = len(robot)
    ob_size = len(ob)
    if robot_size >= ob_size:
        size = ob_size
    else:
        size = robot_size
        
    # 取最小距离    
    min_dist = 111.0      

    for id in range(ob_num_max):
        for i in range(size):
            obs = (ob[f'x{id}'][i], ob[f'y{id}'][i], ob[f'r{id}'][i])

            dist = sqrt((robot['x'][i] - obs[0])**2 + (robot['y'][i] - obs[1])**2) - (robot_r + obs[2])

            if dist < 0:
                print("--------clash---------")
            if dist < min_dist:
                min_dist = dist

    return min_dist



''' @brief: Plotting the current state of the robot (x, y, theta)'''
def slackPlotting(slack_history, obs_num_max, results_rootpath):
    slack_length = len(slack_history)

    # Make sure obs_num_max is an integer
    if hasattr(obs_num_max, 'item'):
        obs_num_max = obs_num_max.item()  # Extract integer if it's an array
    
    print(f'obs_num_max: {obs_num_max}, type: {type(obs_num_max)}')  # Debugging line

    # Initialize a color map
    cmap = cm.get_cmap('tab10', obs_num_max)  # Using the 'tab10' colormap for distinct colors

    # States
    plt.figure(figsize=(10, 2 * obs_num_max))  # Adjust the size of the figure based on the number of slacks
    plt.subplots_adjust(hspace=0.4)  # Adjust spacing between subplots
    plt.suptitle('Obstacle Slack', fontsize=16)  # Title for the whole figure

    # Plot each slack curve on a separate subplot
    for id in range(obs_num_max):
        plt.subplot(obs_num_max, 1, id + 1)  # Create a subplot for each slack
        plt.grid(True)
        slack = [s[id] for s in slack_history]  # Extract the slack values for the current id
        plt.plot(range(slack_length), slack, label=f'Slack {id+1}', color=cmap(id))  # Plot with distinct color
        plt.legend(loc='upper right')
        plt.xlabel('Time Step')
        plt.ylabel('Slack Value')
        plt.title(f'Slack {id+1}')  # Title for each individual subplot

    # Save the figure as an image
    plt.savefig(results_rootpath + '/slacks.png')
    plt.close()
    #plt.show()



''' @brief: Plotting the current state of the robot (x, y, theta)'''
def statePlotting(state_history, results_rootpath):
    state_length = len(state_history)

    x, y, Yaw = [s[0]
                 for s in state_history], [s[1] for s in state_history
                                           ], [s[2] for s in state_history]
    # States
    plt.figure()
    plt.subplots_adjust(hspace=0.4)
    plt.title('Jackal Full States')
    plt.subplot(211)
    plt.grid(True)
    plt.plot(range(state_length), x, 'r--', range(state_length), y, 'b--')
    plt.legend(('$x$ [m]', '$y$ [m]'))
    plt.subplot(212)
    plt.grid(True)
    plt.plot(range(state_length), Yaw, 'g-')
    plt.legend(('$\theta$ [rad.]'))
    plt.savefig(results_rootpath + '/states.png')
    plt.close()
    #plt.show()



''' @brief: Plotting the control input (linear and angular velocities)'''
def controlPlotting(control_history, results_rootpath):
    control_length = len(control_history)

    U0, U1 = [U[0] for U in control_history], [U[1] for U in control_history]

    plt.figure()
    plt.subplots_adjust(hspace=0.4)
    plt.subplot(211)
    plt.grid(True)
    plt.plot(range(control_length), U0, '-', label='$v_x$')
    plt.legend(loc='upper right')
    #plt.title('Control Signal')
    plt.subplot(212)
    plt.grid(True)
    plt.plot(range(control_length), U1, 'r-', label='$w_z$')
    plt.legend(loc='upper right')
    plt.savefig(results_rootpath + '/control.png')
    plt.close()

''' @brief: Plotting the running cost (state and control costs), the minimum sampled cost, 
           and the average execution time of MPPI per iteration '''
def costPlotting(state_cost_history, control_cost_history, min_cost_history,
                 iter_time_history, results_rootpath):
    cost_length = len(state_cost_history)
    iter_length = len(iter_time_history)

    plt.figure()
    plt.subplots_adjust(hspace=0.4)
    plt.subplot(2, 2, 1)
    plt.grid(True)
    plt.plot(range(cost_length), state_cost_history, '-')
    plt.legend(('State Cost'))
    #plt.xlabel('Iterations')
    plt.ylabel('Average Running Cost')
    #plt.title('Average Cost of the Optimal Control $u^*_0$')

    plt.subplot(2, 2, 2)
    plt.grid(True)
    plt.plot(range(cost_length), min_cost_history, '-')
    plt.legend(('Min Cost'))
    #plt.xlabel('Iterations')
    plt.ylabel('Min Cost')

    plt.subplot(2, 2, 3)
    plt.grid(True)
    plt.plot(range(cost_length), control_cost_history, '-')
    plt.legend(('Control Cost'))
    #plt.xlabel('Iterations')
    plt.ylabel('Ctr. Cost')

    # computation time
    plt.subplot(2, 2, 4)
    plt.grid(True)
    plt.plot(range(iter_length), iter_time_history, '-')
    plt.xlabel('Iter')
    plt.ylabel('Time [s]')
    #plt.title('GPU Computation Time per Iteration')
    plt.savefig(results_rootpath + '/costs.png')

''' @brief: Plotting the robot trajectory generated by the controller within a given map'''
def trajectoryPath(state_history, obstacle_grid, map_size, results_rootpath):
    x, y = [s[0] for s in state_history], [s[1] for s in state_history]
    # Get the xy indices of the obstacles map
    obstacle_idx = grid2index(obstacle_grid, map_size)

    plt.figure()
    plt.plot(x, y, color='k', linewidth=1)
    plt.scatter([o[1] for o in obstacle_idx], [o[0] for o in obstacle_idx],
                s=3,
                c='r')
    plt.savefig(results_rootpath + '/obs_map.png')
    #plt.show()
    plt.close()

''' @brief: Retrieving the controllers' parameters, the costmap information, and summary of the performance''' 
def testSummaryMPPI(init_pose, pose_desired, max_linear_velocity,
                    max_angular_velocity, time_horizon, hz, weights,
                    num_trajectories, R, Sigma_du, exploration_variance, gamma, 
                    slack_weight, std_n_sla, atau,
                    beta_1, beta_2, beta_3, check_dist,
                    SG_window, SG_PolyOrder, Distibution_Type, CBF_Type, local_minima,
                    violate_ctrl_const, pathLength, var_v,average_v,mindist,
                    av_t_mppi, real_time_mppi, sys_run_time, robot_run_time,
                    results_rootpath):
    # For MPPI
    summary = [{
        'Initial Pose [m & Rad]':
        '[' + str(init_pose[0]) + ',' + str(init_pose[1]) + ',' +
        str(init_pose[2]) + ']'
    }, {
        'Desired Pose [m & Rad]':
        '[' + str(pose_desired[0]) + ',' + str(pose_desired[1]) + ',' +
        str(pose_desired[2]) + ']'
    }, {
        'Max linear Velocity': str(max_linear_velocity)
    }, {
        'Max Angular Velocity': str(max_angular_velocity)
    }, {
        'Parameters': 'SG Filter',
        'Window Length': str(SG_window),
        'Poly. Order': str(SG_PolyOrder)
    }, {
        'Distibution Type (0: Normal, 1: Normal & Log-Normal, 2: ODIA)': Distibution_Type,
        'CBF Type (0: CE, 1: COS_E, 2: PJ_E, 3: DC, 4: Ellispe )': CBF_Type
    }, {
        'Control Parameters': 'MPPI/log-MPPI',
        'Time Horizon [s]': str(time_horizon),
        'Sampling Rate [Hz]': str(hz),
        'Num of Trajectories': num_trajectories,
        'Weights': str(weights),
        'R': str(R),
        'Sigma_du': str(Sigma_du),
        'Exploration Variance': str(exploration_variance),
        'Lambda': str(gamma)
    }, {
        'slack_weight': str(slack_weight),
        'std_n_sla': str(std_n_sla),
        'atau': str(atau),
        'beta_1': str(beta_1),
        'beta_2': str(beta_2),
        'beta_3': str(beta_3),
        'check_dist': str(check_dist),
    },{
        'Traveled Distance [m]': str(pathLength)
    }, {
        'velocity_var [m2/s2]': str(var_v)
    }, {
        'velocity_average [m/s]': str(average_v)
    }, {
        'Min distance [m]': str(mindist)
    }, {
        'Average MPPI Excution Time [ms]': str(av_t_mppi)
    }, {        
        'Once sys run Time [s]': str(sys_run_time)
    }, {
        'Once robot run Time [s]': str(robot_run_time)
    }, {
        'Real Time MPPI': real_time_mppi
    }, {
        'Reach Local Minima': local_minima
    }, {
        'Violate Ctrl Constraints': violate_ctrl_const
    }]

    with open(results_rootpath + '/test_summary.yaml', 'w') as f:
        yaml.dump(summary, f, sort_keys=False)

''' @brief: Saving the results of the control mission in txt file''' 
def save_results(state_history, desired_state_history, state_cost_history,
                 min_cost_history, control_history, iter_time_history,
                 results_rootpath):
    # Get Control inputs
    v, w = [U[0] for U in control_history], [U[1] for U in control_history]
    x, y, theta, dx, dy = [s[0]for s in state_history], [s[1] for s in state_history
                                                         ], [s[2] for s in state_history
                                                             ], [s[3] for s in state_history
                                                                ], [s[4] for s in state_history]
    x_d, y_d, theta_d = [s[0] for s in desired_state_history
                         ], [s[1] for s in desired_state_history
                             ], [s[2] for s in desired_state_history]

    df = pd.DataFrame()
    df['x'] = x
    df['y'] = y
    df['theta'] = np.asarray(theta) 
    df['dx'] = dx
    df['dy'] = dy    
    df['x_d'] = x_d
    df['y_d'] = y_d
    df['theta_d'] = np.asarray(theta_d) 
    df['v'] = v
    df['w'] = w
    df['state_cost'] = state_cost_history
    df['min_Traj_cost'] = min_cost_history
    df['t_mppi'] = iter_time_history
    df.to_csv(results_rootpath + '/results.csv', index=False)

''' @brief: Saving the results of the control mission in txt file''' 
def save_obs_results(obs_num_max, results_rootpath, obs_state_history, obs_slack_history = []):
    # obs_state_history 包含 x,y,r,dx,dy
    # obs_slack_history 包含 松弛变量
    # 
    # Get Control inputs
    df = pd.DataFrame()

    for id in range(obs_num_max):
        x, y, r, dx, dy = [s[id][0] for s in obs_state_history
                        ], [s[id][1] for s in obs_state_history
                            ], [s[id][2] for s in obs_state_history
                                    ], [s[id][5] for s in obs_state_history
                                        ], [s[id][6] for s in obs_state_history] 
         

        df[f'x{id}'] = x
        df[f'y{id}'] = y
        df[f'r{id}'] = r
        df[f'dx{id}'] = dx
        df[f'dy{id}'] = dy
        if len(obs_slack_history) != 0:
            slack = [sla[id] for sla in obs_slack_history]
            df[f'slack{id}'] = slack


    df.to_csv(results_rootpath + '/obs_results.csv', index=False)

def save_time_results(results_rootpath, mppi_time_history):
    df = pd.DataFrame()
    df['t_mppi'] = mppi_time_history
    df.to_csv(results_rootpath + '/time_results.csv', index=False)


''' @brief: Retrieving the travelled distance by the robot'''
def getTraveledDistances(folder_path):
    TraveledDistances = readDataFromFile(folder_path +
                                         '/average_travelled_distance.csv')
    return TraveledDistances

''' @brief: Retrieving the summary over the intensive simulations'''
def intensiveSimulationSummary(folder_path):
    counter_unreachableGoals, counter_reachableGoals, counter_successful_tests, counter_semi_successful_tests, counter_real_time_MPPI, counter_unreal_time_MPPI, violate_Ctrl_constraints = 0, 0, 0, 0, 0, 0, 0
    folder_unreachableGoals, folder_reachableGoals, folder_success_tests, folder_semi_success_tests, folder_violate_Ctrl_constraints, folder_real_time_MPPI, folder_unreal_time_MPPI = [], [], [], [], [], [], []
    save_average_real_time_MPPI, save_average_unreal_time_MPPI, save_average_travelled_distance = [], [], []
    # List all directories in a certain folder
    for root, dirs, files in os.walk(folder_path, topdown=True):
        dirs.sort()  # sort files in order
        for name in dirs:
            roots = os.path.join(root, name)
            data = getDataFromYamlFile(roots + '/test_summary.yaml')

            # Extract the relevent information
            Traveled_Distance = data[8]
            Traveled_Distance = Traveled_Distance['Traveled Distance [m]']
            save_average_travelled_distance.append(Traveled_Distance)

            Average_MPPI_Excution_Time = data[9]
            Average_MPPI_Excution_Time = Average_MPPI_Excution_Time[
                'Average MPPI Excution Time [ms]']

            Real_Time_MPPI = data[10]
            Real_Time_MPPI = Real_Time_MPPI['Real Time MPPI']

            Reach_Local_Minima = data[11]
            Reach_Local_Minima = Reach_Local_Minima['Reach Local Minima']

            Violate_Ctrl_Constraints = data[12]
            Violate_Ctrl_Constraints = Violate_Ctrl_Constraints[
                'Violate Ctrl Constraints']

            # Count the unreachable goal tests and save the corresponding folder
            if Reach_Local_Minima == True:
                counter_unreachableGoals += 1
                folder_unreachableGoals.append(name)
            else:
                counter_reachableGoals += 1
                folder_reachableGoals.append(name)

            if Violate_Ctrl_Constraints == True:
                violate_Ctrl_constraints += 1
                folder_violate_Ctrl_constraints.append(name)

            if Real_Time_MPPI == True:
                counter_real_time_MPPI += 1
                folder_real_time_MPPI.append(name)
                save_average_real_time_MPPI.append(Average_MPPI_Excution_Time)
            else:
                counter_unreal_time_MPPI += 1
                folder_unreal_time_MPPI.append(name)
                save_average_unreal_time_MPPI.append(
                    Average_MPPI_Excution_Time)

            if Violate_Ctrl_Constraints == False and Reach_Local_Minima == False:
                counter_successful_tests += 1
                folder_success_tests.append(name)

            if Violate_Ctrl_Constraints == True and Reach_Local_Minima == False:
                counter_semi_successful_tests += 1
                folder_semi_success_tests.append(name)

    # Compute the average traveled distance over all tasks, average MPPI excution time
    average_traveled_distance = np.mean(
        np.array(save_average_travelled_distance).astype(np.float))
    total_traveled_distance = np.sum(
        np.array(save_average_travelled_distance).astype(np.float))
    average_real_time_MPPI = np.mean(
        np.array(save_average_real_time_MPPI).astype(np.float))
    average_unreal_time_MPPI = np.mean(
        np.array(save_average_unreal_time_MPPI).astype(np.float))
    # Save all results in a Yaml file
    summary = [{
        '# of Tasks': counter_unreachableGoals + counter_reachableGoals
    }, {
        'Reachable Goal Tasks': counter_reachableGoals
    }, {
        'Non-Reachable Goal Tasks': counter_unreachableGoals
    }, {
        'Violated Ctrl Constraints Tasks': violate_Ctrl_constraints
    }, {
        'Successful Tasks (Unviolated Ctrl Const. + Reachable)':
        counter_successful_tests
    }, {
        'Semi-Successful Tasks (Violated Ctrl Const. + Reachable)':
        counter_semi_successful_tests
    }, {
        'Average Traveled Distance [m]': str(average_traveled_distance)
    }, {
        'Total Traveled Distance [m]': str(total_traveled_distance)
    }, {
        'Real-Time Tasks': counter_real_time_MPPI
    }, {
        'Average Real-Time MPPI Excution Time [ms]':
        str(average_real_time_MPPI)
    }, {
        'Average Un-Real-Time MPPI Excution Time [ms]':
        str(average_unreal_time_MPPI)
    }]

    with open(folder_path + '/intensive_simulation_summary.yaml', 'w') as f:
        yaml.dump(summary, f, sort_keys=True)
    # Save convergence index and their folder name
    np.savetxt(folder_path + '/unreachableGoals_tests_folders.csv',
               folder_unreachableGoals,
               fmt='%s')
    np.savetxt(folder_path + '/reachableGoals_tests_folders.csv',
               folder_reachableGoals,
               fmt='%s')
    np.savetxt(folder_path + '/real_time_mppi.csv',
               folder_real_time_MPPI,
               fmt='%s')
    np.savetxt(folder_path + '/unreal_time_mppi.csv',
               folder_unreal_time_MPPI,
               fmt='%s')
    np.savetxt(folder_path + '/violate_ctrl_constraints.csv',
               folder_violate_Ctrl_constraints,
               fmt='%s')
    np.savetxt(folder_path + '/success_tests_folders.csv',
               folder_success_tests,
               fmt='%s')
    np.savetxt(folder_path + '/semi_success_tests_folders.csv',
               folder_semi_success_tests,
               fmt='%s')
    np.savetxt(folder_path + '/average_travelled_distance.csv',
               save_average_travelled_distance,
               fmt='%s')

def readDataFromFile(data_path):
    filename = data_path
    with open(filename, 'r') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=' ')
        data = list(readCSV)
        data = np.array(data).astype(float)
        #data = np.array(data)
    return data

def readStringDataFromFile(data_path):
    filename = data_path
    with open(filename, 'r') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=' ')
        data = list(readCSV)
    return data

def getDataFromYamlFile(fileName):
    data = yaml.load(open(fileName), Loader=yaml.FullLoader)
    return data

''' @brief: Returning the mean and standard deviation of the lognormal distribution, 
           given mean and variance of Normal distribution'''
def Normal2LogN(m, v):
    """ m: mean, v: variance
    Return: mu: mean, sigma: standard deviation of LN dist"""
    mu = np.exp(m + 0.5 * v)
    var = np.exp(2 * m + v) * (np.exp(v) - 1)
    sigma = np.sqrt(var)
    return mu, sigma

''' @brief: Returning the mean and variance of the Normal distribution, 
           given mean and standard deviation of lognormal distribution'''
def LogN2Normal(m, v):
    """ m: mean, v: standard deviation of LN dist.
    Return: mu: mean, var: variance"""
    mu = 2 * np.log(m) - 0.5 * np.log(np.square(m) + np.square(v))
    var = -2 * np.log(m) + np.log(np.square(m) + np.square(v))
    return mu, var

''' @brief: Returning the mean and variance of the product of lognormal and Normal distributions, 
           given the mean and variance of both Normal and Log-Normal distributions.
'''
def NLN(mu_N, var_N, mu_LN, var_LN):
    mu = mu_N * np.exp(mu_LN + 0.5 * var_LN)
    var = (np.square(mu_N) + var_N) * np.exp(
        2 * mu_LN + 2 * var_LN) - np.square(mu_N) * np.exp(2 * mu_LN + var_LN)
    return mu, var

''' @brief: Mapping from real-world poses (in meters) into positive grid '''
def map2grid(obstacles_gazebo_map, params):
    Xmin, Xmax, Ymin, Ymax = params[0], params[1], params[2], params[3]
    XmapSize, YmapSize = params[4], params[5]

    # Mapping from Gazebo obstacles pose into positive grid
    Xmap = XmapSize * (obstacles_gazebo_map[:, 0] - Xmin) / (Xmax - Xmin)
    Ymap = YmapSize * (obstacles_gazebo_map[:, 1] - Ymin) / (Ymax - Ymin)
    Xmap = np.floor(Xmap)
    Ymap = np.floor(Ymap)

    obstacle_grid = np.array([Xmap, Ymap])
    return obstacle_grid.T

''' @brief: Getting the 2D coordinates (x,y) of the obstacles in the map '''
def grid2index(obstacle_grid, map_size):
    obstacle_idx = []
    for i in range(0, map_size):
        for j in range(0, map_size):
            if obstacle_grid[i, j] > 0:
                obstacle_idx.append(np.array([i, j]))
    return obstacle_idx

# @brief:  Converting from map coordinates to world coordinates
def mapToWorld(mx, my, params):
    origin_x, origin_y, resolution = params[0], params[1], params[2]
    wx = origin_x + (mx + 0.5) * resolution
    wy = origin_y + (my + 0.5) * resolution
    return wx, wy

''' @brief:  Converting from world coordinates to map coordinates'''
def worldToMap(wx, wy, params):
    origin_x, origin_y, resolution = params[0], params[1], params[2]
    mx = int((wx - origin_x) / resolution)
    my = int((wy - origin_y) / resolution)
    return mx, my

''' @brief: Returning the pose of the obstacles, given the costmap in cells '''
def mapInMeters(costmap, params):
    map_size = params[3]
    obstacles_xy = []
    for i in range(0, int(map_size)):
        for j in range(0, int(map_size)):
            if costmap[i, j] > 0:
                wx, wy = mapToWorld(i, j, params)
                obstacles_xy.append(np.array([wx, wy]))
    return obstacles_xy

''' @brief:  Given an index... compute the associated costmap coordinates '''
def index2cell(index, size_x):
    my = np.floor(index / size_x)
    mx = index - (my * size_x)
    return mx, my


''' @brief: Calculating the travalled distance between set of points '''
def pathLength(x, y):
    n = len(x)
    lv = [
        np.sqrt((x[i + 1] - x[i])**2 + (y[i + 1] - y[i])**2)
        for i in range(n - 1)
    ]
    L = sum(lv)
    return L

''' @brief: Updating the robot states '''
def update_kinematics(s, u, dt):
    # Set the control input: linear and angular velocities
    v, w = u
    # Update x, y, yaw of the vehicle
    s[0] += np.cos(s[2]) * v * dt
    s[1] += np.sin(s[2]) * v * dt
    s[2] += w * dt
    return s

''' @brief: 计算 速度 方差 '''
def var_v(controlhistory):

    # 获取历史控制量中的速度 v
    v = [u[0] for u in controlhistory]  

    # 计算均值
    mean_v = np.mean(v)

    # 计算方差
    var_v = np.mean((v - mean_v) ** 2)

    # 获取历史控制量中的速度 w
    w = [u[1] for u in controlhistory]  

    # 计算均值
    mean_w = np.mean(w)

    # 计算方差
    var_w = np.mean((w - mean_w) ** 2)   
    var = [var_v,var_w]
    return var 

''' @brief: 计算 速度 平均值'''
def average_v(controlhistory):

    # 获取历史控制量中的速度 v
    velocities = [u[0] for u in controlhistory]  

    # 计算均值
    mean_v = np.mean(velocities)

    return mean_v 
