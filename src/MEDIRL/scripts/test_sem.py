import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from network.env_dilated import OnlyEnvDilated
from torch.autograd import Variable
import torch
from os.path import join


# initialize parameters
grid_size = 160

net = OnlyEnvDilated(feat_in_size=4,feat_out_size=25)
net.init_weights()
#加载训练好的模型参数
net.load_state_dict(torch.load(join('/home/zle/Documents/Geo_Semantic_fusion_nav_ws/src/MEDIRL/picture_process_keji/exp/train-4.7-canghai-2', 'step30-loss0.pth'))['net_state'])
# net.load_state_dict(torch.load(join('/home/zle/Documents/Geo_Semantic_fusion_nav_ws/src/MEDIRL/picture_process_keji/exp/train-3.13-kejipo2', 'step40-loss0.pth'))['net_state'])
# net.load_state_dict(torch.load(join('/home/zle/Documents/Geo_Semantic_fusion_nav_ws/src/MEDIRL/picture_process_keji/exp/train-3.13-kejipo3', 'step20-loss0.pth'))['net_state'])

#将模型切换到评估模式（与训练模式相对）
net.eval()


def load(grid_size):
    
    feat = np.zeros((4, grid_size, grid_size))

    #library jinggai
    feat[0] = np.loadtxt('/home/zle/Documents/Geo_Semantic_fusion_nav_ws/src/MEDIRL/picture_process_keji/inital/canghai/slope.txt') #slope
    feat[1] = np.loadtxt('/home/zle/Documents/Geo_Semantic_fusion_nav_ws/src/MEDIRL/picture_process_keji/inital/canghai/step.txt')  #height
    feat[2] = np.loadtxt('/home/zle/Documents/Geo_Semantic_fusion_nav_ws/src/MEDIRL/picture_process_keji/inital/canghai/rough.txt')
    feat[3] = np.loadtxt('/home/zle/Documents/Geo_Semantic_fusion_nav_ws/src/MEDIRL/picture_process_keji/inital/canghai/sem.txt')
    future_traj = np.loadtxt('/home/zle/Documents/Geo_Semantic_fusion_nav_ws/src/MEDIRL/picture_process_keji/inital/canghai/odom_4a.txt')

    # feat[0] = (feat[0] - 0.16004319296682) / 0.43282621367170965
    # feat[1] = (feat[1] - 0.15061161074218749) / 0.41371947577389556
    # feat[2] = (feat[2] - 0.2311822556544169) / 0.4288381059949021

    # 前3个特征减去均值并除以标准查进行归一化 
    for i in range(3):
        a = np.mean(feat[i])
        b = np.std(feat[i])
        feat[i] = (feat[i] - np.mean(feat[i])) / np.std(feat[i])

    return feat, future_traj

feat, future_traj = load(grid_size)
#将feat转换为PyTorch张量，并扩展维度以适应神经网络的输入形状。
feat_var = Variable(torch.from_numpy(np.expand_dims(feat, axis=0)).float())
r_var = net(feat_var) #（1，1，80，80）
# mask = np.loadtxt('/home/zle/Documents/Geo_Semantic_fusion_nav_ws/src/MEDIRL/picture_process_keji/inital/library-kejipo/step2-2.txt')
# a = torch.from_numpy(mask)
r_var = 50*(r_var - r_var.min())/(r_var.max() - r_var.min())-51
# r_var = torch.where( (a==2), torch.tensor(-60, dtype=torch.float32), r_var) #高度空白区域赋值

r = r_var[0].data.numpy().squeeze() #转换为（80，80）的numpy
# r = np.flipud(r)    #上下翻转

# np.savetxt('/home/zle/Documents/Geo_Semantic_fusion_nav_ws/src/MEDIRL/picture_process_keji/inital/canghai/reward2-2.txt',r)

plt.clf()
plt.pcolor(r,cmap="plasma")
plt.axis('off')
plt.gca().set_aspect('equal', adjustable='box')  # 设置坐标轴等比例
plt.colorbar()
# 雨娟：(-30,0)  #科技：(-30,3)  #library:(-8,-2.5)
plt.clim(-10, -0.1) 
plt.draw()
plt.show()
