"""
@author: Ihab S. Mohamed, Vehicle Autonomy and Intelligence Lab - Indiana University, Bloomington
"""
"""
@brief: The  definition of the kinematics model of a differential wheeled robot, e.g., Jackal Robot,
as well as the definition of the state-dependent running cost function. 
"""
import numpy as np
from jinja2 import Template
import matplotlib.pyplot as plt



# CUDA内核模板生成函数和机器人运动学及成本计算。
class Obstacle:
    def __init__(self,
                 dt, # 时间步长，用于更新位置。
                 time_horizon,
                 control_freq,
                 slack_weight,
                 max_slack_vari, # 定义了松弛变量最大变化率
                 num_max = 10, # 松弛变量状态向量的最大维数
                 soft_cbf=False,
                 IF_virtual_ob=True,#虚拟障碍物
                 virtual_obstacles=None,
                 seed=123): # 用于随机数生成。
        self.num_max = num_max 
        self.dt = dt
        self.max_slack_vari = max_slack_vari


        self.num_timesteps = int(time_horizon * control_freq)
        self.id = [] # 障碍物编号
        self.id_last = [] # 上次回调障碍物编号
        self.num = 0 # 障碍物数量
        self.data = [] # 障碍物数据
        self.cur_state = [] # 障碍物当前位置数据
        self.all_cur_state = []
        self.U = np.zeros((self.num_timesteps, self.num_max),
                dtype=np.float32)
        # 非线性软约束
        # self.target_sla = 1.0
        # 线性软约束
        self.target_sla = 1.0

        self.weights = slack_weight
        # print(f"self.weights:{self.weights}")
        ''' \param "self.Control_Constraints = true" means that the control constraints are considered
            in the control law design. More precisely, an element-wise clamping function is used to restrict
            the control input to remain within a given range, for all samples drawn from the dynamics system'''
        self.Control_Constraints = 'true'
        self.soft_cbf = soft_cbf
        self.ctrl_scale = 0.8

        self.state_sla = np.ones(self.num_max, dtype=np.float32)
        #由于懒得写python仿真的mppi，在这里偷懒，设定虚拟的virtual 障碍物
        self.IF_virtual_ob=IF_virtual_ob
        # x y a b theta dx dy   

        # self.virtual_ob_params = [[9, 6, 0.4, 0, 0, -0.20, -1.1],
        #                           [18, 2.0, 0.4, 0, 0, -1.1, 0.0],
        #                           [21, -1.0, 0.4, 0, 0, -1.1, 0.0],
        #                           [26, 1.6, 0.4, 0, 0, -1.1, 0.0]]    
        #纯迎面
        # self.virtual_ob_params = [[7, 1.0, 0.4, 0, 0, -0.8, 0.0],
        #                           [13, -1.0, 0.4, 0, 0, -0.6, 0.0],                                  
        #                           [20, 1.0, 0.4, 0, 0, -0.4, 0.0],]
        

                                #  [25, -1.2, 0.4, 0, 0, -1.2, 0.0],]
                                #   [30, 0.8, 0.4, 0, 0, -1.2, -0.0],]             
        # self.virtual_ob_params = [[12, 10.5, 0.4, 0, 0, -0.02, -1.18],]  #单垂直
        # self.virtual_ob_params = [[10, -0.5, 0.4, 0, 0, -1.2, 0.0],]  #单迎面
        # self.virtual_ob_params = [[200, -0.5, 0.4, 0, 0, -1.0, 0.0],]  #单迎面
        # self.virtual_ob_params = [[12, 10.5, 0.4, 0, 0, -0.02, -1.18],
        #                           [30, 1.0, 0.4, 0, 0, -1.1, 0]]        

        # # ACC场景
        # self.virtual_ob_params = [[4.0, 5.5, 0.4, 0, 0, -0.5, -0.5],
        #                           [8.0, 6.5, 0.4, 0, 0, -0.2, -0.2],                                  
        #                           [10.1, 11.5, 0.4, 0, 0, -0.05, -0.05],]
        # # STATIC场景
        # self.virtual_ob_params = [[4.0, 5.0, 0.4, 0, 0, 0, 0],
        #                           [8.0, 6.5, 0.4, 0, 0, 0, 0],                                  
        #                           [8.0, 10.0, 0.4, 0, 0, 0, 0],]
        
        # # AVE场景
        # self.virtual_ob_params = [[4.0, 5.5, 0.4, 0, 0, -0.9, -0.9],
        #                           [8.0, 6.5, 0.4, 0, 0, -0.8, -0.8],                                  
        #                           [11.2, 11.8, 0.4, 0, 0,-0.6, -0.6],]
        
        # DACC场景
        # self.virtual_ob_params = [[4.0, 5.5, 0.4, 0, 0, -0.9, -0.9],
        #                           [8.0, 6.5, 0.4, 0, 0, -0.8, -0.8],                                  
        #                           [10.2, 11.4, 0.4, 0, 0,-0.93, -0.93],]        

        # 单ACC场景
        self.virtual_ob_params = virtual_obstacles or [[10.0, -0.6, 0.4, 0, 0, -0.5, 0.0]]

        
        self.virtual_ob_state = [list(ob) for ob in self.virtual_ob_params]
        self.virtual_data = []


    def update_predict_state_virtual(self, virtual_ob_state):
        # x y a b theta dx dy
        data = []
        for id in range(len(self.virtual_ob_params)):
            ob_state_predict = []
            for i in range(self.num_timesteps):
                # virtual_ob_state[id][5] = virtual_ob_state[id][5] + 0.001
                # if virtual_ob_state[id][5] > 0:
                #     virtual_ob_state[id][5] = 0 

                dx = virtual_ob_state[id][5]
                dy = virtual_ob_state[id][6]                
                x  = virtual_ob_state[id][0] + dx * self.dt * i
                y  = virtual_ob_state[id][1] + dy * self.dt * i 
                a  = virtual_ob_state[id][2]
                b  = virtual_ob_state[id][3]
                theta  = virtual_ob_state[id][4]
                ob_state_predict = [x,y,a,b,theta,dx,dy]
                data.append(ob_state_predict)
        # 将嵌套列表转换为 numpy 数组
        data_array = np.array(data)
        
        # 将 numpy 数组展平成一维数组
        flat_data = data_array.reshape(-1)
        return flat_data
    
    def update_cur_state_virtual(self, virtual_ob_state):
        for id in range(len(self.virtual_ob_params)):
            # virtual_ob_state[id][5] = virtual_ob_state[id][5] + 0.01
            # if virtual_ob_state[id][5] > 0:
            #     virtual_ob_state[id][5] = 0 

            # # ACC场景
            # self.virtual_ob_params = [[4.0, 5.5, 0.4, 0, 0, -0.5, -0.5],
            #                           [8.0, 6.5, 0.4, 0, 0, -0.3, -0.3],                                  
            #                           [11.2, 11.8, 0.4, 0, 0, -0.1, -0.1],]
            virtual_ob_state[id][5] = virtual_ob_state[id][5] - 0.01
            if virtual_ob_state[id][5] < -1.2:
                virtual_ob_state[id][5] = -1.2 
            # virtual_ob_state[id][6] = virtual_ob_state[id][6] - 0.01
            # if virtual_ob_state[id][6] < -1.2:
            #     virtual_ob_state[id][6] = -1.2                 

            # # DACC场景
            # self.virtual_ob_params = [[4.0, 5.5, 0.4, 0, 0, -0.9, -0.9],
            #                         [8.0, 6.5, 0.4, 0, 0, -0.8, -0.8],                                  
            #                         [11.2, 11.8, 0.4, 0, 0,-0.94, -0.94],]   
            # virtual_ob_state[id][5] = virtual_ob_state[id][5] + 0.01
            # if virtual_ob_state[id][5] > 0:
            #     virtual_ob_state[id][5] = 0 
            # virtual_ob_state[id][6] = virtual_ob_state[id][6] + 0.01
            # if virtual_ob_state[id][6] > 0:
            #     virtual_ob_state[id][6] = 0      


            dx = virtual_ob_state[id][5]
            dy = virtual_ob_state[id][6] 
            x  = virtual_ob_state[id][0] + dx * self.dt
            y  = virtual_ob_state[id][1] + dy * self.dt
            virtual_ob_state[id][0] = x
            virtual_ob_state[id][1] = y
        return virtual_ob_state

    ''' @brief: Updatting the robot state (x, y, theta) '''    
    def update_state(self, s, u):
        self.kinematics(s, u)
        self.state_sla = s
        return s

    def kinematics(self, s, u):
        # 创建一个布尔掩码，表示需要设置为1的位置
        mask = ~np.isin(np.arange(self.num_max), self.id)
        # 非线性软约束为 1， 线性软约束为 0
        s[mask] = self.target_sla  # 将 mask 为 True 的位置的 s 值设置为 1

        s += u * self.dt # 直接更新 s
        s[:] = np.clip(s, 0, self.max_slack_vari)

    ''' @brief: Returning the next state of the vehicle (x, y, theta) '''    
    def update_kinematics(self, s, u):
        s += u * self.dt
        return s

    ''' @brief: Calculating the running state ''' 
    """ cost 函数计算状态。状态成本使用二次差方差形式表示，控制成本为二次控制输入项。数学公式如下： """
    def cost(self, s, cost_params):
        weights, targets = cost_params  
        state_cost = np.sum(weights * (s- targets)**2)#松弛变量目标为 1 

        return state_cost

    ''' @brief: The CUDA code of the robot kinematics '''    
    '''cuda_kinematics 函数生成一个 CUDA 设备函数，用于在 GPU 上执行运动学计算。这个函数通过应用控制输入 ,v,w 更新状态 
        s 的位置和方向。这个模板还包含了控制约束 clamping 以确保速度在限定范围内。''' 
    def cuda_kinematics(self):
        slack_kinematics_template = Template("""
                                             
            // 限制输入的 float 值在 0.0 到1的范围
        __device__ float clamp_(float value, float limit) {
                if (value < 0.0f) {
                    return 0.0f;
                }
                else if (value > limit) {
                    return limit;
                }
                else{                             
                    return value;
                }
            }   
            // 数组元素相加                                 
        __device__ void array_add(float *result, float *arr1, float *arr2, int size) {
            for (int i = 0; i < size; i++) {
                result[i] = arr1[i] + arr2[i];
            }
        }
           // 检查id                               
        __device__ bool array_equal(float *arr, float id, int num) {
            for (int i = 0; i < num; i++) {
                if (id == arr[i]) {
                    return true; // 如果有元素相等，则返回 true
                }
            }
            return false; // 元素不相等，返回 false
        }                                  
        __device__ void all_slack_kinematics(float* s, float* u, float* du, float* id, int num, float dt, int timestep)
        {

            // The command is given as (u[0] + du[0]) for u0, and so on.
            float* v;   
                                                 
            float ctrl_scale = {{ctrl_scale}};
            bool Ctrl_Const = {{Control_Constraints}};   
                                               
            for(int i = 0; i < {{num_max}}; i++){
                if(array_equal(id,i,num)){
                    v[i] = u[i] + du[i];
                    // Handling Control Constraints: the maximum linear and angular velocities (clamping function)
                    if (Ctrl_Const)
                    {
                        if (v[i] > {{max_slack_vari}}){
                            v[i] = ctrl_scale*{{max_slack_vari}};
                        }
                        if (v[i] < - {{max_slack_vari}}){
                            v[i] = - ctrl_scale*{{max_slack_vari}};
                        }
                    }
                    s[i] += v[i]*dt;
                    s[i] = clamp_(s[i], {{max_slack_vari}});                                                                         
                }
                //id不在,则更新slack为 1
                else{
                    if(s[i] != 1.0f){
                        v[i] = (1.0f-s[i])/dt;
                        s[i] = 1.0f;}
                    else{
                        v[i] = 0;
                    }
                }
            }
                                             
            //printf("dx=%f           |",s[3]); 
            //printf("v=%f           |",u[0]);                                                                                     
         }
                                             
        __device__ void slack_kinematics(float* s, float* u, float* du, int ob_num, float dt, int timestep)
        {

            // The command is given as (u[0] + du[0]) for u0, and so on.
            float v[{{num_max}}];   
                                             
            float ctrl_scale = {{ctrl_scale}};
            bool Ctrl_Const = {{Control_Constraints}};   
                                               
            for(int i = 0; i < ob_num; i++){
                v[i] = u[i] + du[i];
                // Handling Control Constraints: the maximum linear and angular velocities (clamping function)
                if (Ctrl_Const)
                {
                    if (v[i] > {{max_slack_vari}}){
                        v[i] = ctrl_scale*{{max_slack_vari}};
                    }
                    if (v[i] < - {{max_slack_vari}}){
                        v[i] = - ctrl_scale*{{max_slack_vari}};
                    }
                }
                s[i] += v[i] * dt;
                s[i] = clamp_(s[i],{{max_slack_vari}});        
                                                                     
                //printf("x=%f           |",s[0]); 
                //printf("v=%f           |",v[0]);                                                                         
            }
                                                                                    
         }        

         """)
        return (slack_kinematics_template.render(
            num_max = self.num_max,
            max_slack_vari = self.max_slack_vari,
            Control_Constraints = self.Control_Constraints,
            ctrl_scale = self.ctrl_scale))
    
    def cuda_state_cost(self):
        state_cost_template = Template("""
        /*                        
        @brief: slack_get_state_cost(): for computing the state-dependent running cost. 
                                    计算状态成本，公式如下 state_cost= ∑_i weights_i⋅(s_i - targets_i) ^2
        */
                                       
        __device__ float slack_get_state_cost(float* s, int ob_num)
        {
            float state_cost = 0;

            //for (int i = 0; i < ob_num; i++) {
            for (int i = 0; i < ob_num; i++) {                       
                state_cost += {{weights}}*(s[i] - {{target_sla}})*(s[i] - {{target_sla}}); //松弛变量目标为 1
                //printf("s[%d]=%f",i,s[i]); 
                //printf("state_cost=%f",state_cost);                          
            }                 
            //printf("state_cost=%f",state_cost);                                    
            return state_cost;
                                       
        }
        """)
        return (state_cost_template.render(
            target_sla = self.target_sla,
            weights = self.weights           
            ))
