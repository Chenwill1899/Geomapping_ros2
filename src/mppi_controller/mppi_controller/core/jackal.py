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
class Jackal:
    def __init__(self,
                 state_dim, # 状态向量的维数（例如，机器人的位置和朝向）。
                 dt, # 时间步长，用于更新位置。
                 max_linear_velocity, # 定义了机器人的最大线速度和角速度。
                 max_angular_velocity,
                 r, # 机器人半径
                 safety_dist, # 安全距离
                 atau, # 超前时间
                 ob_states_num, # 障碍物状态数量
                 map_info = [], #  地图信息，包括地图尺寸、障碍物信息、成本等。
                 cbf_type=0,
                 dcbf_alpha=0.1,
                 dcbf_weight=10000,
                 seed=123): # 用于随机数生成。
        self.state_dim = state_dim 
        self.dt = dt
        self.max_linear_velocity = max_linear_velocity
        self.max_angular_velocity = max_angular_velocity

        self.r = r
        self.safety_dist = safety_dist
        self.atau = atau
        self.ob_states_num = ob_states_num
        self.ob_id = []
        self.ob_num = 0
        self.ob = []
        self.dcbf_alpha = dcbf_alpha
        self.dcbf_weight = dcbf_weight

        self.cbf_type = cbf_type
        ''' \param "self.Control_Constraints = true" means that the control constraints are considered
            in the control law design. More precisely, an element-wise clamping function is used to restrict
            the control input to remain within a given range, for all samples drawn from the dynamics system'''
        self.Control_Constraints = 'true'
        '''Since the control constraints, in this case, acting as soft constraints. It is notoriously difficult to ensure
            that the control input, obtained by the controller, remains always within its allowed bounds, even after rejecting
            each trajectory that violates the control input limits. For this reason, \param "self.ctrl_scale" is here used. 
            Another way is to apply the clamping function to the optimal control sequence obtained by the controller'''
        self.ctrl_scale = 1.0

        self.state = np.zeros(self.state_dim, dtype=np.float32)

    ''' @brief: Updatting the robot state (x, y, theta) '''    
    def update_state(self, s, u):
        self.kinematics(s, u)
        self.state = s
        return s

    def kinematics(self, s, u):
        # Set the control input: linear and angular velocities
        v, w = u
        # Update x, y, yaw of the vehicle

        s[0] += np.cos(s[2]) * v * self.dt
        s[1] += np.sin(s[2]) * v * self.dt
        s[2] += w * self.dt
        s[3] = np.cos(s[2]) * v
        s[4] = np.sin(s[2]) * v
      

    ''' @brief: Returning the next state of the vehicle (x, y, theta) '''    
    def update_kinematics(self, s, u):
        # Set the control input: linear and angular velocities
        v, w = u
        # Update x, y, yaw of the vehicle
        s[0] += np.cos(s[2]) * v * self.dt
        s[1] += np.sin(s[2]) * v * self.dt
        s[2] += w * self.dt
        s[3] = np.cos(s[2]) * v
        s[4] = np.sin(s[2]) * v
        return s

    ''' @brief: Calculating the running state and control costs ''' 
    """ cost 函数计算状态和控制成本。状态成本使用二次差方差形式表示，控制成本为二次控制输入项。数学公式如下： """
    def cost(self, s, u, cost_params):
        weights, targets, R = cost_params
        state_cost = np.sum(weights[:3] * (s[:3] - targets[:3])**2)
        control_cost = np.sum(0.5 * u * R * u)
        return state_cost, control_cost

    def plot_grid(self, results_rootpath, counter):
        plt.imshow(self.obstacle_grid, interpolation="nearest")
        plt.savefig(results_rootpath + '/costmap_{n}.png'.format(n=counter))
        np.savetxt(results_rootpath + '/costmap.csv',
                   self.obstacle_grid,
                   fmt='%s')
        # plt.show()

    def param_getter(self):
        return {
            "obstacle_grid": self.obstacle_grid,
            "origin": self.local_costmap_origin
        }

    ''' @brief: Updatting the local costmap each timestep '''    
    def update_obstacle_grid(self, local_costmap, costmap_updated_origin):
        self.obstacle_grid = local_costmap
        self.local_costmap_origin_x = np.array([costmap_updated_origin[0]])
        self.local_costmap_origin_y = np.array([costmap_updated_origin[1]])
        self.local_costmap_origin = costmap_updated_origin

    ''' @brief: The CUDA code of the robot kinematics '''    
    '''cuda_kinematics 函数生成一个 CUDA 设备函数，用于在 GPU 上执行运动学计算。这个函数通过应用控制输入 ,v,w 更新状态 
        s 的位置和方向。这个模板还包含了控制约束 clamping 以确保速度在限定范围内。''' 
    def cuda_kinematics(self):
        kinematics_template = Template("""
        __device__ void kinematics(float* s, float* u, float* du, float dt, int timestep)
        {

            // The command is given as (u[0] + du[0]) for u0, and so on.
            float v = (u[0] + du[0]);
            float w = (u[1] + du[1]);

            float ctrl_scale = {{ctrl_scale}};
            bool Ctrl_Const = {{Control_Constraints}};

           // Handling Control Constraints: the maximum linear and angular velocities (clamping function)
            if (Ctrl_Const)
            {
                if (v > {{max_linear_velocity}}){
                    v = ctrl_scale*{{max_linear_velocity}};
                }
                if (v < - {{max_linear_velocity}}){
                    v = - ctrl_scale*{{max_linear_velocity}};
                }
                //------------------------------------------------------------
                if (w > {{max_angular_velocity}}){
                    w = ctrl_scale*{{max_angular_velocity}};
                }
                if (w < - {{max_angular_velocity}}){
                    w = - ctrl_scale*{{max_angular_velocity}};
                }
            }
            float cos_ = __cosf(s[2]);
            float sin_ = __sinf(s[2]);      
                                                            
            s[0] += cos_ * v *dt;
            s[1] += sin_ * v *dt;
            s[2] += w * dt;
            s[3] = cos_ * v;
            s[4] = sin_ * v;
            //printf("dx=%f           |",s[3]); 
            //if(v>={{max_linear_velocity}})
                //printf("v=%f           |",v);                                                                                     
         }
         """)
        return kinematics_template.render(
            max_linear_velocity=self.max_linear_velocity,
            max_angular_velocity=self.max_angular_velocity,
            Control_Constraints=self.Control_Constraints,
            ctrl_scale=self.ctrl_scale)

    ''' @brief: The CUDA code of the state-dependent cost and collision indicator function '''   
    ''' cuda_state_cost 函数定义了一个 CUDA 模板，用于计算状态成本并检查碰撞。它包括以下部分： '''  

    # ##############以下是椭圆cbf
    def cuda_state_cost(self):
        state_cost_template = Template("""
                                     
            // 快速平方根倒数函数
            __device__ float inv_Sqrt(float x) {
                float halfx = 0.5f * x;
                float y = x;
                int i = *(int*)&y;  // 使用 int 类型
                i = 0x5f3759df - (i>>1);
                y = *(float*)&i;
                y = y * (1.5f - (halfx * y * y));
                return y;
            }


            __device__ float distance(float* p1, float* p2) {
                return sqrt(pow(p1[0] - p2[0], 2) + pow(p1[1] - p2[1], 2));
            }                           

            // 限制输入的 float 值在 0.0 到正无穷的范围
            __device__ float clamp_to_non_negative(float value) {
                if (value < 0.0f) {
                    return 0.0f;
                }
                return value;
            }         
            __device__ void exp_ellipse(float *s, float *ob, float slack) {
                float F1[2], F2[2];
                                       
                float angle_rad;
                float deltaX, deltaY;

                // 计算焦点 F1 和 F2
                F1[0] = ob[0];
                F1[1] = ob[1];
                F2[0] = ob[0] + slack * {{atau}} * (ob[5]-s[3]);
                F2[1] = ob[1] + slack * {{atau}} * (ob[6]-s[4]);

                                       
                if (F1[0] == F2[0] && F1[1] == F2[1]) {
                    //printf("Foci are coincident, no angle can be computed.");
                    ob[3] = ob[2];
                    ob[4] = 0;
                    return;                      
                } else {
                    deltaX = F2[0] - F1[0];
                    deltaY = F2[1] - F1[1];
                    angle_rad = atan2(deltaY, deltaX);
                }

                // 生成点集                    
                float points1[NUMS][2];
                float points2[NUMS][2];
                for (int i = 0; i < NUMS; i++) {
                    float theta = angle_rad + M_PI * 0.7 + i * (M_PI * 0.6 / (NUMS-1));
                    points1[i][0] = F1[0] + ob[2] * cos(theta);
                    points1[i][1] = F1[1] + ob[2] * sin(theta);

                    theta = angle_rad - M_PI * 0.3 + i * (M_PI * 0.6 / (NUMS-1));
                    points2[i][0] = F2[0] + ob[2] * cos(theta);
                    points2[i][1] = F2[1] + ob[2] * sin(theta);
                }

                // 计算半长轴 a 和半短轴 b
                float sum_distances = 0.0;
                for (int i = 0; i < NUMS; i++) {
                    float dist_F1 = distance(points1[i], F1);
                    float dist_F2 = distance(points1[i], F2);
                    sum_distances += dist_F1 + dist_F2;
                }
                for (int i = 0; i < NUMS; i++) {
                    float dist_F1_ = distance(points2[i], F1);
                    float dist_F2_ = distance(points2[i], F2);
                    sum_distances += dist_F1_ + dist_F2_;
                }
                                                                             
                float average_2a = sum_distances/(NUMS*2);
                float a = average_2a / 2.0; //平方
                float c = distance(F1, F2) / 2.0;
                float err = ob[2] - a + c;
                a += err;
                float a2 = a*a;
                float b2 = a2 - c*c;                                              
                                                      
                    
                // 计算椭圆的中心
                float center[2];
                center[0] = (F1[0] + F2[0]) / 2.0;
                center[1] = (F1[1] + F2[1]) / 2.0;

                // 更新结果
                ob[0] = center[0];
                ob[1] = center[1];
                ob[2] = a2;
                ob[3] = b2;
                ob[4] = angle_rad;
            }              
                                                    
            // 计算椭圆与机器人的角度delta , 参考dcbf中的图解
            __device__ float delta(float* s, float* ob) {
                // 以下是求机器人到椭圆直线，角度的求解方法
                 float dx = ob[0] - s[0];
                 float dy = ob[1] - s[1];

                 // 使用 atan2 计算直线方向向量与 x 轴正方向的夹角
                 float angle_to_ob = atan2(dy, dx);

                 // 计算障碍物长轴与 x 轴正方向的夹角
                 float angle_of_major_axis = ob[4];

                 // 计算两个角度之间的差值
                 float delta = angle_to_ob - angle_of_major_axis;

                 // 将角度差转换为 [-pi, pi] 范围内
                 delta = fmod(delta + M_PI, 2 * M_PI) - M_PI;

                 // 确保 delta 在 [-pi/2, pi/2] 范围内，即在 [-90°, 90°] 之间
                 if (delta > M_PI / 2) {
                     delta -= M_PI;  // 将大于 90° 的角度调整到 -90° 到 90° 范围
                 }
                 else if (delta < -M_PI / 2) {
                     delta += M_PI;  // 将小于 -90° 的角度调整到 -90° 到 90° 范围
                 } 
                                             
                return delta;
            }         

            // 计算椭圆与机器人 直线方向，椭圆中心到椭圆边界的距离 , 参考dcbf中的图解
            __device__ float l_x(float* ob) {
                // 从障碍物参数数组中获取椭圆的半长轴 a，半短轴 b 和夹角 delta
                float a2 = ob[2];
                float b2 = ob[3];
                float delta = ob[5]; // 假设 delta 已经由之前的函数计算得到

                // 计算椭圆中心到椭圆边界的距离 l_i(k)
                float l_i = (a2 *b2  * (1 + tan(delta) * tan(delta))) /
                                (b2 + a2 * tan(delta) * tan(delta));

                return l_i;
            } 

            /*
            @brief: h_x_(): for calculating the value of cbf_h, 
                      计算cbf值，包含椭圆距离。                 
            */
            __device__ float h_x_(float *s, float *ob)
            {
                float h,dist;
                dist = (s[0] - ob[0])*(s[0] - ob[0]) + (s[1] - ob[1])*(s[1] - ob[1]);
                //h = dist - (ob[6]+{{robot_r}}*{{robot_r}}+{{safety_dist}}*{{safety_dist}}); 
                h = sqrt(dist) - (sqrt(ob[6])+{{robot_r}}+{{safety_dist}});                               
                return h;            
            }  
                                       
            /*
            @brief: h_x(): for calculating the value of cbf_h, 
                      计算cbf值，包含普通距离。                 
            */
            __device__ float h_x(float *s, float *ob)
            {
                float h,dist;
                                       
                // x,y上障碍物与机器人相对位置
                float dx = ob[0] - s[0];
                float dy = ob[1] - s[1];
                                       
                dist = sqrt(dx*dx + dy*dy);        
                h = dist - (ob[2]+{{robot_r}}+{{safety_dist}});                        
                return h;            
            }          
                                                                      
            /*
            @brief: h_ax(): for calculating the value of cbf_h, 
                      计算cbf值，包含超前距离，安全裕度线性可调    hzh             
            */
            __device__ float h_csx(float *s, float *ob)
            {
                float h,dist;
                                       
                // x,y上障碍物与机器人相对位置
                float dx = ob[0] - s[0];
                float dy = ob[1] - s[1];
                // x,y上障碍物与机器人相对速度
                float dvx = ob[5] - s[3];
                float dvy = ob[6] - s[4];    
                //点积小于0，则说明两者正在靠近                      
                float dot = dx * dvx + dy * dvy;  
                if(dot<0){                    
                    dist = sqrt((s[0] - ob[0])*(s[0] - ob[0]) + (s[1] - ob[1])*(s[1] - ob[1]));    
                                                                             
                    float cos_v = dot / dist;       

                    float tau = 0.1*dist/(cos_v+0.001f);
                    if(abs(tau) > {{atau}}){
                        h = dist + cos_v * {{atau}}  - (ob[2]+{{robot_r}}+{{safety_dist}});}
                    else{
                        h = dist + cos_v * abs(tau) - (ob[2]+{{robot_r}}+{{safety_dist}});}
                                                          
                }
                else{
                    dist = sqrt(dx*dx + dy*dy);        
                    h = dist - (ob[2]+{{robot_r}}+{{safety_dist}});                                       
                                       }                                    
                return h;            
            }       
                                       
            /*
            @brief: h_ax(): for calculating the value of cbf_h, 
                      计算cbf值，包含超前距离，直接相对速度        pj         
            */
            __device__ float h_ex(float *s, float *ob)
            {
                float h,dist;
                                       
                // x,y上障碍物与机器人相对位置
                float dx = ob[0] - s[0];
                float dy = ob[1] - s[1];
                // x,y上障碍物与机器人相对速度
                float dvx = ob[5] - s[3];
                float dvy = ob[6] - s[4];    
                //点积小于0，则说明两者正在靠近                      
                float dot = dx * dvx + dy * dvy;  
                if(dot<0){                       
                    dist = sqrt((dx + dvx * {{atau}})*(dx + dvx * {{atau}}) + (dy + dvy * {{atau}})*(dy + dvy * {{atau}}));   
                    //printf("{{cbf_type}}:= %d", {{cbf_type}});                       
                }
                else{
                    dist = sqrt(dx*dx + dy*dy);                                                                                    
                }                                                                                                  
                h = dist - (ob[2]+{{robot_r}}+{{safety_dist}});
                             
                return h;            
            }          
                                                                                                           
            // 检查危险性，判断是否需要扩展椭圆
            __device__ bool check_danger(float* s, float* ob) {
                // x,y上障碍物与机器人相对位置
                float dx = ob[0] - s[0];
                float dy = ob[1] - s[1];
                // x,y上障碍物与机器人相对速度
                float dvx = ob[5] - s[3];
                float dvy = ob[6] - s[4];    
                //点积小于0，则说明两者正在靠近                      
                float dot_product = dx * dvx + dy * dvy;
                if(dot_product < 0)
                    return true;
                else
                    return false;         
            } 
                                       
            /*                        
            @brief: get_state_cost(): for computing the state-dependent running cost. 
                                       计算状态成本，公式如下 state_cost= ∑_i weights_i⋅(s_i - targets_i) ^2
            */
            __device__ float get_state_cost(float* next_s, float* s, float* weights, float* targets, int timestep, int timesteps, float* ob, int ob_num)
            {
                float state_cost = 0;

                //for (int i = 0; i < {{state_dim}}; i++) {
                for (int i = 0; i < 3; i++) {                       
                    state_cost += weights[i]*(s[i] - targets[i])*(s[i] - targets[i]);
                }
                //printf("timesteps=%d",timesteps);  
                                       
                float ob_i[7] = {0,0,0,0,0,0,0};
                float ob_i_next[7] = {0,0,0,0,0,0,0};                              
                float h,h_next,cbf_cost_i,cbf_cost=0;
                                       
                //printf("ob_num=%d",ob_num);
                if(ob_num!=0){
                    for (int i = 0; i < ob_num; i++){
                        // 从这里开始                     
                        ob_i[0] = ob[ 7 * timesteps * i + 7 * timestep + 0 ];// x
                        ob_i[1] = ob[ 7 * timesteps * i + 7 * timestep + 1 ];// y   
                        ob_i[2] = ob[ 7 * timesteps * i + 7 * timestep + 2 ];// a
                        //ob_i[3] = ob[ 7 * timesteps * i + 7 * timestep + 3 ];// b 感知无数据
                        //ob_i[4] = ob[ 7 * timesteps * i + 7 * timestep + 4 ];// theta 感知无数据
                        ob_i[5] = ob[ 7 * timesteps * i + 7 * timestep + 5 ];// dx  
                        ob_i[6] = ob[ 7 * timesteps * i + 7 * timestep + 6 ];// dy    
                                        
                        ob_i_next[0] = ob[ 7 * timesteps * i + 7 * (timestep+1) + 0 ];
                        ob_i_next[1] = ob[ 7 * timesteps * i + 7 * (timestep+1) + 1 ];   
                        ob_i_next[2] = ob[ 7 * timesteps * i + 7 * (timestep+1) + 2 ];
                        //ob_i_next[3] = ob[ 7 * timesteps * i + 7 * (timestep+1) + 3 ];   
                        //ob_i_next[4] = ob[ 7 * timesteps * i + 7 * (timestep+1) + 4 ];                                       
                        ob_i_next[5] = ob[ 7 * timesteps * i + 7 * (timestep+1) + 5 ];
                        ob_i_next[6] = ob[ 7 * timesteps * i + 7 * (timestep+1) + 6 ];
         
                        //扩展椭圆距离                
                        if({{cbf_type}} == 4){
                            if( (check_danger(s,ob_i)) && ({{atau}} != 0.f)){
                                //扩展障碍为椭圆               
                                exp_ellipse(s,ob_i,1.0f);
                                exp_ellipse(next_s,ob_i_next,1.0f);
                                if(ob_i[2]!=ob_i[3] && ob_i_next[2]!=ob_i_next[3]){
                                    ob_i[5] = delta(s,ob_i);//椭圆与机器人夹角      
                                    ob_i[6] = l_x(ob_i);//椭圆与机器人 直线方向，椭圆中心到椭圆边界的距离的平方                                                    
                                    ob_i_next[5] = delta(s,ob_i_next);//椭圆与机器人夹角      
                                    ob_i_next[6] = l_x(ob_i_next);//椭圆与机器人 直线方向，椭圆中心到椭圆边界的距离的平方 
                                    h = h_x_(s,ob_i);               
                                    h_next = h_x_(next_s,ob_i_next); 
                                            }              
                                else{
                                    h = h_x(s,ob_i);               
                                    h_next = h_x(next_s,ob_i_next);                                         
                                            }    
                                ob_i[5] = delta(s,ob_i);//椭圆与机器人夹角      
                                ob_i[6] = l_x(ob_i);//椭圆与机器人 直线方向，椭圆中心到椭圆边界的距离的平方                                                    
                                ob_i_next[5] = delta(s,ob_i_next);//椭圆与机器人夹角      
                                ob_i_next[6] = l_x(ob_i_next);//椭圆与机器人 直线方向，椭圆中心到椭圆边界的距离的平方 
                                h = h_x_(s,ob_i);               
                                h_next = h_x_(next_s,ob_i_next);
                            }
                            else{
                                h = h_x(s,ob_i);               
                                h_next = h_x(next_s,ob_i_next);           
                            }                                                   
                        }       
                                       
                        // DC距离               
                        else if({{cbf_type}} == 3){ 
                            h = h_x(s,ob_i);               
                            h_next = h_x(next_s,ob_i_next);        
                        }                                         
                        //pj超前距离               
                        else if({{cbf_type}} == 2){ 
                            h = h_ex(s,ob_i);               
                            h_next = h_ex(next_s,ob_i_next); 
                                           
                        }  
                        //cos超前距离               
                        else if({{cbf_type}} == 1){ 
                            h = h_csx(s,ob_i);               
                            h_next = h_csx(next_s,ob_i_next);      
                            //printf("{{cbf_type}}:= %d", {{cbf_type}});                                            
                        }                                                      
                        else{
                            h = h_x(s,ob_i);               
                            h_next = h_x(next_s,ob_i_next);           
                        }

                        //h = 0;
                        //h_next = 0;
                        if({{cbf_type}} == 3){ 
                            cbf_cost_i = - h;  // 计算 DC 的代价           
                        }
                        else{
                            cbf_cost_i = -h_next + {{dcbf_alpha}} * h;  // 计算 CBF 的代价 
                        }
                                       
                        cbf_cost_i = {{dcbf_weight}} * clamp_to_non_negative(cbf_cost_i);  // 将 CBF 代价乘以权重，并限制其值
                        cbf_cost += cbf_cost_i;  // 将 CBF 代价加到新的代价中                                               
                        }                                       
                    }

                state_cost += cbf_cost;             

                float dv2 = (next_s[3] - s[3])*(next_s[3] - s[3]) +  (next_s[4] - s[4])*(next_s[4] - s[4]);
                float dw = (next_s[2] - s[2])*(next_s[2] - s[2]);                                 

                float acc_cost = 0;
                acc_cost = {{dcbf_weight}} * (clamp_to_non_negative(dv2-0.01) + clamp_to_non_negative(dw-0.01));                      
                state_cost += acc_cost;

                return state_cost;
            }
            /*                        
            @brief: get_state_cost(): for computing the state-dependent running cost. 
                                       计算状态成本，公式如下 state_cost= ∑_i weights_i⋅(s_i - targets_i) ^2
                                       加入扩展椭圆的动态cbf
                                       引入松弛变量
            */
            __device__ float get_state_cost_with_slack(float* next_s, float* s, float* weights, float* targets, int timestep, int timesteps, float* ob, int ob_num, float* ob_slack)
            {
                float state_cost = 0;

                //for (int i = 0; i < {{state_dim}}; i++) {
                for (int i = 0; i < 3; i++) {                       
                    state_cost += weights[i]*(s[i] - targets[i])*(s[i] - targets[i]);                
                }
                //printf("state_cost=%f",state_cost);                       
                //printf("timesteps=%d",timesteps);  
                float ob_i[7] = {0,0,0,0,0,0,0};
                float ob_i_next[7] = {0,0,0,0,0,0,0};                              
                float h,h_next,cbf_cost_i,cbf_cost=0;

                //printf("ob_num=%d",ob_num);
                if(ob_num!=0){
                    for (int i = 0; i < ob_num; i++){
                        //ob_i[0] = ob[ 7 * timesteps * i + 7 * timestep + 0 ];// x
                        //ob_i[1] = ob[ 7 * timesteps * i + 7 * timestep + 1 ];// y   
                        //ob_i[2] = ob[ 7 * timesteps * i + 7 * timestep + 2 ];// a
                        //ob_i[3] = ob[ 7 * timesteps * i + 7 * timestep + 3 ];// b
                        //ob_i[4] = ob[ 7 * timesteps * i + 7 * timestep + 4 ];// theta
                        //ob_i[5] = delta(s,ob_i);//椭圆与机器人夹角      
                        //ob_i[6] = l_x(ob_i);//椭圆与机器人 直线方向，椭圆中心到椭圆边界的距离的平方 
                        //printf("b=%f           |",ob_i[3]);               
                        //printf("l=%f           |",ob_i[6]);
                                                      
                        //h = h_x_(s,ob_i);
                        //printf("h=%f           |",h);
                        //ob_i_next[0] = ob[ 7 * timesteps * i + 7 * (timestep+1) + 0 ];
                        //ob_i_next[1] = ob[ 7 * timesteps * i + 7 * (timestep+1) + 1 ];   
                        //ob_i_next[2] = ob[ 7 * timesteps * i + 7 * (timestep+1) + 2 ];
                        //ob_i_next[3] = ob[ 7 * timesteps * i + 7 * (timestep+1) + 3 ];   
                        //ob_i_next[4] = ob[ 7 * timesteps * i + 7 * (timestep+1) + 4 ];                                       
                        //ob_i_next[5] = delta(s,ob_i_next);//椭圆与机器人夹角      
                        //ob_i_next[6] = l_x(ob_i_next);//椭圆与机器人 直线方向，椭圆中心到椭圆边界的距离的平方 
                        //h_next = h_x_(next_s,ob_i_next);
                        // 从这里开始
                                       
                        ob_i[0] = ob[ 7 * timesteps * i + 7 * timestep + 0 ];// x
                        ob_i[1] = ob[ 7 * timesteps * i + 7 * timestep + 1 ];// y   
                        ob_i[2] = ob[ 7 * timesteps * i + 7 * timestep + 2 ];// a
                        //ob_i[3] = ob[ 7 * timesteps * i + 7 * timestep + 3 ];// b 感知无数据
                        //ob_i[4] = ob[ 7 * timesteps * i + 7 * timestep + 4 ];// theta 感知无数据
                        ob_i[5] = ob[ 7 * timesteps * i + 7 * timestep + 5 ];// dx  
                        ob_i[6] = ob[ 7 * timesteps * i + 7 * timestep + 6 ];// dy    
                                        
                        ob_i_next[0] = ob[ 7 * timesteps * i + 7 * (timestep+1) + 0 ];
                        ob_i_next[1] = ob[ 7 * timesteps * i + 7 * (timestep+1) + 1 ];   
                        ob_i_next[2] = ob[ 7 * timesteps * i + 7 * (timestep+1) + 2 ];
                        //ob_i_next[3] = ob[ 7 * timesteps * i + 7 * (timestep+1) + 3 ];   
                        //ob_i_next[4] = ob[ 7 * timesteps * i + 7 * (timestep+1) + 4 ];                                       
                        ob_i_next[5] = ob[ 7 * timesteps * i + 7 * (timestep+1) + 5 ];
                        ob_i_next[6] = ob[ 7 * timesteps * i + 7 * (timestep+1) + 6 ];
                        //检查障碍物与机器人是否正在接近               
                        if(check_danger(s,ob_i)){
                            //扩展障碍为椭圆               
                            exp_ellipse(s,ob_i,ob_slack[i]);
                            exp_ellipse(next_s,ob_i_next,ob_slack[i]);
                            if(ob_i[2]!=ob_i[3] && ob_i_next[2]!=ob_i_next[3]){
                                ob_i[5] = delta(s,ob_i);//椭圆与机器人夹角      
                                ob_i[6] = l_x(ob_i);//椭圆与机器人 直线方向，椭圆中心到椭圆边界的距离的平方                                                    
                                ob_i_next[5] = delta(s,ob_i_next);//椭圆与机器人夹角      
                                ob_i_next[6] = l_x(ob_i_next);//椭圆与机器人 直线方向，椭圆中心到椭圆边界的距离的平方 
                                h = h_x_(s,ob_i);               
                                h_next = h_x_(next_s,ob_i_next); 
                                        }              
                            else{
                                h = h_x(s,ob_i);               
                                h_next = h_x(next_s,ob_i_next);                                         
                                        }    
                            ob_i[5] = delta(s,ob_i);//椭圆与机器人夹角      
                            ob_i[6] = l_x(ob_i);//椭圆与机器人 直线方向，椭圆中心到椭圆边界的距离的平方                                                    
                            ob_i_next[5] = delta(s,ob_i_next);//椭圆与机器人夹角      
                            ob_i_next[6] = l_x(ob_i_next);//椭圆与机器人 直线方向，椭圆中心到椭圆边界的距离的平方 
                            h = h_x_(s,ob_i);               
                            h_next = h_x_(next_s,ob_i_next);
                        }
                        else{
                            h = h_x(s,ob_i);               
                            h_next = h_x(next_s,ob_i_next);           
                                       }                              
                        
                            //h = 0;
                            //h_next = 0;

                                       // 非线性软约束为 * || 线性软约束为 -           
                            //cbf_cost_i = -h_next - ob_slack[i] + {{dcbf_alpha}} * h;  // 计算 CBF 的代价
                                       
                            cbf_cost_i = -h_next + {{dcbf_alpha}} *  h;  // 计算 CBF 的代价           
                                       
                            //if(cbf_cost_i > 0 ){printf("cbf_cost_i=%f           |",cbf_cost_i); }
                                       
                            cbf_cost_i = {{dcbf_weight}} * clamp_to_non_negative(cbf_cost_i);  // 将 CBF 代价乘以权重，并限制其值
                                       
                            cbf_cost += cbf_cost_i;  // 将 CBF 代价加到新的代价中                                               
                        }                                       
                    }

                                                      
                state_cost += cbf_cost;

                return state_cost;
            }
            """)
        return (state_cost_template.render(
            state_dim=self.state_dim,
            robot_r=self.r,
            safety_dist=self.safety_dist,
            atau = self.atau,
            ob_states_num=self.ob_states_num,
            dcbf_alpha = self.dcbf_alpha,
            cbf_type = self.cbf_type,
            dcbf_weight = self.dcbf_weight))