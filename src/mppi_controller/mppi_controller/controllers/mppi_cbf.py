"""
@authors: Grady Williams, Autonomous Control and Decision Systems Laboratory, Georgia Institute of Technology, USA
          Ihab S. Mohamed, Vehicle Autonomy and Intelligence Lab, Indiana University, Bloomington, USA        
"""
"""
@brief: The implementation of the MPPI control strategy as proposed by Williams in "Model predictive
        path integral control: From theory to parallel computation", as well as its extension to log-MPPI. 
"""
import numpy as np
from scipy import interpolate
import scipy.signal

from jinja2 import Template

import warnings
import sys
import logging

import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

# Import pyCuda modules for computations
from pycuda.curandom import XORWOWRandomNumberGenerator
from pycuda import gpuarray

logger = logging.getLogger(__name__)

class MPPI_Controller:
    """
    Model predictive path integral controller. Computes an approximation of an optimal open loop
    control sequence by sampling and evaluation trajectories from the system dynamics. Sampling
    is performed in parallel on a GPU using the pycuda module, a python interface to Nvidia's CUDA
    architecture, this Requires an Nvidia GPU. The costs, dynamics, and (optionally) an initial
    policy to sample around are given as arguments to the constructor.

    Methods:
    default_policy -- The default control policy (All zeros) which the controller samples around. 
    default_initialization_policy -- Policy for initializing new controls.
    initialize_controls -- Allow the controller iterate many times when starting a new task.
    reset_controls -- Reset all the control commands to zero.
    get_cuda_functions -- Generate and compiles CUDA code.
    debug_printout -- Give a nice printout of the generated CUDA code with line numbers.
    params_to_cuda -- Transfer cost, policy, and dynamics parameters to device (GPU) memory.
    numerical_check -- Check for Nans/Infs after transferring variables from device to host memory.
    rollouts -- Sample and evaluate trajectories and compute the weighted average over control sequences.
    spline_controls -- Smooth the resulting control sequence by fitting a spline.
    polyfit -- Smooth the resulting control sequence by fitting a polynomial.
    savitsky_galoy --  Smooth the resulting control sequence by Savitsky Galoy filter.
    compute_control -- Given the current state, return an approximation to the optimal controls.
    on_gpu -- Transform numpy array into a gpuarray.
    default_cuda_policy -- CUDA code version of default_policy.
    cuda_rollouts -- CUDA code for sampling and evaluating system trajectories.

    这段代码定义了一个名为 MPPI_Controller 的类，主要用于模型预测路径积分控制（MPPI）。
    该控制器通过从系统动态中采样和评估轨迹来计算最优开放循环控制序列的近似值，
    特别适合于使用 Nvidia GPU 进行并行计算。以下是对类和构造函数的逐步解释：

    类注释
        MPPI_Controller: 计算最优控制序列的类，支持 GPU 加速。
    方法概述
        default_policy: 默认控制策略（全零）。
        initialize_controls: 初始化控制以允许多次迭代。
        reset_controls: 重置控制命令为零。
        rollouts: 采样和评估轨迹，计算控制序列的加权平均值。
        spline_controls: 通过拟合样条平滑控制序列。
        compute_control: 返回当前状态下的近似最优控制。
    构造函数解释
        参数: 包括状态维度、控制维度、样本数、时间范围等多个输入参数，定义控制器的行为。
        状态和控制: 初始化控制序列和状态序列为零。
        GPU 随机数生成器: 使用 XORWOWRandomNumberGenerator 生成随机数。
        分布类型: 根据 LogN_info 中的分布类型，生成正态分布或对数正态分布的样本。
        参数设置: 设置平滑选项、控制滤波器等。
        动态和成本代码: 根据提供的动态和成本参数，保存编译后的 CUDA 代码。
        重要性采样策略: 设置默认策略或用户定义策略。
    """
    """
    模型预测路径积分控制器。通过从系统动态中采样和评估轨迹来计算最优开环控制序列的近似值。
    采样是在GPU上并行执行的，使用pycuda模块，这是Nvidia CUDA架构的Python接口，
    这需要一个Nvidia GPU。成本、动态和（可选）初始策略作为参数传递给构造函数。

    方法：
        default_policy -- 控制器采样周围的默认控制策略（全部为零）。
        default_initialization_policy -- 初始化新控制的策略。
        initialize_controls -- 当开始新任务时，允许控制器多次迭代。
        reset_controls -- 将所有控制命令重置为零。
        get_cuda_functions -- 生成并编译CUDA代码。
        debug_printout -- 带有行号的生成CUDA代码的漂亮打印输出。
        params_to_cuda -- 将成本、策略和动态参数传输到设备（GPU）内存。
        numerical_check -- 在从设备传输变量到主机内存后检查Nans/Infs。
        rollouts -- 采样和评估轨迹，并计算控制序列的加权平均值。
        spline_controls -- 通过拟合样条平滑结果控制序列。
        polyfit -- 通过拟合多项式平滑结果控制序列。
        savitsky_galoy -- 通过Savitsky Galoy滤波器平滑结果控制序列。
        compute_control -- 给定当前状态，返回最优控制的近似值。
        on_gpu -- 将numpy数组转换为gpuarray。
        default_cuda_policy -- CUDA代码版本的default_policy。
        cuda_rollouts -- 用于采样和评估系统轨迹的CUDA代码。
    """
    def __init__(self,
                 state_dim,
                 control_dim,
                 num_samples,
                 draw_num_traj,
                 time_horizon,
                 control_freq,
                 exploration_variance,
                 kinematics,
                 state_costs,
                 SG_window,
                 SG_PolyOrder,
                 LogN_info,
                 cbf_type,        
                 initialization_policy=None,
                 num_optimization_iterations=1,
                 policy_args=None,
                 block_dim=(32, 1, 1),
                 spline_smoothing=True,
                 check_dist = 6.0,                 
                 beta_1 = 0.5,
                 beta_2 = 0.5,
                 beta_3 = 0.5,
                 lambda_=1.0,
                 atau = 0.0,
                 cost_range=(-100000000, 10000000000)):
        """
        初始化类字段并编译并保存CUDA函数以供后续使用。

        参数：
            state_dim, control_dim -- 状态和控制维度大小。
            num_samples -- 每个时间步采样的轨迹数量。
            time_horizon -- 每个轨迹样本的长度（秒）。
            control_freq -- 控制输入需要返回的频率。
            exploration_variance -- 控制系统的自然随机方差。
            dynamics -- 可以编译的cuda代码 -OR- 包含可编译cuda代码和返回参数名称/值对字典的可调用对象的元组/列表。这些名称应该与返回的cuda代码中的全局常量数组名称匹配。
            state_costs -- 与dynamics相同，但是返回的是成本而不是下一个状态。
            SG_window, SG_PolyOrder -- Savitzky-Golay滤波器的参数（如果使用），其中
                    SG_window：滤波器窗口的长度，
                    SG_PolyOrder：用于拟合样本的多项式的阶数。
            LogN_info -- [dist_type, mu_LogN, std_LogN]列表，其中
                        dist_type：0（正态），1：（正态和对数正态），
                        mu_LogN：对数正态分布的“均值”，
                        std_LogN：对数正态分布的“标准差”。

        关键字参数：
            initialization_policy -- 可调用对象，它接受当前状态并返回控制输入。
            num_optimization_iterations -- 每个时间步运行的采样迭代次数。
            policy_args -- 一个元组/列表，包括（1）一个可调用对象，它根据当前状态返回控制输入，（2）执行相同功能的cuda代码，以及（3）返回参数名称和值的参数更新器。这些参数名称应该与返回的cuda代码中的全局常量数组名称匹配。
            block_dim -- cuda块在x方向的维度。 （Y和Z目前为1。）
            spline_smoothing -- 是否在优化后平滑控制序列。
            lambda_ -- 当计算加权平均值时的softmax温度。零对应于未加权平均值，无穷大对应于max函数。
            cost_range -- 成本函数的有效值范围。
        """
        self.state_dim = state_dim
        self.dist_type, self.mu_LogN, self.std_LogN = LogN_info
        self.cbf_type = cbf_type
        self.atau = atau
        self.control_dim = control_dim
        self.block_dim = block_dim
        """ Note that the actual number of sampled trajectories depends on the dimension of Cuda blocks in the x-direction """  
        self.num_samples = block_dim[0] * (
            (num_samples - 1) // self.block_dim[0] + 1)
        print(f"num_samples =: {self.num_samples}")
        self.num_timesteps = int(time_horizon * control_freq)
        self.draw_num_traj = draw_num_traj
        # Nominal control sequence.
        self.U = np.zeros((self.num_timesteps, self.control_dim),
                          dtype=np.float32)
        
        # 初始化控制信号序列 加快模拟
        self.U[:, 0] = 1  # 将第一列设置为1，初始化

        self.last_U = np.zeros((self.num_timesteps, self.control_dim),
                               dtype=np.float32)
        self.draw_sample_U = [np.zeros((self.draw_num_traj, self.num_timesteps * self.control_dim),
                          dtype=np.float32)] # 绘制50个轨迹
        # Nominal sequence of states.
        self.nominal_sequence = np.zeros(self.num_timesteps * self.state_dim,
                                         dtype=np.float32)
        self.dt = 1.0 / control_freq
        # GPU random number generator
        self.generator = XORWOWRandomNumberGenerator()
        # For MPPI
        if self.dist_type == 0:
            du_d = self.generator.gen_normal(
                self.num_samples * self.num_timesteps * self.control_dim,
                np.float32)
            logger.info("Trajectories are sampled from Normal dist.")
        # log-MPPI
        elif self.dist_type == 1:
            logger.info(
                "Trajectories are sampled from Normal & Log-normal dist.")
            du_LogN_d = self.generator.gen_log_normal(
                self.num_samples * self.num_timesteps * self.control_dim,
                np.float32, self.mu_LogN, self.std_LogN)
            du_d = du_LogN_d * self.generator.gen_normal(
                self.num_samples * self.num_timesteps * self.control_dim,
                np.float32)
        elif self.dist_type == 2: # 
            du_d = self.generator.gen_normal(
                self.num_samples * self.num_timesteps * self.control_dim,
                np.float32)
            
        self.du = du_d.get()
        self.exploration_variance = exploration_variance
        self.num_optimization_iterations = num_optimization_iterations
        self.spline_smoothing = spline_smoothing
        self.check_dist = check_dist
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.beta_3 = beta_3
        self.lambda_ = lambda_
        self.SG_window = SG_window
        self.SG_PolyOrder = SG_PolyOrder
        self.cost_range = cost_range
        self.control_filter = np.convolve(np.array([0, 0, 1, 0, 0]),
                                          np.array([1, 0, 0]))
        if initialization_policy is None:
            self.initialization_policy = self.default_initialization_policy
        else:
            self.initialization_policy = initialization_policy
        self.param_dict = {
            "lambda_": np.array([self.lambda_]),
            "control_filter": self.control_filter,
            "du": self.du
        }

        # Save the dynamics code and dynamics param updater if there is one to save.
        if (type(kinematics) is list or type(kinematics) is tuple):
            kinematics_code = kinematics[0]
            self.kinematics_param_getter = kinematics[1]
            self.kinematics_arrs = self.kinematics_param_getter()
        else:
            kinematics_code = kinematics
            self.kinematics_param_getter = None
            self.kinematics_arrs = {}

        # Save the costs code and cost param updater
        if (type(state_costs) is list or type(state_costs) is tuple):
            costs_code = state_costs[0]
            self.costs_param_getter = state_costs[1]
            self.costs_arrs = self.costs_param_getter()
        else:
            costs_code = state_costs
            self.costs_param_getter = None
            self.costs_arrs = {}
        
        # Set the importance sampling policy.
        if (policy_args is None):
            self.policy = self.default_policy
            cuda_policy = self.default_cuda_policy()
            self.policy_param_getter = None
            self.policy_arrs = {}
        if (policy_args is not None):
            self.policy = policy_args[0]
            cuda_policy = policy_args[1]
            self.policy_param_getter = policy_args[2]
            self.policy_arrs = self.policy_param_getter()
        
        # Generate and compile CUDA code for sampling and evaluating trajectories.
        self.cuda_functions = self.get_cuda_functions(cuda_policy,
                                                      kinematics_code,
                                                      costs_code)
    """ @brief: The default control policy (All zeros) which the controller samples around """    
    def default_policy(self, x):
        """Returns a zero control command
            这些函数定义了默认控制策略，返回全零控制向量，适用于所有状态。"""
        return np.zeros(self.control_dim)

    def default_initialization_policy(self, x, args=[]):
        """Returns a zero control command
            可能用于初始化控制器的起始控制输入。在系统尚未学习到任何控制策略时，默认使用零输入作为控制信号"""
        return np.zeros(self.control_dim)

    """ @brief: Allow the controller to iterate to convergence before starting a new task """
    def initialize_controls(self,
                            R,
                            std_n,
                            weights,
                            targets,
                            ob,
                            init_state=None,
                            num_iters=350):
        """
        Allow the controller to iterate to convergence before starting a new task, (i.e. let it sit and
        think for a moment before acting). This is plausible for some tasks (swinging up a cart pole for
        instance), but less so for others. Usage is optional.

        **功能: 控制器在执行任务之前通过多次迭代 num_iters 来初始化控制信号 
                这种方法允许在执行新任务之前有一定的“预思考时间”。

        Arguments:
        R -- Control cost matrix.
        std_n -- Standard deviation of the injected control noise.
        weights, targets -- Cost parameters.
        """

        # 初始化控制信号序列为零
        self.U = np.zeros((self.num_timesteps,self.control_dim), dtype = np.float32)
        # self.U = np.random.randn(self.num_timesteps, self.control_dim) * 0.05
        """for i in range(self.num_timesteps):
            self.U[i,:] = self.initialization_policy(init_state)"""
        if init_state is not None: # 如果传递了初始状态，那么使用初始状态进行轨迹模拟。
            state = np.copy(init_state)
            print(state)
            # 通过多次调用轨迹采样函数 rollouts 来优化控制策略，生成并评估不同的控制信号扰动（带噪声的控制轨迹）。
            for i in range(num_iters):
                self.rollouts(state, std_n, R, weights, targets, ob)

    """ @brief: Reset the whole control sequence to the initial policy.
            重置控制器的控制信号序列到默认初始化策略（即全零控制策略）。"""           
    def reset_controls(self):
        for i in range(self.num_timesteps):
            #  遍历每个时间步 i，为每个时间步的控制信号设置为初始化策略（通常为零信号）。
            self.U[i, :] = self.initialization_policy(np.zeros(self.state_dim))

    """ @brief: Compile the CUDA code and returns a callable function to perform parallel sampling, and 
            returns addresses for CUDA constant arrays U_d and policy_params 
                编译CUDA代码 并返回用于执行并行轨迹采样的CUDA函数指针。"""
    def get_cuda_functions(self, cuda_policy, kinematics, state_costs):
        """
        Generate compiled cuda code which can be called during optimization.

        Arguments:
        cuda_policy -- string of cuda code for making policy predictions.
        dynamics -- string of cuda code for making dynamics predictions.
        state_costs -- string of cuda code for computing state_costs
        """
        # 将各部分的CUDA代码（控制策略、运动学预测、状态成本计算）拼接起来，形成一个完整的CUDA核函数。
        rollout_kernel = self.cuda_headers(
        ) + cuda_policy + kinematics + state_costs + self.cuda_rollouts()
        # First see if the compilation is successful
        # 尝试编译CUDA代码。如果编译失败，会通过 debug_printout 打印出带有行号的CUDA代码，帮助调试。
        try:
            SourceModule(rollout_kernel)
        except cuda.CompileError:
            """ If compilation is not succesful print the code with linenumbers and the pycuda error message """
            self.debug_printout(rollout_kernel)
        """ If we were successful compile the code, otherwise this will tell us the error
            and linenumber where it occured. """
        mod = SourceModule(rollout_kernel)

            # 返回已编译的CUDA核函数，用于执行轨迹采样。
        func = mod.get_function("rollout_kernel") 
             # 控制信号的GPU地址。
        U_d = mod.get_global("U_d")[0] 

        # Get the addresses for policy, dynamics and cost parameters
        policy_adrs = {} # 返回控制策略 的GPU地址
        for arr_name in self.policy_arrs:
            policy_adrs[arr_name] = mod.get_global(arr_name)[0]

        kinematics_adrs = {} # 运动学 的GPU地址
        for arr_name in self.kinematics_arrs:
            kinematics_adrs[arr_name] = mod.get_global(arr_name)[0]

        costs_adrs = {} # 成本参数 的GPU地址
        for arr_name in self.costs_arrs:
            costs_adrs[arr_name] = mod.get_global(arr_name)[0]
            # 以便在CUDA核中访问这些参数
        return func, U_d, policy_adrs, kinematics_adrs, costs_adrs

    """ @brief: Print out cuda code in an easy to read format"""
    def debug_printout(self, rollout_kernel):
        rollout_kernel_debug = rollout_kernel.split('\n')
        count = 0
        print("CUDA compilation failed")
        print()
        print("=====================================")
        print()
        for line in rollout_kernel_debug:
            sys.stdout.write("%d %s \n" % (count, line))
            count += 1
        print()
        print("=====================================")

    """ @brief: Transfer policy, dynamics, and cost parameters to device memory 
            将控制策略、运动学和成本相关的参数从CPU内存转移到GPU设备内存。"""    
    def params_to_cuda(self, policy_params_adrs, kinematics_params_adrs,
                       costs_params_adrs):
        # policy_param_getter returns a dict of keynames and arrays
         # self.policy_param_getter() 获取策略参数并通过 cuda.memcpy_dtod 将其从CPU复制到GPU上的指定地址。
        if (self.policy_param_getter is not None): 
            policy_params = self.policy_param_getter()
            for key in policy_params_adrs:
                gpu_arr = self.on_gpu(policy_params[key]) # 将数组转换为GPU上的数据结构。
                cuda.memcpy_dtod(policy_params_adrs[key], gpu_arr.ptr,
                                 gpu_arr.nbytes)
        
        # Transfer the kinematics parameters to CUDA constant memory
        if (self.kinematics_param_getter is not None):
            kinematics_params = self.kinematics_param_getter()
            for key in kinematics_params_adrs:
                gpu_arr = self.on_gpu(kinematics_params[key])
                cuda.memcpy_dtod(kinematics_params_adrs[key], gpu_arr.ptr,
                                 gpu_arr.nbytes)

        # Transfer the cost parameters to CUDA constant memory
        if (self.costs_param_getter is not None):
            costs_params = self.costs_param_getter()
            for key in costs_params_adrs:
                gpu_arr = self.on_gpu(costs_params[key])
                cuda.memcpy_dtod(costs_params_adrs[key], gpu_arr.ptr,
                                 gpu_arr.nbytes)

    """ @brief: Check if any returned values have Nans/Infs
        检查成本或控制扰动中是否存在非数值 (NaN) 或无穷大 (Inf) 的值，以确保数值计算的稳定性。 """ 
    def numerical_check(self, costs, control_variations):
        # Do some NaN/Infinity checking
        fail = False
        # 检查 costs 或 control_variations 中是否有NaN。
        # 如果有，将其替换为无穷大或使用 nan_to_num 进行修正
        if np.isnan(np.sum(costs)):
            warnings.warn("Nan Deteced in Costs", UserWarning)
            indices = np.isfinite(costs)
            for i in range(self.num_samples):
                if (not indices[i]):
                    costs[i] = float("inf")
            fail = True
        if (np.isnan(np.sum(control_variations))):
            control_variations = np.nan_to_num(control_variations)
            warnings.warn("Nan/Infinity deteced in control variations",
                          UserWarning)
            fail = True
        if (np.sum(costs) == 0): # 如果所有轨迹的成本和为0，说明有异常情况，发出警告。
            warnings.warn("Normalizer is zero", UserWarning)
            fail = True
        return fail

    """ @brief: 计算控制器内部维护的控制序列的更新 """
    def rollouts(self, state, std_n, R, weights, targets, ob, ob_num, cost_baseline=None):
        """
        在 GPU 上生成和评估轨迹。然后计算轨迹的加权平均奖励，并根据路径
        积分更新法则更新名义控制序列。

          参数：
            state -- 当前状态。
            R -- 控制成本矩阵。
            std_n -- 注入控制噪声的标准差。
            weights, targets -- 成本参数。
            cost_baseline：成本基线，若为空则使用最小成本作为基线。
        返回：
            normalizer -- 轨迹的指数成本之和。当控制器表现良好时，这个值非常高，否则很小。
                    Normalizer < 1 是一个坏信号。
            min_cost -- 所有轨迹的最小成本。
        """

        # 初始化和噪声生成
        self.lambda_ = self.param_dict["lambda_"][0]
        if self.dist_type == 0: # 
            du_d = self.generator.gen_normal(
                self.num_samples * self.num_timesteps * self.control_dim,
                np.float32)
        elif self.dist_type == 1:
            du_LogN_d = self.generator.gen_log_normal(
                self.num_samples * self.num_timesteps * self.control_dim,
                np.float32, self.mu_LogN, self.std_LogN)
            du_d = du_LogN_d * self.generator.gen_normal(
                self.num_samples * self.num_timesteps * self.control_dim,
                np.float32)
        elif self.dist_type == 2: # 
            du_d = self.generator.gen_normal(
                self.num_samples * self.num_timesteps * self.control_dim,
                np.float32)
        self.du = du_d.get()

        self.control_filter = self.param_dict["control_filter"]
        # 将数组传输到 GPU
        std_nd = self.on_gpu(std_n, dtype=np.float32)
        R_d = self.on_gpu(R, dtype=np.float32)
        du_d = self.on_gpu(self.du, dtype=np.float32)
        # print(f"dx: {state[3]}")
        # print(f"dy: {state[4]}")
        state_d = self.on_gpu(state)
        costs_d = gpuarray.zeros(self.num_samples, dtype=np.float32)
        nominal_sequence_d = gpuarray.zeros(self.num_timesteps *
                                            self.state_dim,
                                            dtype=np.float32)
        
        ob = np.array(ob)
        ob_num = np.array(ob_num)
        # print(f"ob:{ob}")
        # print(f"ob_num:{ob_num}")
        ob_d = self.on_gpu(ob)# 障碍物
        ob_num_d = self.on_gpu(ob_num)

        U_d = self.on_gpu(self.U)
        weights_d = self.on_gpu(weights)
        targets_d = self.on_gpu(targets)
        # 卸载 CUDA 函数和地址
        rollout_kernel, U_d_adr, policy_params_adrs, kinematics_params_adrs, costs_params_adrs = self.cuda_functions
        # 将当前控制序列传输到 CUDA 常量内存
        cuda.memcpy_dtod(U_d_adr, U_d.ptr, U_d.nbytes)
        # 将策略参数传输到 CUDA 常量内存
        self.params_to_cuda(policy_params_adrs, kinematics_params_adrs,
                            costs_params_adrs)
        
            # 这一部分通过CUDA调用，设置并行计算的块大小（blocksize）和网格大小（gridsize），
            # 并通过rollout_kernel内核函数在GPU上并行模拟多个轨迹（rollouts）。
            # rollout_kernel会模拟每个轨迹并计算其成本
        # 设置 rollout 和成本-to-go 核的 blocksize 和 gridsize
        blocksize = self.block_dim
        gridsize = ((self.num_samples - 1) // self.block_dim[0] + 1, 1, 1)
        # 启动模拟 rollouts 的核
        rollout_kernel(costs_d,
                       du_d,
                       state_d,
                       std_nd,
                       R_d,
                       weights_d,
                       targets_d,
                       nominal_sequence_d,
                       ob_d,
                       ob_num_d,
                       grid=gridsize,
                       block=blocksize)
        cuda.Context.synchronize() # 强制CUDA核完成所有并行计算，以确保所有数据已计算完毕，避免数据冲突。
        # 从 GPU 检索成本和控制变化
        costs = costs_d.get() #  从GPU中检索每条轨迹的成本，表示为一个向量，每个元素对应一条轨迹的累积成本。
        
         # 从GPU中检索生成的控制扰动 du_d，并重新reshape为形状 (控制维度×时间步×样本数)
        control_variations = du_d.get().reshape(  # 
            (self.num_samples, self.num_timesteps * self.control_dim)).T # 生成的控制信号扰动。
        
        indices = np.linspace(0, self.num_samples - 1, self.draw_num_traj).astype(int)
        self.draw_sample_U = np.copy(control_variations.T[indices, :])

        # self.draw_sample_U = np.copy(control_variations.T[:self.draw_num_traj, :]) # 用于绘制动图
        # 从GPU中检索“标称轨迹”，即在不加入噪声的情况下得到的系统状态序列。
        self.nominal_sequence = nominal_sequence_d.get().reshape(
            (self.num_timesteps, self.state_dim)) 
        # 计算控制更新
        min_cost = np.min(costs) # 计算最小成本 min_cost
        if (cost_baseline is None):
            cost_baseline = min_cost # 并确定基准成本 cost_baseline

        
        """
                *使用路径积分方法，通过指数变换来将成本转化为权重 transformed_costs。

                *数学原理： 此时，成本值 J_i已经计算完成。
                为了将这些成本转化为权重，我们需要一个基线（即成本的参考值），
                这里选择最小成本 min(J_i) 作为基线。
                基线成本的选择会影响后续的指数加权计算。
        """
        transformed_costs = np.exp(-(1.0 / self.lambda_) *
                                   (costs - cost_baseline))
        # 其中 
            # λ 是温度参数，控制权重的衰减速度。较大的 
            # λ 会使得权重分布更均匀，而较小的 
            # λ 则会使成本较低的轨迹占据更大权重。
        normalizer = np.sum(transformed_costs) # 计算所有轨迹的权重之和，用于归一化这些权重。
        """
                w_i除以 normalizer 使其总和为1 
                归一化后的权重 w_i表示轨迹 i 对最终控制更新的贡献程度。
        """
        costs = transformed_costs # 
        fail = self.numerical_check(costs, control_variations) # 检查数值稳定性，确保权重和控制变化没有出现数值问题。
        costs /= normalizer # 
        """
            将每条轨迹的控制扰动​du_t按照权重w_i进行加权平均:
            这一步计算的是每个时间步 t 的最终控制更新 Δu_t
             ，结果保存在 control_update 中。 
        """
        control_variations *= costs
        control_update = np.sum(control_variations, axis=1).reshape(
            (self.num_timesteps, self.control_dim))
        control_update = control_update.reshape(
            (self.num_timesteps, self.control_dim))
        # control_update = self.spline_controls(control_update)

        # 使用 Savitzky-Golay 滤波器对控制信号进行平滑处理。
        # 这种滤波器通过多项式拟合来消除控制信号中的噪声，使得控制信号更加平滑，适合实际应用。
        control_update = self.savitsky_galoy(control_update) 
        # control_update = self.polyfit(control_update)

        # 检查control_update的值是否有效（即没有无穷大或NaN值）
        # 然后将更新后的控制序列赋值给 self.U。
        if np.isfinite(control_update).all():
            self.U = control_update.reshape(
                (self.num_timesteps, self.control_dim))
        return normalizer, min_cost

    """@brief: Smooth the resulting control sequence by Savitsky Galoy filter
        使用 Savitzky-Golay 滤波器对控制信号 control_update 进行平滑处理。"""    
    def savitsky_galoy(self, control_update):
        new_update = np.zeros_like(control_update)
        for i in range(self.control_dim): # 对每个控制维度单独应用滤波器。
            new_update[:, i] = scipy.signal.savgol_filter(control_update[:, i], # 
                                                          self.SG_window,# 滤波器窗口大小
                                                          self.SG_PolyOrder, # 边界处理方式为镜像扩展，用于减少边界效应。
                                                          mode='mirror') # 边界处理方式为镜像扩展，用于减少边界效应
        return new_update # 边界处理方式为镜像扩展，用于减少边界效应

    """@brief: Smooth the resulting control sequence by fitting a polynomial 
            使用五阶多项式拟合对控制信号进行平滑处理。"""
    def polyfit(self, control_update):
        new_update = np.zeros_like(control_update)
        for i in range(self.control_dim):
            # fit = np.polyfit(range(len(new_update)), control_update[:,i], 2) # get the coeff. of the poly (2nd order)
            # 对每个维度的控制信号 control_update 拟合一个五阶多项式（5个系数）。
            fit = np.polyfit(range(len(new_update)), control_update[:, i],
                             5)  # get the coeff. of the poly (3nd order)
            for j in range(len(new_update)):
                # new_update[j,i] = fit[0]* np.square(j) + fit[1]*j + fit[2] # 2nd order poly
                # new_update[j,i] = fit[0] * np.square(j) * j + fit[1] * np.square(j) + fit[2] * j + fit[3] # 3rd order poly
                # 对于每个时间步 j，根据拟合出的多项式系数 fit 计算新的平滑控制信号。
                new_update[j, i] = fit[0] * np.square(j) * np.square(
                    j) * j + fit[1] * np.square(j) * np.square(
                        j) + fit[2] * np.square(j) * j + fit[3] * np.square(
                            j) + fit[4] * j + fit[5]  # 5rd order poly
        return new_update # 返回平滑后的控制信号

    """@brief: Smooth the resulting control sequence by fitting a spline 
            使用样条曲线拟合对控制信号进行平滑处理。""" 
    def spline_controls(self, control_update, spline_pwr=3):
        """Fits a spline to the current nominal control sequence"""
        # knots: 生成用于样条拟合的节点，节点数与时间步相关，按 .1 秒间隔采样。
        knots = np.linspace(0, self.num_timesteps,
                            int((self.num_timesteps * self.dt) / .1))[1:-1]
        old_update = np.copy(control_update)
        for i in range(self.control_dim):
            # 对控制信号进行样条拟合，k=spline_pwr 表示样条的阶数（默认为3，即三次样条）。
            spline_params = interpolate.splrep(range(self.num_timesteps),
                                               old_update[:, i],
                                               k=spline_pwr,
                                               t=knots)
            # 使用拟合参数 spline_params 计算在每个时间步的平滑控制信号。
            control_update[:, i] = interpolate.splev(range(self.num_timesteps),
                                                     spline_params)
        return control_update # 返回经过样条拟合后的平滑控制信号 control_update。

    """@brief: Compute an approximation to the optimal control """   
    def compute_control(self, state, cost_params, cost_baseline=None):
        """

        Arguments:
        state -- The current state  
        cost_params -- tuple/list containing R (control cost matrix), std_n (exploration standard deviation),
                       and the weights and targets (cost parameters).

        Keywards Arguments:
        cost_baseline -- Value to subtract from the costs when computing the control update. If None
                         then the minimum value of all the sampled trajectories is used.

        Returns:
        u - control to execute.
        normalizer - sum of the exponentiated trajectory costs.
        min_cost - the minimum cost over all the trajectories.
        """

        """
            Arguments:
            state -- 当前状态
            cost_params -- 包含 R (控制代价矩阵), 
                                std_n (探索标准差),
                                    weights 和 targets (成本参数) 的元组。
            
            Keywards Arguments:
            cost_baseline -- 用于减去的成本基线值，如果为 None 则使用所有采样轨迹的最小值。
            Returns:
            u -- 执行的控制信号
            normalizer -- 轨迹成本的指数和
            min_cost -- 所有轨迹中的最小成本
        """
        std_n, R, weights, targets, ob, ob_num = cost_params
        # print(f"ob:{ob}")
        # print(f"ob_id:{ob_id}")
        # print(f"dx: {state[3]}")
        # print(f"dy: {state[4]}")     
          
        for i in range(self.num_optimization_iterations):
            normalizer, min_cost = self.rollouts(state,
                                                 std_n,
                                                 R,
                                                 weights,
                                                 targets,
                                                 ob,
                                                 ob_num,
                                                 cost_baseline=cost_baseline)
        u = self.U[0, :] # # 选取第一个时间步的控制信号作为当前执行的控制
        self.last_U = np.copy(self.U) # 保存当前的控制序列
        
        # Slide the control sequence down one-timestep
        # 滤波控制序列，滑动控制信号一个时间步
        U_new = np.zeros_like(self.U)
        for i in range(self.control_dim):
            # 卷积平滑控制信号
            U_new[:, i] = np.convolve(self.U[:, i], self.control_filter)[3:-3] 
        
        optimal_U = U_new
        # # 最后一个时间步初始化为零
        U_new[-1, :] = self.initialization_policy(state)
        self.U = U_new # 更新控制序列
        # Update and compute the control to be excuted
            # 通过策略函数计算当前状态下的额外控制信号修正并叠加到 u 上
        u += self.policy(state.reshape((1, self.state_dim))).flatten()
        return u, optimal_U, self.draw_sample_U, normalizer, min_cost

    """@brief: Transform a numpy array into a gpuarray 
            将一个numpy数组 a 转换为GPU数组，以便后续的CUDA计算使用。"""    
    def on_gpu(self, a, dtype=np.float32):
        a = a.flatten() # 将输入数组展平成一维。
        # 确保数组的内存布局满足一定的要求。A（任意顺序）、O（所有权）、W（可写）、C（C连续的内存布局）。
        a = np.require(a, dtype=dtype, requirements=['A', 'O', 'W', 'C'])
        a_d = gpuarray.to_gpu(a) # 将numpy数组拷贝到GPU内存，并返回GPU上的数组 a_d。
        return a_d

    """@brief: Cuda code for the default return zero policy.
            提供一个CUDA设备上的默认控制策略。该策略简单地将输出控制 u 初始化为全零。"""    
    def default_cuda_policy(self):
        default_policy_template = Template("""
        // 该函数将控制变量 u 的所有元素设置为零。
        __device__ void compute_policy(float* input, float* u, int input_dim, int output_dim)
        {
            int i;
            for (i = 0; i < output_dim; i++) {
                u[i] = 0;
            }
        }

        """)
        return default_policy_template.render()

    """@brief: Headers to include for cuda compilation 
            生成CUDA代码的头文件 包含了路径积分控制需要的基本常量和库。"""
    def cuda_headers(self):
        header_template = Template("""                       
        #include <math.h>
        #include <stdio.h>
        #include <stdlib.h>
        #include <float.h>  // import FLT_EPSILON
        #define ATAU 0.1  // 自定义atau值
        #define NUMS 3  // 自定义拟合椭圆的点集                         
                                   
        // 包含数学函数、标准输入输出和浮点数定义的头文件。
        // __device__ __constant__: 定义一个常量 U_d 用于存储控制信号 分配在GPU的常量内存中 供内核访问。                           
        __device__ __constant__ float U_d[{{timesteps}}*{{control_dim}}];
                                   
        """)
        return header_template.render(timesteps=self.num_timesteps,
                                      control_dim=self.control_dim)

    """@brief: Path integral control cuda code for sampling and evaluating trajectories
            生成路径积分控制算法的CUDA内核代码 进行控制采样和轨迹评估。 """
    
    def cuda_rollouts(self):
        rollout_kernel_template = Template("""
        __device__ float get_control_costs(float* u, float* du, float* R)
        {
            // 计算控制成本（控制信号 u 和控制更新 du 的加权成本）。
            // 使用控制代价矩阵 R 和控制信号 u、控制差值 du 计算控制成本。
            int i;
            float cost = 0;
            for (i = 0; i < {{control_dim}}; i++) {
                cost += 0.5*(1 - (1.0/{{exploration_variance}}))*(R[i]*du[i]*du[i]) + R[i]*du[i]*u[i] + 0.5*R[i]*u[i]*u[i];
            }
            return cost;
        }    
                                           
            // 限制输入的 float 值在 0.0 到1的范围
        __device__ float clamp(float value, float limit) {
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


                                           
        //加权平均算距离，dial_# = -(D-d)/(beta_3*D)   D是检测距离              
        __device__ float update_dial_3(float* s, float* ob, int timestep, int timesteps , int ob_num)
        {
            float ob_i[7] = {0,0,0,0,0,0,0};
            float dist_i = {{D}};
            float odia = 0; 
            float d = 0;                                                                                                                                     
            if(ob_num!=0){
                for (int i = 0; i < ob_num; i++){     
                    ob_i[0] = ob[ 7 * timesteps * i + 7 * timestep + 0 ];// x
                    ob_i[1] = ob[ 7 * timesteps * i + 7 * timestep + 1 ];// y
                    ob_i[2] = ob[ 7 * timesteps * i + 7 * timestep + 1 ];// r                      
                    ob_i[5] = ob[ 7 * timesteps * i + 7 * timestep + 5 ];// dx
                    ob_i[6] = ob[ 7 * timesteps * i + 7 * timestep + 6 ];// dy                                                                        
              
                    // x,y上障碍物与机器人相对位置
                    float dx = ob_i[0] - s[0];
                    float dy = ob_i[1] - s[1];
                    // x,y上障碍物与机器人相对速度
                    float dvx = ob_i[5] - s[3];
                    float dvy = ob_i[6] - s[4];   

                    // 普通距离                        
                    if({{cbf_type}} == 0){
                        dist_i = sqrt((dx*dx + dy*dy));                               
                        if (dist_i == 0) {
                            // 防止除以零
                            printf("Error: Distance cannot be zero.");
                            return 0;}                                       
                        d = clamp(dist_i-ob_i[2]-0.9f, {{D}});                                              
                    }

                    // cos距离
                    else if({{cbf_type}} == 1){
                                           
                        //点积小于0，则说明两者正在靠近                      
                        float dot = dx * dvx + dy * dvy;        
                                                                    
                        //printf("dot:= %f  ||",dot);  
                                            
                        dist_i = sqrt((dx*dx + dy*dy));  
                                           
                        if (dist_i == 0) {
                            // 防止除以零
                            printf("Error: Distance cannot be zero.");
                            return 0;}                                       
                        float cos_v = dot / dist_i;  

                        d = clamp(dist_i + {{atau}}*cos_v -ob_i[2]-0.9f, {{D}});          
                    }                 
                                                                        
                    // 超前距离
                    else if({{cbf_type}} == 2){
                                           
                        //点积小于0，则说明两者正在靠近                      
                                            
                        dist_i = sqrt( (dx + dvx * {{atau}})*(dx + dvx * {{atau}}) + (dy + dvy * {{atau}})*(dy + dvy * {{atau}}) );  
                        if (dist_i == 0) {
                            // 防止除以零
                            printf("Error: Distance cannot be zero.");
                            return 0;}                                       
                        d = clamp(dist_i -ob_i[2]-0.9f, {{D}});          
                    }
                    else{
                        dist_i = sqrt(dx*dx + dy*dy);                               
                        if (dist_i == 0) {
                            // 防止除以零
                            printf("Error: Distance cannot be zero.");
                            return 0;}                                       
                        d = clamp(dist_i, {{D}});                      
                    }                                                                   
                    //float odia_i = ({{D}}-d)/({{D}}*{{beta_3}});
                    float odia_i = -d/({{D}}*{{beta_3}});                       
                    if(odia<odia_i){
                        odia = odia_i;                   
                    }                       
                    
                }                                                                                            
            }
            //无障碍物
            else{
                   odia = 0; 
            }                        
            return odia;
        }      

                                                                             
        __global__ void rollout_kernel(float* costs_d, float* ran_vec, float* init_state,
                                       float* std_nd, float* R_d, float* weights, float* targets, float* nominal_seq_d, float* ob, float* ob_num)
        {
            //__global__ 表示这是一个CUDA全局内核函数 在GPU上并行执行。
            //Get thread and block index 通过线程索引 tdx, bdx, tdy 获取当前线程在块和网格中的位置。
            int tdx = threadIdx.x;
            int bdx = blockIdx.x;
            int tdy = threadIdx.y;
                                           
            int global_index = blockIdx.x * blockDim.x + threadIdx.x;   
            //全局样本索引   
            if(global_index == 2495){
                printf("global_index:%d",global_index);}                               

            int ob_num_s = int(ob_num[0]);
            //Initialize block wide state and control variables
                //初始化共享内存 _shared_ 存储状态和控制变量。
            __shared__ float state_shared[{{BLOCK_DIM_X}}*({{state_dim}} + {{control_dim}})];  //存储每个线程的状态 s 和控制输入 u。
            __shared__ float control_var_shared[{{BLOCK_DIM_X}}*{{control_dim}}]; //存储控制扰动 du。
            __shared__ float std_n[{{control_dim}}]; //存储控制输入的标准差和控制成本矩阵的权重，用于所有线程共享的代价计算。
            __shared__ float R[{{control_dim}}];

            //Initialize local state
            float *s, *u, *du;

            //Initialize local state and control variables //这些指针允许每个线程独立访问其对应的状态和控制输入。
            s = &state_shared[tdx*({{state_dim}} + {{control_dim}})]; // s 指向状态变量的起始地址
            u = &state_shared[tdx*({{state_dim}} + {{control_dim}}) + {{state_dim}}];// u 指向控制变量的起始地址
            du = &control_var_shared[tdx*{{control_dim}}];// du 指向控制扰动的起始地址

            //Initialize trajectory cost
            float running_cost = 0; //running_cost 保存当前时间步的代价。
            float cost = 0; // cost 累计整个轨迹的总代价。

            // normal || log-normal
            if({{dist_type}} == 0 || {{dist_type}} == 1){                                   
                // Load std_n, R, and the initial state
                for (int i = tdy; i < {{control_dim}}; i+= blockDim.y){
                    std_n[i] = std_nd[i]; //将 std_nd 控制标准差 和 R_d 控制成本矩阵 加载到共享内存 std_n 和 R 中。
                    R[i] = R_d[i];
                }
            }
                                           
            for (int i = tdy; i < {{state_dim}}; i+=blockDim.y)
            {
                s[i] = init_state[i];//加载到状态 s 中。
            }                          


            __syncthreads();// 确保所有线程都完成加载，避免数据冲突。
            
                                           
             /*<----Start of simulation loop (i.e., the main program loop) -----> */
            for (int i = 0; i < {{timesteps}}; i++)
            {   // 开始模拟轨迹的主循环，循环次数为 timesteps。
                // Get the initial control estimate from the feedback controller
                compute_policy(s, u, {{state_dim}}, {{control_dim}}); //使用 compute_policy 函数从反馈控制器获取初始控制估计 u。
                                           
                // ODIA
                if({{dist_type}} == 2){                             
                    // 更新轨迹退火 指数 
                    float dial_1,dial_2,dial_3,Covariance;
                                            
                    dial_1 = -global_index/({{beta_1}}*{{num_rollouts}});
                    dial_2 =  -({{timesteps}}-i)/({{timesteps}}*{{beta_2}});                                                                            
                    dial_3 = update_dial_3(s, ob, i, {{timesteps}}, ob_num_s);

                    dial_1 = 0;                       
                    //dial_2 = 0;
                    //dial_3 = 0;                                                                 
                    Covariance = exp(dial_1+dial_2+dial_3); 
                                            
                    //Covariance = exp(dial_3);                                                       

                    for (int j = tdy; j < {{control_dim}}; j+= blockDim.y)
                    {
                        std_n[j] = clamp(0.6*Covariance, 0.6); //将 std_nd 控制标准差 和 R_d 控制成本矩阵 加载到共享内存 std_n 和 R 中。
                        //std_n[j] = clamp(Covariance, 0.5); //将 std_nd 控制标准差 和 R_d 控制成本矩阵 加载到共享内存 std_n 和 R 中。                                     
                        R[j] = 1 * {{lambda_}} / std_n[j];                                           
                    }                     
                }
                    //printf("Covariance:=%f",Covariance);
                __syncthreads();

                // Get the control and control variation
                for (int j = tdy; j < {{control_dim}}; j+=blockDim.y)
                {
                    u[j] += U_d[i*{{control_dim}} + j]; // 将当前时间步的控制序列 U_d 添加到控制输入 u。
                    // Noise free rollout
                    if ((tdx == 0 && bdx == 0))
                    {
                        du[j] = 0; //如果是主线程（tdx == 0 && bdx == 0），则 du[j]（控制扰动）设为零，表示无扰动的情况。
                    }
                    else
                    { 
                        //其他线程根据 ran_vec 和标准差 std_n 生成随机扰动 du。
                        du[j] = std_n[j]*ran_vec[(blockDim.x*bdx + tdx)*{{timesteps}}*{{control_dim}} + i*{{control_dim}} + j];
                        //du[j] = Covariance*ran_vec[(blockDim.x*bdx + tdx)*{{timesteps}}*{{control_dim}} + i*{{control_dim}} + j];  
                                                    
                    }
                }

                // Save the random control variation (i.e.e, control updates)
                for (int j = tdy; j < {{control_dim}}; j+= blockDim.y)
                {
                    float udu = 0;
                    if(j ==0 ){
                        if(du[j]>0.1){                         
                            udu = u[j] + 0.1;}  
                        else if(du[j]<-0.1){
                            udu = u[j] - 0.1;}  
                        else{
                            udu = u[j] + du[j];}   
                                           }                                                              
                    else{
                        if(du[j]>0.4){                         
                            udu = u[j] + 0.4;}  
                        else if(du[j]<-0.4){
                            udu = u[j] - 0.4;}  
                        else{
                            udu = u[j] + du[j];}   
                                           } 
                                           
                    if (udu>1.2 && j==0){udu = 1.2;}
                    if (udu<-1.2 && j==0){udu = -1.2;}
                    if (udu>1.5 && j==1){udu = 1.5;}
                    if (udu<-1.5 && j==1){udu = -1.5;}  
                             
                    // 保存每个时间步的控制输入和扰动到 ran_vec 中，便于后续计算。
                    ran_vec[(blockDim.x*bdx + tdx)*{{timesteps}}*{{control_dim}} + i*{{control_dim}} + j] = udu;
                }

                // Save the nominal sequence
                if (tdx == 0 && bdx == 0 && tdy == 0)
                {
                    // 如果是主线程，将当前状态 s 保存到 nominal_seq_d 以记录无扰动的状态序列。
                    for (int j = 0; j < {{state_dim}}; j++)
                    {
                        nominal_seq_d[i*{{state_dim}} + j] = s[j];
                    }
                }
                __syncthreads();


                if (tdy == 0){
                    // 使用 kinematics 函数根据当前控制输入 u 和扰动 du 更新状态 s。
                    kinematics(s, u, du, {{dt}}, i);
                    //printf("dx=%f           |",s[3]);
                    //printf("dy=%f           |",s[4]);                                              
                }
                __syncthreads();

                // Get state and control costs
                if (tdy == 0)
                {
                    //使用 get_state_cost 函数计算状态代价，并限制其在 [min_cost, max_cost] 范围内。
                                           
                    float next_s[5] = {0,0,0,0,0};
                    next_s[0] = s[0];
                    next_s[1] = s[1]; 
                    next_s[2] = s[2];
                    next_s[3] = s[3]; 
                    next_s[4] = s[4];                       
                    kinematics(next_s, u, du, {{dt}}, i); 

					running_cost = get_state_cost(next_s, s, weights, targets, i, {{timesteps}}, ob, ob_num_s);
                    if (isnan(running_cost) || running_cost > {{max_cost}})
                    {
                        running_cost = {{max_cost}};
                    }
                    else if (running_cost < {{min_cost}})
                    {
                        running_cost = {{min_cost}};
                    }
                    cost += running_cost;
                    cost += get_control_costs(u, du, R);
                }
                __syncthreads();
            }
            /* <------- End of the simulation loop ----------> */

            // Write back the cost results to global memory.
            if (tdy == 0)
            {
                // 将每条轨迹的累计代价 cost 存储到全局内存 costs_d 中，保存每个采样轨迹的最终代价。
                costs_d[(blockDim.x*bdx + tdx)] = cost*{{dt}};
            }
        }
        """)
        # 并行采样: 每个线程处理一个轨迹的采样，更新状态和控制信号，并计算轨迹成本。
            # 共享内存: 利用CUDA的共享内存存储中间变量，减少全局内存访问。  
        return rollout_kernel_template.render(
            dt=self.dt,
            state_dim=self.state_dim,
            control_dim=self.control_dim,
            timesteps=self.num_timesteps,
            num_rollouts=self.num_samples,
            exploration_variance=self.exploration_variance,
            max_cost=self.cost_range[1],
            min_cost=self.cost_range[0],
            D = self.check_dist,
            beta_1 = self.beta_1,
            beta_2 = self.beta_2,
            beta_3 = self.beta_3,
            lambda_= self.lambda_,
            atau = self.atau,
            cbf_type = self.cbf_type,
            dist_type = self.dist_type,
            BLOCK_DIM_X=self.block_dim[0]) 
