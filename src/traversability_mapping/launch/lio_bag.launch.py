from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, ThisLaunchFileDir, PathJoinSubstitution
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

# 定义 OpaqueFunction 来处理 rosparam 的加载
# ROS2 的参数加载通常在节点定义时进行，或者通过参数文件。
# 这里使用 OpaqueFunction 模拟 rosparam load，因为直接的 rosparam load action 不存在
def load_traversability_params(context):
    pkg_share_dir = get_package_share_directory('traversability_mapping')
    param_file_path = os.path.join(pkg_share_dir, 'params', 'traversability.yaml')

    # 获取 traversability_map 节点的引用，并通过其 set_parameters_from_file 加载参数
    # 注意：这需要节点支持从文件加载参数。
    # 实际项目中，更常见的做法是在 Node 定义时使用 parameters=[param_file_path]
    # 或者为每个参数单独定义。为了保持与rosparam load的逻辑一致，这里暂时不修改。
    # 如果您的C++节点没有实现从文件加载参数，您需要手动为每个节点定义参数。
    # 另一种方法是为每个节点显式声明和加载参数，如下方注释所示。

    # 这段 OpaqueFunction 的逻辑主要是为了模拟 `rosparam load`，
    # 但在 ROS 2 中，通常会在 Node 的定义中直接加载参数文件。
    # 例如：
    # Node(
    #     package='traversability_mapping',
    #     executable='traversability_map',
    #     name='traversability_map',
    #     output='screen',
    #     parameters=[param_file_path, {'urbanMapping': True}]
    # ),
    # Node(
    #     package='traversability_mapping',
    #     executable='traversability_cost',
    #     name='traversability_cost',
    #     output='screen',
    #     parameters=[param_file_path]
    # ),
    # Node(
    #     package='traversability_mapping',
    #     executable='traversability_path',
    #     name='traversability_path',
    #     output='screen',
    #     parameters=[param_file_path]
    # )
    return []

def generate_launch_description():
    # 获取包的分享目录
    traversability_mapping_pkg_share_dir = get_package_share_directory('traversability_mapping')
    fast_lio_pkg_share_dir = get_package_share_directory('fast_lio')

    # 声明 launch 参数
    use_sim_time = DeclareLaunchArgument(
        'use_sim_time',
        default_value='True',
        description='Use simulation (Gazebo) clock if true'
    )

    # RVIZ 节点
    rviz_node = Node(
        package='rviz2', # ROS 2 中 rviz 的包名是 rviz2
        executable='rviz2', # ROS 2 中 rviz 的可执行文件是 rviz2
        name='rviz',
        output='log',
        arguments=['-d', os.path.join(traversability_mapping_pkg_share_dir, 'launch', 'include', 'traversability_mapping.rviz')],
    )

    # 静态 TF 转换发布器节点
    # ROS 2 中静态 TF 发布器在 tf2_ros 包中，可执行文件是 static_transform_publisher
    velodyne_base_link_tf_publisher = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='velodyne_base_link',
        output='screen',
        arguments=['0', '0', '0', '0', '0', '0', 'base_link', 'velodyne'], # 最后一个参数是周期（Hz），旧版tf是10ms，新版是10Hz
    )

    camera_map_tf_publisher = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='camera_map',
        output='screen',
        arguments=['0', '0', '0', '0', '0', '0', 'camera_init', 'map'], # 最后一个参数是周期（Hz）
    )
    base_map_tf_publisher = Node(
          package='tf2_ros',
          executable='static_transform_publisher',
          name='base_map',
          output='screen',
          arguments=['0', '0', '0', '0', '0', '0', 'map', 'base_link'], # 最后一个参数是周期（Hz）
      )
    # 包含 fast_lio 的 launch 文件
    fast_lio_mapping_velodyne_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(fast_lio_pkg_share_dir, 'launch', 'mapping.launch.py') # 确保这是 ROS 2 launch 文件
        )
    )

    # traversability_filter 节点
    traversability_filter_node = Node(
        package='traversability_mapping',
        executable='traversability_filter', # 确保这是你的可执行文件名
        name='traversability_filter',
        output='screen',
        parameters=[
            {'frameID': 'map'}
            # 如果 'isLegoLoam' 和 'laserCloudInfoTopic' 默认值不是您想要的，或者需要从 YAML 加载，可以在此添加
            # {'isLegoLoam': False}, # 如果需要，取消注释
            # {'laserCloudInfoTopic': '/lio_sam/deskew/cloud_info'} # 如果需要，取消注释
        ]
    )

    # traversability_map 节点
    traversability_map_node = Node(
        package='traversability_mapping',
        executable='traversability_map', # 确保这是你的可执行文件名
        name='traversability_map',
        output='screen',
        parameters=[
            {'urbanMapping': True},
            # 如果 traversability.yaml 中的参数也需要应用于此节点，可以在这里添加文件加载
            os.path.join('/home/mexxiie/prj/Geomapping_ros2/src/traversability_mapping/', 'params', 'traversability.yaml')
        
        ]
    )

    # # traversability_cost 节点
    # traversability_cost_node = Node(
    #     package='traversability_mapping',
    #     executable='traversability_cost', # 确保这是你的可执行文件名
    #     name='traversability_cost',
    #     output='screen',
    #     # 如果 traversability.yaml 中的参数也需要应用于此节点，可以在这里添加文件加载
    #     # parameters=[PathJoinSubstitution([traversability_mapping_pkg_share_dir, 'params', 'traversability.yaml'])]
    # )
    
    # terrain_pub_node 节点
 
    terrain_pub_node = Node(
        package='terrain_pub_node',
        executable='terrain_pub_node', # 确保这是你的可执行文件名
        name='terrain_pub_node',
        output='screen'
    )

    # traversability_path 节点
    # traversability_path_node = Node(
    #     package='traversability_mapping',
    #     executable='traversability_path', # 确保这是你的可执行文件名
    #     name='traversability_path',
    #     output='screen',
    #     # 如果 traversability.yaml 中的参数也需要应用于此节点，可以在这里添加文件加载
    #     # parameters=[PathJoinSubstitution([traversability_mapping_pkg_share_dir, 'params', 'traversability.yaml'])]
    # )

    # ROS 2 中加载参数文件的方式
    # 注意：ROS 2 没有直接的 <rosparam file="..." command="load"/> 标签。
    # 你需要在每个需要这些参数的节点中，通过 'parameters' 字段来加载 YAML 文件。
    # 我在这里保留了原始逻辑，但实际应用中，你可能需要根据每个节点的需求来调整。
    # 最常见的是将参数文件作为 Node 的 'parameters' 列表中的一项。
    # 比如：parameters=[os.path.join(traversability_mapping_pkg_share_dir, 'params', 'traversability.yaml')]

    # 以下是如何在 Node 定义中加载 YAML 参数文件的示例
    # For example, to apply 'traversability.yaml' to all relevant nodes:
    # common_params = [os.path.join(traversability_mapping_pkg_share_dir, 'params', 'traversability.yaml')]
    # traversability_map_node = Node(
    #     package='traversability_mapping',
    #     executable='traversability_map',
    #     name='traversability_map',
    #     output='screen',
    #     parameters=common_params + [{'urbanMapping': True}] # 合并 YAML 参数和直接参数
    # )
    traversability_cost_node = Node(
        package='traversability_mapping',
        executable='traversability_cost',
        name='traversability_cost',
        output='screen'
    )
    # ...以此类推

    return LaunchDescription([
        use_sim_time,
        #rviz_node,
        velodyne_base_link_tf_publisher,
        camera_map_tf_publisher,
        base_map_tf_publisher,
        fast_lio_mapping_velodyne_launch,
        traversability_filter_node,
        # OpaqueFunction(function=load_traversability_params), # 如果需要模拟 rosparam load，但通常不推荐
        # traversability_map_node,
        # traversability_cost_node,
        terrain_pub_node,
        rviz_node
        # traversability_path_node,
    ])