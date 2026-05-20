from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


GEOMAPPING_ROOT = "/home/mexxiie/prj/Geomapping_ros2"
TRAVERSABILITY_PARAM_FILE = (
    GEOMAPPING_ROOT + "/src/traversability_mapping/params/traversability.yaml"
)
TRAVERSABILITY_RVIZ_FILE = (
    GEOMAPPING_ROOT + "/src/traversability_mapping/launch/include/traversability_mapping.rviz"
)


def generate_ausim_scout_localmap_launch_description():
    use_sim_time = LaunchConfiguration("use_sim_time")
    launch_rviz = LaunchConfiguration("launch_rviz")
    use_medirl = LaunchConfiguration("use_medirl")
    medirl_device = LaunchConfiguration("medirl_device")
    launch_mppi = LaunchConfiguration("launch_mppi")
    use_frontend = LaunchConfiguration("use_frontend")
    mppi_profile = LaunchConfiguration("mppi_profile")
    mppi_controller = LaunchConfiguration("mppi_controller")
    mppi_cmd_vel_topic = LaunchConfiguration("mppi_cmd_vel_topic")

    ausim_lidar_topic = LaunchConfiguration("ausim_lidar_topic")
    ausim_odom_frame = LaunchConfiguration("ausim_odom_frame")
    ausim_base_frame = LaunchConfiguration("ausim_base_frame")
    planner_base_frame = LaunchConfiguration("planner_base_frame")
    planner_lidar_frame = LaunchConfiguration("planner_lidar_frame")
    terrain_output_frame = LaunchConfiguration("terrain_output_frame")

    scan_lines = LaunchConfiguration("scan_lines")
    target_scan_lines = LaunchConfiguration("target_scan_lines")
    horizon_scan = LaunchConfiguration("horizon_scan")
    vertical_fov_deg = LaunchConfiguration("vertical_fov_deg")
    traversability_param_file = LaunchConfiguration("traversability_param_file")
    rviz_config_file = LaunchConfiguration("rviz_config_file")

    use_sim_time_param = ParameterValue(use_sim_time, value_type=bool)
    scan_lines_param = ParameterValue(scan_lines, value_type=int)
    target_scan_lines_param = ParameterValue(target_scan_lines, value_type=int)
    horizon_scan_param = ParameterValue(horizon_scan, value_type=int)
    vertical_fov_deg_param = ParameterValue(vertical_fov_deg, value_type=float)

    declared_arguments = [
        DeclareLaunchArgument(
            "use_sim_time",
            default_value="true",
            description="Use ausim2 /clock.",
        ),
        DeclareLaunchArgument(
            "launch_rviz",
            default_value="true",
            description="Launch RViz with the Geomapping view.",
        ),
        DeclareLaunchArgument(
            "use_medirl",
            default_value="true",
            description="Run MEDIRL to convert /msg_local_feature into /msg_local_reward.",
        ),
        DeclareLaunchArgument(
            "medirl_device",
            default_value="cuda",
            description="MEDIRL inference device.",
        ),
        DeclareLaunchArgument(
            "launch_mppi",
            default_value="true",
            description="Launch the MPPI closed-loop controller.",
        ),
        DeclareLaunchArgument(
            "use_frontend",
            default_value="false",
            description="Launch traversability_path and publish /tltrajectory for MPPI path tracking.",
        ),
        DeclareLaunchArgument(
            "mppi_profile",
            default_value=PathJoinSubstitution(
                [FindPackageShare("mppi_controller"), "configs", "mujoco_rviz_goal.yaml"]
            ),
            description="MPPI experiment profile for MuJoCo closed-loop control.",
        ),
        DeclareLaunchArgument(
            "mppi_controller",
            default_value="nominal_cuda",
            description="Controller entry from the MPPI profile.",
        ),
        DeclareLaunchArgument(
            "mppi_cmd_vel_topic",
            default_value="/joy/cmd_vel",
            description="Command topic consumed by ausim2 Scout.",
        ),
        DeclareLaunchArgument(
            "ausim_lidar_topic",
            default_value="/scout1/lidar/points",
            description="PointCloud2 topic published by ausim2.",
        ),
        DeclareLaunchArgument(
            "ausim_odom_frame",
            default_value="scout1/odom",
            description="Odometry frame published by ausim2.",
        ),
        DeclareLaunchArgument(
            "ausim_base_frame",
            default_value="scout1/base_link",
            description="Base frame published by ausim2.",
        ),
        DeclareLaunchArgument(
            "planner_base_frame",
            default_value="base_link",
            description="Base frame expected by Geomapping local-map nodes.",
        ),
        DeclareLaunchArgument(
            "planner_lidar_frame",
            default_value="velodyne",
            description="LiDAR frame expected by Geomapping local-map nodes.",
        ),
        DeclareLaunchArgument(
            "terrain_output_frame",
            default_value="/base_link",
            description="Output frame used by terrain_pub_node.",
        ),
        DeclareLaunchArgument(
            "scan_lines",
            default_value="16",
            description="Input organized scan line count for the ausim2 Scout LiDAR.",
        ),
        DeclareLaunchArgument(
            "target_scan_lines",
            default_value="16",
            description="Projected scan line count consumed by traversability_filter.",
        ),
        DeclareLaunchArgument(
            "horizon_scan",
            default_value="900",
            description="Projected horizontal scan bins consumed by traversability_filter.",
        ),
        DeclareLaunchArgument(
            "vertical_fov_deg",
            default_value="30.0",
            description="Vertical field of view used to project the ausim2 point cloud.",
        ),
        DeclareLaunchArgument(
            "traversability_param_file",
            default_value=TRAVERSABILITY_PARAM_FILE,
            description="Parameter file for traversability_mapping.",
        ),
        DeclareLaunchArgument(
            "rviz_config_file",
            default_value=TRAVERSABILITY_RVIZ_FILE,
            description="RViz config for the Geomapping view.",
        ),
    ]

    map_to_odom = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="ausim_map_to_odom",
        output="screen",
        arguments=[
            "--x",
            "0",
            "--y",
            "0",
            "--z",
            "0",
            "--roll",
            "0",
            "--pitch",
            "0",
            "--yaw",
            "0",
            "--frame-id",
            "map",
            "--child-frame-id",
            ausim_odom_frame,
        ],
        parameters=[{"use_sim_time": use_sim_time_param}],
    )

    ausim_base_to_planner_base = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="ausim_base_to_geomapping_base",
        output="screen",
        arguments=[
            "--x",
            "0",
            "--y",
            "0",
            "--z",
            "0",
            "--roll",
            "0",
            "--pitch",
            "0",
            "--yaw",
            "0",
            "--frame-id",
            ausim_base_frame,
            "--child-frame-id",
            planner_base_frame,
        ],
        parameters=[{"use_sim_time": use_sim_time_param}],
    )

    planner_base_to_lidar = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="geomapping_base_to_lidar",
        output="screen",
        arguments=[
            "--x",
            "0",
            "--y",
            "0",
            "--z",
            "0",
            "--roll",
            "0",
            "--pitch",
            "0",
            "--yaw",
            "0",
            "--frame-id",
            planner_base_frame,
            "--child-frame-id",
            planner_lidar_frame,
        ],
        parameters=[{"use_sim_time": use_sim_time_param}],
    )

    terrain_pub = Node(
        package="terrain_pub_node",
        executable="terrain_pub_node",
        name="terrain_pub_node",
        output="screen",
        remappings=[("/velodyne_points", ausim_lidar_topic)],
        parameters=[
            {"use_sim_time": use_sim_time_param},
            {"scan_lines": scan_lines_param},
            {"target_scan_lines": target_scan_lines_param},
            {"horizon_scan": horizon_scan_param},
            {"vertical_fov_deg": vertical_fov_deg_param},
            {"output_frame": terrain_output_frame},
        ],
    )

    traversability_filter = Node(
        package="traversability_mapping",
        executable="traversability_filter",
        name="traversability_filter",
        output="screen",
        parameters=[{"frameID": "map"}, {"use_sim_time": use_sim_time_param}],
    )

    traversability_map = Node(
        package="traversability_mapping",
        executable="traversability_map",
        name="traversability_map",
        output="screen",
        parameters=[
            traversability_param_file,
            {"urbanMapping": True},
            {"use_sim_time": use_sim_time_param},
            {"planning.time_roll": 0.8},
            {"planning.path_valid_check_distance": 6.0},
            {"planning.inflate_r": 0.0},
            {"mapping.local_publish_every_scan": True},
            {"mapping.visualization_hz": 2.0},
            {"mapping.publish_global_debug": False},
        ],
    )

    traversability_cost = Node(
        package="traversability_mapping",
        executable="traversability_cost",
        name="traversability_cost",
        output="screen",
        parameters=[{"use_sim_time": use_sim_time_param}],
    )

    traversability_path = Node(
        package="traversability_mapping",
        executable="traversability_path",
        name="traversability_path",
        output="screen",
        parameters=[
            {"use_sim_time": use_sim_time_param},
            {"map_frame": "map"},
            {"base_frame": planner_base_frame},
            {"goal_topic": "/RRT_goal"},
            {"local_costmap_topic": "/msg_local_reward"},
            {"local_plan.radius": 0.3},
            {"local_plan.robot_radius": 0.55},
            {"local_plan.visitflag": False},
            {"local_plan.inflation_radius": 0.0},
            {"local_plan.omni_mode": True},
            {"local_plan.omni_path_length": 3.0},
            {"local_plan.omni_path_spacing": 0.25},
            {"planning.cost_max": 50.0},
        ],
        condition=IfCondition(use_frontend),
    )

    medirl = Node(
        package="medirl",
        executable="MEDIRL.py",
        name="MEDIRL",
        output="screen",
        parameters=[
            {"use_sim_time": use_sim_time_param},
            {"device": medirl_device},
        ],
        condition=IfCondition(use_medirl),
    )

    rviz = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz",
        output="log",
        arguments=["-d", rviz_config_file],
        parameters=[{"use_sim_time": use_sim_time_param}],
        condition=IfCondition(launch_rviz),
    )

    mppi = Node(
        package="mppi_controller",
        executable="fdm_mppi",
        name="mppi_closed_loop",
        output="screen",
        arguments=[
            "mujoco-closed-loop",
            "--profile",
            mppi_profile,
            "--controller",
            mppi_controller,
        ],
        remappings=[("/scout1/cmd_vel", mppi_cmd_vel_topic)],
        parameters=[{"use_sim_time": use_sim_time_param}],
        condition=IfCondition(launch_mppi),
    )

    return LaunchDescription(
        declared_arguments
        + [
            map_to_odom,
            ausim_base_to_planner_base,
            planner_base_to_lidar,
            terrain_pub,
            traversability_filter,
            traversability_map,
            traversability_cost,
            traversability_path,
            medirl,
            rviz,
            mppi,
        ]
    )
