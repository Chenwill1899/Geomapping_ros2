from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution


def generate_launch_description():
    launch_rviz = LaunchConfiguration("launch_rviz")
    mujoco_base_frame = LaunchConfiguration("mujoco_base_frame")
    mujoco_odom_frame = LaunchConfiguration("mujoco_odom_frame")
    planner_base_frame = LaunchConfiguration("planner_base_frame")
    use_medirl = LaunchConfiguration("use_medirl")
    use_sim_time = LaunchConfiguration("use_sim_time")
    launch_mppi = LaunchConfiguration("launch_mppi")
    mppi_controller = LaunchConfiguration("mppi_controller")
    mppi_profile = LaunchConfiguration("mppi_profile")
    mppi_cmd_vel_topic = LaunchConfiguration("mppi_cmd_vel_topic")

    use_sim_time_arg = DeclareLaunchArgument(
        "use_sim_time",
        default_value="true",
        description="Use simulation time.",
    )
    launch_rviz_arg = DeclareLaunchArgument(
        "launch_rviz",
        default_value="true",
        description="Launch RViz with the Geomapping view.",
    )
    use_medirl_arg = DeclareLaunchArgument(
        "use_medirl",
        default_value="true",
        description="Run MEDIRL to convert local terrain features into /msg_local_reward.",
    )
    launch_mppi_arg = DeclareLaunchArgument(
        "launch_mppi",
        default_value="true",
        description="Launch the MPPI closed-loop controller.",
    )
    mppi_profile_arg = DeclareLaunchArgument(
        "mppi_profile",
        default_value=PathJoinSubstitution(
            [FindPackageShare("mppi_controller"), "configs", "mujoco_rviz_goal.yaml"]
        ),
        description="MPPI experiment profile for MuJoCo closed-loop control.",
    )
    mppi_controller_arg = DeclareLaunchArgument(
        "mppi_controller",
        default_value="nominal_numpy",
        description="Controller entry from the MPPI profile.",
    )
    mppi_cmd_vel_topic_arg = DeclareLaunchArgument(
        "mppi_cmd_vel_topic",
        default_value="/joy/cmd_vel",
        description="Command topic consumed by ausim2 Scout.",
    )
    mujoco_odom_frame_arg = DeclareLaunchArgument(
        "mujoco_odom_frame",
        default_value="scout1/odom",
        description="MuJoCo truth odometry frame published by ausim2.",
    )
    mujoco_base_frame_arg = DeclareLaunchArgument(
        "mujoco_base_frame",
        default_value="scout1/base_link",
        description="MuJoCo truth base frame published by ausim2.",
    )
    planner_base_frame_arg = DeclareLaunchArgument(
        "planner_base_frame",
        default_value="base_link",
        description="Base frame name expected by traversability_mapping.",
    )

    map_to_odom = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="map_to_scout_odom",
        output="screen",
        arguments=["0", "0", "0", "0", "0", "0", "map", mujoco_odom_frame],
        parameters=[{"use_sim_time": use_sim_time}],
    )
    scout_base_to_base = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="scout_base_to_base_link",
        output="screen",
        arguments=["0", "0", "0", "0", "0", "0", mujoco_base_frame, planner_base_frame],
        parameters=[{"use_sim_time": use_sim_time}],
    )
    base_to_velodyne = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="base_to_velodyne",
        output="screen",
        arguments=["0", "0", "0", "0", "0", "0", planner_base_frame, "velodyne"],
        parameters=[{"use_sim_time": use_sim_time}],
    )

    terrain_pub = Node(
        package="terrain_pub_node",
        executable="terrain_pub_node",
        name="terrain_pub_node",
        output="screen",
        remappings=[("/velodyne_points", "/scout1/lidar/points")],
        parameters=[
            {"use_sim_time": use_sim_time},
            {"scan_lines": 16},
            {"target_scan_lines": 16},
            {"horizon_scan": 900},
            {"vertical_fov_deg": 30.0},
            {"output_frame": "/base_link"},
        ],
    )
    traversability_filter = Node(
        package="traversability_mapping",
        executable="traversability_filter",
        name="traversability_filter",
        output="screen",
        parameters=[{"frameID": "map"}, {"use_sim_time": use_sim_time}],
    )
    traversability_map = Node(
        package="traversability_mapping",
        executable="traversability_map",
        name="traversability_map",
        output="screen",
        parameters=[
            "/home/mexxiie/prj/Geomapping_ros2/src/traversability_mapping/params/traversability.yaml",
            {"urbanMapping": True},
            {"use_sim_time": use_sim_time},
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
        parameters=[{"use_sim_time": use_sim_time}],
    )
    medirl = Node(
        package="medirl",
        executable="MEDIRL.py",
        name="MEDIRL",
        output="screen",
        parameters=[{"use_sim_time": use_sim_time}, {"device": "cuda"}],
        condition=IfCondition(use_medirl),
    )
    rviz = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz",
        output="log",
        arguments=[
            "-d",
            "/home/mexxiie/prj/Geomapping_ros2/src/traversability_mapping/launch/include/traversability_mapping.rviz",
        ],
        parameters=[{"use_sim_time": use_sim_time}],
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
        parameters=[{"use_sim_time": use_sim_time}],
        condition=IfCondition(launch_mppi),
    )

    return LaunchDescription(
        [
            use_sim_time_arg,
            launch_rviz_arg,
            use_medirl_arg,
            launch_mppi_arg,
            mppi_profile_arg,
            mppi_controller_arg,
            mppi_cmd_vel_topic_arg,
            mujoco_odom_frame_arg,
            mujoco_base_frame_arg,
            planner_base_frame_arg,
            map_to_odom,
            scout_base_to_base,
            base_to_velodyne,
            terrain_pub,
            traversability_filter,
            traversability_map,
            traversability_cost,
            medirl,
            rviz,
            mppi,
        ]
    )
