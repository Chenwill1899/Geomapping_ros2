from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    base_launch = PathJoinSubstitution(
        [
            FindPackageShare("ausim_geomapping_adapter"),
            "launch",
            "ausim_scout_localmap.launch.py",
        ]
    )
    hfdm_profile = PathJoinSubstitution(
        [
            FindPackageShare("mppi_controller"),
            "configs",
            "mujoco_rviz_goal_hfdm_h25.yaml",
        ]
    )

    return LaunchDescription(
        [
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(base_launch),
                launch_arguments={
                    "mppi_profile": hfdm_profile,
                    "mppi_controller": "learned_hfdm_h25",
                    "use_frontend": "false",
                }.items(),
            )
        ]
    )
