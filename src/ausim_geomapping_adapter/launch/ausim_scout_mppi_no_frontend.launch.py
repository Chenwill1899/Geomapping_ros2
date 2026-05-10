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
    mppi_profile = PathJoinSubstitution(
        [
            FindPackageShare("mppi_controller"),
            "configs",
            "mujoco_rviz_goal_no_frontend.yaml",
        ]
    )

    return LaunchDescription(
        [
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(base_launch),
                launch_arguments={
                    "mppi_profile": mppi_profile,
                    "use_frontend": "false",
                }.items(),
            )
        ]
    )
