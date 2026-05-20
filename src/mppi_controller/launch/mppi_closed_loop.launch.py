from launch import LaunchDescription
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution


def generate_launch_description():
    profile = PathJoinSubstitution(
        [FindPackageShare("mppi_controller"), "configs", "mujoco_rviz_goal.yaml"]
    )

    return LaunchDescription(
        [
            Node(
                package="mppi_controller",
                executable="fdm_mppi",
                name="mppi_closed_loop",
                output="screen",
                arguments=[
                    "mujoco-closed-loop",
                    "--profile",
                    profile,
                    "--controller",
                    "nominal_cuda",
                ],
            ),
        ]
    )
