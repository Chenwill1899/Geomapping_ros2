from glob import glob
import os

from setuptools import setup

package_name = "ausim_geomapping_adapter"

setup(
    name=package_name,
    version="0.0.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="mexxiie",
    maintainer_email="2579171559@qq.com",
    description="Launch-level adapter from ausim2 MuJoCo ROS outputs to Geomapping local map nodes.",
    license="TODO",
    entry_points={"console_scripts": []},
)
