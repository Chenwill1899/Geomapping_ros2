import os
from glob import glob
from setuptools import find_packages, setup

package_name = "mppi_controller"

def get_files(directory, prefix=""):
    files = []
    for root, _, filenames in os.walk(directory):
        for f in filenames:
            full = os.path.join(root, f)
            rel = os.path.relpath(full, ".")
            if prefix:
                dest = os.path.join(prefix, os.path.relpath(root, directory))
            else:
                dest = os.path.relpath(root, ".")
            files.append((dest, [rel]))
    return files

setup(
    name=package_name,
    version="0.0.1",
    packages=find_packages(),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/configs", glob("configs/*")),
        ("share/" + package_name + "/launch", glob("launch/*")),
        ("lib/" + package_name, glob("tools/*")),
    ],
    install_requires=[
        "setuptools",
        "numpy",
        "pandas",
        "pyyaml",
        "torch",
        "matplotlib",
    ],
    zip_safe=True,
    maintainer="mexxiie",
    maintainer_email="mexxiie@example.com",
    description="MPPI closed-loop controller for MuJoCo + local_costmap navigation",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "fdm_mppi=mppi_controller.cli:main",
        ],
    },
)
