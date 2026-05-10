cmake_minimum_required(VERSION 3.8)
project(traversability_mapping) # 确认项目名称与 package.xml 中的 <name> 标签完全一致

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# 查找 ROS 2 核心依赖
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(interactive_markers REQUIRED)
find_package(pluginlib REQUIRED)
find_package(pcl_ros REQUIRED)
find_package(nav2_core REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(tf2_geometry_msgs REQUIRED) # 用于几何消息的 TF2 转换
find_package(rosidl_default_generators REQUIRED)
# 查找你的自定义消息包 (如果 elevation_msgs 是一个独立的 ROS 2 包，且其名称就是 'elevation_msgs')
find_package(elevation_msgs REQUIRED) # 假设 ROS 2 版本的名称就是 elevation_msgs
find_package(tf2_eigen REQUIRED)
find_package(grid_map_ros REQUIRED)
# 查找外部库
find_package(PCL REQUIRED QUIET)
find_package(OpenCV REQUIRED QUIET)
find_package(pcl_conversions REQUIRED) # PCL 和 ROS 消息转换
find_package(OpenMP REQUIRED)
# Include Directories - ROS 2 通常会自动处理 ROS 包的头文件路径
# 这里只添加你自己的本地头文件目录和非 ROS 库的头文件目录
include_directories(
  include # 你的本地头文件目录 (例如 src/traversability_mapping/include)
  ${PCL_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  # IMPORTANT: 移除以下行，ROS 2 通常会自动处理生成消息的头文件路径
  # ${CMAKE_CURRENT_BINARY_DIR}/rosidl_generated_cpp
  # ${CMAKE_CURRENT_BINARY_DIR}/ament_cmake_core
)

# --- MESSAGE GENERATION ---
# Call rosidl_generate_interfaces for your custom messages
rosidl_generate_interfaces(
  ${PROJECT_NAME} # ${PROJECT_NAME} 应该就是 'traversability_mapping'
  "msg/CloudInfo.msg"
  "msg/Polynome.msg"
  DEPENDENCIES
  sensor_msgs    # CloudInfo 依赖 sensor_msgs/PointCloud2
  std_msgs       # CloudInfo 依赖 std_msgs/Header
  geometry_msgs  # CloudInfo 依赖 geometry_msgs/Point, geometry_msgs/Vector3
  builtin_interfaces # Polynome 可能依赖 builtin_interfaces/Time
  nav_msgs
)


# --- ROS 2 PACKAGE EXPORT ---
# 声明本包向其他依赖本包的包提供的运行时依赖
# ament_export_dependencies(
#   rosidl_default_runtime # 必须导出，因为你生成了消息
#   std_msgs
#   geometry_msgs
#   sensor_msgs
#   builtin_interfaces
#   elevation_msgs # 假设 ROS 2 elevation_msgs 包的名称就是 'elevation_msgs'
#   # 注意：这里不需要列出自己这个包名 (${PROJECT_NAME})
# )

# --- TESTING ---
# ament_lint_auto 必须在 ament_package() 之前调用
# if(BUILD_TESTING)
#   find_package(ament_lint_auto REQUIRED)
#   set(ament_cmake_copyright_FOUND TRUE)
#   set(ament_cmake_cpplint_FOUND TRUE)
#   ament_lint_auto_find_test_dependencies()
# endif()
# 安装 launch 文件
install(DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}
  PATTERN "*.launch.py"
)
ament_package()

# --- BUILD SECTION ---
# 定义所有可执行文件的通用 ROS 2 依赖列表
# 注意：这里不能包含 ${PROJECT_NAME} 本身，因为它不是外部依赖。
set(ROS2_COMMON_DEPS
  rclcpp
  std_msgs
  geometry_msgs
  nav_msgs
  sensor_msgs
  tf2_ros
  cv_bridge
  image_transport
  pcl_conversions
  interactive_markers
  pluginlib
  elevation_msgs
  tf2_geometry_msgs
  tf2_eigen
  pcl_ros
  rosidl_default_runtime
  grid_map_ros
  OpenMP  
)

# Range Image Filter
add_executable(traversability_filter src/traversability_filter.cpp)

# # 添加头文件路径（如果使用本包消息）
# target_include_directories(traversability_filter PRIVATE
#   ${CMAKE_CURRENT_BINARY_DIR}/rosidl_generator_cpp
# )

ament_target_dependencies(traversability_filter ${ROS2_COMMON_DEPS})

target_link_libraries(traversability_filter
  ${PCL_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${PROJECT_NAME}__rosidl_generator_cpp  # 只有使用本包消息时才需要
)
install(TARGETS traversability_filter
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

add_executable(traversability_cost src/traversability_cost.cpp)

ament_target_dependencies(traversability_cost ${ROS2_COMMON_DEPS})

target_link_libraries(traversability_cost
  ${PCL_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${PROJECT_NAME}__rosidl_generator_cpp  # 只有使用本包消息时才需要
)
install(TARGETS traversability_cost
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

# # Mapping
add_executable(traversability_map src/traversability_map.cpp src/library/raycast.cpp src/library/kdtree.c src/library/Dstar.cpp src/library/ikd_Tree.cpp)
ament_target_dependencies(traversability_map ${ROS2_COMMON_DEPS})
target_link_libraries(traversability_map ${PCL_LIBRARIES} ${OpenCV_LIBRARIES} )
install(TARGETS traversability_map
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)
# # OccupancyGrid_save
# add_executable(traversability_OccuGridSave src/traversability_OccuGridSave.cpp)
# ament_target_dependencies(traversability_OccuGridSave ${ROS2_COMMON_DEPS})
# target_link_libraries(traversability_OccuGridSave ${PCL_LIBRARIES} ${OpenCV_LIBRARIES})

# # PRM Planner
# add_executable(traversability_prm src/traversability_prm.cpp src/library/kdtree.c)
# ament_target_dependencies(traversability_prm ${ROS2_COMMON_DEPS})
# target_link_libraries(traversability_prm ${PCL_LIBRARIES} ${OpenCV_LIBRARIES})

# # Path Planner
add_executable(traversability_path src/traversability_path.cpp)
ament_target_dependencies(traversability_path ${ROS2_COMMON_DEPS})
target_link_libraries(traversability_path
  ${PCL_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${PROJECT_NAME}__rosidl_generator_cpp
  ${PROJECT_NAME}__rosidl_typesupport_cpp
)
install(TARGETS traversability_path
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

# add_executable(traversability_record src/traversability_record.cpp)
# ament_target_dependencies(traversability_record ${ROS2_COMMON_DEPS})
# target_link_libraries(traversability_record ${PCL_LIBRARIES} ${OpenCV_LIBRARIES})

# add_executable(traversability_cost src/traversability_cost.cpp)
# ament_target_dependencies(traversability_cost ${ROS2_COMMON_DEPS})
# target_link_libraries(traversability_cost ${PCL_LIBRARIES} ${OpenCV_LIBRARIES})

# add_executable(traversability_visualrecord src/traversability_visualrecord.cpp)
# ament_target_dependencies(traversability_visualrecord ${ROS2_COMMON_DEPS})
# target_link_libraries(traversability_visualrecord ${PCL_LIBRARIES} ${OpenCV_LIBRARIES})

# add_executable(pathFollower src/pathFollower.cpp)
# ament_target_dependencies(pathFollower ${ROS2_COMMON_DEPS})
# target_link_libraries(pathFollower ${PCL_LIBRARIES})

# # move_base Planner Plugin
# add_library(tm_planner_lib SHARED src/library/tm_planner.cpp)
# ament_target_dependencies(tm_planner_lib ${ROS2_COMMON_DEPS})
# If it's a Nav2 plugin, you might also need to link against nav2_core:
# target_link_libraries(tm_planner_lib nav2_core)
