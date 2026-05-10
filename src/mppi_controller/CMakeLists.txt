cmake_minimum_required(VERSION 3.8)
project(mppi_controller)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

ament_python_install_package(${PROJECT_NAME})

install(DIRECTORY configs/
  DESTINATION share/${PROJECT_NAME}/configs
)

install(DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch
)

install(PROGRAMS
  tools/fdm_mppi
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
