cmake_minimum_required(VERSION 3.8)
project(interact_with_depth)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(PCL REQUIRED)

add_executable(depthProcess src/depthProcess.cpp)
target_include_directories(depthProcess PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${PCL_INCLUDE_DIRS})
target_compile_features(depthProcess PUBLIC c_std_99 cxx_std_17)  # Require C99 and C++17
ament_target_dependencies(depthProcess
  std_msgs
  sensor_msgs
  visualization_msgs
  pcl_conversions
)
target_link_libraries(depthProcess
  ${PCL_LIBRARIES}
)

add_executable(depthTrackingProcess src/depthTrackingProcess.cpp)
target_include_directories(depthTrackingProcess PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${PCL_INCLUDE_DIRS})
target_compile_features(depthTrackingProcess PUBLIC c_std_99 cxx_std_17)  # Require C99 and C++17
ament_target_dependencies(depthTrackingProcess
  std_msgs
  sensor_msgs
  visualization_msgs
  pcl_conversions
)
target_link_libraries(depthProcess
  ${PCL_LIBRARIES}
)

install(TARGETS
  depthProcess
  depthTrackingProcess
  DESTINATION lib/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
