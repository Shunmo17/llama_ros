cmake_minimum_required(VERSION 3.0.2)
project(llama_ros)

find_package(catkin REQUIRED COMPONENTS
  roscpp 
  llama_msgs
  actionlib
  actionlib_msgs
)

catkin_package(
  CATKIN_DEPENDS actionlib_msgs
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)

add_subdirectory(llama_cpp)
add_subdirectory(llama_cpp/examples)

add_executable(llama_node src/llama_node.cpp src/llama.cpp)
target_compile_options(llama_node PUBLIC ${CXXMAKE_CXX_FLAGS} -DLLAMA_CUBLAS=ON)
target_link_libraries(llama_node llama common ${catkin_LIBRARIES})
