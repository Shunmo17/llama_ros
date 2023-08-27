cmake_minimum_required(VERSION 3.0.2)
project(llama_msgs)

find_package(catkin REQUIRED COMPONENTS
  message_generation
  std_msgs
  actionlib_msgs 
)

add_message_files(
  FILES
  TokenProb.msg
  TokenProbArray.msg
  Response.msg
  PartialResponse.msg
  LogitBias.msg
  LogitBiasArray.msg
  SamplingConfig.msg
)

add_service_files(
  FILES
  GenerateEmbeddings.srv
  Tokenize.srv
)

add_action_files(
  FILES
  GenerateResponse.action
)

generate_messages(
  DEPENDENCIES
  std_msgs
  actionlib_msgs 
)

catkin_package(
  CATKIN_DEPENDS 
  message_runtime
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)
