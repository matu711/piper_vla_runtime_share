cmake_minimum_required(VERSION 3.0.2)
project(piper_vla_runtime_msgs)

find_package(catkin REQUIRED COMPONENTS
  message_generation
)

add_service_files(
  FILES
  StartInference.srv
)

generate_messages()

catkin_package(
  CATKIN_DEPENDS message_runtime
)
