cmake_minimum_required(VERSION 3.10)
set(CMAKE_CXX_STANDARD 11)

find_package(OpenVINO REQUIRED)
find_package(OpenCV REQUIRED)

set (TARGET_NAME "fuel-app")

add_executable(${TARGET_NAME} src/main.cpp)

target_link_libraries(${TARGET_NAME} PRIVATE
	openvino::runtime
	${OpenCV_LIBRARIES}
)
