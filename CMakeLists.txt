cmake_minimum_required(VERSION 3.2.0)

project(example)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

set(CMAKE_PREFIX_PATH ../libtorch)
find_package(Torch REQUIRED)
message(STATUS "Pytorch status:")
message(STATUS "    libraries: ${TORCH_LIBRARIES}")

# set(OpenCV_DIR "/Users/jingxiaofei/Downloads/opencv-4.1.0/build")
# find_package(OpenCV REQUIRED PATHS OpenCV_DIR)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

# add_executable(torch_ext tensor_test.cpp)
# target_link_libraries(torch_ext "${TORCH_LIBRARIES}")
# set_property(TARGET torch_ext PROPERTY CXX_STANDARD 11)

# add_executable(test2 cxx_inference.cpp)
# target_link_libraries(test2  ${TORCH_LIBRARIES} ${OpenCV_LIBS})
# set_property(TARGET test2 PROPERTY CXX_STANDARD 11)

add_executable(test3 cxx_resnet.cpp)
target_link_libraries(test3 ${TORCH_LIBRARIES} ${OpenCV_LIBS})
set_property(TARGET test3 PROPERTY CXX_STANDARD 11)

# add_executable(test4 cxx_simple.cpp)
# target_link_libraries(test4 ${TORCH_LIBRARIES})
# set_property(TARGET test4 PROPERTY CXX_STANDARD 11)

# add_executable(test5 cxx_video.cpp)
# target_link_libraries(test5 ${TORCH_LIBRARIES} ${OpenCV_LIBS})
# set_property(TARGET test5 PROPERTY CXX_STANDARD 11)