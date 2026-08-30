cmake_minimum_required(VERSION 3.13)
project(multi_frame_sphere_fit LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(
    PCL 1.8 REQUIRED
    COMPONENTS
        common
        io
        filters
        sample_consensus
        segmentation
)

add_executable(
    multi_frame_sphere_fit
    multi_frame_sphere_fit.cpp
)

target_include_directories(
    multi_frame_sphere_fit
    PRIVATE
        ${PCL_INCLUDE_DIRS}
)

target_link_directories(
    multi_frame_sphere_fit
    PRIVATE
        ${PCL_LIBRARY_DIRS}
)

target_compile_options(
    multi_frame_sphere_fit
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

add_definitions(${PCL_DEFINITIONS})
target_link_libraries(
    multi_frame_sphere_fit
    PRIVATE
        ${PCL_LIBRARIES}
)
