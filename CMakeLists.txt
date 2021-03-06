cmake_minimum_required(VERSION 3.15)
project(myslam)

set(CMAKE_CXX_STANDARD 14)

############### dependencies ######################
# Eigen
include_directories( "/usr/include/eigen3" )
# OpenCV
find_package( OpenCV 4.0 REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
# Ceres
find_package( Ceres REQUIRED )
include_directories( ${Ceres_INCLUDE_DIRS} )
# PCL
find_package( PCL 1.8 REQUIRED )
include_directories(${PCL_INCLUDE_DIRS})

set( THIRD_PARTY_LIBS
        ${OpenCV_LIBS}
        ${Ceres_LIBRARIES}
        ${PCL_LIBRARY_DIRS}
        )
############### dependencies ######################

include_directories(
        ${PROJECT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/include)
add_subdirectory( src )
add_subdirectory( test )

add_executable(myslam main.cpp)
target_link_libraries(myslam
        myslam_lib)