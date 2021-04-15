//
// Created by jihaorui on 2/23/21.
//

#ifndef MYSLAM_COMMON_INCLUDE_H
#define MYSLAM_COMMON_INCLUDE_H

// define the commonly included file to avoid a long include list
// for Eigen
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/Dense>

// for cv
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
using cv::Mat;

// std
#include <vector>
#include <list>
#include <memory>
#include <string>
#include <iostream>
#include <set>
#include <unordered_map>
#include <map>
#include <fstream>
#include <boost/timer.hpp>

// pcl
#include <pcl/point_types.h>
#include <pcl/registration/ndt.h>
#include <pcl/filters/approximate_voxel_grid.h>

// for ceres
#include <ceres/ceres.h>

using namespace std;

#endif //MYSLAM_COMMON_INCLUDE_H
