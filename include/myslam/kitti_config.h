//
// Created by jihaorui on 2/3/21.
//

#ifndef MYSLAM_KITTI_H
#define MYSLAM_KITTI_H

#include "myslam/config.h"

namespace myslam
{

class kittiConfig: public Config {
public:

    static void loadCalibration(const std::string & dataset);

    static void loadTimes(const std::string & dataset);

    static void loadPoints(
            const pcl::PointCloud<pcl::PointXYZ>::Ptr& point_cloud,
            const string & dataset,
            int n);

    static Mat loadImage(
            const string & dataset,
            const int cam,
            const int n);

    static Mat loadDepth(
            const string & dataset,
            const int n);

    static int img_width, img_height;

    static vector<double> times;

    static vector<Eigen::Matrix<double, 3, 4>,
            Eigen::aligned_allocator<Eigen::Matrix<double, 3, 4>>> cam_mat;

    static vector<Eigen::Matrix3d,
            Eigen::aligned_allocator<Eigen::Matrix3d>> cam_intrinsic;

    static vector<Eigen::Matrix3d,
            Eigen::aligned_allocator<Eigen::Matrix3d>> cam_intrinsic_inv;

    static vector<Eigen::Vector3d,
            Eigen::aligned_allocator<Eigen::Vector3d>> cam_trans;

    static vector<Eigen::Matrix4d,
            Eigen::aligned_allocator<Eigen::Vector3d>> cam_pose;

    static Eigen::Matrix4d velo_to_cam, cam_to_velo;

    static vector<double> min_x, max_x, min_y, max_y;
};

}


#endif //MYSLAM_KITTI_H
