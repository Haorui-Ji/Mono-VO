//
// Created by jihaorui on 2/25/21.
//

#ifndef MYSLAM_EPIPOLAR_GEOMETRY_H
#define MYSLAM_EPIPOLAR_GEOMETRY_H

#endif //MYSLAM_EPIPOLAR_GEOMETRY_H

#include "myslam/common_include.h"
#include "myslam/utils.h"
#include "myslam/camera.h"
#include "myslam/config.h"

namespace myslam
{

void estiMotionByEssential(
        const vector<cv::Point2f> &pts_in_img1, const vector<cv::Point2f> &pts_in_img2,
        const cv::Mat &camera_intrinsics,
        cv::Mat &essential_matrix,
        cv::Mat &R21, cv::Mat &t21,    // R_curr_to_prev, t_curr_to_prev
        vector<int> &inliers_index // the inliers used in estimating Essential
);

void doTriangulation(
        const vector<cv::Point2f> &pts_on_img1,
        const vector<cv::Point2f> &pts_on_img2,
        const cv::Mat &camera_intrinsic,
        const cv::Mat &R21, const cv::Mat &t21,
        const vector<int> &inliers,
        vector<cv::Point3f> &pts3d_in_cam1);

vector<cv::DMatch> helperfindInlierMatchesByEpipolarConstraints(
        const vector<cv::KeyPoint> &keypoints_1,
        const vector<cv::KeyPoint> &keypoints_2,
        const vector<cv::DMatch> &matches,
        const cv::Mat &K);

vector<cv::Point3f> helperTriangulatePoints(
        const vector<cv::KeyPoint> &prev_kpts, const vector<cv::KeyPoint> &curr_kpts,
        const vector<cv::DMatch> &curr_inlier_matches,
        const cv::Mat &T_curr_to_prev,
        const cv::Mat &K);

}

