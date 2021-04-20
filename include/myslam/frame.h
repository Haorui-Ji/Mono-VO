//
// Created by jihaorui on 2/13/21.
//

#ifndef MYSLAM_FRAME_H
#define MYSLAM_FRAME_H

#include "myslam/common_include.h"
#include "myslam/camera.h"
#include "myslam/feature_matching.h"

namespace myslam
{

typedef struct PtConn_
{
    int pt_ref_idx;
    int pt_map_idx;
} PtConn;

class Frame {
public:
    typedef std::shared_ptr<Frame> Ptr;
    static int factory_id_;

public:
    int id_;            // id of this frame
    double time_stamp_; // when it is recorded

    // -- image features
    cv::Mat rgb_img_;
    cv::Mat depth_img_;
    vector<cv::KeyPoint> keypoints_;
    cv::Mat descriptors_;

    // -- Matches with reference keyframe (for E/H or PnP)
    //  for (1) E/H at initialization stage and (2) triangulating 3d points at all stages.
    vector<cv::DMatch> matches_with_ref_;         // matches with reference frame
    vector<cv::DMatch> inliers_matches_with_ref_; // matches that satisify E or H's constraints

    // -- vectors for triangulation
    vector<double> triangulation_angles_of_inliers_;
    vector<cv::DMatch> inliers_matches_for_3d_;                    // matches whose triangulation result is good.
    vector<cv::Point3f> inliers_pts3d_;                            // 3d points triangulated from inliers_matches_for_3d_
    std::unordered_map<int, PtConn> inliers_to_mappt_connections_; // curr idx -> idx in ref, and map

    // -- Matches with map points (for PnP)
    vector<cv::DMatch> matches_with_map_; // inliers matches index with respect to all the points

    // -- Camera
    Camera::Ptr camera_;

    // -- Current pose (This is a 4x4 matrix)
    cv::Mat T_c_w_; // transform from world to camera

    // -- Velocity
    cv::Mat velocity_;

public:
    Frame() {}

    ~Frame() {}

    static Frame::Ptr createFrame(cv::Mat rgb_img, cv::Mat depth_img, Camera::Ptr camera, double time_stamp = -1);
    void calcKeypointsDescriptors();

    bool isInFrame(const cv::Point3f &p_world);
    bool isInFrame(const cv::Mat &p_world);
    bool isMappoint(int idx);

    cv::Mat getCamCenter();
};

}


#endif //MYSLAM_FRAME_H
