//
// Created by jihaorui on 2/13/21.
//

#ifndef MYSLAM_VISUALODOMETRY_H
#define MYSLAM_VISUALODOMETRY_H

#include "myslam/common_include.h"
#include "myslam/frame.h"
#include "myslam/mappoint.h"
#include "myslam/map.h"

namespace myslam
{

class VisualOdometry
    {
    public:
        typedef shared_ptr<VisualOdometry> Ptr;
        enum VOState {
            BLANK,
            INITIALIZING,
            TRACKING,
            LOST
        };
        VOState     vo_state_;     // current VO status

        // Frame
        Frame::Ptr curr_ = nullptr;         // current frame
        Frame::Ptr prev_ = nullptr;         // previous frame
        Frame::Ptr ref_ = nullptr;          // reference keyframe
        std::deque<Frame::Ptr> frames_buff_;

        // Map
        Map::Ptr map_;

        cv::Mat T_c_w_estimated_;      // the estimated pose of current frame
        int num_inliers_;               // number of inlier features in icp
        int num_lost_;                  // number of lost times

        // Parameters
        const int kBuffSize_ = 20; // How much prev frames to store (used in bundle adjustment)

    public: // functions
        VisualOdometry();

        void addFrame( Frame::Ptr frame );      // add a new frame

        bool isInitialized();

    private: // functions

        // Push a frame to the buff.
        void pushFrameToBuff_(Frame::Ptr frame)
        {
            frames_buff_.push_back(frame);
            if (frames_buff_.size() > kBuffSize_)
                frames_buff_.pop_front();
        }

        //  ------------------- Initialization -------------------------
        void estimateMotionAnd3DPoints_();

        // Check if visual odmetry is good to be initialized.
        bool isVoGoodToInit_();

        // Remove bad triangulation points
        // Change "pts3d_in_curr", return a new "inlier_matches".
        void retainGoodTriangulationResult_();
        // -------------------------------------------------------------

        // ------------------------- Tracking --------------------------
        bool poseEstimationPnP_();
        bool checkLargeMoveForAddKeyFrame_();
        void callBundleAdjustment_();
        // -------------------------------------------------------------

        // ------------------------------- Mapping -------------------------------
        void addKeyFrame_(Frame::Ptr keyframe);

        void optimizeMap_();

        void pushCurrPointsToMap_();

        void getMappointsInCurrentView_(vector<MapPoint::Ptr> &candidate_mappoints_in_map,
                                        vector<cv::Point2f> &candidate_2d_pts_in_image,
                                        Mat &corresponding_mappoints_descriptors);

        double getViewAngle_(Frame::Ptr frame, MapPoint::Ptr point);
};

// Read image paths from config_file by the key_dataset_dir and key_num_images.
// The image should be named as
vector<string> readImagePaths(
        const string &dataset_dir,
        int num_images,
        const string &image_formatting,
        bool is_print_res);

// Write camera pose to file.
//      Numbers of each row:  1st row of T, 2nd row of T, 3rd row of T
void writePoseToFile(const string filename, vector<cv::Mat> list_T);


}


#endif // MYSLAM_VISUALODOMETRY_H