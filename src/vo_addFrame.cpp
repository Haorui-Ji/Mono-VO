//
// Created by jihaorui on 3/4/21.
//

#include "myslam/visual_odometry.h"
#include "myslam/feature_matching.h"
#include "myslam/epipolar_geometry.h"
#include "myslam/kitti_config.h"
#include "myslam/utils.h"

namespace myslam
{

void VisualOdometry::addFrame(Frame::Ptr frame)
    {
        cout << "VO State is: " << vo_state_ << endl;
        // Settings
        pushFrameToBuff_(frame);

        // Renamed vars
        curr_ = frame;
        const int img_id = curr_->id_;
        const cv::Mat &K = curr_->camera_->K_;

        // Start
        printf("\n\n=============================================\n");
        printf("Start processing the %dth image.\n", img_id);

        curr_->calcKeypointsDescriptors();
        cout << "Number of keypoints: " << curr_->keypoints_.size() << endl;

        switch (vo_state_)
        {
            case BLANK:
            {
                curr_->T_c_w_ = cv::Mat::eye(4, 4, CV_64F);
                vo_state_ = INITIALIZING;
                addKeyFrame_(curr_);
                ref_ = curr_;
                break;
            }
            case INITIALIZING:
            {
                // Match features
                matchFeatures(
                        ref_->descriptors_, curr_->descriptors_, curr_->matches_with_ref_);
                printf("Number of matches with the 1st frame: %d\n", (int)curr_->matches_with_ref_.size());

                // Estimae motion and triangulate points
                estimateMotionAnd3DPoints_();
                printf("Number of inlier matches: %d\n", (int)curr_->inliers_matches_for_3d_.size());

                // Check initialization condition:
                printf("\nCheck VO init conditions: \n");
                ////////////////////////////////////
                // These criteria still needs to be further adjusted
                if (isVoGoodToInit_())
                ///////////////////////////////////
                {
                    cout << "Large movement detected at frame " << img_id << ". Start initialization" << endl;
                    pushCurrPointsToMap_();
                    addKeyFrame_(curr_);
                    vo_state_ = TRACKING;
                    cout << "Inilialiation success !!!" << endl;
                    cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << endl;
                }
                else // skip this frame
                {
                    curr_->T_c_w_ = ref_->T_c_w_.clone();
                    cout << "Not initialize VO ..." << endl;
                }
                break;
            }
            case TRACKING:
            {
                printf("\nDoing tracking\n");
                curr_->T_c_w_ = prev_->T_c_w_.clone(); // Initial estimation of the current pose
                bool is_pnp_good = poseEstimationPnP_();

                if (!is_pnp_good) // pnp failed. Print log.
                {
                    int num_matches = curr_->matches_with_map_.size();
                    constexpr int kMinPtsForPnP = 5;
                    printf("PnP failed.\n");
                    printf("    Num inlier matches with map: %d.\n", num_matches);
                    if (num_matches >= kMinPtsForPnP)
                    {
                        printf("    Computed world to camera transformation:\n");
                        std::cout << curr_->T_c_w_ << std::endl;
                    }
//                    printf("PnP result has been reset as R=identity, t=zero.\n");
                    printf("PnP result has been reset as previous frame. \n");

//                    // Use motion model to set pose for this unsuccessful frame and implement re-initialization
//                    printf(" -------------- Re-initializing ---------------- \n");
//                    vo_state_ = INITIALIZING;
                }
                else
                {
                    // -- Insert a keyframe if motion is large.
                    // Call BA, then triangulate more points
                    if (checkLargeMoveForAddKeyFrame_())
                    {
                        printf("Detected key frame \n");
                        callBundleAdjustment_();
                        // Feature matching betwwen current and reference keyframe
                        matchFeatures(
                                ref_->descriptors_, curr_->descriptors_, curr_->matches_with_ref_);

                        // Find inliers with reference frame by epipolar constraint
                        curr_->inliers_matches_with_ref_ = helperfindInlierMatchesByEpipolarConstraints(
                                ref_->keypoints_, curr_->keypoints_, curr_->matches_with_ref_, K);

                        printf("For triangulation: Matches with prev keyframe: %d; Num inliers: %d \n",
                               (int)curr_->matches_with_ref_.size(), (int)curr_->inliers_matches_with_ref_.size());

                        // Triangulate points
                        cv::Mat T_cur_prev = curr_->T_c_w_ * ref_->T_c_w_.inv();
                        curr_->inliers_pts3d_ = helperTriangulatePoints(
                                ref_->keypoints_, curr_->keypoints_,
                                curr_->inliers_matches_with_ref_, T_cur_prev, K);

                        retainGoodTriangulationResult_();

                        // -- Update state
                        pushCurrPointsToMap_();
                        optimizeMap_();
                        addKeyFrame_(curr_);
                    }
                }

                break;
            }
        }

        // Print relative motion
        if (vo_state_ == TRACKING)
        {
            const cv::Mat T_prev_w = prev_->T_c_w_;
            const cv::Mat T_curr_w = curr_->T_c_w_;
            cv::Mat T_curr_prev = T_curr_w * T_prev_w.inv();
            cv::Mat R, t;
            getRtFromT(T_curr_prev, R, t);
            cout << "\nCamera motion:" << endl;
            cout << "R_curr_prev:\n " << R << endl;
            cout << "t_curr_prev:\n " << t << endl;
        }
        prev_ = curr_;
        cout << "\nEnd of a frame" << endl;
    }

}
