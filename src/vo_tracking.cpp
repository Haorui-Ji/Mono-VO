//
// Created by jihaorui on 3/9/21.
//

#include "myslam/visual_odometry.h"
#include "myslam/feature_matching.h"
#include "myslam/epipolar_geometry.h"
#include "myslam/kitti_config.h"
#include "myslam/utils.h"
#include "myslam/ceres_ba.h"

namespace myslam
{
    bool VisualOdometry::poseEstimationPnP_()
    {
        // From the local map, find the mappoints that fall into the current view
        vector<MapPoint::Ptr> candidate_mappoints_in_map;
        vector<cv::Point2f> candidate_2d_pts_in_image;
        cv::Mat corresponding_mappoints_descriptors;

//        // Check if mappoints positions are correct
//        for (auto &iter_map_point : map_->map_points_) {
//            MapPoint::Ptr &p_world = iter_map_point.second;
//
//            cout << p_world->pos_ << '\t' << transCoord(p_world->pos_, curr_->T_c_w_) << endl;
//        }

        getMappointsInCurrentView_(
                candidate_mappoints_in_map,
                candidate_2d_pts_in_image,
                corresponding_mappoints_descriptors);

        // Compare descriptors to find matches, and extract 3d 2d correspondance
        // match.queryIdx is the index of mappoints in candidate_mappoints_in_map
        // match.trainIdx is the index of 2D keypoints
        matchFeatures(
                corresponding_mappoints_descriptors,
                curr_->descriptors_,
                curr_->matches_with_map_);

        const int num_matches = curr_->matches_with_map_.size();
        cout << "Number of 3d-2d pairs: " << num_matches << endl;
        vector<cv::Point3f> pts_3d;
        vector<cv::Point2f> pts_2d;
        for (int i = 0; i < num_matches; i++)
        {
            cv::DMatch &match = curr_->matches_with_map_[i];
            MapPoint::Ptr mappoint = candidate_mappoints_in_map[match.queryIdx];
            pts_3d.push_back(mappoint->pos_);
            pts_2d.push_back(curr_->keypoints_[match.trainIdx].pt);
        }

        // -- Solve PnP, get T_c_w_
        constexpr int kMinPtsForPnP = 5;
        static const double max_possible_dist_to_prev_frame =
                Config::get<double>("max_possible_dist_to_prev_frame");

        cv::Mat pnp_inliers_mask; // type = 32SC1, size = 999x1
        cv::Mat rvec, t;

        bool is_pnp_good = num_matches >= kMinPtsForPnP;
        if (is_pnp_good)
        {
            bool useExtrinsicGuess = false;
            int iterationsCount = 100;
            float reprojectionError = 2.0;
            double confidence = 0.999;
            cv::solvePnPRansac(pts_3d, pts_2d, curr_->camera_->K_, cv::Mat(), rvec, t,
                               useExtrinsicGuess,
                               iterationsCount, reprojectionError, confidence, pnp_inliers_mask, cv::SOLVEPNP_EPNP);
            // Output two variables:
            //      1. curr_->matches_with_map_
            //      2. curr_->T_c_w_

            cv::Mat R;
            cv::Rodrigues(rvec, R); // angle-axis rotation to 3x3 rotation matrix

            // -- Get inlier matches used in PnP
            vector<cv::Point2f> tmp_pts_2d;
            vector<cv::Point3f> inlier_candidates_pos;
            vector<MapPoint::Ptr> inlier_candidates;
            vector<cv::DMatch> tmp_matches_with_map_;
            int num_inliers = pnp_inliers_mask.rows;
            for (int i = 0; i < num_inliers; i++)
            {
                int good_idx = pnp_inliers_mask.at<int>(i, 0);

                // good match
                cv::DMatch &match = curr_->matches_with_map_[good_idx];
                tmp_matches_with_map_.push_back(match);

                // good pts 3d
                MapPoint::Ptr inlier_mappoint = candidate_mappoints_in_map[match.queryIdx];
                inlier_candidates_pos.push_back(inlier_mappoint->pos_);
                inlier_mappoint->matched_times_++;

                // good pts 2d
                tmp_pts_2d.push_back(pts_2d[good_idx]);

                // Update graph info
                curr_->inliers_to_mappt_connections_[match.trainIdx] = PtConn{-1, inlier_mappoint->id_};
            }
            pts_2d.swap(tmp_pts_2d);
            curr_->matches_with_map_.swap(tmp_matches_with_map_);

            // -- Update current camera pos
            curr_->T_c_w_ = convertRt2T(R, t);

            // -- Check relative motion with previous frame
            cv::Mat t_prev, t_curr;
            t_curr = getPosFromT(curr_->T_c_w_);
            t_prev = getPosFromT(prev_->T_c_w_);
            double dist_to_prev_keyframe = cv::norm(t_prev - t_curr);
            printf("PnP: distance with prev frame is %.3f. Threshold is %.3f.\n",
                   dist_to_prev_keyframe, max_possible_dist_to_prev_frame);
            if (dist_to_prev_keyframe >= max_possible_dist_to_prev_frame)
            {
                is_pnp_good = false;
            }
        }
        else
        {
            printf("PnP num inlier matches: %d.\n", num_matches);
        }

        if (!is_pnp_good) // Set this frame's pose the same as previous frame
        {
            curr_->T_c_w_ = prev_->T_c_w_.clone();
        }
        return is_pnp_good;
    }

    bool VisualOdometry::checkLargeMoveForAddKeyFrame_()
    {
        cv::Mat T_c_r = curr_->T_c_w_ * ref_->T_c_w_.inv();
        cv::Mat R, t, R_vec;
        getRtFromT(T_c_r, R, t);
        cv::Rodrigues(R, R_vec);

        static const double min_dist_between_two_keyframes = Config::get<double>("min_dist_between_two_keyframes");
        static const double min_rotation_angle_betwen_two_keyframes = Config::get<double>("min_rotation_angle_betwen_two_keyframes");

        double moved_dist = cv::norm(t);
        double rotated_angle = cv::norm(R_vec);

        printf("Wrt prev keyframe, relative dist = %.5f, angle = %.5f\n", moved_dist, rotated_angle);

        // Satisfy each one will be a good keyframe
        bool res = false;
        if (moved_dist > min_dist_between_two_keyframes ||
            rotated_angle > min_rotation_angle_betwen_two_keyframes)
        {
            res = true;
        }
        return res;
    }

    // bundle adjustment (optimize local map)
    void VisualOdometry::callBundleAdjustment_()
    {
        // Read settings from config.yaml
        static const bool is_ba_fix_map_points = Config::getBool("is_ba_fix_map_points");
        static const bool is_ba_update_map_points = !is_ba_fix_map_points;

        // Set params
        const int kNumFramesForBA = frames_buff_.size();

        printf("\nCalling bundle adjustment on %d frames ... \n", kNumFramesForBA - 1);

        // Measurement (which is fixed; ground truth)
        vector<vector<cv::Point2f>> v_pts_2d;
        vector<vector<int>> v_pts_2d_to_3d_idx;

        // Things to to optimize
        std::unordered_map<int, cv::Point3f *> um_pts_3d_in_map;
        vector<cv::Point3f> v_pts_3d_only_in_curr;
        vector<cv::Mat *> v_camera_poses;

        // Set up input vars
        // Use all the frames in frame_buffer for BA
        for (int ith_frame_in_buff = kNumFramesForBA - 1;
             ith_frame_in_buff >= 0;
             ith_frame_in_buff--)
        {
            Frame::Ptr frame = frames_buff_[ith_frame_in_buff];
            int num_mappt_in_frame = frame->inliers_to_mappt_connections_.size();
            if (num_mappt_in_frame < 3)
            {
                continue; // Too few mappoints. Not optimizing this frame
            }
            printf("Frame id: %d, num map points = %d\n", frame->id_, num_mappt_in_frame);
            v_pts_2d.push_back(vector<cv::Point2f>());
            v_pts_2d_to_3d_idx.push_back(vector<int>());

            // Get camera poses
            v_camera_poses.push_back(&frame->T_c_w_);

            // Iterate through this camera's mappoints
            for (std::unordered_map<int, PtConn>::iterator it = frame->inliers_to_mappt_connections_.begin();
                 it != frame->inliers_to_mappt_connections_.end(); it++)
            {
                int kpt_idx = it->first;
                int mappt_idx = it->second.pt_map_idx;
                if (map_->map_points_.find(mappt_idx) == map_->map_points_.end())
                {
                    continue; // point has been deleted
                }

                // Get 2d pos
                v_pts_2d.back().push_back(frame->keypoints_[kpt_idx].pt);
                v_pts_2d_to_3d_idx.back().push_back(mappt_idx);

                // Get 3d pos
                cv::Point3f *p = &(map_->map_points_[mappt_idx]->pos_);
                um_pts_3d_in_map[mappt_idx] = p;
            }
        }

        // Bundle Adjustment
        ceres::Problem problem;
        cv::Mat K = curr_->camera_->K_;

        // Things to optimize
        int num_frames = v_camera_poses.size();

        vector<double *> camera_rvec(num_frames);
        vector<double *> camera_t(num_frames);

        // Initialization
        for (int i = 0; i < num_frames; i++)
        {
            camera_rvec[i] = new double[3];
            camera_t[i] = new double[3];
        }

        for (int ith_frame = 0; ith_frame < num_frames; ith_frame++)
        {
            int num_pts_2d = v_pts_2d[ith_frame].size();

            // Retrieve camera pose
            cv::Mat pose = *v_camera_poses[ith_frame];
            cv::Mat R, t;
            getRtFromT(pose, R, t);

            cv::Mat rvec;
            Rodrigues(R, rvec);

            camera_rvec[ith_frame][0] = rvec.at<double>(0, 0);
            camera_rvec[ith_frame][1] = rvec.at<double>(1, 0);
            camera_rvec[ith_frame][2] = rvec.at<double>(2, 0);
            camera_t[ith_frame][0] = t.at<double>(0, 0);
            camera_t[ith_frame][1] = t.at<double>(1, 0);
            camera_t[ith_frame][2] = t.at<double>(2, 0);

            // Retrieve 3d mappoints and corresponding 2d keypoints
            for (int j = 0; j < num_pts_2d; j++)
            {
                const cv::Point2f pt2d = v_pts_2d[ith_frame][j];
                int pt3d_id = v_pts_2d_to_3d_idx[ith_frame][j];
                const cv::Point3f *pt3d = um_pts_3d_in_map[pt3d_id];

                ceres::CostFunction* cost_function = ReprojectionErrorFixPoints::create(*pt3d, pt2d, K);
                problem.AddResidualBlock(cost_function, NULL, camera_rvec[ith_frame], camera_t[ith_frame]);
            }
        }

        ceres::Solver::Options options;
        options.linear_solver_type = ceres::DENSE_SCHUR;
        options.max_num_iterations = 100;
        options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
        options.minimizer_progress_to_stdout = true;

        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);
        std::cout << summary.BriefReport() << std::endl;

        // Update the result for camera pose in every frame(copy them back)
        for (int ith_frame = 0; ith_frame < num_frames; ith_frame++)
        {
            cv::Mat rvec = (cv::Mat_<double>(3, 1) << camera_rvec[ith_frame][0],
                                                                    camera_rvec[ith_frame][1],
                                                                    camera_rvec[ith_frame][2]);
            cv::Mat t = (cv::Mat_<double>(3, 1) << camera_t[ith_frame][0],
                                                                    camera_t[ith_frame][1],
                                                                    camera_t[ith_frame][2]);;
            cv::Mat R;
            Rodrigues(rvec, R);
            cv::Mat T = convertRt2T(R, t);

//            cout << "Num " << ith_frame << '\t' << *v_camera_poses[ith_frame] << endl;
            *v_camera_poses[ith_frame] = convertRt2T(R, t);
//            for (int i = 0; i < T.rows; i++)
//            {
//                for (int j = 0; j < T.cols; j++)
//                {
//                    v_camera_poses[ith_frame]->at<double>(i, j) = T.at<double>(i, j);
//                }
//            }
//            cout << "Num " << ith_frame << '\t' << *v_camera_poses[ith_frame] << endl;
        }
    }
}

