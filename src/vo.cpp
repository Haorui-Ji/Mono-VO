//
// Created by jihaorui on 3/4/21.
//

#include "myslam/visual_odometry.h"
#include "myslam/feature_matching.h"
#include "myslam/epipolar_geometry.h"
#include "myslam/map.h"
#include "myslam/kitti_config.h"
#include "myslam/utils.h"

namespace myslam
{

VisualOdometry::VisualOdometry() : map_(new (Map))
{
    vo_state_ = BLANK;
}

bool VisualOdometry::isInitialized()
{
    return vo_state_ == TRACKING;
}

void VisualOdometry::updateVelocity()
{
    if (prev_ == nullptr)
    {
        curr_->velocity_ = (cv::Mat_<double>(3, 1) << 0, 0, 0);
        return;
    }
    cv::Mat t_diff = getPosFromT(curr_->T_c_w_) - getPosFromT(prev_->T_c_w_);
    double time_interval = curr_->time_stamp_ - prev_->time_stamp_;
    prev_->velocity_ = (cv::Mat_<double>(3, 1) <<
            t_diff.at<double>(0, 0) / time_interval,
            t_diff.at<double>(1, 0) / time_interval,
            t_diff.at<double>(2, 0) / time_interval);
    curr_->velocity_ = prev_->velocity_.clone();
}

// ------------------- Mapping -------------------

void VisualOdometry::addKeyFrame_(Frame::Ptr keyframe)
{
    map_->insertKeyFrame(keyframe);
    ref_ = keyframe;
}


void VisualOdometry::optimizeMap_()
{
    static const double default_erase = 0.1;
    static double map_point_erase_ratio = default_erase;

    printf("Optimizing Map: \nTotal points in map %d\n", int(map_->map_points_.size()));

    int cnt1 = 0;
    int cnt2 = 0;
    int cnt3 = 0;
    // remove the hardly seen and no visible points
    for (auto iter = map_->map_points_.begin(); iter != map_->map_points_.end();)
    {
        if (!curr_->isInFrame(iter->second->pos_))
        {
            cv::Point3f p_cam = transCoord(iter->second->pos_, curr_->T_c_w_);
//            cout << p_cam << '\t' << camera2pixel(p_cam, curr_->camera_->K_) << endl;
            iter = map_->map_points_.erase(iter);
            cnt1++;
            continue;
        }

        float match_ratio = float(iter->second->matched_times_) / iter->second->visible_times_;
        if (match_ratio < map_point_erase_ratio)
        {
            iter = map_->map_points_.erase(iter);
            cnt2++;
            continue;
        }

        double angle = getViewAngle_(curr_, iter->second);
        if (angle > M_PI / 4.)
        {
            iter = map_->map_points_.erase(iter);
            cnt3++;
            continue;
        }
        iter++;
    }
    printf("Each type of removal:\n Type 1: %d \t Type 2: %d \t Type 3: %d \n", cnt1, cnt2, cnt3);

    if (map_->map_points_.size() > 1000)
    {
        // TODO map is too large, remove some one
        map_point_erase_ratio += 0.05;
    }
    else
        map_point_erase_ratio = default_erase;
    cout << "map points: " << map_->map_points_.size() << endl;
}


void VisualOdometry::pushCurrPointsToMap_()
{
    // -- Input
    const vector<cv::Point3f> &inliers_pts3d_in_curr = curr_->inliers_pts3d_;
    const cv::Mat &T_c_w_curr_ = curr_->T_c_w_;
    const cv::Mat &descriptors = curr_->descriptors_;
    const vector<cv::DMatch> &inliers_matches_for_3d = curr_->inliers_matches_for_3d_;

    // -- Output
    std::unordered_map<int, PtConn> &inliers_to_mappt_connections = curr_->inliers_to_mappt_connections_;

    for (int i = 0; i < inliers_matches_for_3d.size(); i++)
    {
        const cv::DMatch &dm = inliers_matches_for_3d[i];
        int pt_idx_curr = dm.trainIdx;
        int map_point_id;

        // If this point already triangulated in previous frames
        // Then just find the mappoint, and update its descriptor as the descriptor in current frame's keypoint
        if (ref_->isMappoint(dm.queryIdx))
        {
            map_point_id = ref_->inliers_to_mappt_connections_[dm.queryIdx].pt_map_idx;
            MapPoint::Ptr map_point = map_->map_points_[map_point_id];
            map_point->descriptor_ = descriptors.row(pt_idx_curr).clone();
        }
        else // Not triangulated before. Create and push to map.
        {

            // Change coordinate of 3d points to world frame
            cv::Point3f world_pos = transCoord(inliers_pts3d_in_curr[i], T_c_w_curr_.inv());

            // Create map point
            MapPoint::Ptr map_point(new MapPoint( // createMapPoint
                    world_pos,
                    descriptors.row(pt_idx_curr).clone(),
                    getNormalizedMat(point3f_to_mat3x1(world_pos) - curr_->getCamCenter()) // view direction of the point
            ));
            map_point_id = map_point->id_;

            // Push to map
            map_->insertMapPoint(map_point);
        }
        // Update graph connection of current frame
        inliers_to_mappt_connections.insert({pt_idx_curr, PtConn{dm.queryIdx, map_point_id}});
    }
    return;
}


double VisualOdometry::getViewAngle_(Frame::Ptr frame, MapPoint::Ptr point)
{
    cv::Mat n = point3f_to_mat3x1(point->pos_) - frame->getCamCenter();
    n = getNormalizedMat(n);
    cv::Mat vector_dot_product = n.t() * point->norm_;
    return acos(vector_dot_product.at<double>(0, 0));
}


void VisualOdometry::getMappointsInCurrentView_(
        vector<MapPoint::Ptr> &candidate_mappoints_in_map,
        vector<cv::Point2f> &candidate_2d_pts_in_image,
        cv::Mat &corresponding_mappoints_descriptors)
{
    // vector<MapPoint::Ptr> candidate_mappoints_in_map;
    // cv::Mat corresponding_mappoints_descriptors;
    candidate_mappoints_in_map.clear();
    corresponding_mappoints_descriptors.release();

    for (auto &iter_map_point : map_->map_points_)
    {
        MapPoint::Ptr &p_world = iter_map_point.second;

        // -- Check if p in curr frame image
        bool is_p_in_curr_frame = true;

        cv::Point3f p_cam = transCoord(p_world->pos_, curr_->T_c_w_);
        if (p_cam.z < 0)
        {
            is_p_in_curr_frame = false;
        }


        cv::Point2f pixel = camera2pixel(p_cam, curr_->camera_->K_);
        const bool is_inside_image = pixel.x > 0 && pixel.y > 0 && pixel.x < curr_->rgb_img_.cols && pixel.y < curr_->rgb_img_.rows;
        if (!is_inside_image)
        {
            is_p_in_curr_frame = false;
        }

        // -- If is in current frame,
        //      then add this point to candidate_mappoints_in_map
        if (is_p_in_curr_frame)
        {
            candidate_mappoints_in_map.push_back(p_world);
            candidate_2d_pts_in_image.push_back(pixel);
            corresponding_mappoints_descriptors.push_back(p_world->descriptor_);
            p_world->visible_times_++;
        }
    }
}

}