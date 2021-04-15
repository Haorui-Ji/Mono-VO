//
// Created by jihaorui on 2/25/21.
//

#include "myslam/epipolar_geometry.h"
#include "myslam/feature_matching.h"

namespace myslam
{
    void estiMotionByEssential(
            const vector<cv::Point2f> &pts_in_img1,
            const vector<cv::Point2f> &pts_in_img2,
            const cv::Mat &camera_intrinsics,
            cv::Mat &essential_matrix,
            cv::Mat &R21, cv::Mat &t21,
            vector<int> &inliers_index)
    {
        inliers_index.clear();
        cv::Mat K = camera_intrinsics;

        // -- Essential matrix
        int method = cv::RANSAC;
        static double findEssentialMat_prob = Config::get<double>("findEssentialMat_prob");
        static double findEssentialMat_threshold = Config::get<double>("findEssentialMat_threshold");
        cv::Mat inliers_mask;
        essential_matrix = findEssentialMat(
                pts_in_img1, pts_in_img2, K,
                method, findEssentialMat_prob, findEssentialMat_threshold, inliers_mask);
        essential_matrix /= essential_matrix.at<double>(2, 2);

        // Get inliers
        for (int i = 0; i < inliers_mask.rows; i++)
        {
            if ((int)inliers_mask.at<unsigned char>(i, 0) == 1)
            {
                inliers_index.push_back(i);
            }
        }

        // Recover R,t from Essential matrix
        recoverPose(essential_matrix, pts_in_img1, pts_in_img2, K, R21, t21, inliers_mask);

        // Normalize t
        t21 = t21 / cv::norm(t21);
    }


    void doTriangulation(
            const vector<cv::Point2f> &pts_in_img1,
            const vector<cv::Point2f> &pts_in_img2,
            const cv::Mat &camera_intrinsic,
            const cv::Mat &R21, const cv::Mat &t21,
            const vector<int> &inliers,
            vector<cv::Point3f> &pts3d_in_cam1)
    {
        const cv::Mat &K = camera_intrinsic;

        // extract inliers points
        vector<cv::Point2f> inlier_pts_in_img1, inlier_pts_in_img2;
        for (auto idx : inliers)
        {
            inlier_pts_in_img1.push_back(pts_in_img1[idx]);
            inlier_pts_in_img2.push_back(pts_in_img2[idx]);
        }

        // back project to camera coordinates on normalized plane
        vector<cv::Point2f> inlier_pts_in_cam1, inlier_pts_in_cam2;
        for (int idx = 0; idx < inlier_pts_in_img1.size(); idx++)
        {
            cv::Point3f tmp1 = pixel2camera(inlier_pts_in_img1[idx], K, 1);
            cv::Point3f tmp2 = pixel2camera(inlier_pts_in_img2[idx], K, 1);
            inlier_pts_in_cam1.push_back(cv::Point2f(tmp1.x, tmp1.y));
            inlier_pts_in_cam2.push_back(cv::Point2f(tmp2.x, tmp2.y));
        }

        // set up
        cv::Mat T_c1_w =
                (cv::Mat_<double>(3, 4) <<
                    1, 0, 0, 0,
                    0, 1, 0, 0,
                    0, 0, 1, 0);
        cv::Mat T_c2_w = convertRt2T_3x4(R21, t21);

        // triangulartion
        cv::Mat pts4d_in_world;
        cv::triangulatePoints(
                T_c1_w, T_c2_w, inlier_pts_in_cam1, inlier_pts_in_cam2, pts4d_in_world);

        // change to homogeneous coords
        vector<cv::Point3f> pts3d_in_world;
        for (int i = 0; i < pts4d_in_world.cols; i++)
        {
            cv::Mat x = pts4d_in_world.col(i);
            x /= x.at<float>(3, 0);
            cv::Point3f pt3d_in_world(
                    x.at<float>(0, 0),
                    x.at<float>(1, 0),
                    x.at<float>(2, 0));
            pts3d_in_world.push_back(pt3d_in_world);
        }

        // return
        pts3d_in_cam1 = pts3d_in_world;
    }


    vector<cv::DMatch> helperfindInlierMatchesByEpipolarConstraints(
            const vector<cv::KeyPoint> &keypoints_1,
            const vector<cv::KeyPoint> &keypoints_2,
            const vector<cv::DMatch> &matches,
            const cv::Mat &K)
    {
        // Output
        vector<cv::DMatch> inlier_matches;

        // Estimate Essential to get inlier matches;
        vector<cv::Point2f> pts_in_img1, pts_in_img2;
        extractPtsFromMatches(keypoints_1, keypoints_2, matches, pts_in_img1, pts_in_img2);

        cv::Mat R, t;
        cv::Mat essential_matrix;
        vector<int> inliers_index;
        estiMotionByEssential(pts_in_img1, pts_in_img2, K, essential_matrix, R, t, inliers_index);

        inlier_matches.clear();
        for (int idx : inliers_index)
        {
            const cv::DMatch &m = matches[idx];
            inlier_matches.push_back(
                    cv::DMatch(m.queryIdx, m.trainIdx, m.distance));
        }
        return inlier_matches;
    }


    // Triangulate points in PnP
    vector<cv::Point3f> helperTriangulatePoints(
            const vector<cv::KeyPoint> &prev_kpts, const vector<cv::KeyPoint> &curr_kpts,
            const vector<cv::DMatch> &curr_inlier_matches,
            const cv::Mat &T_curr_prev,
            const cv::Mat &K)
    {
        cv::Mat R_curr_prev, t_curr_prev;
        getRtFromT(T_curr_prev, R_curr_prev, t_curr_prev);

        // Extract matched keypoints, and convert to camera normalized plane
        vector<cv::Point2f> pts_img1, pts_img2;
        extractPtsFromMatches(prev_kpts, curr_kpts, curr_inlier_matches, pts_img1, pts_img2);

        // Set inliers indices
        const cv::Mat &R = R_curr_prev, &t = t_curr_prev; //rename
        vector<int> inliers;
        for (int i = 0; i < pts_img1.size(); i++)
            inliers.push_back(i); // all are inliers

        // Do triangulation
        vector<cv::Point3f> pts_3d_in_prev; // pts 3d pos to compute
        doTriangulation(pts_img1, pts_img2, K, R, t, inliers, pts_3d_in_prev);

        // Change pos to current frame
        vector<cv::Point3f> pts_3d_in_curr;
        for (const cv::Point3f &pt3d : pts_3d_in_prev)
            pts_3d_in_curr.push_back(transCoord(pt3d, R, t));

        // Return
        return pts_3d_in_curr;
    }
}