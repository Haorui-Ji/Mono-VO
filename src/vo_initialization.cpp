//
// Created by jihaorui on 3/2/21.
//

#include "myslam/visual_odometry.h"
#include "myslam/feature_matching.h"
#include "myslam/epipolar_geometry.h"
#include "myslam/kitti_config.h"
#include "myslam/utils.h"

namespace myslam
{

void VisualOdometry::estimateMotionAnd3DPoints_()
    {
        // -- Rename output
        vector<cv::DMatch> &inlier_matches = curr_->inliers_matches_with_ref_;
        vector<cv::Point3f> &pts3d_in_curr = curr_->inliers_pts3d_;
        vector<cv::DMatch> &inliers_matches_for_3d = curr_->inliers_matches_for_3d_;
        cv::Mat &T = curr_->T_c_w_;

        // (1) motion from Essential, (2) inliers indices, (3) triangulated points
        // Motion estimation & inliers
        vector<cv::KeyPoint> pts_img1_all = ref_->keypoints_;
        vector<cv::KeyPoint> pts_img2_all = curr_->keypoints_;
        vector<cv::DMatch> matches = curr_->matches_with_ref_;
        vector<cv::Point2f> pts_img1, pts_img2; // matched points
        extractPtsFromMatches(pts_img1_all, pts_img2_all, matches, pts_img1, pts_img2);
        cv::Mat &K = curr_->camera_->K_;

        cv::Mat R21_e, t21_e, essential_matrix;
        vector<int> inliers_index_e; // index of the inliers

        estiMotionByEssential(pts_img1, pts_img2, K,
                                essential_matrix, R21_e, t21_e, inliers_index_e);

//        cout << "R: " << endl << R21_e << endl;
//        cout << "t: " << endl << t21_e << endl;

        inlier_matches.clear();
        // Convert [inliers of matches] to the [cv::DMatch of all kpts]
        for (const int &idx : inliers_index_e)
        {
            inlier_matches.push_back(
                    cv::DMatch(matches[idx].queryIdx, matches[idx].trainIdx, matches[idx].distance));
        }

        // Triangulation
        const cv::Mat &R_curr_prev = R21_e;
        const cv::Mat &t_curr_prev = t21_e;
        vector<cv::Point3f> pts3d_in_cam1;
        doTriangulation(pts_img1, pts_img2, K, R_curr_prev, t_curr_prev, inliers_index_e, pts3d_in_cam1);

        pts3d_in_curr.clear();
        for (const cv::Point3f &p1 : pts3d_in_cam1)
        {
            pts3d_in_curr.push_back(transCoord(p1, R_curr_prev, t_curr_prev));
        }

//        for (int i = 0; i < pts_img1.size(); i++)
//        {
//            cout << pts_img1[i] << '\t' << pts3d_in_cam1[i] << endl;
//            cout << pts_img2[i] << '\t' << pts3d_in_curr[i] << endl;
//        }

        // compute camera pose
        T = convertRt2T(R_curr_prev, t_curr_prev) * ref_->T_c_w_;

        // Get points that are used for triangulating new map points
        retainGoodTriangulationResult_();

        int N = curr_->inliers_pts3d_.size();
        if (N < 20)
        {
            printf("After remove bad triag, only %d pts. It's too few ...\n", N);
            return;
        }

//        // Normalize average points depth to the one we set, and adjust translation accordingly
//        double mean_depth_without_scale = calcMeanDepth(curr_->inliers_pts3d_);
//        cout << "Mean depth without scale: " << mean_depth_without_scale << endl;
//        static const double assumed_mean_pts_depth_during_vo_init =
//                Config::get<double>("assumed_mean_pts_depth_during_vo_init");
//        double scale = assumed_mean_pts_depth_during_vo_init / mean_depth_without_scale;
//        t_curr_prev *= scale;
//        for (cv::Point3f &p : curr_->inliers_pts3d_)
//        {
//            scalePointPos(p, scale);
//        }
//        T = convertRt2T(R_curr_prev, t_curr_prev) * ref_->T_c_w_;
    }

// Compute the triangulation angle of each point, and get the statistics.
void VisualOdometry::retainGoodTriangulationResult_()
{
    int N = (int)curr_->inliers_pts3d_.size();
    if (N == 0)
        return;

    // Step 1: Remove triangulation results whose depth < 0 or any component is infinite
    vector<bool> feasibility;
    for (int i = 0; i < N; i++)
    {
        cv::Point3f &p_in_curr = curr_->inliers_pts3d_[i];
        feasibility.push_back(p_in_curr.z >= 0 &&
                              isfinite(p_in_curr.x) &&
                              isfinite(p_in_curr.y) &&
                              isfinite(p_in_curr.z));
    }

    // Step 2: Remove those with a too large or too small parallax angle.
    static const double min_triang_angle = Config::get<double>("min_triang_angle");
    static const double max_ratio_between_max_angle_and_median_angle =
            Config::get<double>("max_ratio_between_max_angle_and_median_angle");

    vector<double> &angles = curr_->triangulation_angles_of_inliers_;
    // -- Compute parallax angles
    for (int i = 0; i < N; i++)
    {
        cv::Point3f &p_in_curr = curr_->inliers_pts3d_[i];
        cv::Mat p_in_world = point3f_to_mat3x1(transCoord(p_in_curr, curr_->T_c_w_.inv()));
        cv::Mat vec_p_to_cam_curr = getPosFromT(curr_->T_c_w_.inv()) - p_in_world;
        cv::Mat vec_p_to_cam_prev = getPosFromT(ref_->T_c_w_.inv()) - p_in_world;
        double angle = calcAngleBetweenTwoVectors(vec_p_to_cam_curr, vec_p_to_cam_prev);
        angles.push_back(angle / 3.1415926 * 180.0);
    }

    // Get statistics
    vector<double> sort_a = angles;
    sort(sort_a.begin(), sort_a.end());
    double mean_angle = accumulate(sort_a.begin(), sort_a.end(), 0.0) / N;
    double median_angle = sort_a[N / 2];
    printf("Triangulation angle: mean=%f, median=%f, min=%f, max=%f\n",
           mean_angle,   // mean
           median_angle, // median
           sort_a[0],    // min
           sort_a[N - 1] // max
    );

    for (int i = 0; i < N; i++)
    {
        if (angles[i] < min_triang_angle ||
            angles[i] / median_angle > max_ratio_between_max_angle_and_median_angle)
        {
            feasibility[i] = false;
        }
    }
//    ///////////////////////////////////
//    // Test
//    int cnt_false_2 = 0;
//    int cnt_true_2 = 0;
//    for (int i = 0; i < N; i++)
//    {
//        if (feasibility[i] == false)
//        {
//            cnt_false_2++;
//        }
//        else
//        {
//            cnt_true_2++;
//        }
//    }
//    printf("Stage 2:\n Total: %d, true: %d, false: %d\n", N, cnt_true_2, cnt_false_2);
//    ///////////////////////////////////


    // Step 3: Remove those reprojection error is too large
    cv::Mat &K = curr_->camera_->K_;
    static const double sigma = Config::get<double>("initialization_sigma");
    double sigma2 = sigma * sigma;
    cv::Mat T12 = ref_->T_c_w_ * curr_->T_c_w_.inv();
    for (int i = 0; i < N; i++)
    {
        cv::Point3f p_cam2 = curr_->inliers_pts3d_[i];
        cv::Point2f p_img2_proj = camera2pixel(p_cam2, K);

        cv::Point3f p_cam1 = transCoord(p_cam2, T12);
        cv::Point2f p_img1_proj = camera2pixel(p_cam1, K);

        // Check frame1
        cv::KeyPoint kp1 = ref_->keypoints_[curr_->inliers_matches_with_ref_[i].queryIdx];
        float squareError1 = (p_img1_proj.x - kp1.pt.x) * (p_img1_proj.x - kp1.pt.x) \
                                + (p_img1_proj.y - kp1.pt.y) * (p_img1_proj.y - kp1.pt.y);

        // Check frame 2
        cv::KeyPoint kp2 = curr_->keypoints_[curr_->inliers_matches_with_ref_[i].trainIdx];
        float squareError2 = (p_img2_proj.x - kp2.pt.x) * (p_img2_proj.x - kp2.pt.x) \
                                + (p_img2_proj.y - kp2.pt.y) * (p_img2_proj.y - kp2.pt.y);

//        printf("Reprojection 1: %f, Reprojection 2: %f \n", squareError1, squareError2);

        if (squareError1 > 4*sigma2 || squareError2 > 4*sigma2)
        {
            feasibility[i] = false;
        }
    }

//    ///////////////////////////////////
//    // Test
//    int cnt_false_3 = 0;
//    int cnt_true_3 = 0;
//    for (int i = 0; i < N; i++)
//    {
//        if (feasibility[i] == false)
//        {
//            cnt_false_3++;
//        }
//        else
//        {
//            cnt_true_3++;
//        }
//    }
//    printf("Stage 3:\n Total: %d, true: %d, false: %d\n", N, cnt_true_3, cnt_false_3);
//    ///////////////////////////////////

    // Get good triangulation points
    vector<cv::Point3f> old_inlier_points = curr_->inliers_pts3d_;
    curr_->inliers_pts3d_.clear();

    vector<double> old_angles = angles;
    angles.clear();

    for (int i = 0; i < N; i++)
    {
        if (feasibility[i] == true)
        {
            cv::DMatch dmatch = curr_->inliers_matches_with_ref_[i];
            curr_->inliers_matches_for_3d_.push_back(dmatch);
            curr_->inliers_pts3d_.push_back(old_inlier_points[i]);
            angles.push_back(old_angles[i]);
        }
    }
}

bool VisualOdometry::isVoGoodToInit_()
{

    // -- Rename input
    const vector<cv::KeyPoint> &init_kpts = ref_->keypoints_;
    const vector<cv::KeyPoint> &curr_kpts = curr_->keypoints_;
    const vector<cv::DMatch> &matches = curr_->inliers_matches_for_3d_;

    // Params
    static const int min_inlier_matches = Config::get<int>("min_inlier_matches");
    static const double min_pixel_dist = Config::get<double>("min_pixel_dist");
    static const double min_median_triangulation_angle = Config::get<double>("min_median_triangulation_angle");

    // -- Check CRITERIA_0: num inliers should be large
    bool criteria_0 = true;
    if (matches.size() < min_inlier_matches)
    {
        printf("%d inlier points are too few... threshold is %d.\n",
               int(matches.size()), min_inlier_matches);
        criteria_0 = false;
    }

    // -- Check criteria_1: init vo only when distance between matched keypoints are large
    bool criteria_1 = true;
    vector<double> dists_between_kpts;
    double mean_dist = computeMeanDistBetweenKeypoints(init_kpts, curr_kpts, matches);
    printf("Pixel movement of matched keypoints: %.1f. Threshold is %.1f\n", mean_dist, min_pixel_dist);
    if (mean_dist > min_pixel_dist)
    {
        criteria_1 = true;
    }

    // -- Check criteria_2: triangulation angle of each point should be larger than threshold.
    bool criteria_2 = true;
    if (curr_->triangulation_angles_of_inliers_.size() > 0)
    {
        vector<double> sort_a = curr_->triangulation_angles_of_inliers_; // a copy of angles
        int N = sort_a.size();                                           // num of 3d points triangulated from inlier points
        sort(sort_a.begin(), sort_a.end());
        double mean_angle = accumulate(sort_a.begin(), sort_a.end(), 0.0) / N;
        double median_angle = sort_a[N / 2];
        printf("Triangulation angle: mean=%f, median=%f, min=%f, max=%f.\n",
               mean_angle,   // mean
               median_angle, // median
               sort_a[0],    // min
               sort_a[N - 1] // max
        );

        // Thresholding
        printf("    median_angle is %.2f, threshold is %.2f.\n",
               median_angle, min_median_triangulation_angle);
        if (median_angle > min_median_triangulation_angle)
        {
            criteria_2 = true;
        }
    }

    // -- Return
    return criteria_0 && criteria_1 && criteria_2;
}


}