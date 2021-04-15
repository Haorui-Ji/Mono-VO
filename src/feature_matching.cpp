//
// Created by jihaorui on 2/23/21.
//

#include "myslam/feature_matching.h"
#include "myslam/utils.h"

namespace myslam
{

void extractFeatures(
        const cv::Mat &image,
        vector<cv::KeyPoint> &keypoints,
        cv::Mat &descriptors)
{
    // -- Set arguments
    static const int num_keypoints = Config::get<int>("number_of_features");
    static const float scale_factor = Config::get<float>("scale_factor");
    static const int level_pyramid = Config::get<int>("level_pyramid");
    static const int score_threshold = Config::get<int>("score_threshold");
//    cout << num_keypoints << '\t' << scale_factor << '\t' << level_pyramid << '\t' << score_threshold <<endl;

    // -- Create ORB
    static cv::Ptr<cv::ORB> orb = cv::ORB::create(num_keypoints, scale_factor, level_pyramid,
                                                  31, 0, 2, cv::ORB::HARRIS_SCORE, 31, score_threshold);
    // Default arguments of ORB:
    //          int 	nlevels = 8,
    //          int 	edgeThreshold = 31,
    //          int 	firstLevel = 0,
    //          int 	WTA_K = 2,
    //          ORB::ScoreType 	scoreType = ORB::HARRIS_SCORE,
    //          int 	patchSize = 31,
    //          int 	fastThreshold = 20

    // detect keypoints
    orb->detect(image, keypoints);
//        selectUniformKptsByGrid(keypoints, image.rows, image.cols);

    // compute descriptors
    orb->compute(image, keypoints, descriptors);
}

std::pair<vector<int>, vector<double>> findBestMatches(
        const cv::Mat &descriptors_1,
        const cv::Mat &descriptors_2,
        vector<cv::DMatch> &matches)
{
    // find best matches
    vector<int> idx(descriptors_1.rows, -1);
    vector<double> dist(descriptors_1.rows, -1);
    for (int i =0; i < descriptors_1.rows; i++)
    {
        double bestDist = 1e10;
        int bestMatch = -1;
        for (int j = 0; j < matches.size(); j++)
        {
            if (matches[j].queryIdx == i && matches[j].distance < bestDist)
            {
                bestDist = matches[j].distance;
                bestMatch = matches[j].trainIdx;
            }
        }
        idx[i] = bestMatch;
        dist[i] = bestDist;
    }

    return std::make_pair(idx, dist);
}

std::pair<vector<int>, vector<double>> matchFeaturesHelper(
        const cv::Mat &descriptors_1,
        const cv::Mat &descriptors_2,
        vector<cv::DMatch> &matches)
{
    double match_ratio = Config::get<double> ( "match_ratio" );
    cv::FlannBasedMatcher matcher_flann(new cv::flann::LshIndexParams(5, 10, 2));
    vector<cv::DMatch> all_matches;

    matcher_flann.match ( descriptors_1, descriptors_2, all_matches );
//    cout << "all_matches size" << '\t' << all_matches.size() << endl;
    double min_dist=10000, max_dist=0;

    for ( int i = 0; i < all_matches.size(); i++)
    {
        double dist = all_matches[i].distance;
        if ( dist < min_dist ) min_dist = dist;
        if ( dist > max_dist ) max_dist = dist;
    }

    // Select good matches and push to the result vector.
    for ( cv::DMatch& m : all_matches  )
    {
        if ( m.distance <= max ( min_dist*match_ratio, 50.0 ) )
        {
            matches.push_back( m );
        }
    }
//    cout << "refine size" << '\t' << matches.size() << endl;

    // find best matches
    return findBestMatches(descriptors_1, descriptors_2, matches);
}

void matchFeatures(
        const cv::Mat &descriptors_1,
        const cv::Mat &descriptors_2,
        vector<cv::DMatch> &matches)
{
    vector<cv::DMatch> all_matches_12, all_matches_21;

    std::pair<vector<int>, vector<double>> best_matches_12 = matchFeaturesHelper(descriptors_1, descriptors_2, all_matches_12);
    std::pair<vector<int>, vector<double>> best_matches_21 = matchFeaturesHelper(descriptors_2, descriptors_1, all_matches_21);

    for (int idx1 = 0; idx1 < (int)best_matches_12.first.size(); idx1++) {
        int idx2 = best_matches_12.first[idx1];
        if (best_matches_21.first[idx2] == idx1)
        {
            cv::DMatch match = cv::DMatch(idx1, idx2, best_matches_12.second[idx1]);
            matches.push_back(match);
        }
    }
}

void extractPtsFromMatches(
        const vector<cv::Point2f> &points_1,
        const vector<cv::Point2f> &points_2,
        const vector<cv::DMatch> &matches,
        vector<cv::Point2f> &pts1,
        vector<cv::Point2f> &pts2)
{
    pts1.clear();
    pts2.clear();
    for (auto &m : matches)
    {
        pts1.push_back(points_1[m.queryIdx]);
        pts2.push_back(points_2[m.trainIdx]);
    }
}

void extractPtsFromMatches(
        const vector<cv::KeyPoint> &keypoints_1,
        const vector<cv::KeyPoint> &keypoints_2,
        const vector<cv::DMatch> &matches,
        vector<cv::Point2f> &pts1,
        vector<cv::Point2f> &pts2)
{
    pts1.clear();
    pts2.clear();
    for (auto &m : matches)
    {
        pts1.push_back(keypoints_1[m.queryIdx].pt);
        pts2.push_back(keypoints_2[m.trainIdx].pt);
    }
}

double computeMeanDistBetweenKeypoints(
        const vector<cv::KeyPoint> &kpts1,
        const vector<cv::KeyPoint> &kpts2,
        const vector<cv::DMatch> &matches)
{

    vector<double> dists_between_kpts;
    for (const cv::DMatch &d : matches)
    {
        cv::Point2f p1 = kpts1[d.queryIdx].pt;
        cv::Point2f p2 = kpts2[d.trainIdx].pt;
        dists_between_kpts.push_back(calcDist(p1, p2));
    }
    double mean_dist = 0;
    for (double d : dists_between_kpts)
        mean_dist += d;
    mean_dist /= dists_between_kpts.size();
    return mean_dist;
}


}

