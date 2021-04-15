//
// Created by jihaorui on 2/23/21.
//

#ifndef MYSLAM_FEATURE_MATCHING_H
#define MYSLAM_FEATURE_MATCHING_H

#include "myslam/common_include.h"
#include "myslam/config.h"

namespace myslam
{

void extractFeatures(
        const cv::Mat &image,
        vector<cv::KeyPoint> &keypoints,
        cv::Mat &descriptors);

void matchFeatures(
        const cv::Mat &descriptors_1,
        const cv::Mat &descriptors_2,
        vector<cv::DMatch> &matches);

void extractPtsFromMatches(
        const vector<cv::Point2f> &points_1,
        const vector<cv::Point2f> &points_2,
        const vector<cv::DMatch> &matches,
        vector<cv::Point2f> &pts1,
        vector<cv::Point2f> &pts2);

void extractPtsFromMatches(
        const vector<cv::KeyPoint> &keypoints_1,
        const vector<cv::KeyPoint> &keypoints_2,
        const vector<cv::DMatch> &matches,
        vector<cv::Point2f> &pts1,
        vector<cv::Point2f> &pts2);

double computeMeanDistBetweenKeypoints(
        const vector<cv::KeyPoint> &kpts1,
        const vector<cv::KeyPoint> &kpts2,
        const vector<cv::DMatch> &matches);

// Use a grid to remove the keypoints that are too close to each other.
void selectUniformKptsByGrid(vector<cv::KeyPoint> &keypoints,
                             int image_rows, int image_cols);

// --------------------- Other assistant functions ---------------------
std::pair<vector<int>, vector<double>> findBestMatches(
        const cv::Mat &descriptors_1,
        const cv::Mat &descriptors_2,
        vector<cv::DMatch> &matches);

std::pair<vector<int>, vector<double>> matchFeaturesHelper(
        const cv::Mat &descriptors_1,
        const cv::Mat &descriptors_2,
        vector<cv::DMatch> &matches);

} // namespace myslam

#endif MYSLAM_FEATURE_MATCHING_H
