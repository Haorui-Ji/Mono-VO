//
// Created by jihaorui on 2/23/21.
//

#include "myslam/camera.h"

namespace myslam
{

cv::Point3f world2camera(const cv::Point3f &p_c, const cv::Mat &T_c_w)
{
    cv::Mat p_c_h = (cv::Mat_<double>(4, 1) << p_c.x, p_c.y, p_c.z, 1);
    cv::Mat p_w_h = T_c_w * p_w_h;
    return cv::Point3f(p_c_h.at<double>(1, 0),
                       p_c_h.at<double>(2, 0),
                       p_c_h.at<double>(3, 0));
}

cv::Point3f camera2world(const cv::Point3f &p_c, const cv::Mat &T_c_w)
{
    cv::Mat R = T_c_w.rowRange(0, 3).colRange(0, 3);
    cv::Mat t = T_c_w.rowRange(0, 3).col(3);

    cv::Mat T_c_w_4x4 = cv::Mat::eye(4, 4, CV_64F);
    R.copyTo(T_c_w.rowRange(0, 3).colRange(0, 3));
    t.copyTo(T_c_w.rowRange(0, 3).col(3));

    cv::Mat p_c_h = (cv::Mat_<double>(4, 1) << p_c.x, p_c.y, p_c.z, 1);
    cv::Mat p_w_h = T_c_w_4x4.inv() * p_c_h;
    return cv::Point3f(p_w_h.at<double>(1, 0),
                       p_w_h.at<double>(2, 0),
                       p_w_h.at<double>(3, 0));
}

cv::Point3f pixel2camera(const cv::Point2f &p, const cv::Mat &K, double depth)
{
    return cv::Point3f(
            depth * (p.x - K.at<double>(0, 2)) / K.at<double>(0, 0),
            depth * (p.y - K.at<double>(1, 2)) / K.at<double>(1, 1),
            depth);
}

cv::Point2f camera2pixel(const cv::Point3f &p, const cv::Mat &K)
{
    return cv::Point2f(
            K.at<double>(0, 0) * p.x / p.z + K.at<double>(0, 2),
            K.at<double>(1, 1) * p.y / p.z + K.at<double>(1, 2));
}

cv::Point2f world2pixel ( const cv::Point3f &p, const cv::Mat &T_c_w, const cv::Mat &K )
{
    return camera2pixel ( world2camera(p, T_c_w), K );
}

cv::Point3f pixel2world ( const cv::Point2f &p, const cv::Mat &K, double depth, const cv::Mat &T_c_w )
{
    return camera2world ( pixel2camera ( p, K, depth ), T_c_w );
}

}

