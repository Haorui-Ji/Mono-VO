//
// Created by jihaorui on 2/23/21.
//

#ifndef MYSLAM_CAMERA_H
#define MYSLAM_CAMERA_H

#include "myslam/common_include.h"
#include "myslam/config.h"

using namespace myslam;

namespace myslam
{

// coordinate transform: world, camera, pixel
cv::Point2f pixel2CamNormPlane(const cv::Point2f &p, const cv::Mat &K);
cv::Point3f world2camera(const cv::Point3f &p, const cv::Mat &T_c_w);
cv::Point3f camera2world(const cv::Point3f &p, const cv::Mat &T_c_w);
cv::Point3f pixel2camera(const cv::Point2f &p, const cv::Mat &K, double depth = 1);
cv::Point2f camera2pixel(const cv::Point3f &p, const cv::Mat &K);
cv::Point2f world2pixel(const cv::Point3f &p, const cv::Mat &T_c_w, const cv::Mat &K);
cv::Point3f pixel2world (const cv::Point2f &p, const cv::Mat &K, double depth, const cv::Mat &T_c_w);


// Pinhole RGBD camera model
class Camera
{
public:
    typedef std::shared_ptr<Camera> Ptr;
    double  fx_, fy_, cx_, cy_;
    cv::Mat K_;

    Camera()
    {
        fx_ = Config::get<double>("camera.fx");
        fy_ = Config::get<double>("camera.fy");
        cx_ = Config::get<double>("camera.cx");
        cy_ = Config::get<double>("camera.cy");
        K_ = (cv::Mat_<double>(3, 3) << fx_, 0, cx_, 0, fy_, cy_, 0, 0, 1);
    }

    Camera(double fx, double fy, double cx, double cy) :
    fx_(fx), fy_(fy), cx_(cx), cy_(cy)
    {
        K_ = (cv::Mat_<double>(3, 3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);
    }

    Camera(cv::Mat K)
    {
        fx_ = K.at<double>(0, 0);
        fy_ = K.at<double>(1, 1);
        cx_ = K.at<double>(0, 2);
        cy_ = K.at<double>(1, 2);
        K_ = K;
    }

};

}


#endif //MYSLAM_CAMERA_H
