//
// Created by jihaorui on 3/29/21.
//

#ifndef MYSLAM_CERES_BA_H
#define MYSLAM_CERES_BA_H

#include "myslam/common_include.h"
#include "ceres/rotation.h"

namespace myslam
{

    class ReprojectionError
    {
    public:

        ReprojectionError(cv::Point2f observed, cv::Mat K)
                : observed_(observed) {
            fx_ = K.at<double>(0, 0);
            fy_ = K.at<double>(1, 1);
            cx_ = K.at<double>(0, 2);
            cy_ = K.at<double>(1, 2);
        }

        template<typename T>
        bool operator()(const T *const rvec,
                        const T *const tvec,
                        const T *const pt1,
                        T *residuals) const
        {

            T pt2[3];
            ceres::AngleAxisRotatePoint(rvec, pt1, pt2);

            pt2[0] = pt2[0] + tvec[0];
            pt2[1] = pt2[1] + tvec[1];
            pt2[2] = pt2[2] + tvec[2];

            const T xProj = T(fx_ * (pt2[0] / pt2[2]) + cx_);
            const T yProj = T(fy_ * (pt2[1] / pt2[2]) + cy_);

            const T u = T(observed_.x);
            const T v = T(observed_.y);

            residuals[0] = u - xProj;
            residuals[1] = v - yProj;

//            LOG(INFO) << residuals[0] << " " << residuals[1];

            return true;
        }

        static ceres::CostFunction *create(cv::Point2f observed, cv::Mat K) {
            return (new ceres::AutoDiffCostFunction<ReprojectionError, 2, 3, 3, 3>(
                    new ReprojectionError(observed, K)));
        }

    private:
        cv::Point2f observed_;
        // Camera intrinsics
        double fx_, fy_, cx_, cy_;
    };

    class ReprojectionErrorFixPoints {
    public:

        ReprojectionErrorFixPoints(cv::Point3f point, cv::Point2f observed, cv::Mat K)
                : point_(point), observed_(observed) {
            fx_ = K.at<double>(0, 0);
            fy_ = K.at<double>(1, 1);
            cx_ = K.at<double>(0, 2);
            cy_ = K.at<double>(1, 2);
        }

        template<typename T>
        bool operator()(
                const T *const camera_r,
                const T *const camera_t,
                T *residuals) const
        {
            T pt1[3];
            pt1[0] = T(point_.x);
            pt1[1] = T(point_.y);
            pt1[2] = T(point_.z);

            T pt2[3];
            ceres::AngleAxisRotatePoint(camera_r, pt1, pt2);

            pt2[0] = pt2[0] + camera_t[0];
            pt2[1] = pt2[1] + camera_t[1];
            pt2[2] = pt2[2] + camera_t[2];

            const T xProj = T(fx_ * (pt2[0] / pt2[2]) + cx_);
            const T yProj = T(fy_ * (pt2[1] / pt2[2]) + cy_);

            const T u = T(observed_.x);
            const T v = T(observed_.y);

            residuals[0] = u - xProj;
            residuals[1] = v - yProj;

//            LOG(INFO) << residuals[0] << " " << residuals[1];

            return true;
        }

        static ceres::CostFunction *create(cv::Point3f point, cv::Point2f observed, cv::Mat K) {
            return (new ceres::AutoDiffCostFunction<ReprojectionErrorFixPoints, 2, 3, 3>(
                    new ReprojectionErrorFixPoints(point, observed, K)));
        }

    private:
        cv::Point3f point_;
        cv::Point2f observed_;
        // Camera intrinsics
        double fx_, fy_, cx_, cy_;
    };
}

#endif //MYSLAM_CERES_BA_H
