//
// Created by jihaorui on 2/25/21.
//

#include "myslam/utils.h"

namespace myslam
{

    // ---------------- datatype conversion ----------------

    cv::Mat point3f_to_mat3x1(const cv::Point3f &p)
    {
        return (cv::Mat_<double>(3, 1) << p.x, p.y, p.z);
    }
    cv::Mat point3f_to_mat4x1(const cv::Point3f &p)
    {
        return (cv::Mat_<double>(4, 1) << p.x, p.y, p.z, 1);
    }

    cv::Point3f Mat3x1_to_Point3f(const cv::Mat &p)
    {
        return cv::Point3f(p.at<double>(0, 0), p.at<double>(1, 0), p.at<double>(2, 0));
    }

    cv::Mat point2f_to_mat2x1(const cv::Point2f &p)
    {
        return (cv::Mat_<double>(2, 1) << p.x, p.y);
    }

    cv::Mat getZerosMat(int rows, int cols, int type)
    {
        return cv::Mat::zeros(cv::Size(rows, cols), type);
    }
    // -----------------------------------------------------

    // ---------------- Transformation --------------------------------
    cv::Mat convertRt2T_3x4(const cv::Mat &R, const cv::Mat &t)
    {
        cv::Mat T = (cv::Mat_<double>(3, 4) <<
                R.at<double>(0, 0), R.at<double>(0, 1), R.at<double>(0, 2), t.at<double>(0, 0),
                R.at<double>(1, 0), R.at<double>(1, 1), R.at<double>(1, 2), t.at<double>(1, 0),
                R.at<double>(2, 0), R.at<double>(2, 1), R.at<double>(2, 2), t.at<double>(2, 0));
        return T;
    }

    cv::Mat convertRt2T(const cv::Mat &R, const cv::Mat &t)
    {
        cv::Mat T = (cv::Mat_<double>(4, 4) <<
                R.at<double>(0, 0), R.at<double>(0, 1), R.at<double>(0, 2), t.at<double>(0, 0),
                R.at<double>(1, 0), R.at<double>(1, 1), R.at<double>(1, 2), t.at<double>(1, 0),
                R.at<double>(2, 0), R.at<double>(2, 1), R.at<double>(2, 2), t.at<double>(2, 0),
                0, 0, 0, 1);
        return T;
    }

    void getRtFromT(const cv::Mat &T, cv::Mat &R, cv::Mat &t)
    {
        R = (cv::Mat_<double>(3, 3) << T.at<double>(0, 0), T.at<double>(0, 1), T.at<double>(0, 2),
                T.at<double>(1, 0), T.at<double>(1, 1), T.at<double>(1, 2),
                T.at<double>(2, 0), T.at<double>(2, 1), T.at<double>(2, 2));
        t = (cv::Mat_<double>(3, 1) << T.at<double>(0, 3),
                T.at<double>(1, 3),
                T.at<double>(2, 3));
    }

    cv::Mat getPosFromT(const cv::Mat &T)
    {
        return T(cv::Rect(3, 0, 1, 3)).clone();
    }

    vector<cv::Point2f> convertkeypointsToPoint2f(const vector<cv::KeyPoint, std::allocator<cv::KeyPoint>> kpts)
    {
        vector<cv::Point2f> pts;
        for (auto &kpt : kpts)
            pts.push_back(kpt.pt);
        return pts;
    }

    cv::Point3f transCoord(const cv::Point3f &p, const cv::Mat &R, const cv::Mat &t)
    {
        cv::Mat p1 = (cv::Mat_<double>(3, 1) << p.x, p.y, p.z);
        cv::Mat p2 = R * p1 + t;
        return cv::Point3f(p2.at<double>(0, 0), p2.at<double>(1, 0), p2.at<double>(2, 0));
    }

    cv::Point3f transCoord(const cv::Point3f &p, const cv::Mat T4x4)
    {
        const cv::Mat &T = T4x4;
        cv::Mat p1 = (cv::Mat_<double>(4, 1) << p.x, p.y, p.z, 1);
        cv::Mat p2 = T4x4 * p1;
        return cv::Point3f(p2.at<double>(0, 0), p2.at<double>(1, 0), p2.at<double>(2, 0));
    }

    // ---------------------------------------------------------------------

    double scalePointPos(cv::Point3f &p, double scale)
    {
        p.x *= scale;
        p.y *= scale;
        p.z *= scale;
    }

    double calcDist(const cv::Point2f &p1, const cv::Point2f &p2)
    {
        double dx = p1.x - p2.x, dy = p1.y - p2.y;
        return sqrt(dx * dx + dy * dy);
    }

    double calcMeanDepth(const vector<cv::Point3f> &pts_3d)
    {
        double mean_depth = 0;
        for (const cv::Point3f &p : pts_3d)
            mean_depth += p.z;
        mean_depth /= pts_3d.size();
        return mean_depth;
    }

    double calcAngleBetweenTwoVectors(const cv::Mat &vec1, const cv::Mat &vec2)
    {
        // cos(angle) = vec1.dot(vec2) / (||vec1|| * ||vec2||)
        assert(vec1.rows == vec2.rows && vec1.rows > vec1.cols);
        int N = vec1.rows;
        double res = 0;
        for (int i = 0; i < N; i++)
        {
            res += vec1.at<double>(i, 0) * vec2.at<double>(i, 0);
        }
        double norm = cv::norm(vec1) * cv::norm(vec2);
        assert(norm != 0);
        return acos(res / norm);
    }

    cv::Mat getNormalizedMat(const cv::Mat& mat)
    {
        double len = cv::norm(mat);
        cv::Mat res = mat / len;
        return res;
    }

}

