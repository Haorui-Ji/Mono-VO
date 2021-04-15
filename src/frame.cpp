
#include "myslam/frame.h"
#include "myslam/feature_matching.h"
#include "myslam/camera.h"
#include "myslam/utils.h"

namespace myslam
{

int Frame::factory_id_ = 0;

Frame::Ptr Frame::createFrame(cv::Mat rgb_img, cv::Mat depth_img, Camera::Ptr camera, double time_stamp)
{
    Frame::Ptr frame(new Frame());
    frame->rgb_img_ = rgb_img;
    frame->depth_img_ = depth_img;
    frame->id_ = factory_id_++;
    frame->time_stamp_ = time_stamp;
    frame->camera_ = camera;
    return frame;
}

void Frame::calcKeypointsDescriptors()
{
    extractFeatures(rgb_img_, keypoints_, descriptors_);
}

bool Frame::isInFrame(const cv::Point3f &p_world)
{
    cv::Point3f p_cam = transCoord(p_world, T_c_w_);
    if (p_cam.z < 0)
        return false;

    cv::Point2f pixel = camera2pixel(p_cam, camera_->K_);
    return pixel.x > 0 && pixel.y > 0 && pixel.x < rgb_img_.cols && pixel.y < rgb_img_.rows;
}

bool Frame::isInFrame(const cv::Mat &p_world)
{
    return isInFrame(Mat3x1_to_Point3f(p_world));
}

bool Frame::isMappoint(int idx)
{
    bool not_find = inliers_to_mappt_connections_.find(idx) == inliers_to_mappt_connections_.end();
    return !not_find;
}

cv::Mat Frame::getCamCenter()
{
    return getPosFromT(T_c_w_.inv());
}

}

