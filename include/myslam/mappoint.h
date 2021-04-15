#ifndef MYSLAM_MAPPOINT_H
#define MYSLAM_MAPPOINT_H

#include "myslam/common_include.h"
#include "myslam/frame.h"

namespace myslam
{
class MapPoint
{
public: // Basics Properties
    typedef std::shared_ptr<MapPoint> Ptr;

    static int factory_id_;
    int id_;
    cv::Point3f pos_;
    cv::Mat norm_;                      // Vector pointing from camera center to the point
    cv::Mat descriptor_;                // Descriptor for matching

public: // Properties for constructing local mapping
    bool good_;                         // TODO: determine wheter a good point
//    list<Frame*> observed_frames_;      // key-frames that can observe this point
    int matched_times_;                 // being an inliner in pose estimation
    int visible_times_;                 // being visible in current frame

public: // Functions
    MapPoint(const cv::Point3f &pos, const cv::Mat &descriptor, const cv::Mat &norm);
    void setPos(const cv::Point3f &pos);
};

} // namespace my_slam
#endif // MYSLAM_MAPPOINT_H
