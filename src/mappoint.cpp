//
// Created by jihaorui on 3/9/21.
//

#include "myslam/mappoint.h"

namespace myslam
{

int MapPoint::factory_id_ = 0;

MapPoint::MapPoint(
    const cv::Point3f &pos, const cv::Mat &descriptor, const cv::Mat &norm) :
    pos_(pos), descriptor_(descriptor), norm_(norm),
    good_(true), visible_times_(1), matched_times_(1)

{
    id_ = factory_id_++;
}

void MapPoint::setPos(const cv::Point3f &pos)
{
    pos_ = pos;
}

} // namespace my_slam

