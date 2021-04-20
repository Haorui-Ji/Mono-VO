#include "myslam/kitti_config.h"
#include "myslam/visual_odometry.h"

#include <opencv2/core/eigen.hpp>

using namespace myslam;

int main(int argc, char** argv) {

    if (argc < 2) {
        cout << "Usage: config_file kittidatasetnumber. e.g. velo 00" << endl;
        return 1;
    }

    // Kitti config setup
    if (argc < 2) {
        cout << "Usage: config_file kittidatasetnumber. e.g. velo 00" << endl;
        return 1;
    }

    kittiConfig::setParameterFile(argv[1]);

    string dataset_dir = kittiConfig::get<string>("dataset_dir");
    cout << "dataset: " << dataset_dir << endl;

    string dataset = argv[2];
    kittiConfig::loadImage(dataset, 2, 0); // to set width and height
    kittiConfig::loadCalibration(dataset);
    cv::Mat cv_K;
    eigen2cv(kittiConfig::cam_intrinsic[2], cv_K);
    Camera::Ptr camera(new Camera(cv_K));
    cv::Mat K = camera->K_;
    cout << K << endl;

    // -- Setup for vo
    VisualOdometry::Ptr vo(new VisualOdometry);

    kittiConfig::loadTimes(dataset);
    int num_frames = kittiConfig::times.size();
    cout << num_frames << endl;

    // -- Main loop: Iterate through images
    vector<cv::Mat> cam_pose_history;
    for (int img_id = 0; img_id < num_frames; img_id++)
    {
        // Read image.
        cv::Mat rgb_img = kittiConfig::loadImage(dataset, 2, img_id);
        cv::Mat depth_img = kittiConfig::loadImage(dataset, 2, img_id);
        if (rgb_img.data == nullptr)
        {
            cout << "The image file " << img_id << " is empty. Finished." << endl;
            break;
        }

        // run vo
        Frame::Ptr frame = Frame::createFrame(rgb_img, depth_img, camera, kittiConfig::times[img_id]);
        vo->addFrame(frame); // This is the core of my VO !!!

        // Return
        // cout << "Finished an image" << endl;
        cam_pose_history.push_back(frame->T_c_w_.clone().inv());
    }

    // Save camera trajectory
    const string save_predicted_traj_to = kittiConfig::get<string>("save_predicted_traj_to");
    writePoseToFile(save_predicted_traj_to, cam_pose_history);
}