#include "myslam/kitti_config.h"
#include "myslam/frame.h"
#include "myslam/feature_matching.h"
#include "myslam/epipolar_geometry.h"
#include "myslam/visual_odometry.h"
#include "myslam/utils.h"

#include <Eigen/Dense>
#include <opencv2/core/eigen.hpp>

using namespace myslam;

int main(int argc, char** argv) {

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

    // Read test images
    string root = "./test/2d_matching/";
    string imgName1 = root + "000000.png";
    string imgName2 = root + "000001.png";

    cv::Mat rgb_img1 = cv::imread(imgName1);
    cv::Mat rgb_img2 = cv::imread(imgName2);
    cv::Mat depth_img1 = cv::Mat(rgb_img1.rows, rgb_img1.cols, CV_32FC3);
    cv::Mat depth_img2 = cv::Mat(rgb_img2.rows, rgb_img2.cols, CV_32FC3);

    Frame::Ptr frame1 = Frame::createFrame(rgb_img1, depth_img1, camera);
    vo->addFrame(frame1); // This is the core of my VO !!!

    Frame::Ptr frame2 = Frame::createFrame(rgb_img2, depth_img2, camera);
    vo->addFrame(frame2); // This is the core of my VO !!!
//
//    // Extract features
//    vector<cv::KeyPoint> keypoints1, keypoints2;
//    cv::Mat descriptors1, descriptors2;
//    vector<cv::DMatch> matches;
//
//    extractFeatures(img1, keypoints1, descriptors1);
//    extractFeatures(img2, keypoints2, descriptors2);
//
//    matchFeatures(descriptors1, descriptors2, matches);
//    cout << matches.size() << endl;
//
//    vector<cv::Point2f> pts_img1, pts_img2; // matched points
//    extractPtsFromMatches(keypoints1, keypoints2, matches, pts_img1, pts_img2);
//
//    // Motion estimation via essential matrix
//    cv::Mat R_e, t_e, essential_matrix;
//    vector<int> inliers_index_e;
//    estiMotionByEssential(pts_img1, pts_img2, K,
//                          essential_matrix,
//                          R_e, t_e, inliers_index_e);
//
//    cout << R_e << endl;
//    cout << t_e << endl;



//    Mat img_match;
//    cv::drawMatches (img1, keypoints1, img2, keypoints2, matches, img_match );
//    cv::imshow ( "ref", img1 );
//    cv::imshow ( "cur", img2 );
//    imshow ( "matches", img_match );
//    cv::waitKey(0);
    return 0;
}




