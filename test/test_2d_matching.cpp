#include "myslam/common_include.h"
#include "myslam/feature_matching.h"
#include "myslam/config.h"

using namespace myslam;

int main(int argc, char** argv) {

    if (argc < 1) {
        cout << "Usage: config_file " << endl;
        return 1;
    }

    Config::setParameterFile(argv[1]);

    string root = "./test/2d_matching/";
    string imgName1 = root + "000000.png";
    string imgName2 = root + "000001.png";

    cv::Mat img1 = cv::imread(imgName1);
    cv::Mat img2 = cv::imread(imgName2);

    cout << img1.size << endl;
    cout << img2.size << endl;

    vector<cv::KeyPoint> keypoints1, keypoints2;
    cv::Mat descriptors1, descriptors2;
    vector<cv::DMatch> matches;

    extractFeatures(img1, keypoints1, descriptors1);
    extractFeatures(img2, keypoints2, descriptors2);

    cout << keypoints1.size() << '\t' << keypoints1.size() << endl;
    cout << descriptors1.size << '\t' << descriptors2.size << endl;

    matchFeatures(descriptors1, descriptors2, matches);
    cout << matches.size() << endl;

    Mat img_match;
    cv::drawMatches (img1, keypoints1, img2, keypoints2, matches, img_match );
    cv::imshow ( "ref", img1 );
    cv::imshow ( "cur", img2 );
    imshow ( "matches", img_match );
    cv::waitKey(0);
    return 0;
}

