//
// Created by jihaorui on 4/8/21.
//

#include "myslam/visual_odometry.h"
#include <boost/format.hpp> // for setting image filename

namespace myslam
{

void writePoseToFile(const string filename, vector<cv::Mat> list_T)
{
    std::ofstream fout;
    fout.open(filename);
    if (!fout.is_open())
    {
        cout << "my WARNING: failed to store camera trajectory to the wrong file name of:" << endl;
        cout << "    " << filename << endl;
        return;
    }
    for (auto T : list_T)
    {
        fout << T.at<double>(0, 0) << " "
            << T.at<double>(0, 1) << " "
            << T.at<double>(0, 2) << " "
            << T.at<double>(0, 3) << " "
            << T.at<double>(1, 0) << " "
            << T.at<double>(1, 1) << " "
            << T.at<double>(1, 2) << " "
            << T.at<double>(1, 3) << " "
            << T.at<double>(2, 0) << " "
            << T.at<double>(2, 1) << " "
            << T.at<double>(2, 2) << " "
            << T.at<double>(2, 3) << "\n";
    }
    fout.close();
}

}
