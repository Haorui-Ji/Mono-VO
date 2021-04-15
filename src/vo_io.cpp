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
        for (int i = 0; i < 3; i++) // order: 1st row, 2nd row, 3rd row
        {
            for (int j = 0; j < 4; j++)
            {
                fout << T.at<double>(i, j) << " ";
            }
        }
        fout << '\n';
    }
    fout.close();
}

}
