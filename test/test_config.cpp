#include "myslam/common_include.h"
#include "myslam/kitti_config.h"

using namespace myslam;

int main(int argc, char** argv) {

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
    kittiConfig::loadTimes(dataset);

    int num_frames = kittiConfig::times.size();
    for (int i = 0; i < num_frames; i++){
        cout << kittiConfig::times.at(i) << endl;
    }

    cout << num_frames << endl;
    cout << "Kitti Config Test Complete" << endl;
}

