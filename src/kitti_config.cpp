//
// Created by jihaorui on 2/9/21.
//

#include "myslam/kitti_config.h"

using namespace myslam;
using namespace Eigen;

vector<double> kittiConfig::times;

int kittiConfig::img_width, kittiConfig::img_height;

vector<Eigen::Matrix<double, 3, 4>,
        Eigen::aligned_allocator<Eigen::Matrix<double, 3, 4>>> kittiConfig::cam_mat;

vector<Eigen::Matrix3d,
        Eigen::aligned_allocator<Eigen::Matrix3d>> kittiConfig::cam_intrinsic;

vector<Eigen::Matrix3d,
        Eigen::aligned_allocator<Eigen::Matrix3d>> kittiConfig::cam_intrinsic_inv;

vector<Eigen::Vector3d,
        Eigen::aligned_allocator<Eigen::Vector3d>> kittiConfig::cam_trans;

vector<Eigen::Matrix4d,
        Eigen::aligned_allocator<Eigen::Vector3d>> kittiConfig::cam_pose;

Eigen::Matrix4d kittiConfig::velo_to_cam;
Eigen::Matrix4d kittiConfig::cam_to_velo;


vector<double> kittiConfig::min_x, kittiConfig::max_x, kittiConfig::min_y, kittiConfig::max_y;


void kittiConfig::loadCalibration(
        const string& dataset
) {
    string calib_path = config_->get<string>("dataset_dir") + dataset + "/calib.txt";
    ifstream calib_stream(calib_path);
    string P;
    velo_to_cam = Eigen::Matrix4d::Identity();
    for(int cam=0; cam<config_->get<int>("num_cams_actual"); cam++) {
        calib_stream >> P;
        cam_mat.push_back(Eigen::Matrix<double, 3, 4>());
        for(int i=0; i<3; i++) {
            for(int j=0; j<4; j++) {
                calib_stream >> cam_mat[cam](i,j);
            }
        }
        Matrix3d K = cam_mat[cam].block<3,3>(0,0);
        Matrix3d Kinv = K.inverse();
        Vector3d Kt = cam_mat[cam].block<3,1>(0,3);
        Vector3d t = Kinv * Kt;
        cam_trans.push_back(t);
        cam_intrinsic.push_back(K);
        cam_intrinsic_inv.push_back(K.inverse());

        cam_pose.push_back(Eigen::Matrix4d::Identity());
        cam_pose[cam](0, 3) = t(0);
        cam_pose[cam](1, 3) = t(1);
        cam_pose[cam](2, 3) = t(2);

        Vector3d min_pt;
        min_pt << 0, 0, 1;
        min_pt = Kinv * min_pt;
        min_x.push_back(min_pt(0) / min_pt(2));
        min_y.push_back(min_pt(1) / min_pt(2));
        //std::cerr << "min_pt: " << min_pt << std::endl;

        Vector3d max_pt;
        max_pt << img_width, img_height, 1;
        max_pt = Kinv * max_pt;
        max_x.push_back(max_pt(0) / max_pt(2));
        max_y.push_back(max_pt(1) / max_pt(2));
        //std::cerr << "max_pt: " << max_pt << std::endl;
    }

    calib_stream >> P;
    for(int i=0; i<3; i++) {
        for(int j=0; j<4; j++) {
            calib_stream >> velo_to_cam(i,j);
        }
    }

    cam_to_velo = velo_to_cam.inverse();
}

void kittiConfig::loadTimes(
        const string& dataset
) {
    string time_path = config_->get<string>("dataset_dir") + dataset + "/times.txt";
    ifstream time_stream(time_path);
    double t;
    while(time_stream >> t) {
        times.push_back(t);
    }
}

cv::Mat kittiConfig::loadImage(
        const string & dataset,
        const int cam,
        const int n
) {
    std::stringstream ss;
    ss << config_->get<string>("dataset_dir") << dataset << "/image_" << cam << "/"
       << std::setfill('0') << std::setw(6) << n << ".png";
    cv::Mat I = cv::imread(ss.str());
    img_width = I.cols;
    img_height = I.rows;
    return I;
}

cv::Mat kittiConfig::loadDepth(
        const string & dataset,
        const int n
) {
    std::stringstream ss;
    ss << config_->get<string>("dataset_dir") << dataset << "/depth/"
       << std::setfill('0') << std::setw(10) << n << ".png";
    cv::Mat I = cv::imread(ss.str(), 0);
    return I;
}

void kittiConfig::loadPoints(
        const pcl::PointCloud<pcl::PointXYZ>::Ptr& point_cloud,
        const string & dataset,
        int n
) {
    std::stringstream ss;
    ss << config_->get<string>("dataset_dir") << dataset << "/velodyne_points/"
       << std::setfill('0') << std::setw(6) << n << ".bin";

    // allocate 40 MB buffer (only ~1300*4*4 KB are needed)
    int32_t num = 10000000;
    double *data = (double*)malloc(num*sizeof(double));

    // pointers
    double *px = data+0;
    double *py = data+1;
    double *pz = data+2;
    double *pr = data+3;

    // load point cloud
    FILE *stream;
    stream = fopen (ss.str().c_str(),"rb");
    num = fread(data,sizeof(double),num,stream)/4;


    for (int32_t i=0; i<num; i++) {
        point_cloud->points.push_back(pcl::PointXYZ(*px,*py,*pz));
        px+=4; py+=4; pz+=4; pr+=4;
    }

    fclose(stream);
    free(data);
}
