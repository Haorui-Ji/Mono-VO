//
// Created by jihaorui on 2/12/21.
//

#include "myslam/config.h"

using namespace myslam;

void Config::setParameterFile( const string& filename )
{
    if ( config_ == nullptr )
        config_ = shared_ptr<Config>(new Config);
    config_->file_ = cv::FileStorage( filename, cv::FileStorage::READ );
    if (!config_->file_.isOpened())
    {
        std::cerr<<"parameter file "<<filename<<" does not exist."<<std::endl;
        config_->file_.release();
        return;
    }
}

Config::~Config()
{
    if ( file_.isOpened() )
        file_.release();
}

cv::FileNode Config::get_(const std::string &key)
{
    cv::FileNode content = Config::config_->file_[key];
    if (content.empty())
        throw std::runtime_error("Key " + key + " doesn't exist");
    return content;
}

bool Config::getBool(const std::string &key)
{
    std::string val = static_cast<std::string>(Config::get_(key));
    return val == "true" || val == "True";
}

shared_ptr<Config> Config::config_ = nullptr;

