//
// Created by jihaorui on 2/12/21.
//

#ifndef MYSLAM_CONFIG_H
#define MYSLAM_CONFIG_H

#include "myslam/common_include.h"

namespace myslam
{

class Config
{
private:
    Config () {} // private constructor makes a singleton

    static cv::FileNode get_(const std::string &key);

protected:
    static std::shared_ptr<Config> config_;

    // Used for reading .yaml file
    cv::FileStorage file_;

public:
    ~Config();  // close the file when deconstructing

    // set a new config file
    static void setParameterFile( const string& filename );

    // access the parameter values
    template< typename T >
    static T get( const string& key )
    {
        return T( Config::config_->file_[key] );
    }

    static bool getBool(const std::string &key);
};

}

#endif // MYSLAM_CONFIG_H
