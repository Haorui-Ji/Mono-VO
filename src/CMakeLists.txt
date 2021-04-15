add_library( myslam_lib SHARED
        utils.cpp
        config.cpp
        kitti_config.cpp
        camera.cpp
        mappoint.cpp
        map.cpp
        frame.cpp
        feature_matching.cpp
        epipolar_geometry.cpp
        vo.cpp
        vo_addFrame.cpp
        vo_initialization.cpp
        vo_tracking.cpp vo_io.cpp)

target_link_libraries( myslam_lib
        ${THIRD_PARTY_LIBS})