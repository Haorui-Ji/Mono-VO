add_executable( test_config test_config.cpp)
target_link_libraries( test_config myslam_lib)

add_executable( test_2d_matching test_2d_matching.cpp)
target_link_libraries( test_2d_matching myslam_lib)

add_executable( test_epipolar_geometry test_epipolar_geometry.cpp)
target_link_libraries( test_epipolar_geometry myslam_lib)

add_executable( test_vo_initilzation test_vo_initialization.cpp)
target_link_libraries( test_vo_initilzation myslam_lib)