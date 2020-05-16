add_library(utils STATIC registry.cpp tovec.cpp timer.cpp bisect.cpp metric_logger.cpp checkpoint.cpp camera.cpp depth_utils.cpp image_utils.cpp misc.cpp)
target_include_directories(utils 
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../../../include/monodepth/utils/)
target_link_libraries(utils modeling solver)