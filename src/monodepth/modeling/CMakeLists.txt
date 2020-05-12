file(GLOB_RECURSE sources "*.cpp")

add_library(modeling STATIC ${sources})

target_include_directories(modeling  
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../../../include/monodepth/modeling/ )
  
target_link_libraries(modeling 
  PUBLIC layers config PRIVATE utils)