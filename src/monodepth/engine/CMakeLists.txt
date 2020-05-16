add_library(engine STATIC trainer.cpp parallel.cpp)
target_include_directories(engine 
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../../../include/monodepth/engine/)
target_link_libraries(engine modeling utils data config solver)