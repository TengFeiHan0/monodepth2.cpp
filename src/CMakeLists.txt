set(target depth)


add_subdirectory(monodepth/config)
add_subdirectory(monodepth/data)
add_subdirectory(monodepth/layers)
add_subdirectory(monodepth/modeling)
add_subdirectory(monodepth/utils)
add_subdirectory(monodepth/solver)
add_subdirectory(monodepth/engine)

# add_subdirectory(tools)

add_library(${target} INTERFACE)
target_link_libraries(${target} INTERFACE modeling config layers utils data solver engine)