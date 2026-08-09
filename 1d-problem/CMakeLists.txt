cmake_minimum_required(VERSION 3.23)
project(neutron_transport LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(HDF5 REQUIRED COMPONENTS C HL)

add_executable(solver solver.cpp)
target_include_directories(solver PRIVATE ${HDF5_INCLUDE_DIRS})
target_link_libraries(solver PRIVATE ${HDF5_LIBRARIES} ${HDF5_HL_LIBRARIES})