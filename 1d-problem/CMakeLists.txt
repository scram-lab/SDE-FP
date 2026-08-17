cmake_minimum_required(VERSION 3.23)
project(neutron_transport LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(HDF5 REQUIRED COMPONENTS C HL)

if(APPLE)
  execute_process(COMMAND brew --prefix libomp OUTPUT_VARIABLE LIBOMP_PREFIX OUTPUT_STRIP_TRAILING_WHITESPACE)
  set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I${LIBOMP_PREFIX}/include")
  set(OpenMP_CXX_LIB_NAMES "omp")
  set(OpenMP_omp_LIBRARY "${LIBOMP_PREFIX}/lib/libomp.dylib")
endif()
find_package(OpenMP REQUIRED)

# optimizations
set(CMAKE_BUILD_TYPE "Release")
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-O3 -march=native)
endif()

add_executable(solver solver.cpp)
target_include_directories(solver PRIVATE ${HDF5_INCLUDE_DIRS})
target_link_libraries(solver PRIVATE ${HDF5_LIBRARIES} ${HDF5_HL_LIBRARIES} OpenMP::OpenMP_CXX)