cmake_minimum_required(VERSION 3.10)

# Find OpenMP package
find_package(OpenMP REQUIRED)

add_library(studentlib main.cpp)
target_include_directories(studentlib PUBLIC ${CMAKE_SOURCE_DIR}/include)

# CPU-specific optimizations
target_compile_options(studentlib PRIVATE
    -O3 
    -march=native
    -mtune=native
    -flto
    -ffast-math
    -funroll-loops
    -fprefetch-loop-arrays
    -ftree-vectorize
    -mavx512f
    -mavx512cd
    -mavx512bw
    -mavx512dq
    -mavx512vl
)

# Additional compiler flags
target_compile_definitions(studentlib PRIVATE 
    -DNDEBUG
)

# Link time optimization
set_target_properties(studentlib PROPERTIES INTERPROCEDURAL_OPTIMIZATION TRUE)

# Add OpenMP flags
target_link_libraries(studentlib PRIVATE OpenMP::OpenMP_CXX)

# Optional NUMA support if available
find_package(PkgConfig)
pkg_check_modules(NUMA QUIET numa)
if(NUMA_FOUND)
    target_link_libraries(studentlib PRIVATE ${NUMA_LIBRARIES})
    target_include_directories(studentlib PRIVATE ${NUMA_INCLUDE_DIRS})
    target_compile_definitions(studentlib PRIVATE -DUSE_NUMA)
endif()