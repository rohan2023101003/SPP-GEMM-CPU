cmake_minimum_required(VERSION 3.10)
set(CMAKE_CXX_STANDARD 17)

add_library(studentlib main.cpp)
target_include_directories(studentlib PUBLIC ${CMAKE_SOURCE_DIR}/include)

# Optimize for the specific CPU architecture
target_compile_options(studentlib PRIVATE
    -O3
    -march=native
    -mtune=native
    -mavx2
    -mfma
    -mbmi
    -mbmi2
    -mpopcnt
    -funroll-loops
    -ftree-vectorize
    -ffast-math
    -ftree-parallelize-loops=64
    -fno-signed-zeros
)

# Enable OpenMP with aggressive optimization
find_package(OpenMP REQUIRED)
if(OpenMP_FOUND)
    target_compile_options(studentlib PRIVATE ${OpenMP_CXX_FLAGS})
    target_link_libraries(studentlib PRIVATE ${OpenMP_CXX_LIBRARIES})
    
    # Set OpenMP to use all available CPU cores
    add_definitions(-DOMP_NUM_THREADS=64)
endif()

# Enable interprocedural (link-time) optimization
include(CheckIPOSupported)
check_ipo_supported(RESULT ipo_supported)
if(ipo_supported)
    set_property(TARGET studentlib PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
endif()

# Allocator optimization
add_definitions(-DUSE_JEMALLOC)
find_library(JEMALLOC_LIBRARY jemalloc)
if(JEMALLOC_LIBRARY)
    target_link_libraries(studentlib PRIVATE ${JEMALLOC_LIBRARY})
endif()