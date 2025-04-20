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
)

# Enable OpenMP with aggressive optimization
find_package(OpenMP REQUIRED)
if(OpenMP_FOUND)
    target_compile_options(studentlib PRIVATE ${OpenMP_CXX_FLAGS})
    target_link_libraries(studentlib PRIVATE ${OpenMP_CXX_LIBRARIES})
endif()

# Enable interprocedural (link-time) optimization
include(CheckIPOSupported)
check_ipo_supported(RESULT ipo_supported OUTPUT error)
if(ipo_supported)
    set_property(TARGET studentlib PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
else()
    message(STATUS "IPO/LTO not supported: ${error}")
endif()