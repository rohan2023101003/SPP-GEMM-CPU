add_library(studentlib main.cpp)
target_include_directories(studentlib PUBLIC ${CMAKE_SOURCE_DIR}/include)

# Find NUMA library
find_library(NUMA_LIBRARY numa)
if(NUMA_LIBRARY)
    message(STATUS "Found NUMA library: ${NUMA_LIBRARY}")
    target_link_libraries(studentlib PRIVATE ${NUMA_LIBRARY})
    target_compile_definitions(studentlib PRIVATE HAVE_NUMA)
else()
    message(STATUS "NUMA library not found, continuing without NUMA support")
endif()

# More balanced optimization flags focused on numerical stability and performance
target_compile_options(studentlib PRIVATE 
    -O3
    -march=skylake-avx512
    -mtune=skylake-avx512
    -mavx512f
    -mavx512dq
    -mavx512bw
    -mavx512vl
    -mfma
    -mpopcnt
    -mbmi
    -mbmi2
    -funroll-loops
    -ftree-vectorize
    -finline-functions
    -fomit-frame-pointer
    -fopenmp-simd
    # Removed aggressive floating-point optimizations for better numerical stability
)

# Find and link with OpenMP
find_package(OpenMP REQUIRED)
if (OPENMP_FOUND)
    target_compile_options(studentlib PRIVATE ${OpenMP_CXX_FLAGS})
    target_link_libraries(studentlib PRIVATE ${OpenMP_CXX_LIBRARIES})
endif()