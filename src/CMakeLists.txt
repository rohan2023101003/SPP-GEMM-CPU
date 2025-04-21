add_library(studentlib main.cpp)
target_include_directories(studentlib PUBLIC ${CMAKE_SOURCE_DIR}/include)

# More aggressive optimization flags tailored for the Xeon Gold 6226R
target_compile_options(studentlib PRIVATE 
    -O3 
    -march=skylake-avx512
    -mavx512f 
    -mavx512dq 
    -mavx512bw 
    -mavx512vl 
    -mfma
    -funroll-loops
    -ftree-vectorize
    -finline-functions
    -ffast-math
    -fomit-frame-pointer
    -fopenmp-simd
    -fexpensive-optimizations
    -fprefetch-loop-arrays
    -flto
)

# Find and link with OpenMP with optimal settings
find_package(OpenMP REQUIRED)
if (OPENMP_FOUND)
    target_compile_options(studentlib PRIVATE ${OpenMP_CXX_FLAGS})
    target_link_libraries(studentlib PRIVATE ${OpenMP_CXX_LIBRARIES})
    # Add link-time optimization
    target_link_options(studentlib PRIVATE -flto)
endif()