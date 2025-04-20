add_library(studentlib main.cpp)
target_include_directories(studentlib PUBLIC ${CMAKE_SOURCE_DIR}/include)

# Enable aggressive optimization flags
target_compile_options(studentlib PRIVATE 
    -O3 
    -march=native 
    -mtune=native
    -mavx512f 
    -mavx512dq 
    -mavx512bw 
    -mavx512vl 
    -mfma 
    -ffast-math
    -funroll-loops
    -ftree-vectorize
    -fomit-frame-pointer
    -flto
)

# Find and link with OpenMP
find_package(OpenMP REQUIRED)
if (OPENMP_FOUND)
    target_compile_options(studentlib PRIVATE ${OpenMP_CXX_FLAGS})
    target_link_libraries(studentlib PRIVATE ${OpenMP_CXX_LIBRARIES})
endif()

# Set cache optimization flags
target_compile_definitions(studentlib PRIVATE 
    -DCACHE_LINE_SIZE=64
)

# For debugging and profiling (comment out for final submission)
# target_compile_options(studentlib PRIVATE -g -pg)