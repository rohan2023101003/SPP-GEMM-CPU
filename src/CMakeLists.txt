add_library(studentlib main.cpp)
target_include_directories(studentlib PUBLIC ${CMAKE_SOURCE_DIR}/include)

# Enable optimization flags balanced for performance and correctness
target_compile_options(studentlib PRIVATE 
    -O3 
    -march=native 
    -mavx512f 
    -mavx512dq 
    -mavx512bw 
    -mavx512vl 
    -mfma 
    -funroll-loops
    -ftree-vectorize
)

# Find and link with OpenMP
find_package(OpenMP REQUIRED)
if (OPENMP_FOUND)
    target_compile_options(studentlib PRIVATE ${OpenMP_CXX_FLAGS})
    target_link_libraries(studentlib PRIVATE ${OpenMP_CXX_LIBRARIES})
endif()