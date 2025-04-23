add_library(studentlib main.cpp)
target_include_directories(studentlib PUBLIC ${CMAKE_SOURCE_DIR}/include)

# Fully optimized flags for AVX-512 + OpenMP
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
    -fomit-frame-pointer
    -fopenmp
)

# Link OpenMP properly
find_package(OpenMP REQUIRED)
if (OPENMP_FOUND)
    target_compile_options(studentlib PRIVATE ${OpenMP_CXX_FLAGS})
    target_link_libraries(studentlib PRIVATE ${OpenMP_CXX_LIBRARIES})
endif()
