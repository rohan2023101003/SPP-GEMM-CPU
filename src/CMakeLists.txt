cmake_minimum_required(VERSION 3.10)
set(CMAKE_CXX_STANDARD 17)

add_library(studentlib main.cpp)
target_include_directories(studentlib PUBLIC ${CMAKE_SOURCE_DIR}/include)

# Add optimization flags
target_compile_options(studentlib PRIVATE
    -O3
    -march=native
    -mavx2
    -mfma
    -funroll-loops
)

# Enable OpenMP
find_package(OpenMP REQUIRED)
if(OpenMP_FOUND)
    target_compile_options(studentlib PRIVATE ${OpenMP_CXX_FLAGS})
    target_link_libraries(studentlib PRIVATE ${OpenMP_CXX_LIBRARIES})
endif()