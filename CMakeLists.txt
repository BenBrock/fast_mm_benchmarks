cmake_minimum_required(VERSION 3.20)
project(fast_mm_benchmarks)

set(CMAKE_CXX_FLAGS "-O3 -march=native")

set(CMAKE_CXX_STANDARD 23)

include(FetchContent)
FetchContent_Declare(
        fast_matrix_market
        GIT_REPOSITORY https://github.com/alugowski/fast_matrix_market
        GIT_TAG main
        GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(fast_matrix_market)

FetchContent_Declare(
  fmt
  GIT_REPOSITORY https://github.com/fmtlib/fmt.git
  GIT_TAG 10.1.1)
FetchContent_MakeAvailable(fmt)

FetchContent_Declare(
        binsparse-rc
        GIT_REPOSITORY https://github.com/GraphBLAS/binsparse-reference-c.git
        GIT_TAG main
        GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(binsparse-rc)
  
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
  add_subdirectory(examples)
endif()
