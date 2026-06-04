cmake_minimum_required(VERSION 3.25)
project(PunyLM VERSION 0.1.0 LANGUAGES CXX)

# Options
option(PUNYLM_ENABLE_CUDA "Build the optional CUDA backend" OFF)
option(PUNYLM_BUILD_TESTS "Build unit tests" ON)
option(PUNYLM_ENABLE_ASAN "Enable Address/UB sanitisers (CPU)" OFF)
option(PUNYLM_ENABLE_COVERAGE "Enable coverage instrumentation" OFF)
option(PUNYLM_WERROR "Treat warnings as errors" OFF)

# Global settings
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()
set_property(GLOBAL PROPERTY USE_FOLDERS ON)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(CompilerWarnings)

# Dependencies
include(FetchContent)

# Eigen
set(EIGEN_BUILD_DOC OFF CACHE BOOL "" FORCE)
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
FetchContent_Declare(eigen
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG 3.4.0
    GIT_SHALLOW TRUE)
FetchContent_MakeAvailable(eigen)

# Catch2
if(PUNYLM_BUILD_TESTS)
    FetchContent_Declare(catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG v3.6.0
        GIT_SHALLOW TRUE)
    FetchContent_MakeAvailable(catch2)
    list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
endif()

# Library
add_library(punylm INTERFACE)
add_library(punylm::punylm ALIAS punylm)
target_include_directories(punylm INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
target_compile_features(punylm INTERFACE cxx_std_20)
target_link_libraries(punylm INTERFACE Eigen3::Eigen)

# CLI executable
add_executable(punylm_cli src/main.cpp)
set_target_properties(punylm_cli PROPERTIES OUTPUT_NAME punylm)
target_link_libraries(punylm_cli PRIVATE punylm::punylm)
punylm_set_warnings(punylm_cli)

# Tests
if(PUNYLM_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
