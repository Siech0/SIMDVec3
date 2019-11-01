cmake_minimum_required(VERSION 3.1)

#############################################################################
# Start siech-vec3 project
#############################################################################
include(cmake/utils.cmake)
include(cmake/ide.cmake)
siech_vec3_extract_version()

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose Release or Debug" FORCE)
endif()

project(siech-vec3 VERSION ${SIECH_VEC3_VERSION} LANGUAGES CXX)
message(STATUS "Build siech-vec3: ${SIECH_VEC3_VERSION}")

##########################################################
# Compiler setup
##########################################################
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

##########################################################
# Set SIECH_VEC3_MASTER_PROJECT to on building Vec3
##########################################################
if(NOT DEFINED SIECH_VEC3_MASTER_PROJECT)
    if (CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
        set(SIECH_VEC3_MASTER_PROJECT ON)
        message(STATUS "siech_vec3 is master project.")
    else()
        set(SIECH_VEC3_MASTER_PROJECT OFF)
        message(STATUS "siech_vec3 is NOT master project.")
    endif()
endif()

# Example options
option(SIECH_VEC3_BUILD_EXAMPLE "Build example" ${SIECH_VEC3_MASTER_PROJECT})

# Test options
option(SIECH_VEC3_BUILD_TESTS "Build tests" ${SIECH_VEC3_MASTER_PROJECT})

# Bench options
option(SIECH_VEC3_BUILD_BENCH "Build benchmarks (Requires https://github.com/google/benchmark.git to be installed)" OFF)


##########################################################
# Add library to target
##########################################################
add_library(siech-vec3 INTERFACE)
target_include_directories(siech-vec3 INTERFACE
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include>"
        "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>")

##########################################################
# Build binaries
##########################################################
if(SIECH_VEC3_BUILD_EXAMPLE)
    message(STATUS "Generating examples")
    add_subdirectory(example)
endif()

if(SIECH_VEC3_BUILD_TESTS)
    message(STATUS "Generating tests")
    add_subdirectory(tests)
endif()

if(SIECH_VEC3_BUILD_BENCH)
    message(STATUS "Generating benchmarks")
    add_subdirectory(bench)
endif()