# Copyright(c) 2019 Gavin Dunlap
# Distributed under the MIT License (http://opensource.org/licenses/MIT)

cmake_minimum_required(VERSION 3.2)
project(siech-vec3-tests CXX)

include(../cmake/utils.cmake)

set(SIECH_VEC3_UTESTS_SOURCES
    test.cpp)

enable_testing()

if(SIECH_VEC3_BUILD_TESTS)
    add_executable(siech-vec3-utests ${SIECH_VEC3_UTESTS_SOURCES})
    siech_vec3_enable_warnings(siech-vec3-utests)
    target_link_libraries(siech-vec3 INTERFACE siech-vec3)
    add_test(NAME siech-vec3-utests COMMAND siec-vec3-utests)
endif()
