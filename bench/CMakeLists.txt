# Copyright(c) 2019 Gavin Dunlap
# Distributed under the MIT License (http://opensource.org/licenses/MIT)

cmake_minimum_required(VERSION 3.2)
project(siech_vec3_bench CXX)

#find_package(benchmark CONFIG REQUIRED)
add_executable(bench bench.cpp)
target_link_libraries(bench INTERFACE siech-vec3)