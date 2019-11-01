# Copyright(c) 2019 Gavin Dunlap
# Distributed under the MIT License (http://opensource.org/licenses/MIT)

cmake_minimum_required(VERSION 3.2)
project(siech_vec3_examples CXX)

add_executable(example example.cpp)
siech_vec3_enable_warnings(example)
target_link_libraries(example INTERFACE siech-vec3)