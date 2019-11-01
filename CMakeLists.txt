cmake_minimum_required(VERSION 3.1)

set(CMAKE_CXX_STANDARD 11)

project(siech-vector3)

add_library(siech-vector3 INTERFACE)
target_include_directories(siech-vector3 INTERFACE include/)