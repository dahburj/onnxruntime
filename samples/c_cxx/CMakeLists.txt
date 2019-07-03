# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

cmake_minimum_required(VERSION 3.13)

# Project
project(onnxruntime_samples C CXX)
if (WIN32)
 string(APPEND CMAKE_CXX_FLAGS " /W4")
else()
 string(APPEND CMAKE_CXX_FLAGS " -Wall -Wextra")
 string(APPEND CMAKE_C_FLAGS " -Wall -Wextra")
endif()

find_package(JPEG REQUIRED)
find_package(PNG REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_search_module(GLIB REQUIRED glib-2.0)

add_definitions(-DUSE_CUDA)

#TODO: remove this part(need to talk to Ryan and find a solution for this)
if(UNIX)
 include_directories("/usr/local/include/onnxruntime")
endif()


add_subdirectory(imagenet)
add_subdirectory(fns_candy_style_transfer)

