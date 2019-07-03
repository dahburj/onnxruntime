# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

add_definitions(-D_GLIBCXX_DEBUG)
add_executable(image_classifier "main.cc" "jpeg_handle.cc" "jpeg_handle.h" "jpeg_mem.cc" "jpeg_mem.h"
        image_loader.cc image_loader.h CachedInterpolation.h local_filesystem.h local_filesystem.cc single_consumer.h)
target_include_directories(image_classifier PRIVATE ${PROJECT_SOURCE_DIR}/include ${GLIB_INCLUDE_DIRS})
target_link_libraries(image_classifier PRIVATE onnxruntime ${GLIB_LIBRARIES} ${JPEG_LIBRARIES})

add_executable(resize_image "resize_image_cmd.cc" "jpeg_handle.cc" "jpeg_handle.h" "jpeg_mem.cc" "jpeg_mem.h"
        image_loader.cc image_loader.h CachedInterpolation.h local_filesystem.h local_filesystem.cc)
target_link_libraries(resize_image PRIVATE ${JPEG_LIBRARIES} ${PNG_LIBRARIES})



