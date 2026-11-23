# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
cmake_minimum_required(VERSION 3.16.0)
project(device_preprocess)

include(${ASCENDC_KERNEL_CMAKE_DIR}/bisheng_config.cmake)
include(${ASCENDC_KERNEL_CMAKE_DIR}/bisheng_intf.cmake)
include(${DST_DIR}/template_config.cmake)

if(NOT "${TARGET_TYPE}" STREQUAL "OBJECT")
    set(DEFINITION_OPTION --generate-definition)
else()
    set(DEFINITION_OPTION --generate-ktype-section)
endif()

# Generate device-side preprocessed files, and generate host_stub.cpp and header files based on the preprocessed files
add_library(preprocess_obj
    OBJECT ${SOURCES}
)

# Add the -E option based on normal compilation options to generate preprocessed files
target_compile_options(preprocess_obj PRIVATE
    -E -includestdio.h
)

target_compile_definitions(preprocess_obj PRIVATE
    __CHECK_FEATURE_AT_PRECOMPILE
)


if(${BUILD_MODE} STREQUAL "c310" OR ${BUILD_MODE} STREQUAL "c220")
    target_link_libraries(preprocess_obj PRIVATE
        $<BUILD_INTERFACE:${BUILD_MODE}_aiv_intf_pub>
        $<BUILD_INTERFACE:intf_device>
    )
else()
    target_link_libraries(preprocess_obj PRIVATE
        $<BUILD_INTERFACE:${BUILD_MODE}_intf_pub>
        $<BUILD_INTERFACE:intf_device>
    )
endif()

string(REPLACE "," " " preprocess_objs "$<TARGET_OBJECTS:preprocess_obj>")

if(DYNAMIC_MODE)
    add_library(aiv_obj
        OBJECT ${SOURCES}
    )

    # do not modify link order, custom options should be at the end
    target_link_libraries(aiv_obj PRIVATE
        $<BUILD_INTERFACE:${BUILD_MODE}_aiv_intf_pub>
        $<BUILD_INTERFACE:intf_device>
    )

    string(REPLACE "," " " aiv_objs "$<TARGET_OBJECTS:aiv_obj>")

    add_custom_target(merge_aiv_obj_text ALL
        COMMAND bash merge_obj_text.sh -l ${CMAKE_LINKER} ${aiv_objs}
        WORKING_DIRECTORY ${ASCENDC_KERNEL_CMAKE_DIR}/util
        DEPENDS aiv_obj
        COMMAND_EXPAND_LISTS
    )

    add_library(aic_obj
        OBJECT ${SOURCES}
    )

    # do not modify link order, custom options should be at the end
    target_link_libraries(aic_obj PRIVATE
        $<BUILD_INTERFACE:${BUILD_MODE}_aic_intf_pub>
        $<BUILD_INTERFACE:intf_device>
    )

    string(REPLACE "," " " aic_objs "$<TARGET_OBJECTS:aic_obj>")

    add_custom_target(merge_aic_obj_text ALL
        COMMAND bash merge_obj_text.sh -l ${CMAKE_LINKER} ${aic_objs}
        WORKING_DIRECTORY ${ASCENDC_KERNEL_CMAKE_DIR}/util
        DEPENDS aic_obj
        COMMAND_EXPAND_LISTS
    )

    add_custom_target(${device_target}_host_cpp ALL
        COMMAND ${ASCEND_PYTHON_EXECUTABLE} extract_host_stub.py ${preprocess_objs} --dynamic-mode
                    -d ${DST_DIR} -hd ${INCLUDE_DIR} --aiv-o ${aiv_objs} --aic-o ${aic_objs}
                    --build-mode ${BUILD_MODE} --compile-commands ${CMAKE_BINARY_DIR}/compile_commands.json ${DEFINITION_OPTION}
                    --run-mode ${RUN_MODE}
        WORKING_DIRECTORY ${ASCENDC_KERNEL_CMAKE_DIR}/util
        DEPENDS preprocess_obj aiv_obj aic_obj merge_aiv_obj_text merge_aic_obj_text
        COMMAND_EXPAND_LISTS
    )
else()
    add_library(m200_obj
        OBJECT ${SOURCES}
    )

    # do not modify link order, custom options should be at the end
    target_link_libraries(m200_obj PRIVATE
        $<BUILD_INTERFACE:m200_intf_pub>
        $<BUILD_INTERFACE:intf_device>
    )

    string(REPLACE "," " " m200_obj "$<TARGET_OBJECTS:m200_obj>")

    add_custom_target(merge_m200_obj_text ALL
        COMMAND bash merge_obj_text.sh -l ${CMAKE_LINKER} ${m200_obj}
        WORKING_DIRECTORY ${ASCENDC_KERNEL_CMAKE_DIR}/util
        DEPENDS m200_obj
        COMMAND_EXPAND_LISTS
    )

    add_custom_target(${device_target}_host_cpp ALL
    COMMAND ${ASCEND_PYTHON_EXECUTABLE} extract_host_stub.py ${preprocess_objs} -d ${DST_DIR}
                -hd ${INCLUDE_DIR} --aiv-o ${m200_obj}  --aic-o ${m200_obj} --build-mode ${BUILD_MODE} --compile-commands
                ${CMAKE_BINARY_DIR}/compile_commands.json ${DEFINITION_OPTION}
                --run-mode ${RUN_MODE}
    WORKING_DIRECTORY ${ASCENDC_KERNEL_CMAKE_DIR}/util
    DEPENDS preprocess_obj m200_obj merge_m200_obj_text
    COMMAND_EXPAND_LISTS
    )
endif()

# Preprocessing uses default optimization options, Exclude user-defined optimization options
set(OPTIMIZATION_OPTIONS -O0 -O1 -O2 -O3 -Os --cce-ignore-always-inline=true)
set(COMPILE_OPTIONS)
foreach (_option ${OPTIONS})
    if(NOT "${_option}" IN_LIST OPTIMIZATION_OPTIONS)
        list(APPEND COMPILE_OPTIONS ${_option})
    endif()
endforeach()

# Public compile rules of device
add_library(intf_device INTERFACE)

target_compile_options(intf_device INTERFACE
    ${COMPILE_OPTIONS}
)

target_include_directories(intf_device INTERFACE
    ${INCLUDES}
)

target_compile_definitions(intf_device INTERFACE
    ${DEFINITIONS}
)
