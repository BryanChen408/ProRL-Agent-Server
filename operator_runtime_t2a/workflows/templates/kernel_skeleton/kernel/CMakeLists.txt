cmake_minimum_required(VERSION 3.16.0)
project({op_name}_kernel)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(SOC_VERSION "$ENV{SOC_VERSION}" CACHE STRING "system on chip type")
set(ASCEND_CANN_PACKAGE_PATH "$ENV{ASCEND_HOME_PATH}" CACHE PATH "ASCEND CANN package installation directory")
set(RUN_MODE "npu" CACHE STRING "run mode: npu")
set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Build type Release/Debug (default Debug)" FORCE)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")

if(EXISTS ${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/ascendc_kernel_cmake)
    set(ASCENDC_CMAKE_DIR ${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/ascendc_kernel_cmake)
elseif(EXISTS ${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake)
    set(ASCENDC_CMAKE_DIR ${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake)
elseif(EXISTS ${ASCEND_CANN_PACKAGE_PATH}/ascendc_devkit/tikcpp/samples/cmake)
    set(ASCENDC_CMAKE_DIR ${ASCEND_CANN_PACKAGE_PATH}/ascendc_devkit/tikcpp/samples/cmake)
else()
    message(FATAL_ERROR "ascendc_kernel_cmake does not exist, please check whether the cann package is installed.")
endif()
include(${ASCENDC_CMAKE_DIR}/ascendc.cmake)

# device kernel -> 静态库
ascendc_library(kernels STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel/{op_name}_kernel.cpp
)
ascendc_include_directories(kernels PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/utils
    ${ASCEND_CANN_PACKAGE_PATH}/include
    ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/runtime
    ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/msprof
)

# host(op_host + register) -> {op_name}_ext.so  (扁平名,不要用 {op_name}._C)
add_library({op_name}_ext SHARED
    ${CMAKE_CURRENT_SOURCE_DIR}/op_host/{op_name}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/register.cpp
)
target_link_libraries({op_name}_ext PRIVATE
    kernels
    torch_npu
    m
    dl
)

execute_process(COMMAND python3 -c "import os; import torch; print(os.path.dirname(torch.__file__))"
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE TORCH_PATH
)
set(ENV{ASCEND_HOME_PATH} ${ASCEND_CANN_PACKAGE_PATH})
execute_process(COMMAND python3 -c "import os; import torch_npu; print(os.path.dirname(torch_npu.__file__))"
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE TORCH_NPU_PATH
)
target_link_directories({op_name}_ext PRIVATE
    ${TORCH_PATH}/lib
    ${TORCH_NPU_PATH}/lib
)
target_include_directories({op_name}_ext PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/utils
    ${TORCH_NPU_PATH}/include
    ${TORCH_PATH}/include
    ${TORCH_PATH}/include/torch/csrc/api/include
)
execute_process(COMMAND python3 -m pybind11 --includes
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE PYBIND11_INC
)
string(REPLACE " " ";" PYBIND11_INC ${PYBIND11_INC})
target_compile_options({op_name}_ext PRIVATE
    ${PYBIND11_INC}
    -D_GLIBCXX_USE_CXX11_ABI=1
)
execute_process(COMMAND python3 -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX') or '.so')"
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE PYTHON_EXTENSION_SUFFIX
)
set_target_properties({op_name}_ext PROPERTIES
    OUTPUT_NAME {op_name}_ext
    PREFIX ""
    SUFFIX "${PYTHON_EXTENSION_SUFFIX}"
)
