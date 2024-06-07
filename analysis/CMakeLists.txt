cmake_minimum_required(VERSION 3.10.2)
project(FunctionInfo)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(LLVM_VERSION
    "11"
    CACHE STRING "LLVM Version")

# Obtain the build flags from `llvm-config --cxxflags` and forward them to
# variable CMAKE_CXX_FLAGS.
execute_process(
  # COMMAND llvm-config-${LLVM_VERSION} --cxxflags
  COMMAND llvm-config --cxxflags
  OUTPUT_VARIABLE LLVM_CXXFLAGS
  OUTPUT_STRIP_TRAILING_WHITESPACE)
set(CMAKE_CXX_FLAGS "${LLVM_CXXFLAGS}")

if(NOT CMAKE_BUILD_TYPE OR CMAKE_BUILD_TYPE STREQUAL "Debug")
  set(CMAKE_CXX_CLANG_TIDY
      "clang-tidy;--extra-arg-before=-stdlib=libc++")
  message(STATUS "Enabling clang-tidy")

  execute_process(
    COMMAND llvm-config --includedir
    OUTPUT_VARIABLE LLVM_INCLUDEDIR
    OUTPUT_STRIP_TRAILING_WHITESPACE)
    set(CMAKE_CXX_FLAGS
        "${CMAKE_CXX_FLAGS} -isystem ${LLVM_INCLUDEDIR} -Wall -Wextra -Werror")
endif()

add_subdirectory(lib)
