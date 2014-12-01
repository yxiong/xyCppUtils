################################################################
# CMake file for xyCppUtils.
#
# Author: Ying Xiong.
# Created: Dec 01, 2014.
################################################################

cmake_minimum_required (VERSION 3.0)
project (xyCppUtils CXX)

# Compile the library.
set (CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS}")
set (XY_SOURCES CommandLineFlags.cc FileIO.cc LogAndCheck.cc PlyIO.cc
  StringConvert.cc StringUtils.cc)
add_library (xyutils ${XY_SOURCES})

# Compile and run tests.
enable_testing ()

set (XY_TESTS CommandLineFlagsTest QuaternionTest StringUtilsTest
  StringConvertTest TimerTest)

foreach (test ${XY_TESTS})
  add_executable (${test} ${test}.cc)
  target_link_libraries(${test} xyutils)
  add_test (${test} ./${test})
endforeach (test ${XY_TESTS})
