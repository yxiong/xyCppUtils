################################################################
# CMake file for xyCppUtils.
#
# Author: Ying Xiong.
# Created: Dec 01, 2014.
################################################################

cmake_minimum_required (VERSION 3.0)
project (xyCppUtils CXX)

# Find external libraries.
set (CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
find_package (Boost REQUIRED COMPONENTS serialization)
find_package (Eigen3 REQUIRED)
find_package (JPEG REQUIRED)
find_package (PNG REQUIRED)

# Include directories for external libraries.
include_directories (${Boost_INCLUDE_DIRS})
include_directories (${EIGEN3_INCLUDE_DIR})
include_directories (${JPEG_INCLUDE_DIR})
include_directories (${PNG_INCLUDE_DIRS})

# Configure file.
configure_file (
  "${PROJECT_SOURCE_DIR}/xyUtilsConfig.h.in"
  "${PROJECT_SOURCE_DIR}/xyUtilsConfig.h")

# Compile the target library.
set (CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS}")

set (XY_SOURCES Camera.cc CommandLineFlags.cc EigenUtils.cc FileIO.cc
  LogAndCheck.cc NonlinearLeastSquares.cc NumericalCheck.cc PlyIO.cc
  StringConvert.cc StringUtils.cc)

add_library (xyutils ${XY_SOURCES})

################################################################
# Compile and run tests.
# Currently we have three kinds of tests:
#
#   XY_GENERAL_TESTS: a general test that will pass.
#     Run as "./GeneralTest", it will succeed (return 0) and print a "Passed."
#     message on command line.
#
#   XY_FAILURE_TESTS: a test that is intended to fail.
#     Run as "./FilaureTest", it will fail (return non-zero) and print a "Fail."
#       message on command line.
#
#   XY_DATA_TESTS: a test that uses some data.
#     Run as "./DataTest --test_data_dir=/path/to/TestData", it will succeed
#       (return 0) and print a "Passed." message on command line.
################################################################
enable_testing ()

# Define tests.
set (XY_GENERAL_TESTS CommandLineFlagsTest NonlinearLeastSquaresTest
  NumericalCheckTest QuaternionTest StringUtilsTest StringConvertTest TimerTest)
set (XY_FAILURE_TESTS LogAndCheckTest)
set (XY_DATA_TESTS EigenUtilsTest FileIOTest ImageTest ImageUtilsTest PlyIOTest)

# Compile and link tests.
foreach (test ${XY_GENERAL_TESTS} ${XY_FAILURE_TESTS} ${XY_DATA_TESTS})
  add_executable (${test} ${test}.cc)
  target_link_libraries(${test} xyutils)
  target_link_libraries(${test} ${Boost_LIBRARIES})
  target_link_libraries(${test} ${JPEG_LIBRARIES})
  target_link_libraries(${test} ${PNG_LIBRARIES})
endforeach ()

# Run tests.
foreach (test ${XY_GENERAL_TESTS} ${XY_FAILURE_TESTS})
  add_test (${test} ./${test})
endforeach ()

foreach (test ${XY_DATA_TESTS})
  add_test (${test} ./${test} --test_data_dir=${CMAKE_SOURCE_DIR}/TestData)
endforeach ()

# Check output.
set_tests_properties (${XY_GENERAL_TESTS} ${XY_DATA_TESTS}
  PROPERTIES PASS_REGULAR_EXPRESSION "Passed.")
set_tests_properties (${XY_FAILURE_TESTS}
  PROPERTIES WILL_FAIL true FAIL_REGULAR_EXPRESSION "Fail")
