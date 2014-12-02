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
set (XY_GENERAL_TESTS CommandLineFlagsTest QuaternionTest StringUtilsTest
  StringConvertTest TimerTest)
set (XY_FAILURE_TESTS LogAndCheckTest)
set (XY_DATA_TESTS FileIOTest PlyIOTest)

# Compile and link tests.
foreach (test ${XY_GENERAL_TESTS} ${XY_FAILURE_TESTS} ${XY_DATA_TESTS})
  add_executable (${test} ${test}.cc)
  target_link_libraries(${test} xyutils)
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
