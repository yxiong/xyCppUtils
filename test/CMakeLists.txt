################################################################
# Compile and run tests.
#
# Author: Ying Xiong.
# Created: Dec 06, 2014.
################################################################

set (XY_TEST_DATA_DIR ${CMAKE_CURRENT_SOURCE_DIR}/data)

# MACRO: xy_add_test (test_name test_type [depend1 depend2 ...])
# Check `USE_depend1`, `USE_depend2`, ..., and if all passed, add `test_name`
# target according to `test_type`.
#
# Currently we have three `test_type`s:
#   `general`: a general test that should pass.
#     Run as "./GeneralTest", it will succeed (return 0) and print a "Passed."
#     message on command line.
#   `fail`: a test that is intended to fail.
#     Run as "./FilaureTest", it will fail (return non-zero) and print a "Fail."
#       message on command line.
#   `data`: a test that uses some data.
#     Run as "./DataTest --test_data_dir=/path/to/TestData", it will succeed
#       (return 0) and print a "Passed." message on command line.
macro (xy_add_test test_name test_type)
  # Check dependency.
  xy_check_depend (xy_add_test_depend_ok ${ARGN})
  if (xy_add_test_depend_ok)
    # Compile test binary.
    xy_add_executable (${test_name} ${ARGN})
    # Run test.
    if (${test_type} MATCHES general)
      add_test (NAME ${test_name} COMMAND ./${test_name})
      set_tests_properties (${test_name}
        PROPERTIES PASS_REGULAR_EXPRESSION "Passed.")
    elseif (${test_type} MATCHES fail)
      add_test (NAME ${test_name} COMMAND ./${test_name})
      set_tests_properties (${test_name}
        PROPERTIES WILL_FAIL true FAIL_REGULAR_EXPRESSION "Fail.")
    elseif (${test_type} MATCHES data)
      add_test (NAME ${test_name}
        COMMAND ./${test_name} --test_data_dir=${XY_TEST_DATA_DIR})
      set_tests_properties (${test_name}
        PROPERTIES PASS_REGULAR_EXPRESSION "Passed.")
    endif ()
  endif (xy_add_test_depend_ok)
endmacro ()

# MACRO: xy_add_test_script (script_name [depend1 depend2 ...])
#
# The test script will be run as:
#   test_source_dir/script_name project_binary_dir test_data_dir
# and a "Passed." will be checked for output.
macro (xy_add_test_script script_name)
  xy_check_depend (xy_add_test_script_depend_ok ${ARGN})
  if (xy_add_test_script_depend_ok)
    # Run test script.
    set (script_exe ${CMAKE_CURRENT_SOURCE_DIR}/${script_name})
    add_test (NAME ${script_name}
      COMMAND ${script_exe} ${PROJECT_BINARY_DIR} ${XY_TEST_DATA_DIR})
    set_tests_properties (${script_name}
      PROPERTIES PASS_REGULAR_EXPRESSION "Passed.")
  endif (xy_add_test_script_depend_ok)
endmacro ()

# Define tests.
xy_add_test (CommandLineFlagsTest general)
xy_add_test (EigenUtilsTest data EIGEN)
xy_add_test (FileIOTest data)
xy_add_test (ImageTest data)
xy_add_test (ImageUtilsTest data EIGEN)
xy_add_test (LogAndCheckTest fail)
xy_add_test (MultiDimArrayTest general)
xy_add_test (NonlinearLeastSquaresTest general EIGEN)
xy_add_test (NumericalCheckTest general EIGEN)
xy_add_test (PlyIOTest data)
xy_add_test (PointEdgeViewerTest data EIGEN OPENGL SDL)
xy_add_test (QuaternionTest general)
xy_add_test (RangeTest general)
xy_add_test (StringConvertTest general)
xy_add_test (StringUtilsTest general)
xy_add_test (TimerTest general)

# Add test scripts.
xy_add_test_script (CommandLineFlagsTest.sh)
xy_add_test_script (visualize_points_and_cameras_test.sh EIGEN OPENGL SDL)
