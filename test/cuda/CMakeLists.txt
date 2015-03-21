################################################################
# Compile and run tests.
#
# Author: Ying Xiong.
# Created: Mar 20, 2015.
################################################################

# MACRO: xy_add_cuda_test (test_name [depend1 depend2 ...])
#
# Check `USE_depend1`, `USE_depend2`, ..., and if all passed, add
# `cuda::test_name` target.
macro (xy_add_cuda_test test_name)
  # Check dependency.
  xy_check_depend (xy_add_cuda_test_depend_ok ${ARGN})
  if (xy_add_cuda_test_depend_ok)
    # Add executalbe, need to remove -std=c++11 from CMAKE_CXX_FLAGS.
    set (CMAKE_CXX_FLAGS_BACKUP_ ${CMAKE_CXX_FLAGS})
    string (REPLACE "-std=c++11" "" CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})
    cuda_add_executable (Cuda${test_name} ${test_name}.cu)
    set (CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS_BACKUP_})
    # Link to libraries and add test.
    target_link_libraries (Cuda${test_name} xyCppUtils)
    xy_link_libraries (Cuda${test_name})
    add_test (NAME Cuda${test_name} COMMAND ./Cuda${test_name})
    set_tests_properties (Cuda${test_name}
      PROPERTIES PASS_REGULAR_EXPRESSION "Passed.")
  endif (xy_add_cuda_test_depend_ok)
endmacro ()


# Define tests.
xy_add_cuda_test (RangeTest)
xy_add_cuda_test (MultiDimArrayTest)
