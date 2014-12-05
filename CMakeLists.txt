################################################################
# CMake file for xyCppUtils.
#
# Author: Ying Xiong.
# Created: Dec 01, 2014.
################################################################

cmake_minimum_required (VERSION 3.0)
project (xyCppUtils CXX)

################################################################
# Find and configure external libraries.
################################################################
set (CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
include (CMakeDependentOption)

# Find the library and set USE_XXX option, which allows user to disable a found
# library.
find_package (Boost COMPONENTS serialization)
CMAKE_DEPENDENT_OPTION(USE_BOOST "Use Boost." ON Boost_FOUND OFF)

find_package (Eigen3)
CMAKE_DEPENDENT_OPTION(USE_EIGEN "Use Eigen3." ON EIGEN3_FOUND OFF)

find_package (JPEG)
CMAKE_DEPENDENT_OPTION(USE_JPEG "Use JPEG." ON JPEG_FOUND OFF)

find_package (PNG)
CMAKE_DEPENDENT_OPTION(USE_PNG "Use PNG." ON PNG_FOUND OFF)

find_package (OpenGL)
CMAKE_DEPENDENT_OPTION(USE_OPENGL "Use OPENGL." ON OPENGL_FOUND OFF)

find_package (SDL)
CMAKE_DEPENDENT_OPTION(USE_SDL "Use SDL." ON SDL_FOUND OFF)

# Include directories for external libraries.
if (${USE_BOOST})
  include_directories (${Boost_INCLUDE_DIRS})
endif ()

if (${USE_EIGEN})
  include_directories (${EIGEN3_INCLUDE_DIR})
endif ()

if (${USE_JPEG})
  include_directories (${JPEG_INCLUDE_DIR})
endif ()

if (${USE_PNG})
  include_directories (${PNG_INCLUDE_DIRS})
endif ()

if (${USE_OPENGL})
  include_directories (${OPENG_LINCLUDE_DIR})
endif ()

if (${USE_SDL})
  include_directories (${SDL_INCLUDE_DIR})
endif ()

# In Mac OS X, we need to compile SDLMain library with objective C compiler.
if (${USE_SDL} AND ${APPLE})
  set (SDLMAIN_DIR ${PROJECT_SOURCE_DIR}/ThirdParty/SDLMain)
  add_custom_target (SDLMain ALL
    /usr/bin/gcc -g -O2 -W -Wall -Wextra -I${SDL_INCLUDE_DIR} -I${SDLMAIN_DIR} -c -o SDLMain.o ${SDLMAIN_DIR}/SDLMain.m
    COMMAND ar -r libSDLMain.a SDLMain.o
    COMMAND rm SDLMain.o
    DEPENDS ${SDLMAIN_DIR}/SDLMain.h ${SDLMAIN_DIR}/SDLMain.m
    COMMENT "Build SDLMain on Mac OS X.")
endif ()

# Configure file.
configure_file (
  "${PROJECT_SOURCE_DIR}/xyUtilsConfig.h.in"
  "${PROJECT_SOURCE_DIR}/xyUtilsConfig.h")

################################################################
# Compile the target library.
################################################################
set (CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS}")

# MACRO: xy_add_source (source_list source [depend1 depend2 ...])
# Check `USE_depend1`, `USE_depend2`, ..., and if all passed, add `source` to
# `source_list`.
macro (xy_add_source source_list source)
  set (xy_add_source_depend_ok 1)
  foreach (depend ${ARGN})
    if (NOT USE_${depend})
      set (xy_add_source_depend_ok 0)
    endif ()
  endforeach ()
  if (xy_add_source_depend_ok)
    list (APPEND ${source_list} ${source})
  endif ()
endmacro (xy_add_source)

# Add source files.
xy_add_source (XY_SOURCES Camera.cc EIGEN)
xy_add_source (XY_SOURCES CommandLineFlags.cc)
xy_add_source (XY_SOURCES EigenUtils.cc EIGEN)
xy_add_source (XY_SOURCES FileIO.cc)
xy_add_source (XY_SOURCES LogAndCheck.cc)
xy_add_source (XY_SOURCES NonlinearLeastSquares.cc EIGEN)
xy_add_source (XY_SOURCES NumericalCheck.cc EIGEN)
xy_add_source (XY_SOURCES PlyIO.cc)
xy_add_source (XY_SOURCES PointCameraViewer.cc EIGEN)
xy_add_source (XY_SOURCES PointEdgeViewer.cc EIGEN)
xy_add_source (XY_SOURCES SDLViewer.cc)
xy_add_source (XY_SOURCES StringConvert.cc)
xy_add_source (XY_SOURCES StringUtils.cc)

add_library (xyutils ${XY_SOURCES})

################################################################
# Compile and run tests.
################################################################
enable_testing ()

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
  set (xy_add_test_depend_ok 1)
  foreach (depend ${ARGN})
    if (NOT USE_${depend})
      set (xy_add_test_depend_ok 0)
    endif ()
  endforeach ()
  if (xy_add_test_depend_ok)
    # Compile test binary.
    add_executable (${test_name} ${test_name}.cc)
    # Link to libraries.
    # Note that we currently link all the enabled library to the test
    # target. This is because the `depend` arguments are critical libraries
    # without which the test will not build. Optional libraries, if enabled,
    # should also be linked against test target.
    target_link_libraries(${test_name} xyutils)
    if (USE_BOOST)
      target_link_libraries(${test_name} ${Boost_LIBRARIES})
    endif ()
    if (USE_JPEG)
      target_link_libraries(${test_name} ${JPEG_LIBRARIES})
    endif ()
    if (USE_PNG)
      target_link_libraries(${test_name} ${PNG_LIBRARIES})
    endif ()
    if (USE_OPENGL)
      target_link_libraries(${test_name} ${OPENGL_LIBRARIES})
    endif ()
    if (USE_SDL)
      target_link_libraries(${test_name} ${SDL_LIBRARY})
      if (APPLE)
        add_dependencies(${test_name} SDLMain)
        target_link_libraries(${test_name} ${PROJECT_BINARY_DIR}/libSDLMain.a)
      endif ()
    endif ()
    # Run test.
    if (${test_type} MATCHES general)
      add_test (${test_name} ./${test_name})
      set_tests_properties (${test_name}
        PROPERTIES WILL_FAIL true FAIL_REGULAR_EXPRESSION "Passed.")
    elseif (${test_type} MATCHES fail)
      add_test (${test_name} ./${test_name})
      set_tests_properties (${test_name}
        PROPERTIES WILL_FAIL true FAIL_REGULAR_EXPRESSION "Fail.")
    elseif (${test_type} MATCHES data)
      add_test (${test_name} ./${test_name}
        --test_data_dir=${PROJECT_SOURCE_DIR}/TestData)
      set_tests_properties (${test_name}
        PROPERTIES WILL_FAIL true FAIL_REGULAR_EXPRESSION "Passed.")
    endif ()
  endif (xy_add_test_depend_ok)
endmacro ()

# Define tests.
xy_add_test (CommandLineFlagsTest general)
xy_add_test (EigenUtilsTest data EIGEN)
xy_add_test (FileIOTest data)
xy_add_test (ImageTest data)
xy_add_test (ImageUtilsTest data EIGEN)
xy_add_test (LogAndCheckTest fail)
xy_add_test (NonlinearLeastSquaresTest general EIGEN)
xy_add_test (NumericalCheckTest general EIGEN)
xy_add_test (PlyIOTest data)
xy_add_test (PointEdgeViewerTest data EIGEN OPENGL SDL)
xy_add_test (QuaternionTest general)
xy_add_test (StringConvertTest general)
xy_add_test (StringUtilsTest general)
xy_add_test (TimerTest general)
