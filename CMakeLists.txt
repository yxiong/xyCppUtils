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

# MACRO: xy_check_depend(depend_ok [depend1 depend2 ...])
# Check `USE_depend1`, `USE_depend2`, ..., and if all passed, set `depend_ok` to
# 1, otherwise to 0.
macro (xy_check_depend depend_ok)
  set (${depend_ok} 1)
  foreach (depend ${ARGN})
    if (NOT USE_${depend})
      set (${depend_ok} 0)
    endif ()
  endforeach ()
endmacro ()

################################################################
# Compile the target library.
################################################################
set (CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS}")

# MACRO: xy_add_source (source_list source [depend1 depend2 ...])
# Add `source` to `source_list` if the dependency check passed.
macro (xy_add_source source_list source)
  xy_check_depend (xy_add_source_depend_ok ${ARGN})
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
# Compile the binary executables.
################################################################

# MACRO: xy_add_executable (executable [depend1 depend2])
# Add `executable` with dependency check and proper library linking.
macro (xy_add_executable executable)
  xy_check_depend (xy_add_executable_depend_ok ${ARGN})
  if (xy_add_executable_depend_ok)
    add_executable (${executable} ${executable}.cc)
    # Link to libraries.
    # Note that we currently link all the enabled library to the executable
    # target. This is because the `depend` arguments are critical libraries
    # without which the executable will not build. Optional libraries, if
    # enabled, should also be linked against the executable target.
    target_link_libraries(${executable} xyutils)
    if (USE_BOOST)
      target_link_libraries(${executable} ${Boost_LIBRARIES})
    endif ()
    if (USE_JPEG)
      target_link_libraries(${executable} ${JPEG_LIBRARIES})
    endif ()
    if (USE_PNG)
      target_link_libraries(${executable} ${PNG_LIBRARIES})
    endif ()
    if (USE_OPENGL)
      target_link_libraries(${executable} ${OPENGL_LIBRARIES})
    endif ()
    if (USE_SDL)
      target_link_libraries(${executable} ${SDL_LIBRARY})
      if (APPLE)
        add_dependencies(${executable} SDLMain)
        target_link_libraries(${executable} ${PROJECT_BINARY_DIR}/libSDLMain.a)
      endif ()
    endif ()
  endif ()
endmacro ()
xy_add_executable (visualize_points_and_cameras EIGEN OPENGL SDL)

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
  xy_check_depend (xy_add_test_depend_ok ${ARGN})
  if (xy_add_test_depend_ok)
    # Compile test binary.
    xy_add_executable (${test_name} ${ARGN})
    # Run test.
    if (${test_type} MATCHES general)
      add_test (${test_name} ./${test_name})
      set_tests_properties (${test_name}
        PROPERTIES PASS_REGULAR_EXPRESSION "Passed.")
    elseif (${test_type} MATCHES fail)
      add_test (${test_name} ./${test_name})
      set_tests_properties (${test_name}
        PROPERTIES WILL_FAIL true FAIL_REGULAR_EXPRESSION "Fail.")
    elseif (${test_type} MATCHES data)
      add_test (${test_name} ./${test_name}
        --test_data_dir=${PROJECT_SOURCE_DIR}/TestData)
      set_tests_properties (${test_name}
        PROPERTIES PASS_REGULAR_EXPRESSION "Passed.")
    endif ()
  endif (xy_add_test_depend_ok)
endmacro ()

# MACRO: xy_add_test_script (script_name script_type [depend1 depend2 ...])
macro (xy_add_test_script script_name script_type)
  xy_check_depend (xy_add_test_script_depend_ok ${ARGN})
  if (xy_add_test_script_depend_ok)
    # Run test script.
    set (script_exe ${PROJECT_SOURCE_DIR}/${script_name})
    add_test (${script_name} ${script_exe}
      ${PROJECT_BINARY_DIR} ${PROJECT_SOURCE_DIR}/TestData)
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
xy_add_test (NonlinearLeastSquaresTest general EIGEN)
xy_add_test (NumericalCheckTest general EIGEN)
xy_add_test (PlyIOTest data)
xy_add_test (PointEdgeViewerTest data EIGEN OPENGL SDL)
xy_add_test (QuaternionTest general)
xy_add_test (StringConvertTest general)
xy_add_test (StringUtilsTest general)
xy_add_test (TimerTest general)

xy_add_test_script (CommandLineFlagsTest.sh data)
xy_add_test_script (visualize_points_and_cameras_test.sh data EIGEN OPENGL SDL)
