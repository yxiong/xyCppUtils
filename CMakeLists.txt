################################################################
# CMake file for xyCppUtils.
#
# Author: Ying Xiong.
# Created: Dec 01, 2014.
################################################################

cmake_minimum_required (VERSION 2.8)
project (xyCppUtils CXX)
enable_language (C)

# Default confiurations.

if (NOT CMAKE_BUILD_TYPE)
  set (CMAKE_BUILD_TYPE "RELEASE" CACHE STRING
    "Set default build type to RELEASE." FORCE)
endif (NOT CMAKE_BUILD_TYPE)

set (BUILD_SHARED_LIBS "ON" CACHE BOOL "Build shared library by default.")

################################################################
# Find and configure external libraries.
################################################################
set (CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
include (CMakeDependentOption)

# Find the library and set USE_XXX option, which allows user to disable a found
# library.
find_package (Boost COMPONENTS serialization)
CMAKE_DEPENDENT_OPTION(USE_BOOST "Use Boost." ON Boost_FOUND OFF)

find_package (CUDA)
CMAKE_DEPENDENT_OPTION(USE_CUDA "Use CUDA." ON CUDA_FOUND OFF)

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
if (USE_BOOST)
  include_directories (${Boost_INCLUDE_DIRS})
endif ()

if (USE_EIGEN)
  include_directories (${EIGEN3_INCLUDE_DIR})
endif ()

if (USE_JPEG)
  include_directories (${JPEG_INCLUDE_DIR})
endif ()

if (USE_PNG)
  include_directories (${PNG_INCLUDE_DIRS})
endif ()

if (USE_OPENGL)
  include_directories (${OPENG_LINCLUDE_DIR})
endif ()

if (USE_SDL)
  include_directories (${SDL_INCLUDE_DIR})
endif ()

# In Mac OS X, we need to compile SDLMain library with objective C compiler.
if (USE_SDL AND APPLE)
  set (SDLMAIN_DIR ${PROJECT_SOURCE_DIR}/ThirdParty/SDLMain)
  add_custom_target (SDLMain ALL
    /usr/bin/gcc -g -O2 -W -Wall -Wextra -I${SDL_INCLUDE_DIR} -I${SDLMAIN_DIR} -c -o SDLMain.o ${SDLMAIN_DIR}/SDLMain.m
    COMMAND ar -r libSDLMain.a SDLMain.o
    COMMAND rm SDLMain.o
    COMMAND mkdir -p lib
    COMMAND mv libSDLMain.a lib
    DEPENDS ${SDLMAIN_DIR}/SDLMain.h ${SDLMAIN_DIR}/SDLMain.m
    COMMENT "Build SDLMain on Mac OS X.")
endif ()

################################################################
# Utility macros.
################################################################

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

set (CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS}")
set (CUDA_NVCC_FLAGS "-std=c++11 ${CUDA_NVCC_FLAGS}")

# MACRO: xy_link_libraries (target)
# Link the `target` with all enabled libraries.
macro (xy_link_libraries target)
  if (USE_BOOST)
    target_link_libraries(${target} ${Boost_LIBRARIES})
  endif ()
  if (USE_JPEG)
    target_link_libraries(${target} ${JPEG_LIBRARIES})
  endif ()
  if (USE_PNG)
    target_link_libraries(${target} ${PNG_LIBRARIES})
  endif ()
  if (USE_OPENGL)
    target_link_libraries(${target} ${OPENGL_LIBRARIES})
  endif ()
  if (USE_SDL)
    target_link_libraries(${target} ${SDL_LIBRARY})
    if (APPLE)
      add_dependencies(${target} SDLMain)
      target_link_libraries(${target}
        ${PROJECT_BINARY_DIR}/lib/libSDLMain.a)
    endif ()
  endif ()
endmacro()

# MACRO: xy_add_executable (executable [depend1 depend2])
# Add `executable` with dependency check and proper library linking.
macro (xy_add_executable executable)
  xy_check_depend (xy_add_executable_depend_ok ${ARGN})
  if (xy_add_executable_depend_ok)
    add_executable (${executable} ${executable}.cc)
    # Link to libraries.
    target_link_libraries(${executable} xyCppUtils)
    xy_link_libraries(${executable})
  endif ()
endmacro ()

################################################################
# Add subdirectories.
################################################################

# General setup.
include_directories (${PROJECT_SOURCE_DIR}/include ${PROJECT_BINARY_DIR}/include)

# Header files.
add_subdirectory (include)

# Compile the target library and binary executables.
add_subdirectory (src)

# Compile and run tests.
enable_testing ()
add_subdirectory (test)
