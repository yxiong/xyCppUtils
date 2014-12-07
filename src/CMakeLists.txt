################################################################
# Compile the target library and binary executables.
#
# Author: Ying Xiong.
# Created: Dec 06, 2014.
################################################################

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
xy_add_source (XY_SOURCES PointCameraViewer.cc EIGEN SDL)
xy_add_source (XY_SOURCES PointEdgeViewer.cc EIGEN SDL)
xy_add_source (XY_SOURCES SDLViewer.cc SDL)
xy_add_source (XY_SOURCES StringConvert.cc)
xy_add_source (XY_SOURCES StringUtils.cc)

# Add library.
add_library (xyutils ${XY_SOURCES})
set_target_properties (xyutils PROPERTIES
  ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Add binary executable.
xy_add_executable (visualize_points_and_cameras EIGEN OPENGL SDL)
if (xy_add_executable_depend_ok)
  set_target_properties (visualize_points_and_cameras PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
endif ()
