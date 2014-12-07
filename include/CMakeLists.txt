################################################################
# Header files.
#
# Author: Ying Xiong.
# Created: Dec 06, 2014.
################################################################

# Configure file.
configure_file (
  "${CMAKE_CURRENT_SOURCE_DIR}/xyUtilsConfig.h.in"
  "${CMAKE_CURRENT_BINARY_DIR}/xyUtilsConfig.h")

# Install header files.
file (GLOB xy_headers *.h)
install (FILES ${xy_headers} ${CMAKE_CURRENT_BINARY_DIR}/xyUtilsConfig.h
  DESTINATION include/xyCppUtils)
