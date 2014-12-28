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
file (GLOB xy_tcc_files *.tcc)
set (xy_config_file ${CMAKE_CURRENT_BINARY_DIR}/xyUtilsConfig.h)
install (FILES ${xy_headers} ${xy_tcc_files} ${xy_config_file}
  DESTINATION include/xyCppUtils)
