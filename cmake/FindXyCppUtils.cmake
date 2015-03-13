#.rst:
# FindXyCppUtils
# --------------
#
# Find the native xyCppUtils includes and library.
#
# This module searches xyCppUtils, the cpp utilities by Ying Xiong. It defines
# the following variables:
#
# ``XyCppUtils_INCLUDE_DIRS``
#   where to find the xyCppUtils/ folder that contains the header files.
# ``XyCppUtils_LIBRARIES``
#   the libraries to link against to use xyCppUtils.
# ``XyCppUtils_FOUND``
#   true if xyCppUtils was found on the system.

# Author: Ying Xiong.
# Created: Mar 13, 2015.

find_path (XyCppUtils_INCLUDE_DIRS "xyCppUtils/")
mark_as_advanced (XyCppUtils_INCLUDE_DIRS)

find_library (XyCppUtils_LIBRARIES "xyCppUtils")
mark_as_advanced (XyCppUtils_LIBRARIES)

find_package_handle_standard_args (XyCppUtils
  FOUND_VAR XyCppUtils_FOUND
  REQUIRED_VARS XyCppUtils_INCLUDE_DIRS XyCppUtils_LIBRARIES)
