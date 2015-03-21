################################################################
# CUDA header files.
#
# Author: Ying Xiong.
# Created: Mar 20, 2015.
################################################################

# Install CUDA header files.
file (GLOB xy_cuda_headers *.cuh)
file (GLOB xy_cuda_tcc_files *.tcc)
install (FILES ${xy_cuda_headers} ${xy_cuda_tcc_files}
  DESTINATION include/xyCppUtils/cuda)
