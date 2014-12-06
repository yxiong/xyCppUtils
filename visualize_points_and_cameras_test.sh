#!/usr/bin/env bash
#
# Author: Ying Xiong.
# Created: Dec 04, 2014.

set -x -e
exe_path=$1
data_path=$2

${exe_path}/visualize_points_and_cameras                    \
    --pointFileType=ply                                     \
    --pointFile=${data_path}/Models/dinoSparseRing-pmvs.ply \
    --cameraFileType=NumNameKRt                             \
    --cameraFile=${data_path}/Models/dinoSparseRing-cams.txt

set +x +e
echo "Passed."
