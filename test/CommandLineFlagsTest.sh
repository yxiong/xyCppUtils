#!/usr/bin/env bash
#
# Author: Ying Xiong.
# Created: Dec 05, 2014.

set -x -e
project_binary_path=$1
data_path=$2

${project_binary_path}/test/CommandLineFlagsTest               \
           --long_test=123456789 arg1 --string_test="foo bar"  \
           --int_test=0xff arg2 --bool_test --char_test=c      \
           --double_test=23.34 arg3

${project_binary_path}/test/CommandLineFlagsTest               \
           --long_test=123456789 arg1 arg2 arg3                \
           --flagfile=${data_path}/Texts/CommandLineFlagsFileTest.txt \
           arg4 --char_test=c arg5

set +x +e
echo "Passed."
