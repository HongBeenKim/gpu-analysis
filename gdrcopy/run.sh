#!/bin/bash

export CUDA_VISIBLE_DEVICES=MIG-6e5ecf1c-980b-53b4-b79e-df70177fd284
tests/copylat > copylat1.log &

export CUDA_VISIBLE_DEVICES=MIG-3234bc3b-83f3-5e3a-940e-d1c72da74e00
tests/copylat > copylat2.log &

#export CUDA_VISIBLE_DEVICES=MIG-3234bc3b-83f3-5e3a-940e-d1c72da74e00
#tests/copylat > /dev/null &
#
#export CUDA_VISIBLE_DEVICES=MIG-7eafbec4-0f65-573a-9973-027a818826fa
#tests/copylat > /dev/null &
#
#export CUDA_VISIBLE_DEVICES=MIG-6c8f7562-9526-538e-99a5-4d3808b1a9d7
#tests/copylat > /dev/null &
#
#export CUDA_VISIBLE_DEVICES=MIG-f10db30b-8a0a-5ab0-828c-cc1012868e9d
#tests/copylat > /dev/null &
#
#export CUDA_VISIBLE_DEVICES=MIG-f3eff5ff-244b-520a-9681-d6ea24b8c717
#tests/copylat > /dev/null &

