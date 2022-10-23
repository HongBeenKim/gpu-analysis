#!/bin/bash

for ((var=100 ; var < 2001 ; var += 100));
do
  CUDA_VISIBLE_DEVICES=MIG-19fec0b5-36f5-5c6a-b5c3-04b61904bb89 ./publisher pp $var &
  sleep 1
  CUDA_VISIBLE_DEVICES=MIG-2c89215e-4ef9-54ab-8331-a2e8e5173645 ./subscriber pp $var
  wait
done

