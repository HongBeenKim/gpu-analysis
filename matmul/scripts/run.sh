#!/bin/bash

StartTime=$(date +%s%N)
./single 500 0 &
./single 500 0 &
wait
EndTime=$(date +%s%N)

echo $((($EndTime - $StartTime) / 1000000))

