#!/bin/bash

THIS_DIR=$(dirname $0)

# remove driver
grep mig_channel /proc/devices >/dev/null && sudo /sbin/rmmod mig_channel

# insert driver
sudo /sbin/insmod mig_channel.ko

# create device inodes

# remove old inodes just in case
if [ -e /dev/migchannel ]; then
    sudo rm /dev/migchannel
fi

echo "INFO: creating /dev/migchannel inode"
sudo mknod /dev/migchannel c 300 0
sudo chmod a+w+r /dev/migchannel
