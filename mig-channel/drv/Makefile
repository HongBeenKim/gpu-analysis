KERNDIR=/lib/modules/$(shell uname -r)/build
obj-m+=mig_channel.o
PWD=$(shell pwd)

EXTRA_CFLAGS=-I/home/dtb05045/open-gpu-kernel-modules/kernel-open/nvidia
KBUILD_EXTRA_SYMBOLS=/home/dtb05045/open-gpu-kernel-modules/kernel-open/Module.symvers

default:
	make -C $(KERNDIR) M=$(PWD) modules

clean:
	make -C $(KERNDIR) M=$(PWD) clean
	rm -rf *.ko
	rm -rf *.o

