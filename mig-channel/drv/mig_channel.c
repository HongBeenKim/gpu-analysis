#include <linux/init.h>
#include <linux/kernel.h>
#include <linux/module.h>
#include <linux/fs.h>
#include <linux/uaccess.h>
#include "mig_channel.h"
#include "nv-p2p.h"

#define MIG_CHANNEL_MAJOR 300

#define GPU_PAGE_SHIFT   16
#define GPU_PAGE_SIZE    ((u64)1 << GPU_PAGE_SHIFT)
#define GPU_PAGE_OFFSET  (GPU_PAGE_SIZE-1)
#define GPU_PAGE_MASK    (~GPU_PAGE_OFFSET)

struct nvidia_p2p_page_table *pageTables[2];
unsigned long virtAddrs[2];
int clientCnt = 0;

static int channel_drv_open(struct inode *inode, struct file *filp) {
  return 0;
}

static int channel_drv_release(struct inode *inode, struct file *filp) {
  return 0;
}

static void pages_free_callback(void *data) {
  struct nvidia_p2p_page_table *page_table = NULL;
  page_table = (struct nvidia_p2p_page_table *)data;
  if (page_table)
    nvidia_p2p_free_page_table(page_table);
}

static int pin_buffer(void __user *_params) {
  int ret = 0;
  u64 page_virt_start, page_virt_end;
  size_t rounded_size;
  struct nvidia_p2p_page_table *page_table = NULL;

  MIG_CHANNEL_IOC_PIN_BUFFER_PARAMS params = { 0 };
  if (copy_from_user(&params, _params, sizeof(params))) {
    printk("[migchannel] copy from user failed while pin_buffer\n");
    return -1;
  }
  printk("[migchannel] pin addr: 0x%llx --- size: %llu\n", params.addr, params.size);

  page_virt_start = params.addr & GPU_PAGE_MASK;
  page_virt_end   = params.addr + params.size - 1;
  rounded_size    = page_virt_end - page_virt_start + 1;

  ret = nvidia_p2p_get_pages(
    0, 0, page_virt_start, rounded_size, &page_table, pages_free_callback, page_table
  );

  virtAddrs[clientCnt] = page_virt_start;
  pageTables[clientCnt] = page_table;
  clientCnt++;

  printk("[migchannel] p2p_get_pages res: %d --- clientCnt: %d\n", ret, clientCnt);

  return ret;
}

static int unpin_buffer(void) {
  int ret = 0;
  clientCnt--;
  ret = nvidia_p2p_put_pages(0, 0, virtAddrs[clientCnt], pageTables[clientCnt]);
  printk("[migchannel] p2p_put_pages res: %d --- clientCnt: %d\n", ret, clientCnt);
  return ret;
}

static void read_both(void) {
  void *kernelVirtAddr0 = NULL;
  void *kernelVirtAddr1 = NULL;
  kernelVirtAddr0 = ioremap(pageTables[0]->pages[0]->physical_address, pageTables[0]->page_size);
  kernelVirtAddr1 = ioremap(pageTables[1]->pages[0]->physical_address, pageTables[1]->page_size);
  printk("[migchannel] value 1: %u --- value 2: %u\n", 
    *((uint8_t *)kernelVirtAddr0), *((uint8_t *)kernelVirtAddr1)
  );
  iounmap(kernelVirtAddr0);
  iounmap(kernelVirtAddr1);
}

static void write_both(void __user *_params) {
  void *kernelVirtAddr0 = NULL;
  void *kernelVirtAddr1 = NULL;
  MIG_CHANNEL_IOC_PIN_WRITE_BOTH_PARAMS params = { 0 };
  if (copy_from_user(&params, _params, sizeof(params))) {
    printk("[migchannel] copy from user failed while write_both\n");
    return;
  }
  kernelVirtAddr0 = ioremap(pageTables[0]->pages[0]->physical_address, pageTables[0]->page_size);
  kernelVirtAddr1 = ioremap(pageTables[1]->pages[0]->physical_address, pageTables[1]->page_size);
  *((uint8_t *)kernelVirtAddr0) = *((uint8_t *)kernelVirtAddr1) = params.value;
  iounmap(kernelVirtAddr0);
  iounmap(kernelVirtAddr1);
}

static long channel_drv_ioctl(
  struct file *filp, 
  unsigned int cmd, 
  unsigned long arg
) {
  long ret = 0;
  void __user *argp;

  if (_IOC_TYPE(cmd) != MIG_CHANNEL_IOCTL) {
    printk("[migchannel] Invalid IOCTL code type = %08x\n", _IOC_TYPE(cmd));
    return -EINVAL;
  }

  switch (cmd) {
    case MIG_CHANNEL_IOC_PIN_BUFFER:
      argp = (void __user *)arg;
      ret = pin_buffer(argp);
      break;
    case MIG_CHANNEL_IOC_UNPIN_BUFFER:
      ret = unpin_buffer();
      break;
    case MIG_CHANNEL_IOC_READ_BOTH:
      read_both();
      break;
    case MIG_CHANNEL_IOC_WRITE_BOTH:
      argp = (void __user *)arg;
      write_both(argp);
      break;
  }
  return ret;
}

struct file_operations mig_channel_fops = {
  .open = channel_drv_open,
  .release = channel_drv_release,
  .unlocked_ioctl = channel_drv_ioctl
};

static int __init channel_driver_init(void) {
  int result;
  result = register_chrdev(MIG_CHANNEL_MAJOR, "mig_channel", &mig_channel_fops);
  if (result < 0) {
    printk("mig_channel register failed with major number %d\n", MIG_CHANNEL_MAJOR); 
    return result;
  }
  printk("mig_channel registered with major number %d\n", MIG_CHANNEL_MAJOR);
  return 0;
}

static void __exit channel_driver_exit(void) {
  printk("unregistering mig_channel\n");
  unregister_chrdev(MIG_CHANNEL_MAJOR, "mig_channel");
}

module_init(channel_driver_init);
module_exit(channel_driver_exit);

