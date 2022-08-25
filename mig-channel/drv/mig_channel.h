typedef struct {
  __u64 addr;
  __u64 size;
} MIG_CHANNEL_IOC_PIN_BUFFER_PARAMS;

typedef struct {
  __u8 value;
} MIG_CHANNEL_IOC_PIN_WRITE_BOTH_PARAMS;

#define MIG_CHANNEL_IOCTL 0x28
#define MIG_CHANNEL_IOC_PIN_BUFFER    _IOW (MIG_CHANNEL_IOCTL, 1, MIG_CHANNEL_IOC_PIN_BUFFER_PARAMS)
#define MIG_CHANNEL_IOC_UNPIN_BUFFER  _IO  (MIG_CHANNEL_IOCTL, 2)
#define MIG_CHANNEL_IOC_READ_BOTH     _IO  (MIG_CHANNEL_IOCTL, 3)
#define MIG_CHANNEL_IOC_WRITE_BOTH    _IOW (MIG_CHANNEL_IOCTL, 4, MIG_CHANNEL_IOC_PIN_WRITE_BOTH_PARAMS)

