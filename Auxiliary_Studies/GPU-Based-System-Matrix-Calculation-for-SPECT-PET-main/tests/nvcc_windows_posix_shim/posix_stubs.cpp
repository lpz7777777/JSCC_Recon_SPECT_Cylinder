#include "unistd.h"
#include "sys/file.h"

extern "C" int flock(int, int) { return -1; }
extern "C" ssize_t pread(int, void*, size_t, off_t) { return -1; }
extern "C" ssize_t pwrite(int, const void*, size_t, off_t) { return -1; }
extern "C" int ftruncate(int, off_t) { return -1; }
extern "C" int fdatasync(int) { return -1; }
extern "C" int close(int) { return -1; }
extern "C" int open(const char*, int, ...) { return -1; }
