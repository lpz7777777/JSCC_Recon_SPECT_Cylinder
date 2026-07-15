#pragma once

#include <cstddef>
#include <sys/types.h>

#ifndef _SSIZE_T_DEFINED
using ssize_t = long long;
#define _SSIZE_T_DEFINED
#endif

extern "C" ssize_t pread(int descriptor, void* buffer, size_t count, off_t offset);
extern "C" ssize_t pwrite(
    int descriptor, const void* buffer, size_t count, off_t offset);
extern "C" int ftruncate(int descriptor, off_t length);
extern "C" int fdatasync(int descriptor);
extern "C" int close(int descriptor);
extern "C" int open(const char* path, int flags, ...);
