#pragma once

#define LOCK_EX 2
#define LOCK_UN 8

extern "C" int flock(int descriptor, int operation);
