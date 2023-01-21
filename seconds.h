#ifndef __seconds_h__
#define __seconds_h__ 1

#include <chrono>

namespace utils {

#ifdef _WIN32
#include <Windows.h>
#define getpid GetCurrentProcessId
inline double cpuTime() {
  FILETIME a,b,c,d;
  return GetProcessTimes(GetCurrentThread(),&a,&b,&c,&d) != 0 ?
    (double) (d.dwLowDateTime |
              ((unsigned long long)d.dwHighDateTime << 32))*100.0 : 0.0;
}
#else
#include <unistd.h>
#include <time.h>
inline double cpuTime() {
  timespec t;
  clock_gettime(CLOCK_THREAD_CPUTIME_ID,&t);
  return 1.0e9*t.tv_sec+t.tv_nsec;
}
#endif

class stopWatch {
  std::chrono::time_point<std::chrono::steady_clock> Start;

public:
  void reset() {
    Start=std::chrono::steady_clock::now();
  }

  stopWatch() {
    reset();
  }

  double nanoseconds(bool reset=false) {
    auto Stop=std::chrono::steady_clock::now();
    double ns=std::chrono::duration_cast<std::chrono::nanoseconds>
      (Stop-Start).count();
    if(reset) Start=Stop;
    return ns;
  }

  double seconds(bool reset=false) {
    return 1.0e-9*nanoseconds(reset);
  }
};

class cpuTimer {
  double start;
  std::chrono::time_point<std::chrono::steady_clock> Start;

public:
  void reset() {
    start=cpuTime();
    Start=std::chrono::steady_clock::now();
  }

  cpuTimer() {
    reset();
  }

  double nanoseconds() {
    auto Stop=std::chrono::steady_clock::now();
    double stop=cpuTime();

    return
      std::min((double) std::chrono::duration_cast<std::chrono::nanoseconds>
               (Stop-Start).count(),stop-start);
  }

  double seconds() {
    return 1.0e-9*nanoseconds();
  }
};

}

#endif
