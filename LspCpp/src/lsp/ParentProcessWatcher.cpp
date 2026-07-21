#include "LibLsp/lsp/ParentProcessWatcher.h"
#include <algorithm>
#include <cstdlib>              // for std::system
#include <thread>
#include <chrono>
#include <LibLsp/lsp/asio.h>
#include <iostream>

#ifdef _WIN32
#include <windows.h>          // for Windows-specific system exit codes if needed
#else
#include <sys/wait.h>         // for WIFEXITED / WEXITSTATUS
#endif

#include "LibLsp/lsp/ProcessIoService.h"
#include "LibLsp/lsp/SimpleTimer.h"

using namespace asio::ip;
using namespace std;

struct ParentProcessWatcher::ParentProcessWatcherData : std::enable_shared_from_this<ParentProcessWatcherData>
{
    std::unique_ptr<SimpleTimer<std::chrono::seconds>> timer;
    lsp::Log& _log;
    std::function<void()> on_exit;
    lsp::ProcessIoService asio_io;
    int pid;
    int const _poll_delay_secs;
    std::string command;

    ParentProcessWatcherData(lsp::Log& log, int _pid, std::function<void()> const&& callback, uint32_t poll_delay_secs)
        : _log(log), on_exit(callback), pid(_pid), _poll_delay_secs(poll_delay_secs)
    {
#ifdef _WIN32
        command =
            "cmd /c \"tasklist /FI \"PID eq " + std::to_string(pid) + "\" | findstr " + std::to_string(pid) + "\"";
#else
        command = "ps -p " + std::to_string(pid);
#endif
    }

    void run()
    {
        auto self = shared_from_this();
        // launch the check in a detached thread so we don't block the ASIO event loop
        std::thread([self]() {
                        int status = std::system(self->command.c_str());
                        int exit_code = -1;

                        if (status == -1) {
                            // system call failed entirely
                            self->_log.log(lsp::Log::Level::WARNING,
                                           "System call failed for command: '" + self->command + "'.");
                            exit_code = 2; // treat as generic error >1 to re-arm timer
                        } else {
#ifdef _WIN32
                            // On Windows, system() returns the program exit code directly
                            exit_code = status;
#else
                            if (WIFEXITED(status))
                                exit_code = WEXITSTATUS(status);
                            else
                                exit_code = status;
#endif
                        }

                        // now dispatch back into ASIO so timer and on_exit run in the right context
                        asio::post(self->asio_io.getIOService(), [self, exit_code]() {
                                       if (exit_code == 1)  // parent process not found
                                       {
                                           if (self->on_exit)
                                           {
                                               // small delay before notifying
                                               std::thread([self]() {
                                                               std::this_thread::sleep_for(std::chrono::seconds(3));
                                                               asio::post(self->asio_io.getIOService(), [self]() {
                                                                              self->on_exit();
                                                                          });
                                                           }).detach();
                                           }
                                       }
                                       else
                                       {
                                           if (exit_code > 1)
                                           {
                                               self->_log.log(
                                                   lsp::Log::Level::WARNING,
                                                   "The tasklist command: '" + self->command + "' returned " + std::to_string(exit_code)
                                               );
                                           }
                                           // re‑arm the poll timer
                                           self->timer = std::make_unique<SimpleTimer<std::chrono::seconds>>(
                                               self->_poll_delay_secs,
                                               [self]() { self->run(); }
                                           );
                                       }
                                   });
                    }).detach();
    }
};

ParentProcessWatcher::ParentProcessWatcher(
    lsp::Log& log, int pid, std::function<void()> const&& callback, uint32_t poll_delay_secs
)
    : d_ptr(new ParentProcessWatcherData(log, pid, std::move(callback), poll_delay_secs))
{
    d_ptr->run();
}

ParentProcessWatcher::~ParentProcessWatcher()
{
    if (d_ptr->timer)
    {
        d_ptr->timer->Stop();
    }
}