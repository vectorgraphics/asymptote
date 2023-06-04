#include "LibLsp/lsp/ParentProcessWatcher.h"
#include <boost/process.hpp>

#ifdef _WIN32
#include <boost/process/windows.hpp>
#endif

#include <boost/filesystem.hpp>
#include <boost/asio.hpp>
#include <iostream>

#include "LibLsp/lsp/ProcessIoService.h"
#include "LibLsp/lsp/SimpleTimer.h"


using namespace boost::asio::ip;
using namespace std;

struct ParentProcessWatcher::ParentProcessWatcherData : std::enable_shared_from_this<ParentProcessWatcherData>
{
        std::unique_ptr<SimpleTimer<boost::posix_time::seconds>>  timer;
        lsp::Log& _log;
        std::function<void()>  on_exit;
        lsp::ProcessIoService asio_io;
        std::shared_ptr < boost::process::opstream>  write_to_service;
        std::shared_ptr< boost::process::ipstream >   read_from_service;
        int pid;
        const  int _poll_delay_secs /*= 10*/;
        std::string command;
        std::shared_ptr<boost::process::child> c;

        ParentProcessWatcherData(lsp::Log& log, int _pid,
                const std::function<void()>&& callback, uint32_t  poll_delay_secs) :
                _log(log), on_exit(callback), pid(_pid), _poll_delay_secs(poll_delay_secs)
        {
#ifdef _WIN32
                command = "cmd /c \"tasklist /FI \"PID eq " + std::to_string(pid) + "\" | findstr " +
                        std::to_string(pid) + "\"";
#else
                command = "ps -p " + std::to_string(pid);
#endif

        }

        void run()
        {
                write_to_service = std::make_shared<boost::process::opstream>();
                read_from_service = std::make_shared<boost::process::ipstream>();

//              const uint32_t POLL_DELAY_SECS = _poll_delay_secs;
                auto self(shared_from_this());
                std::error_code ec;
                namespace bp = boost::process;
                c = std::make_shared<bp::child>(asio_io.getIOService(), command,
                        ec,
#ifdef _WIN32
                        bp::windows::hide,
#endif
                        bp::std_out > *read_from_service,
                        bp::std_in < *write_to_service,
                        bp::on_exit([self](int exit_code, const std::error_code& ec_in) {
                                // the tasklist command should return 0 (parent process exists) or 1 (parent process doesn't exist)
                                if (exit_code == 1)//
                                {
                                        if (self->on_exit)
                                        {

                                                std::thread([=]()
                                                        {
                                                                std::this_thread::sleep_for(std::chrono::seconds(3));
                                                                self->on_exit();
                                                        }).detach();
                                        }
                                }
                                else
                                {
                                        if (exit_code > 1)
                                        {
                                                self->_log.log(lsp::Log::Level::WARNING, "The tasklist command: '" + self->command + "' returns " + std::to_string(exit_code));
                                        }

                                        self->timer = std::make_unique<SimpleTimer<boost::posix_time::seconds>>(self->_poll_delay_secs, [=]() {
                                                self->run();
                                                });
                                }

                                }));
                if (ec)
                {
                        // fail
                        _log.log(lsp::Log::Level::SEVERE, "Start parent process watcher failed.");
                }
        }
};

ParentProcessWatcher::ParentProcessWatcher(lsp::Log& log, int pid,
        const std::function<void()>&& callback, uint32_t  poll_delay_secs) : d_ptr(new ParentProcessWatcherData(log, pid, std::move(callback), poll_delay_secs))
{
        d_ptr->run();
}

ParentProcessWatcher::~ParentProcessWatcher()
{
        if (d_ptr->timer)
                d_ptr->timer->Stop();
}
