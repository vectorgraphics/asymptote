#pragma once
#include <boost/asio.hpp>
#include <iostream>

namespace lsp
{
class ProcessIoService
{
public:
    using IOService = boost::asio::io_context;
    using Work = boost::asio::io_context::work;
    using WorkPtr = std::unique_ptr<Work>;

    ProcessIoService()
    {

        work_ = std::unique_ptr<Work>(new Work(ioService_));
        auto temp_thread_ = new std::thread([this] { ioService_.run(); });
        thread_ = std::unique_ptr<std::thread>(temp_thread_);
    }

    ProcessIoService(ProcessIoService const&) = delete;
    ProcessIoService& operator=(ProcessIoService const&) = delete;

    boost::asio::io_context& getIOService()
    {
        return ioService_;
    }

    void stop()
    {

        work_.reset();

        thread_->join();
    }

private:
    IOService ioService_;
    WorkPtr work_;
    std::unique_ptr<std::thread> thread_;
};

} // namespace lsp
