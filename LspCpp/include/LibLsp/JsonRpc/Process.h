#pragma once

#include "LibLsp/JsonRpc/stream.h"

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace lsp
{

class ProcessError : public std::runtime_error
{
public:
    using std::runtime_error::runtime_error;
};

class Process
{
public:
    using ArgList = std::vector<std::string>;

    Process() = default;
    Process(std::string executable, ArgList args = {});
    Process(Process const&) = delete;
    Process& operator=(Process const&) = delete;
    Process(Process&& other) noexcept;
    Process& operator=(Process&& other) noexcept;
    ~Process();

    static Process start(std::string const& executable, ArgList args = {});

    bool isRunning() const;
    void wait();
    void terminate();
    int id() const;

    static int currentProcessId();
    static bool exists(int pid);

    std::shared_ptr<istream> input() const;
    std::shared_ptr<ostream> output() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace lsp
