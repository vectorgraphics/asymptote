#include "LibLsp/JsonRpc/Process.h"

#include <atomic>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <limits>
#include <mutex>
#include <utility>
#include <vector>

#if defined(__APPLE__) || defined(__linux__) || defined(__HAIKU__)
#define LSPCPP_PROCESS_POSIX 1
#elif defined(_WIN32)
#define LSPCPP_PROCESS_WIN32 1
#endif

#ifdef LSPCPP_PROCESS_POSIX
#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>
#elif defined(LSPCPP_PROCESS_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <Windows.h>
#endif

namespace lsp
{
namespace
{

#ifdef LSPCPP_PROCESS_POSIX
#ifndef F_SETNOSIGPIPE
void IgnoreSigPipeIfDefault()
{
    static std::once_flag once;
    std::call_once(
        once,
        []()
        {
            struct sigaction current_action {};
            if (sigaction(SIGPIPE, nullptr, &current_action) != 0)
            {
                return;
            }
            if (current_action.sa_handler != SIG_DFL)
            {
                return;
            }

            struct sigaction ignored_action {};
            ignored_action.sa_handler = SIG_IGN;
            sigemptyset(&ignored_action.sa_mask);
            ignored_action.sa_flags = 0;
            sigaction(SIGPIPE, &ignored_action, nullptr);
        });
}
#endif

class pipe_istream : public istream
{
public:
    explicit pipe_istream(int fd) : fd_(fd)
    {
    }

    int get() override
    {
        unsigned char c = 0;
        if (read_some(reinterpret_cast<char*>(&c), 1) == 1)
        {
            return static_cast<int>(c);
        }
        return EOF;
    }

    istream& read(char* str, std::streamsize count) override
    {
        if (interrupted_.load(std::memory_order_relaxed) || count <= 0)
        {
            return *this;
        }

        for (std::streamsize offset = 0; offset < count;)
        {
            auto const chunk = read_some(str + offset, count - offset);
            if (chunk <= 0)
            {
                break;
            }
            offset += chunk;
        }
        return *this;
    }

    std::streamsize read_some(char* str, std::streamsize count) override
    {
        if (count <= 0 || interrupted_.load(std::memory_order_relaxed))
        {
            return 0;
        }

        if (buffer_pos_ < buffer_.size())
        {
            auto const available = static_cast<std::streamsize>(buffer_.size() - buffer_pos_);
            auto const copied = std::min<std::streamsize>(available, count);
            std::memcpy(str, buffer_.data() + buffer_pos_, static_cast<size_t>(copied));
            buffer_pos_ += static_cast<size_t>(copied);
            return copied;
        }

        if (fd_ < 0)
        {
            bad_ = true;
            what_ = "Invalid process stdout pipe.";
            return 0;
        }

        while (!interrupted_.load(std::memory_order_relaxed))
        {
            pollfd read_fd;
            read_fd.fd = fd_;
            read_fd.events = POLLIN;
            read_fd.revents = 0;

            int const ready = ::poll(&read_fd, 1, 100);
            if (interrupted_.load(std::memory_order_relaxed))
            {
                break;
            }
            if (ready == 0)
            {
                continue;
            }
            if (ready < 0)
            {
                if (errno == EINTR)
                {
                    continue;
                }
                bad_ = true;
                what_ = std::strerror(errno);
                return 0;
            }
            if (read_fd.revents & POLLNVAL)
            {
                bad_ = true;
                what_ = "Invalid process stdout pipe.";
                return 0;
            }
            if (read_fd.revents & POLLERR)
            {
                bad_ = true;
                what_ = "Process stdout pipe error.";
                return 0;
            }
            if ((read_fd.revents & (POLLIN | POLLHUP)) == 0)
            {
                continue;
            }

            if (count >= kBufferCapacity)
            {
                auto const request =
                    std::min<std::streamsize>(count, std::numeric_limits<ssize_t>::max());
                ssize_t const bytes = ::read(fd_, str, static_cast<size_t>(request));
                if (bytes > 0)
                {
                    return bytes;
                }
                if (bytes == 0)
                {
                    eof_ = true;
                    return 0;
                }
            }
            else
            {
                buffer_.resize(kBufferCapacity);
                ssize_t const bytes = ::read(fd_, &buffer_[0], buffer_.size());
                if (bytes > 0)
                {
                    buffer_.resize(static_cast<size_t>(bytes));
                    auto const copied = std::min<std::streamsize>(bytes, count);
                    std::memcpy(str, buffer_.data(), static_cast<size_t>(copied));
                    buffer_pos_ = static_cast<size_t>(copied);
                    return copied;
                }
                buffer_.clear();
                buffer_pos_ = 0;
                if (bytes == 0)
                {
                    eof_ = true;
                    return 0;
                }
            }

            if (errno == EINTR)
            {
                continue;
            }
            bad_ = true;
            what_ = std::strerror(errno);
            return 0;
        }

        eof_ = true;
        return 0;
    }

    bool fail() override
    {
        return interrupted_.load(std::memory_order_relaxed);
    }

    bool bad() override
    {
        return bad_;
    }

    bool eof() override
    {
        return eof_ || interrupted_.load(std::memory_order_relaxed);
    }

    bool good() override
    {
        return !fail() && !bad() && !eof();
    }

    void clear() override
    {
        interrupted_.store(false, std::memory_order_relaxed);
        eof_ = false;
        bad_ = false;
        what_.clear();
    }

    std::string what() override
    {
        if (interrupted_.load(std::memory_order_relaxed))
        {
            return "Process input stream interrupted.";
        }
        return what_;
    }

    void interrupt() override
    {
        interrupted_.store(true, std::memory_order_relaxed);
    }

    void close()
    {
        if (fd_ >= 0)
        {
            ::close(fd_);
            fd_ = -1;
        }
    }

private:
    static constexpr std::streamsize kBufferCapacity = 4096;

    int fd_;
    std::atomic<bool> interrupted_ {false};
    bool eof_ = false;
    bool bad_ = false;
    std::string what_;
    std::string buffer_;
    size_t buffer_pos_ = 0;
};

class pipe_ostream : public ostream
{
public:
    explicit pipe_ostream(int fd) : fd_(fd)
    {
#ifdef F_SETNOSIGPIPE
        fcntl(fd_, F_SETNOSIGPIPE, 1);
#endif
    }

    ostream& write(std::string const& value) override
    {
        if (fd_ < 0 || bad_)
        {
            bad_ = true;
            return *this;
        }

#ifndef F_SETNOSIGPIPE
        IgnoreSigPipeIfDefault();
#endif

        char const* data = value.data();
        std::streamsize remaining = static_cast<std::streamsize>(value.size());
        while (remaining > 0)
        {
            auto const chunk =
                std::min<std::streamsize>(remaining, std::numeric_limits<ssize_t>::max());
            ssize_t const written = ::write(fd_, data, static_cast<size_t>(chunk));
            if (written < 0)
            {
                if (errno == EINTR)
                {
                    continue;
                }
                bad_ = true;
                what_ = std::strerror(errno);
                break;
            }
            if (written == 0)
            {
                bad_ = true;
                what_ = "Failed to write to process stdin pipe.";
                break;
            }
            data += written;
            remaining -= written;
        }
        return *this;
    }

    ostream& write(std::streamsize value) override
    {
        return write(std::to_string(value));
    }

    ostream& flush() override
    {
        return *this;
    }

    bool fail() override
    {
        return bad_;
    }

    bool bad() override
    {
        return bad_;
    }

    bool eof() override
    {
        return false;
    }

    bool good() override
    {
        return !bad_;
    }

    void clear() override
    {
        bad_ = false;
        what_.clear();
    }

    std::string what() override
    {
        return what_;
    }

    void close()
    {
        if (fd_ >= 0)
        {
            ::close(fd_);
            fd_ = -1;
        }
    }

private:
    int fd_;
    bool bad_ = false;
    std::string what_;
};
#endif

#ifdef LSPCPP_PROCESS_WIN32
class pipe_istream : public istream
{
public:
    explicit pipe_istream(HANDLE handle) : handle_(handle)
    {
    }

    int get() override
    {
        unsigned char c = 0;
        if (read_some(reinterpret_cast<char*>(&c), 1) == 1)
        {
            return static_cast<int>(c);
        }
        return EOF;
    }

    istream& read(char* str, std::streamsize count) override
    {
        for (std::streamsize offset = 0; offset < count;)
        {
            auto const chunk = read_some(str + offset, count - offset);
            if (chunk <= 0)
            {
                break;
            }
            offset += chunk;
        }
        return *this;
    }

    std::streamsize read_some(char* str, std::streamsize count) override
    {
        if (count <= 0 || interrupted_.load(std::memory_order_relaxed) || handle_ == nullptr)
        {
            return 0;
        }

        DWORD bytes_read = 0;
        if (!ReadFile(handle_, str, static_cast<DWORD>(count), &bytes_read, nullptr))
        {
            bad_ = true;
            what_ = "Failed to read from process stdout.";
            return 0;
        }
        if (bytes_read == 0)
        {
            eof_ = true;
            return 0;
        }
        return static_cast<std::streamsize>(bytes_read);
    }

    bool fail() override
    {
        return interrupted_.load(std::memory_order_relaxed);
    }

    bool bad() override
    {
        return bad_;
    }

    bool eof() override
    {
        return eof_ || interrupted_.load(std::memory_order_relaxed);
    }

    bool good() override
    {
        return !fail() && !bad() && !eof();
    }

    void clear() override
    {
        interrupted_.store(false, std::memory_order_relaxed);
        eof_ = false;
        bad_ = false;
        what_.clear();
    }

    std::string what() override
    {
        if (interrupted_.load(std::memory_order_relaxed))
        {
            return "Process input stream interrupted.";
        }
        return what_;
    }

    void interrupt() override
    {
        interrupted_.store(true, std::memory_order_relaxed);
        HANDLE const handle = handle_;
        if (handle != nullptr)
        {
            CancelIoEx(handle, nullptr);
            close();
        }
    }

    void close()
    {
        if (handle_ != nullptr)
        {
            CloseHandle(handle_);
            handle_ = nullptr;
        }
    }

private:
    HANDLE handle_ = nullptr;
    std::atomic<bool> interrupted_ {false};
    bool eof_ = false;
    bool bad_ = false;
    std::string what_;
};

class pipe_ostream : public ostream
{
public:
    explicit pipe_ostream(HANDLE handle) : handle_(handle)
    {
    }

    ostream& write(std::string const& value) override
    {
        if (handle_ == nullptr)
        {
            bad_ = true;
            return *this;
        }

        char const* data = value.data();
        std::streamsize remaining = static_cast<std::streamsize>(value.size());
        while (remaining > 0)
        {
            DWORD bytes_written = 0;
            DWORD const chunk = static_cast<DWORD>(std::min<std::streamsize>(remaining, 0x7fffffff));
            if (!WriteFile(handle_, data, chunk, &bytes_written, nullptr))
            {
                bad_ = true;
                what_ = "Failed to write to process stdin.";
                break;
            }
            data += bytes_written;
            remaining -= static_cast<std::streamsize>(bytes_written);
        }
        return *this;
    }

    ostream& write(std::streamsize value) override
    {
        return write(std::to_string(value));
    }

    ostream& flush() override
    {
        return *this;
    }

    bool fail() override
    {
        return bad_;
    }

    bool bad() override
    {
        return bad_;
    }

    bool eof() override
    {
        return false;
    }

    bool good() override
    {
        return !bad_;
    }

    void clear() override
    {
        bad_ = false;
        what_.clear();
    }

    std::string what() override
    {
        return what_;
    }

    void close()
    {
        if (handle_ != nullptr)
        {
            CloseHandle(handle_);
            handle_ = nullptr;
        }
    }

private:
    HANDLE handle_ = nullptr;
    bool bad_ = false;
    std::string what_;
};
#endif

} // namespace

struct Process::Impl
{
#if defined(LSPCPP_PROCESS_POSIX) || defined(LSPCPP_PROCESS_WIN32)
    std::shared_ptr<pipe_istream> input_stream;
    std::shared_ptr<pipe_ostream> output_stream;
#endif
    int process_id = -1;

#if defined(LSPCPP_PROCESS_POSIX)
    pid_t pid = -1;

    Impl(std::string const& executable, Process::ArgList const& args)
    {
        int in_pipe[2] = {-1, -1};
        int out_pipe[2] = {-1, -1};
        int err_pipe[2] = {-1, -1};

        if (pipe(in_pipe) == -1)
        {
            throw ProcessError(std::strerror(errno));
        }
        if (pipe(out_pipe) == -1)
        {
            close(in_pipe[0]);
            close(in_pipe[1]);
            throw ProcessError(std::strerror(errno));
        }
        if (pipe(err_pipe) == -1)
        {
            close(in_pipe[0]);
            close(in_pipe[1]);
            close(out_pipe[0]);
            close(out_pipe[1]);
            throw ProcessError(std::strerror(errno));
        }

        std::vector<char*> argv;
        argv.push_back(const_cast<char*>(executable.c_str()));
        for (auto const& arg : args)
        {
            argv.push_back(const_cast<char*>(arg.c_str()));
        }
        argv.push_back(nullptr);

        pid = fork();
        if (pid == -1)
        {
            close(in_pipe[0]);
            close(in_pipe[1]);
            close(out_pipe[0]);
            close(out_pipe[1]);
            close(err_pipe[0]);
            close(err_pipe[1]);
            throw ProcessError(std::strerror(errno));
        }

        if (pid == 0)
        {
            dup2(in_pipe[0], STDIN_FILENO);
            dup2(out_pipe[1], STDOUT_FILENO);

            close(in_pipe[0]);
            close(in_pipe[1]);
            close(out_pipe[0]);
            close(out_pipe[1]);
            close(err_pipe[0]);
            fcntl(err_pipe[1], F_SETFD, FD_CLOEXEC);

            execvp(executable.c_str(), argv.data());

            int const error = errno;
            ::write(err_pipe[1], &error, sizeof(error));
            close(err_pipe[1]);
            _exit(EXIT_FAILURE);
        }

        close(in_pipe[0]);
        close(out_pipe[1]);
        close(err_pipe[1]);

        int error = 0;
        ssize_t const bytes_read = ::read(err_pipe[0], &error, sizeof(error));
        close(err_pipe[0]);

        if (bytes_read > 0)
        {
            close(in_pipe[1]);
            close(out_pipe[0]);
            waitpid(pid, nullptr, 0);
            throw ProcessError(std::strerror(error));
        }

        process_id = static_cast<int>(pid);
        input_stream = std::make_shared<pipe_istream>(out_pipe[0]);
        output_stream = std::make_shared<pipe_ostream>(in_pipe[1]);
    }

    bool checkRunning()
    {
        if (pid != -1)
        {
            int const status = waitpid(pid, nullptr, WNOHANG);
            if (status != 0)
            {
                pid = -1;
                process_id = -1;
                if (input_stream)
                {
                    input_stream->close();
                }
                if (output_stream)
                {
                    output_stream->close();
                }
            }
        }
        return pid != -1;
    }

    void wait()
    {
        if (checkRunning())
        {
            if (input_stream)
            {
                input_stream->close();
            }
            if (output_stream)
            {
                output_stream->close();
            }
            waitpid(pid, nullptr, 0);
            pid = -1;
            process_id = -1;
        }
    }

    void terminate()
    {
        if (checkRunning())
        {
            if (input_stream)
            {
                input_stream->close();
            }
            if (output_stream)
            {
                output_stream->close();
            }
            kill(pid, SIGTERM);
            waitpid(pid, nullptr, 0);
            pid = -1;
            process_id = -1;
        }
    }

    static int currentProcessIdImpl()
    {
        return static_cast<int>(getpid());
    }

    static bool existsImpl(int id)
    {
        return id > 0 && kill(static_cast<pid_t>(id), 0) == 0;
    }
#elif defined(LSPCPP_PROCESS_WIN32)
    PROCESS_INFORMATION process_info {};

    static std::wstring escapeArg(std::wstring const& arg)
    {
        if (arg.find_first_of(L" \t\n\v\\\"") == std::wstring::npos)
        {
            return arg;
        }

        std::wstring escaped;
        escaped.reserve(arg.size());
        escaped += L'"';
        for (auto it = arg.cbegin();; ++it)
        {
            size_t backslashes = 0;
            while (it != arg.cend() && *it == L'\\')
            {
                ++backslashes;
                ++it;
            }
            if (it == arg.cend())
            {
                escaped.append(backslashes * 2, L'\\');
                break;
            }
            if (*it == L'"')
            {
                escaped.append(backslashes * 2 + 1, L'\\');
                escaped += *it;
            }
            else
            {
                escaped.append(backslashes, L'\\');
                escaped += *it;
            }
        }
        escaped += L'"';
        return escaped;
    }

    static std::wstring toWide(std::string const& value)
    {
        if (value.empty())
        {
            return {};
        }
        int const size = MultiByteToWideChar(CP_UTF8, 0, value.c_str(), -1, nullptr, 0);
        if (size <= 0)
        {
            throw ProcessError("Failed to convert process command line.");
        }
        std::wstring wide(static_cast<size_t>(size), L'\0');
        MultiByteToWideChar(CP_UTF8, 0, value.c_str(), -1, &wide[0], size);
        if (!wide.empty() && wide.back() == L'\0')
        {
            wide.pop_back();
        }
        return wide;
    }

    Impl(std::string const& executable, Process::ArgList const& args)
    {
        std::wstring command = escapeArg(toWide(executable));
        for (auto const& arg : args)
        {
            command.push_back(L' ');
            command += escapeArg(toWide(arg));
        }

        SECURITY_ATTRIBUTES security_attributes;
        security_attributes.nLength = sizeof(security_attributes);
        security_attributes.bInheritHandle = TRUE;
        security_attributes.lpSecurityDescriptor = nullptr;

        HANDLE stdin_read = nullptr;
        HANDLE stdin_write = nullptr;
        HANDLE stdout_read = nullptr;
        HANDLE stdout_write = nullptr;

        if (!CreatePipe(&stdin_read, &stdin_write, &security_attributes, 0))
        {
            throw ProcessError("Failed to create stdin pipe.");
        }
        if (!SetHandleInformation(stdin_write, HANDLE_FLAG_INHERIT, 0))
        {
            CloseHandle(stdin_read);
            CloseHandle(stdin_write);
            throw ProcessError("Failed to configure stdin pipe.");
        }
        if (!CreatePipe(&stdout_read, &stdout_write, &security_attributes, 0))
        {
            CloseHandle(stdin_read);
            CloseHandle(stdin_write);
            throw ProcessError("Failed to create stdout pipe.");
        }
        if (!SetHandleInformation(stdout_read, HANDLE_FLAG_INHERIT, 0))
        {
            CloseHandle(stdin_read);
            CloseHandle(stdin_write);
            CloseHandle(stdout_read);
            CloseHandle(stdout_write);
            throw ProcessError("Failed to configure stdout pipe.");
        }

        STARTUPINFOW startup_info {};
        startup_info.cb = sizeof(startup_info);
        startup_info.dwFlags = STARTF_USESTDHANDLES;
        startup_info.hStdInput = stdin_read;
        startup_info.hStdOutput = stdout_write;
        startup_info.hStdError = GetStdHandle(STD_ERROR_HANDLE);

        std::vector<wchar_t> cmd_line(command.begin(), command.end());
        cmd_line.push_back(L'\0');

        if (!CreateProcessW(
                nullptr,
                cmd_line.data(),
                nullptr,
                nullptr,
                TRUE,
                CREATE_NO_WINDOW,
                nullptr,
                nullptr,
                &startup_info,
                &process_info))
        {
            CloseHandle(stdin_read);
            CloseHandle(stdin_write);
            CloseHandle(stdout_read);
            CloseHandle(stdout_write);
            throw ProcessError("Failed to start process.");
        }

        CloseHandle(stdin_read);
        CloseHandle(stdout_write);
        CloseHandle(process_info.hThread);

        process_id = static_cast<int>(process_info.dwProcessId);
        input_stream = std::make_shared<pipe_istream>(stdout_read);
        output_stream = std::make_shared<pipe_ostream>(stdin_write);
    }

    bool checkRunning()
    {
        if (process_info.hProcess != nullptr)
        {
            DWORD exit_code = 0;
            if (GetExitCodeProcess(process_info.hProcess, &exit_code) && exit_code != STILL_ACTIVE)
            {
                CloseHandle(process_info.hProcess);
                process_info.hProcess = nullptr;
                process_id = -1;
                if (input_stream)
                {
                    input_stream->close();
                }
                if (output_stream)
                {
                    output_stream->close();
                }
            }
        }
        return process_info.hProcess != nullptr;
    }

    void wait()
    {
        if (checkRunning())
        {
            if (input_stream)
            {
                input_stream->close();
            }
            if (output_stream)
            {
                output_stream->close();
            }
            WaitForSingleObject(process_info.hProcess, INFINITE);
            CloseHandle(process_info.hProcess);
            process_info.hProcess = nullptr;
            process_id = -1;
        }
    }

    void terminate()
    {
        if (checkRunning())
        {
            if (input_stream)
            {
                input_stream->close();
            }
            if (output_stream)
            {
                output_stream->close();
            }
            TerminateProcess(process_info.hProcess, 0);
            WaitForSingleObject(process_info.hProcess, INFINITE);
            CloseHandle(process_info.hProcess);
            process_info.hProcess = nullptr;
            process_id = -1;
        }
    }

    static int currentProcessIdImpl()
    {
        return static_cast<int>(GetCurrentProcessId());
    }

    static bool existsImpl(int id)
    {
        HANDLE const handle = OpenProcess(SYNCHRONIZE, FALSE, static_cast<DWORD>(id));
        if (handle == nullptr)
        {
            return false;
        }
        CloseHandle(handle);
        return true;
    }
#else
    Impl(std::string const&, Process::ArgList const&)
    {
        throw ProcessError("Process helper is not supported on this platform.");
    }

    bool checkRunning()
    {
        return false;
    }

    void wait() {}
    void terminate() {}

    static int currentProcessIdImpl()
    {
        return -1;
    }

    static bool existsImpl(int)
    {
        return false;
    }
#endif
};

Process::Process(std::string executable, ArgList args) : impl_(new Impl(std::move(executable), args))
{
}

Process::Process(Process&& other) noexcept : impl_(std::move(other.impl_))
{
}

Process& Process::operator=(Process&& other) noexcept
{
    if (impl_ && impl_->checkRunning())
    {
        impl_->terminate();
    }
    impl_ = std::move(other.impl_);
    return *this;
}

Process::~Process()
{
    if (impl_ && impl_->checkRunning())
    {
        impl_->terminate();
    }
}

Process Process::start(std::string const& executable, ArgList args)
{
    return Process(executable, std::move(args));
}

bool Process::isRunning() const
{
    return impl_ && impl_->checkRunning();
}

void Process::wait()
{
    if (impl_)
    {
        impl_->wait();
    }
}

void Process::terminate()
{
    if (impl_)
    {
        impl_->terminate();
    }
}

int Process::id() const
{
    return impl_ ? impl_->process_id : -1;
}

int Process::currentProcessId()
{
    return Impl::currentProcessIdImpl();
}

bool Process::exists(int pid)
{
    return Impl::existsImpl(pid);
}

std::shared_ptr<istream> Process::input() const
{
#if defined(LSPCPP_PROCESS_POSIX) || defined(LSPCPP_PROCESS_WIN32)
    return impl_ ? impl_->input_stream : std::shared_ptr<istream>();
#else
    return {};
#endif
}

std::shared_ptr<ostream> Process::output() const
{
#if defined(LSPCPP_PROCESS_POSIX) || defined(LSPCPP_PROCESS_WIN32)
    return impl_ ? impl_->output_stream : std::shared_ptr<ostream>();
#else
    return {};
#endif
}

} // namespace lsp
