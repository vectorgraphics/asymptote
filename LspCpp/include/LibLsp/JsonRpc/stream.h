#pragma once
#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#ifndef _WIN32
#include <poll.h>
#include <unistd.h>
#endif
namespace lsp
{
class stream
{
public:
    virtual ~stream() = default;
    virtual bool fail() = 0;
    virtual bool bad() = 0;
    virtual bool eof() = 0;
    virtual bool good() = 0;
    virtual void clear() = 0;
    virtual std::string what() = 0;
    virtual bool need_to_clear_the_state()
    {
        return false;
    }
    virtual void interrupt()
    {
    }

    bool operator!()
    {
        return bad();
    }
};
class istream : public stream
{
public:
    virtual int get() = 0;
    virtual ~istream() = default;
    virtual istream& read(char* str, std::streamsize count) = 0;
    virtual std::streamsize read_some(char* str, std::streamsize count)
    {
        if (count <= 0)
        {
            return 0;
        }
        int const c = get();
        if (c == EOF)
        {
            return 0;
        }
        str[0] = static_cast<char>(c);
        return 1;
    }
};
template<class T>
class base_istream : public istream
{
public:
    explicit base_istream(T& _t) : _impl(_t)
    {
    }

    int get() override
    {
        return _impl.get();
    }
    bool fail() override
    {
        return _impl.fail();
    }
    bool bad() override
    {
        return _impl.bad();
    }
    bool eof() override
    {
        return _impl.eof();
    }
    bool good() override
    {
        return _impl.good();
    }
    istream& read(char* str, std::streamsize count) override
    {
        _impl.read(str, count);
        return *this;
    }

    void clear() override
    {
        _impl.clear();
    }
    T& _impl;
};
class ostream : public stream
{
public:
    virtual ~ostream() = default;

    virtual ostream& write(std::string const&) = 0;
    virtual ostream& write(std::streamsize) = 0;
    virtual ostream& flush() = 0;
};
template<class T>
class base_ostream : public ostream
{
public:
    explicit base_ostream(T& _t) : _impl(_t)
    {
    }

    bool fail() override
    {
        return _impl.fail();
    }
    bool good() override
    {
        return _impl.good();
    }
    bool bad() override
    {
        return _impl.bad();
    }
    bool eof() override
    {
        return _impl.eof();
    }

    ostream& write(std::string const& c) override
    {
        _impl << c;
        return *this;
    }

    ostream& write(std::streamsize _s) override
    {

        _impl << std::to_string(_s);
        return *this;
    }

    ostream& flush() override
    {
        _impl.flush();
        return *this;
    }

    void clear() override
    {
        _impl.clear();
    }

protected:
    T& _impl;
};

template<class T>
class base_iostream : public istream, public ostream
{
public:
    explicit base_iostream(T& _t) : _impl(_t)
    {
    }

    int get() override
    {
        return _impl.get();
    }
    bool fail() override
    {
        return _impl.fail();
    }
    bool bad() override
    {
        return _impl.bad();
    }
    bool eof() override
    {
        return _impl.eof();
    }
    bool good() override
    {
        return _impl.good();
    }
    istream& read(char* str, std::streamsize count) override
    {
        _impl.read(str, count);
        return *this;
    }
    ostream& write(std::string const& c) override
    {
        _impl << c;
        return *this;
    }

    ostream& write(std::streamsize _s) override
    {
        _impl << std::to_string(_s);
        return *this;
    }

    ostream& flush() override
    {
        _impl.flush();
        return *this;
    }

    void clear() override
    {
        _impl.clear();
    }

protected:
    T& _impl;
};

class standard_istream : public base_istream<std::istream>
{
public:
    explicit standard_istream(std::istream& stream) : base_istream<std::istream>(stream)
    {
    }

    int get() override
    {
        if (interrupted_.load(std::memory_order_relaxed))
        {
            return EOF;
        }
        int const c = _impl.get();
        if (interrupted_.load(std::memory_order_relaxed))
        {
            return EOF;
        }
        return c;
    }

    istream& read(char* str, std::streamsize count) override
    {
        if (!interrupted_.load(std::memory_order_relaxed))
        {
            _impl.read(str, count);
        }
        return *this;
    }

    bool fail() override
    {
        return interrupted_.load(std::memory_order_relaxed) || _impl.fail();
    }

    bool bad() override
    {
        return _impl.bad();
    }

    bool eof() override
    {
        return interrupted_.load(std::memory_order_relaxed) || _impl.eof();
    }

    bool good() override
    {
        return !interrupted_.load(std::memory_order_relaxed) && _impl.good();
    }

    void clear() override
    {
        interrupted_.store(false, std::memory_order_relaxed);
        _impl.clear();
    }

    std::string what() override
    {
        if (interrupted_.load(std::memory_order_relaxed))
        {
            return "Input stream interrupted.";
        }
        return {};
    }

    std::streamsize read_some(char* str, std::streamsize count) override
    {
        if (count <= 0 || interrupted_.load(std::memory_order_relaxed))
        {
            return 0;
        }

        int const first = _impl.get();
        if (first == EOF || interrupted_.load(std::memory_order_relaxed))
        {
            return 0;
        }

        str[0] = static_cast<char>(first);
        std::streamsize total = 1;
        auto* buffer = _impl.rdbuf();
        if (buffer && total < count)
        {
            auto const available = buffer->in_avail();
            if (available > 0)
            {
                auto const to_read = std::min<std::streamsize>(available, count - total);
                total += buffer->sgetn(str + total, to_read);
            }
        }
        return total;
    }

    void interrupt() override
    {
        interrupted_.store(true, std::memory_order_relaxed);
        _impl.setstate(std::ios::eofbit);
    }

private:
    std::atomic<bool> interrupted_ {false};
};

class standard_ostream : public base_ostream<std::ostream>
{
public:
    explicit standard_ostream(std::ostream& stream) : base_ostream<std::ostream>(stream)
    {
    }

    std::string what() override
    {
        return {};
    }
};

#ifndef _WIN32
// POSIX stdin wrapper used instead of std::cin so RemoteEndPoint::stop() can
// unblock a producer thread blocked on empty stdin. std::cin does not reliably
// wake a thread already blocked inside a libc read(); poll()+read() lets
// interrupt() take effect within one timeout. This exists for shutdown
// semantics; an internal buffer keeps small reads (e.g. get()) from paying a
// poll+read syscall pair per byte. The injectable fd is for tests (pipe)
// without touching the process stdin.
class stdin_istream : public istream
{
public:
    explicit stdin_istream(int fd = STDIN_FILENO) : fd_(fd)
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

        // Serve buffered bytes first so small reads (e.g. get()) do not pay a
        // poll+read syscall pair per byte.
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
            what_ = "Invalid input file descriptor.";
            return 0;
        }

        while (!interrupted_.load(std::memory_order_relaxed))
        {
            pollfd read_fd;
            read_fd.fd = fd_;
            read_fd.events = POLLIN;
            read_fd.revents = 0;

            // Short timeout so interrupt() is observed even when no input is pending.
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
                what_ = "Invalid input file descriptor.";
                return 0;
            }
            if (read_fd.revents & POLLERR)
            {
                bad_ = true;
                what_ = "Input file descriptor error.";
                return 0;
            }
            if ((read_fd.revents & (POLLIN | POLLHUP)) == 0)
            {
                continue;
            }

            if (interrupted_.load(std::memory_order_relaxed))
            {
                break;
            }

            // Large requests read straight into the caller's buffer to avoid a
            // copy; small requests fill the internal buffer so subsequent small
            // reads are served without further syscalls.
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
            return "Input stream interrupted.";
        }
        return what_;
    }

    void interrupt() override
    {
        interrupted_.store(true, std::memory_order_relaxed);
    }

private:
    static constexpr std::streamsize kBufferCapacity = 4096;

    int fd_;
    std::atomic<bool> interrupted_ {false};
    bool eof_ = false;
    bool bad_ = false;
    std::string what_;
    // Owned by the reader thread; interrupt() never touches it.
    std::string buffer_;
    size_t buffer_pos_ = 0;
};
#endif

inline std::shared_ptr<istream> make_istream(std::istream& stream)
{
    return std::make_shared<standard_istream>(stream);
}

inline std::shared_ptr<ostream> make_ostream(std::ostream& stream)
{
    return std::make_shared<standard_ostream>(stream);
}

inline std::shared_ptr<istream> make_stdin_stream()
{
#ifndef _WIN32
    // Use stdin_istream on POSIX for interruptible shutdown; Windows falls back
    // to standard_istream(std::cin), which sets eof on interrupt().
    return std::make_shared<stdin_istream>();
#else
    return make_istream(std::cin);
#endif
}

inline std::shared_ptr<ostream> make_stdout_stream()
{
    return make_ostream(std::cout);
}
} // namespace lsp
