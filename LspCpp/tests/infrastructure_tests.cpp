#include "LibLsp/JsonRpc/Context.h"
#include "LibLsp/LspCpp.h"
#include "LibLsp/JsonRpc/ScopeExit.h"
#include "LibLsp/JsonRpc/lsRequestId.h"
#include "LibLsp/JsonRpc/threaded_queue.h"
#include "LibLsp/lsp/utils.h"
#include "LibLsp/lsp/working_files.h"
#include "test_helpers.h"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
#ifndef _WIN32
#include <array>
#include <unistd.h>
#endif

namespace
{
using test::Expect;

void TestEnqueueAndDequeueBasic()
{
    ThreadedQueue<int> queue;
    queue.Enqueue(1, false);
    queue.Enqueue(2, false);
    queue.Enqueue(3, false);

    Expect(queue.Dequeue() == 1, "queue must dequeue first item first");
    Expect(queue.Dequeue() == 2, "queue must dequeue second item second");
    Expect(queue.Dequeue() == 3, "queue must dequeue third item third");
    Expect(queue.IsEmpty(), "queue must be empty after dequeuing all items");
}

void TestTryDequeueOnEmptyReturnsNullopt()
{
    ThreadedQueue<int> queue;

    auto item = queue.TryDequeue(false);

    Expect(!item.has_value(), "TryDequeue on an empty queue must return nullopt");
}

void TestEnqueueAllAndTryDequeueSome()
{
    ThreadedQueue<int> queue;
    std::vector<int> input {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    queue.EnqueueAll(std::move(input), false);

    auto first_half = queue.TryDequeueSome(5);

    Expect(first_half.size() == 5, "TryDequeueSome must return requested count when available");
    Expect(queue.Size() == 5, "TryDequeueSome must leave the remaining items queued");
    for (int i = 0; i < 5; ++i)
    {
        Expect(first_half[static_cast<size_t>(i)] == i, "TryDequeueSome must preserve FIFO order");
    }
}

void TestDequeueAllReturnsEverything()
{
    ThreadedQueue<int> queue;
    std::vector<int> input {1, 2, 3, 4, 5};
    queue.EnqueueAll(std::move(input), false);

    auto all = queue.DequeueAll();

    Expect(all.size() == 5, "DequeueAll must return all queued items");
    Expect(queue.IsEmpty(), "DequeueAll must empty the queue");
}

void TestPriorityQueueOrdering()
{
    ThreadedQueue<int> queue;
    queue.Enqueue(1, false);
    queue.Enqueue(10, true);
    queue.Enqueue(2, false);
    queue.Enqueue(11, true);

    Expect(queue.Dequeue() == 10, "priority queue items must be dequeued before normal items");
    Expect(queue.Dequeue() == 11, "priority queue must preserve FIFO order within priority items");
    Expect(queue.Dequeue() == 1, "normal queue items must follow priority items");
    Expect(queue.Dequeue() == 2, "normal queue must preserve FIFO order after priority drain");
}

void TestTryDequeuePreferenceAndPriorityBoundary()
{
    ThreadedQueue<int> queue;
    queue.Enqueue(1, false);
    queue.Enqueue(10, true);
    queue.Enqueue(2, false);

    auto normal_first = queue.TryDequeue(false);
    Expect(normal_first && *normal_first == 1, "TryDequeue(false) must prefer normal queue items");

    auto remaining = queue.TryDequeueSome(10);
    Expect(remaining.size() == 2, "TryDequeueSome must return all remaining items when count is larger than size");
    Expect(remaining[0] == 10, "TryDequeueSome must drain priority items before normal items");
    Expect(remaining[1] == 2, "TryDequeueSome must drain normal items after priority items");
    Expect(queue.IsEmpty(), "TryDequeueSome must empty the queue after draining all items");
}

void TestBlockingDequeueWaitsForProducer()
{
    ThreadedQueue<int> queue;
    std::atomic<bool> consumer_started {false};
    int result = 0;

    std::thread consumer(
        [&]()
        {
            consumer_started.store(true, std::memory_order_relaxed);
            result = queue.Dequeue();
        });

    while (!consumer_started.load(std::memory_order_relaxed))
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    queue.Enqueue(42, false);
    consumer.join();

    Expect(result == 42, "blocking Dequeue must wait until a producer enqueues an item");
}

void TestMultiQueueWaiterWakesOnEnqueue()
{
    auto waiter = std::make_shared<MultiQueueWaiter>();
    ThreadedQueue<int> queue(waiter);
    std::atomic<bool> quit {false};

    std::thread producer(
        [&]()
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            queue.Enqueue(7, false);
        });

    bool const interrupted = waiter->Wait(quit, &queue);
    producer.join();

    Expect(!interrupted, "MultiQueueWaiter must wake on enqueue without reporting interrupt");
    Expect(queue.TryDequeue(false).value() == 7, "queued value must be available after waiter wakes");
}

void TestMultiQueueWaiterInterruptReturnsTrue()
{
    auto waiter = std::make_shared<MultiQueueWaiter>();
    ThreadedQueue<int> queue(waiter);
    std::atomic<bool> quit {false};

    std::thread interrupter(
        [&]()
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            quit.store(true, std::memory_order_relaxed);
            waiter->cv.notify_all();
        });

    bool const interrupted = waiter->Wait(quit, &queue);
    interrupter.join();

    Expect(interrupted, "MultiQueueWaiter must report interrupt when quit becomes true");
}

void TestMultiQueueWaiterWakesOnSecondQueue()
{
    auto waiter = std::make_shared<MultiQueueWaiter>();
    ThreadedQueue<int> first(waiter);
    ThreadedQueue<int> second(waiter);
    std::atomic<bool> quit {false};

    std::thread producer(
        [&]()
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            second.Enqueue(99, false);
        });

    bool const interrupted = waiter->Wait(quit, &first, &second);
    producer.join();

    Expect(!interrupted, "MultiQueueWaiter must wake when any watched queue receives state");
    Expect(first.IsEmpty(), "first watched queue must remain empty");
    auto value = second.TryDequeue(false);
    Expect(value && *value == 99, "second watched queue must hold the produced value");
}

void TestThreadedQueueIterateVisitsPriorityThenNormalItems()
{
    ThreadedQueue<int> queue;
    queue.Enqueue(1, false);
    queue.Enqueue(10, true);
    queue.Enqueue(2, false);
    queue.Enqueue(11, true);

    std::vector<int> visited;
    queue.Iterate(
        [&](int const& entry)
        {
            visited.push_back(entry);
        });

    Expect(visited.size() == 4, "Iterate must visit every queued item");
    if (visited.size() == 4)
    {
        Expect(visited[0] == 10, "Iterate must visit priority items before normal items");
        Expect(visited[1] == 11, "Iterate must preserve FIFO order within priority items");
        Expect(visited[2] == 1, "Iterate must visit normal items after priority items");
        Expect(visited[3] == 2, "Iterate must preserve FIFO order within normal items");
    }
    Expect(queue.Size() == 4, "Iterate must not remove queued items");
}

void TestMultiQueueWaiterWaitUntilTimesOutWhenQueuesEmpty()
{
    auto waiter = std::make_shared<MultiQueueWaiter>();
    ThreadedQueue<int> queue(waiter);

    auto const start = std::chrono::steady_clock::now();
    waiter->WaitUntil(start + std::chrono::milliseconds(50), &queue);
    auto const elapsed = std::chrono::steady_clock::now() - start;

    Expect(elapsed >= std::chrono::milliseconds(40), "WaitUntil must block until timeout when queues are empty");
    Expect(queue.IsEmpty(), "WaitUntil must not dequeue items while waiting");
}

void TestMultiQueueWaiterWaitUntilReturnsImmediatelyWhenQueueHasState()
{
    auto waiter = std::make_shared<MultiQueueWaiter>();
    ThreadedQueue<int> queue(waiter);
    queue.Enqueue(42, false);

    auto const start = std::chrono::steady_clock::now();
    waiter->WaitUntil(start + std::chrono::seconds(10), &queue);
    auto const elapsed = std::chrono::steady_clock::now() - start;

    Expect(elapsed < std::chrono::milliseconds(100), "WaitUntil must return immediately when queue has state");
    auto value = queue.TryDequeue(false);
    Expect(value && *value == 42, "WaitUntil must not consume queue items");
}

void TestContextDeriveAndGet()
{
    static lsp::Key<int> key;
    auto root = lsp::Context::empty();
    auto child = root.derive(key, 42);

    Expect(root.get(key) == nullptr, "parent context must not contain child value");
    Expect(child.get(key) != nullptr, "derived context must contain value");
    if (child.get(key))
    {
        Expect(*child.get(key) == 42, "derived context value mismatch");
    }
}

void TestWithContextRestoresOnDestruction()
{
    static lsp::Key<int> key;
    auto original = lsp::Context::current().clone();
    {
        lsp::WithContext with_context(lsp::Context::current().derive(key, 99));
        auto const* value = lsp::Context::current().get(key);
        Expect(value != nullptr && *value == 99, "WithContext must install derived context");
    }

    Expect(
        lsp::Context::current().get(key) == original.get(key),
        "WithContext must restore previous context on destruction");
}

void TestScopeExitRunsOnDestruction()
{
    bool ran = false;
    {
        auto cleanup = lsp::make_scope_exit([&]() { ran = true; });
        (void)cleanup;
    }

    Expect(ran, "scope_exit must run at scope exit");
}

void TestScopeExitReleasePreventsExecution()
{
    bool ran = false;
    {
        auto cleanup = lsp::make_scope_exit([&]() { ran = true; });
        (void)cleanup;
        cleanup.release();
    }

    Expect(!ran, "scope_exit release must prevent execution");
}

void TestToStringIntId()
{
    lsRequestId id;
    id.set(123);

    Expect(ToString(id) == "123", "integer request id ToString mismatch");
}

void TestToStringStringId()
{
    lsRequestId id;
    id.set("abc");

    Expect(ToString(id) == "abc", "string request id ToString mismatch");
}

void TestToStringNoneId()
{
    lsRequestId id;

    Expect(ToString(id).empty(), "empty request id ToString must be empty");
}

void TestPublicLogAndStreamHelpers()
{
    lsp::NullLog null_log;
    null_log.info("ignored");

    std::istringstream input("abc");
    std::ostringstream output;
    auto input_stream = lsp::make_istream(input);
    auto output_stream = lsp::make_ostream(output);

    char buffer[3] = {};
    input_stream->read(buffer, 3);
    output_stream->write(std::string(buffer, 3)).flush();

    Expect(output.str() == "abc", "public stdio stream helpers must wrap standard streams");
}

void TestStandardIStreamInterruptStopsFurtherReads()
{
    std::istringstream input("abc");
    auto input_stream = lsp::make_istream(input);

    input_stream->interrupt();

    char buffer = 0;
    auto const read = input_stream->read_some(&buffer, 1);
    Expect(read == 0, "interrupted standard istream must not read more bytes");
    Expect(input_stream->eof(), "interrupted standard istream must report eof");
    Expect(input_stream->fail(), "interrupted standard istream must report fail");
    Expect(
        input_stream->what().find("interrupted") != std::string::npos,
        "interrupted standard istream must explain interruption");

    input_stream->clear();
    auto const read_after_clear = input_stream->read_some(&buffer, 1);
    Expect(read_after_clear == 1 && buffer == 'a', "clear must make standard istream readable again");
}

void TestStandardIStreamReadSomeBulkReadsMultipleBytes()
{
    std::string const payload = "hello world payload";
    std::istringstream input(payload);
    auto input_stream = lsp::make_istream(input);

    char buffer[32] = {};
    auto const read = input_stream->read_some(buffer, static_cast<std::streamsize>(sizeof(buffer)));
    Expect(read > 1, "standard istream read_some must bulk-read multiple buffered bytes");
    Expect(
        read == static_cast<std::streamsize>(payload.size()),
        "standard istream read_some must read all available buffered bytes in one call");
    Expect(
        std::string(buffer, static_cast<size_t>(read)) == payload,
        "standard istream read_some bulk read must preserve byte order");
}

#ifndef _WIN32
class PipePair
{
public:
    PipePair()
    {
        Expect(::pipe(fds_) == 0, "pipe must succeed");
    }

    ~PipePair()
    {
        if (fds_[0] >= 0)
        {
            ::close(fds_[0]);
        }
        if (fds_[1] >= 0)
        {
            ::close(fds_[1]);
        }
    }

    int read_fd() const
    {
        return fds_[0];
    }

    void write(std::string const& data)
    {
        Expect(!data.empty(), "pipe write must receive data");
        write(data.data(), data.size());
    }

    void write(char const* data, size_t size)
    {
        Expect(size > 0, "pipe write must receive data");
        size_t offset = 0;
        while (offset < size)
        {
            ssize_t const written = ::write(fds_[1], data + offset, size - offset);
            if (written < 0 && errno == EINTR)
            {
                continue;
            }
            Expect(written > 0, "pipe write must succeed");
            if (written <= 0)
            {
                return;
            }
            offset += static_cast<size_t>(written);
        }
    }

    void close_write()
    {
        if (fds_[1] >= 0)
        {
            ::close(fds_[1]);
            fds_[1] = -1;
        }
    }

private:
    int fds_[2] = {-1, -1};
};

char PatternByte(size_t index)
{
    return static_cast<char>((index * 131 + 17) & 0xff);
}

std::string MakePattern(size_t size)
{
    std::string data(size, '\0');
    for (size_t i = 0; i < size; ++i)
    {
        data[i] = PatternByte(i);
    }
    return data;
}

void ExpectPattern(std::string const& data, size_t offset, char const* message)
{
    for (size_t i = 0; i < data.size(); ++i)
    {
        if (data[i] != PatternByte(offset + i))
        {
            Expect(false, message);
            return;
        }
    }
}

void TestStdinIStreamReadsAvailableBytes()
{
    PipePair pipe;
    pipe.write("hello");
    auto input_stream = std::make_shared<lsp::stdin_istream>(pipe.read_fd());

    std::array<char, 16> buffer {};
    auto const read = input_stream->read_some(buffer.data(), static_cast<std::streamsize>(buffer.size()));
    Expect(read == 5, "stdin istream read_some must batch-read available pipe bytes");
    Expect(std::string(buffer.data(), static_cast<size_t>(read)) == "hello", "stdin istream must preserve pipe payload");
    Expect(input_stream->good(), "stdin istream must stay good after a successful read");
}

void TestStdinIStreamGetAndReadConsumePipeData()
{
    PipePair pipe;
    pipe.write("hello");
    auto input_stream = std::make_shared<lsp::stdin_istream>(pipe.read_fd());

    Expect(input_stream->get() == 'h', "stdin istream get must read the first byte");

    std::array<char, 8> buffer {};
    input_stream->read(buffer.data(), 4);
    Expect(std::string(buffer.data(), 4) == "ello", "stdin istream read must consume the remaining bytes");
}

void TestStdinIStreamBufferedSmallReadsPreserveOrder()
{
    PipePair pipe;
    std::string const payload = "abcdefghij";
    pipe.write(payload);
    auto input_stream = std::make_shared<lsp::stdin_istream>(pipe.read_fd());

    // First byte triggers one syscall that fills the internal buffer; the rest
    // must be served from the buffer in order, without blocking.
    std::string consumed;
    for (size_t i = 0; i < payload.size(); ++i)
    {
        int const c = input_stream->get();
        Expect(c != EOF, "buffered stdin istream get must not hit EOF while data remains");
        consumed.push_back(static_cast<char>(c));
    }
    Expect(consumed == payload, "buffered stdin istream must preserve byte order across small reads");

    pipe.close_write();
    Expect(input_stream->get() == EOF, "buffered stdin istream must reach EOF after buffer and pipe drain");
    Expect(input_stream->eof(), "buffered stdin istream must report eof after draining");
}

void TestStdinIStreamMixedBufferedAndBulkReads()
{
    PipePair pipe;
    pipe.write("header\r\n\r\nbody-payload");
    auto input_stream = std::make_shared<lsp::stdin_istream>(pipe.read_fd());

    // Small read buffers the rest internally.
    std::array<char, 6> head {};
    auto const head_read = input_stream->read_some(head.data(), static_cast<std::streamsize>(head.size()));
    Expect(head_read == 6, "stdin istream small read must return requested bytes");
    Expect(std::string(head.data(), 6) == "header", "stdin istream small read content mismatch");

    // Bulk read() must drain the internal buffer before touching the fd again.
    std::array<char, 16> rest {};
    input_stream->read(rest.data(), 16);
    Expect(
        std::string(rest.data(), 16) == "\r\n\r\nbody-payload",
        "stdin istream bulk read must drain buffered bytes before reading the fd");
}

void TestStdinIStreamStressMixedReads()
{
    PipePair pipe;
    size_t const total_size = 2 * 1024 * 1024;
    std::string const payload = MakePattern(total_size);
    auto input_stream = std::make_shared<lsp::stdin_istream>(pipe.read_fd());

    std::thread writer(
        [&]()
        {
            size_t offset = 0;
            size_t const chunks[] = {1, 17, 257, 4093, 8192, 333};
            for (size_t i = 0; offset < payload.size(); ++i)
            {
                size_t const chunk = std::min(chunks[i % 6], payload.size() - offset);
                pipe.write(payload.substr(offset, chunk));
                offset += chunk;
            }
            pipe.close_write();
        });

    std::array<char, 8192> buffer {};
    size_t offset = 0;
    size_t const read_some_sizes[] = {1, 7, 64, 4096, 8192};
    while (offset < total_size)
    {
        if (offset % 9973 == 0)
        {
            int const c = input_stream->get();
            Expect(c != EOF, "stdin istream stress get must not hit EOF before payload end");
            if (c == EOF)
            {
                break;
            }
            Expect(static_cast<char>(c) == PatternByte(offset), "stdin istream stress get byte mismatch");
            ++offset;
            continue;
        }

        if (offset % 5 == 0)
        {
            auto const requested = std::min<size_t>(7000, total_size - offset);
            input_stream->read(buffer.data(), static_cast<std::streamsize>(requested));
            ExpectPattern(std::string(buffer.data(), requested), offset, "stdin istream stress read byte mismatch");
            offset += requested;
            continue;
        }

        auto const requested =
            std::min(read_some_sizes[offset % 5], std::min<size_t>(buffer.size(), total_size - offset));
        auto const read = input_stream->read_some(buffer.data(), static_cast<std::streamsize>(requested));
        Expect(read > 0, "stdin istream stress read_some must make progress before payload end");
        if (read <= 0)
        {
            break;
        }
        ExpectPattern(std::string(buffer.data(), static_cast<size_t>(read)), offset, "stdin istream stress read_some byte mismatch");
        offset += static_cast<size_t>(read);
    }

    writer.join();

    Expect(offset == total_size, "stdin istream stress must consume the whole payload");
    char tail = 0;
    Expect(input_stream->read_some(&tail, 1) == 0, "stdin istream stress must reach EOF after payload");
    Expect(input_stream->eof(), "stdin istream stress must report EOF after pipe close");
    Expect(!input_stream->fail(), "stdin istream stress must not report fail after clean EOF");
    Expect(!input_stream->bad(), "stdin istream stress must not report bad after clean EOF");
}

void TestStdinIStreamPseudoRandomLengthStress()
{
    PipePair pipe;
    size_t const total_size = 3 * 1024 * 1024 + 123;
    std::string const payload = MakePattern(total_size);
    auto input_stream = std::make_shared<lsp::stdin_istream>(pipe.read_fd());

    std::thread writer(
        [&]()
        {
            std::mt19937 rng(0xC0FFEEu);
            std::uniform_int_distribution<size_t> chunk_dist(1, 16384);
            std::uniform_int_distribution<size_t> large_chunk_dist(1, 131072);
            size_t offset = 0;
            while (offset < payload.size())
            {
                size_t chunk = chunk_dist(rng);
                if ((offset & 0xffff) == 0)
                {
                    chunk = large_chunk_dist(rng);
                }
                chunk = std::min(chunk, payload.size() - offset);
                pipe.write(payload.data() + offset, chunk);
                offset += chunk;
            }
            pipe.close_write();
        });

    std::mt19937 rng(0xBAD5EEDu);
    std::uniform_int_distribution<int> read_kind_dist(0, 10);
    std::vector<char> buffer(32768);
    std::uniform_int_distribution<size_t> read_size_dist(1, buffer.size());
    size_t offset = 0;
    while (offset < total_size)
    {
        int const choice = read_kind_dist(rng);
        if (choice == 0)
        {
            int const c = input_stream->get();
            Expect(c != EOF, "stdin istream random stress get must not hit EOF before payload end");
            if (c == EOF)
            {
                break;
            }
            Expect(static_cast<char>(c) == PatternByte(offset), "stdin istream random stress get byte mismatch");
            ++offset;
            continue;
        }

        size_t requested = read_size_dist(rng);
        requested = std::min(requested, total_size - offset);
        if (choice <= 4)
        {
            input_stream->read(buffer.data(), static_cast<std::streamsize>(requested));
            ExpectPattern(
                std::string(buffer.data(), requested),
                offset,
                "stdin istream random stress read byte mismatch");
            offset += requested;
        }
        else
        {
            auto const read = input_stream->read_some(buffer.data(), static_cast<std::streamsize>(requested));
            Expect(read > 0, "stdin istream random stress read_some must make progress before payload end");
            if (read <= 0)
            {
                break;
            }
            ExpectPattern(
                std::string(buffer.data(), static_cast<size_t>(read)),
                offset,
                "stdin istream random stress read_some byte mismatch");
            offset += static_cast<size_t>(read);
        }
    }

    writer.join();

    Expect(offset == total_size, "stdin istream random stress must consume the whole payload");
    char tail = 0;
    Expect(input_stream->read_some(&tail, 1) == 0, "stdin istream random stress must reach EOF after payload");
    Expect(input_stream->eof(), "stdin istream random stress must report EOF after pipe close");
    Expect(!input_stream->fail(), "stdin istream random stress must not report fail after clean EOF");
    Expect(!input_stream->bad(), "stdin istream random stress must not report bad after clean EOF");
}

void TestStdinIStreamEofOnClosedPipe()
{
    PipePair pipe;
    pipe.close_write();
    auto input_stream = std::make_shared<lsp::stdin_istream>(pipe.read_fd());

    char buffer = 0;
    auto const read = input_stream->read_some(&buffer, 1);
    Expect(read == 0, "stdin istream must return zero bytes when the pipe is closed");
    Expect(input_stream->eof(), "stdin istream must report eof when the pipe is closed");
    Expect(!input_stream->fail(), "stdin istream must not report fail on a clean pipe eof");
    Expect(!input_stream->bad(), "stdin istream must not report bad on a clean pipe eof");
}

void TestStdinIStreamRejectsInvalidFileDescriptor()
{
    auto input_stream = std::make_shared<lsp::stdin_istream>(-1);

    char buffer = 0;
    auto const read = input_stream->read_some(&buffer, 1);
    Expect(read == 0, "stdin istream must not read from an invalid file descriptor");
    Expect(input_stream->bad(), "stdin istream must report bad on an invalid file descriptor");
    Expect(!input_stream->fail(), "stdin istream invalid file descriptor must not look like interrupt");
    Expect(!input_stream->what().empty(), "stdin istream invalid file descriptor must explain the error");
}

void TestStdinIStreamInterruptWhileBlockedInSelect()
{
    PipePair pipe;
    auto input_stream = std::make_shared<lsp::stdin_istream>(pipe.read_fd());

    std::atomic<bool> reader_started {false};
    std::atomic<bool> reader_done {false};
    std::streamsize read_count = -1;
    char buffer = 0;

    std::thread reader(
        [&]()
        {
            reader_started.store(true, std::memory_order_release);
            read_count = input_stream->read_some(&buffer, 1);
            reader_done.store(true, std::memory_order_release);
        });

    for (int i = 0; i < 200 && !reader_started.load(std::memory_order_acquire); ++i)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    Expect(reader_started.load(std::memory_order_acquire), "stdin istream reader must start blocking in select");

    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    input_stream->interrupt();
    reader.join();

    Expect(reader_done.load(std::memory_order_acquire), "stdin istream reader must finish after interrupt");
    Expect(read_count == 0, "stdin istream must not read bytes when interrupted while blocked");
    Expect(input_stream->fail(), "stdin istream must report fail after interrupt");
    Expect(input_stream->eof(), "stdin istream must report eof after interrupt");
    Expect(
        input_stream->what().find("interrupted") != std::string::npos,
        "stdin istream must explain interruption after a blocked read");
}

void TestStdinIStreamInterruptClearAndResume()
{
    PipePair pipe;
    pipe.write("abc");
    auto input_stream = std::make_shared<lsp::stdin_istream>(pipe.read_fd());

    input_stream->interrupt();

    char buffer = 0;
    Expect(input_stream->read_some(&buffer, 1) == 0, "interrupted stdin istream must not read bytes");
    Expect(input_stream->fail(), "interrupted stdin istream must report fail");
    Expect(input_stream->eof(), "interrupted stdin istream must report eof");
    Expect(
        input_stream->what().find("interrupted") != std::string::npos,
        "interrupted stdin istream must explain interruption");

    input_stream->clear();
    auto const read_after_clear = input_stream->read_some(&buffer, 1);
    Expect(read_after_clear == 1 && buffer == 'a', "clear must make stdin istream readable again");
    Expect(input_stream->good(), "stdin istream must be good after clear and a successful read");
}

void TestStdinIStreamRepeatedInterruptWhileBlocked()
{
    int const attempts = 25;
    for (int i = 0; i < attempts; ++i)
    {
        PipePair pipe;
        auto input_stream = std::make_shared<lsp::stdin_istream>(pipe.read_fd());

        std::atomic<bool> reader_started {false};
        std::atomic<bool> reader_done {false};
        std::streamsize read_count = -1;
        char buffer = 0;

        std::thread reader(
            [&]()
            {
                reader_started.store(true, std::memory_order_release);
                read_count = input_stream->read_some(&buffer, 1);
                reader_done.store(true, std::memory_order_release);
            });

        for (int spin = 0; spin < 200 && !reader_started.load(std::memory_order_acquire); ++spin)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        Expect(reader_started.load(std::memory_order_acquire), "stdin istream repeated interrupt reader must start");

        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        input_stream->interrupt();
        reader.join();

        Expect(reader_done.load(std::memory_order_acquire), "stdin istream repeated interrupt reader must finish");
        Expect(read_count == 0, "stdin istream repeated interrupt must return zero bytes");
        Expect(input_stream->fail(), "stdin istream repeated interrupt must report fail");
        Expect(input_stream->eof(), "stdin istream repeated interrupt must report eof");
    }
}
#endif

void TestStdinStreamInterruptStopsReads()
{
#ifndef _WIN32
    PipePair pipe;
    auto input_stream = std::make_shared<lsp::stdin_istream>(pipe.read_fd());
#else
    auto input_stream = lsp::make_stdin_stream();
#endif

    input_stream->interrupt();

    char buffer = 0;
    auto const read = input_stream->read_some(&buffer, 1);
    Expect(read == 0, "interrupted stdin stream must not read bytes");
    Expect(input_stream->eof(), "interrupted stdin stream must report eof");
    Expect(input_stream->fail(), "interrupted stdin stream must report fail");
    Expect(
        input_stream->what().find("interrupted") != std::string::npos,
        "interrupted stdin stream must explain interruption");
}

void TestStderrLogWritesMessages()
{
    std::ostringstream narrow_output;
    std::wostringstream wide_output;
    auto* old_cerr = std::cerr.rdbuf(narrow_output.rdbuf());
    auto* old_wcerr = std::wcerr.rdbuf(wide_output.rdbuf());
    {
        auto restore = lsp::make_scope_exit(
            [&]()
            {
                std::cerr.rdbuf(old_cerr);
                std::wcerr.rdbuf(old_wcerr);
            });

        lsp::StderrLog log;
        log.error("narrow message");
        log.warning(L"wide message");
    }

    Expect(
        narrow_output.str().find("narrow message") != std::string::npos,
        "StderrLog must write narrow messages to std::cerr");
    Expect(
        wide_output.str().find(L"wide message") != std::wstring::npos,
        "StderrLog must write wide messages to std::wcerr");
}

void TestLanguageSessionFacadeKeepsEndpointAccessible()
{
    lsp::NullLog log;
    lsp::LanguageSession session(log);

    bool initialized = false;
    session.on(
        [&](td_initialize::request const& req)
        {
            initialized = true;
            td_initialize::response rsp;
            rsp.id = req.id;
            return rsp;
        });

    Expect(session.protocolJsonHandler() != nullptr, "LanguageSession must expose its protocol handler");
    Expect(session.localEndpoint() != nullptr, "LanguageSession must expose its local endpoint");
    session.endpoint().getNextRequestId();
    Expect(
        session.protocolJsonHandler()->GetRequestJsonHandler(td_initialize::request::kMethodInfo) != nullptr,
        "LanguageSession must register handlers on the underlying protocol handler");
    Expect(!initialized, "registering a LanguageSession handler must not invoke it immediately");
}

void TestWorkingFilesRangeChangeUsesCachedLineOffsets()
{
    WorkingFiles files;
    AbsolutePath path("/tmp/lspcpp-working-files-test.cpp");

    lsTextDocumentItem open;
    open.uri = lsDocumentUri(path);
    open.languageId = "cpp";
    open.version = 1;
    open.text = "alpha\nbeta\ngamma\n";
    auto file = files.OnOpen(open);

    lsTextDocumentDidChangeParams change;
    change.textDocument.uri = lsDocumentUri(path);
    change.textDocument.version = optional<int>(2);

    lsTextDocumentContentChangeEvent edit;
    edit.range = lsRange(lsPosition(1, 1), lsPosition(1, 3));
    edit.text = "EE";
    change.contentChanges.push_back(edit);
    files.OnChange(change);

    std::string content;
    bool const got_content = files.GetFileBufferContent(file, content);
    Expect(got_content, "WorkingFiles must return content for an open file");
    Expect(content == "alpha\nbEEa\ngamma\n", "WorkingFiles ranged change must preserve old API behavior");
}

void TestWorkingFilesMaintainsLineOffsetIndex()
{
    // Uses a large file to verify the cached line-offset index is built and
    // rebuilt after both ranged and full-content changes.
    WorkingFiles files;
    AbsolutePath path("/tmp/lspcpp-working-files-large-test.cpp");

    std::string text;
    for (int i = 0; i < 10000; ++i)
    {
        text += "line ";
        text += std::to_string(i);
        text += "\n";
    }

    lsTextDocumentItem open;
    open.uri = lsDocumentUri(path);
    open.languageId = "cpp";
    open.version = 1;
    open.text = text;
    auto file = files.OnOpen(open);

    Expect(file->LineOffsetCountForTest() == 10001, "WorkingFiles must build one cached line offset per line start");

    lsTextDocumentDidChangeParams tail_change;
    tail_change.textDocument.uri = lsDocumentUri(path);
    tail_change.textDocument.version = optional<int>(2);

    lsTextDocumentContentChangeEvent tail_edit;
    tail_edit.range = lsRange(lsPosition(9999, 5), lsPosition(9999, 9));
    tail_edit.text = "tail";
    tail_change.contentChanges.push_back(tail_edit);
    files.OnChange(tail_change);

    Expect(
        file->LineOffsetCountForTest() == 10001,
        "WorkingFiles must keep cached line offsets rebuilt after ranged changes");

    lsTextDocumentDidChangeParams full_change;
    full_change.textDocument.uri = lsDocumentUri(path);
    full_change.textDocument.version = optional<int>(3);

    lsTextDocumentContentChangeEvent full_edit;
    full_edit.text = "one\ntwo\n";
    full_change.contentChanges.push_back(full_edit);
    files.OnChange(full_change);

    Expect(file->LineOffsetCountForTest() == 3, "WorkingFiles must rebuild line offsets after full-content changes");
}

void TestAsciiPositionOffsetRoundTrip()
{
    std::string const content = "abc\ndef\nlast";
    for (int offset = 0; offset <= static_cast<int>(content.size()); ++offset)
    {
        lsPosition const position = lsp::GetPositionForOffset(static_cast<size_t>(offset), content);
        int const round_trip = lsp::GetOffsetForPosition(position, content);
        Expect(round_trip == offset, "ASCII position/offset conversion must round-trip every byte offset");
    }
}

void TestStringAndPathUtilities()
{
    Expect(lsp::StartsWith("alpha.cpp", "alpha"), "StartsWith must detect prefixes");
    Expect(lsp::EndsWith("alpha.cpp", ".cpp"), "EndsWith must detect suffixes");
    Expect(!lsp::StartsWith("alpha.cpp", "beta"), "StartsWith must reject non-matching prefixes");
    Expect(!lsp::StartsWith("short", "longer-prefix"), "StartsWith must reject prefixes longer than the value");
    Expect(!lsp::EndsWith("alpha.cpp", ".h"), "EndsWith must reject non-matching suffixes");
    Expect(!lsp::EndsWith("x", "xx"), "EndsWith must reject suffixes longer than the value");

    Expect(
        lsp::AnyStartsWith({"alpha.cpp", "beta.h"}, "alpha"),
        "AnyStartsWith must detect a matching prefix in the collection");
    Expect(
        !lsp::AnyStartsWith({"alpha.cpp", "beta.h"}, "gamma"),
        "AnyStartsWith must reject when no value matches");
    Expect(
        lsp::StartsWithAny("alpha.cpp", {"beta.h", "alpha"}),
        "StartsWithAny must detect a matching prefix in the candidate list");
    Expect(
        !lsp::StartsWithAny("alpha.cpp", {"beta", "gamma"}),
        "StartsWithAny must reject when no candidate matches");
    Expect(
        lsp::EndsWithAny("alpha.cpp", {".h", ".cpp"}),
        "EndsWithAny must detect a matching suffix in the candidate list");
    Expect(
        !lsp::EndsWithAny("alpha.cpp", {".h", ".txt"}),
        "EndsWithAny must reject when no candidate matches");

    Expect(lsp::GetDirName("/tmp/lspcpp/file.cpp") == "/tmp/lspcpp/", "GetDirName must keep trailing slash");
    Expect(lsp::GetDirName("foo") == "./", "GetDirName must return ./ for bare filenames");
    Expect(lsp::GetDirName("/foo") == "/", "GetDirName must return / for root-level absolute paths");
    Expect(lsp::GetDirName("foo/bar") == "foo/", "GetDirName must return parent directory with trailing slash");
    Expect(lsp::GetDirName("foo/") == "./", "GetDirName must treat trailing slash as directory-only path");
    Expect(lsp::GetDirName("") == "./", "GetDirName must return ./ for empty paths");
    Expect(lsp::GetDirName("/") == "/", "GetDirName must preserve the filesystem root");
    Expect(
        lsp::GetDirName("foo\\bar\\baz.cpp") == "foo/bar/",
        "GetDirName must normalize backslashes before returning the parent directory");
    Expect(
        lsp::GetDirName("foo/bar\\baz.cpp") == "foo/bar/",
        "GetDirName must honor forward slashes even when backslashes are present");

    Expect(lsp::GetBaseName("/tmp/lspcpp/file.cpp") == "file.cpp", "GetBaseName must return final path component");
    Expect(lsp::GetBaseName("file.cpp") == "file.cpp", "GetBaseName must return the whole path when no slash is present");
    Expect(lsp::GetBaseName("/tmp/") == "/tmp/", "GetBaseName must return trailing-slash paths unchanged");
    Expect(lsp::GetBaseName("foo/bar/") == "foo/bar/", "GetBaseName must return directory-only paths unchanged");

    Expect(lsp::StripFileType("/tmp/lspcpp/file.cpp") == "/tmp/lspcpp/file", "StripFileType must remove suffix");
    Expect(lsp::StripFileType("file") == "file", "StripFileType must leave extensionless paths unchanged");
    Expect(
        lsp::StripFileType("archive.tar.gz") == "archive.tar",
        "StripFileType must remove only the last dotted suffix");

    Expect(
        lsp::ReplaceAll("a-b-c", "-", "::") == "a::b::c",
        "ReplaceAll must replace every occurrence of the source token");
    Expect(
        lsp::ReplaceAll("unchanged", "missing", "replacement") == "unchanged",
        "ReplaceAll must leave the source unchanged when the pattern is absent");
    Expect(
        lsp::ReplaceAll("unchanged", "", "replacement") == "unchanged",
        "ReplaceAll must leave the source unchanged when the pattern is empty");

    std::string with_slash = "foo";
    lsp::EnsureEndsInSlash(with_slash);
    Expect(with_slash == "foo/", "EnsureEndsInSlash must append a slash when missing");

    std::string already_slashed = "foo/";
    lsp::EnsureEndsInSlash(already_slashed);
    Expect(already_slashed == "foo/", "EnsureEndsInSlash must not duplicate an existing trailing slash");

    std::string empty_path;
    lsp::EnsureEndsInSlash(empty_path);
    Expect(empty_path == "/", "EnsureEndsInSlash must append a slash to empty paths");
}

void TestUtf8PositionOffsetUtilities()
{
    std::string const content = "a\xc3\xa9\nb\xf0\x9f\x98\x80z";

    Expect(
        lsp::GetOffsetForPosition(lsPosition(0, 1), content) == 1,
        "UTF-8 GetOffsetForPosition must treat ASCII characters as one code point");
    Expect(
        lsp::GetOffsetForPosition(lsPosition(0, 2), content) == 3,
        "UTF-8 GetOffsetForPosition must advance two-byte code points as one character");
    Expect(
        lsp::GetOffsetForPosition(lsPosition(1, 3), content) == 9,
        "GetOffsetForPosition must count four-byte code points as two UTF-16 code units");
    Expect(
        lsp::GetOffsetForPosition(lsPosition(1, 4), content) == 10,
        "GetOffsetForPosition must resume single-unit counting after a surrogate pair");
    Expect(
        lsp::GetOffsetForPosition(lsPosition(1, 2), content) == 9,
        "character indexes inside a surrogate pair must snap past the whole code point");
    Expect(
        lsp::GetPositionForOffset(9, content) == lsPosition(1, 3),
        "GetPositionForOffset must count four-byte code points as two UTF-16 code units");

    for (int offset : {0, 1, 3, 4, 5, 9, 10})
    {
        lsPosition const position = lsp::GetPositionForOffset(static_cast<size_t>(offset), content);
        int const round_trip = lsp::GetOffsetForPosition(position, content);
        Expect(
            round_trip == offset,
            "UTF-8 code-point boundary offsets must round-trip through GetPositionForOffset");
    }

    lsPosition const inside_two_byte = lsp::GetPositionForOffset(2, content);
    Expect(
        inside_two_byte == lsPosition(0, 2),
        "GetPositionForOffset must count a started multi-byte code point as one character");
    Expect(
        lsp::GetOffsetForPosition(inside_two_byte, content) == 3,
        "UTF-8 offsets inside multi-byte code points must snap to the enclosing code point boundary");
    lsPosition const inside_four_byte = lsp::GetPositionForOffset(7, content);
    Expect(
        lsp::GetOffsetForPosition(inside_four_byte, content) == 9,
        "UTF-8 offsets inside four-byte code points must snap to the enclosing code point boundary");
}

void TestDirectGetOffsetForPositionCrlfAndBounds()
{
    std::string const content = "a\r\nbb\nccc";

    Expect(
        lsp::GetOffsetForPosition(lsPosition(0, 0), content) == 0,
        "direct GetOffsetForPosition must map the document start to offset zero");
    Expect(
        lsp::GetOffsetForPosition(lsPosition(0, 2), content) == 2,
        "direct GetOffsetForPosition must count carriage returns as characters on CRLF lines");
    Expect(
        lsp::GetOffsetForPosition(lsPosition(1, 1), content) == 4,
        "direct GetOffsetForPosition must locate offsets on later lines");
    Expect(
        lsp::GetOffsetForPosition(lsPosition(2, 3), content) == 9,
        "direct GetOffsetForPosition must reach the end of the final line");

    Expect(
        lsp::GetOffsetForPosition(lsPosition(20, 0), content) == static_cast<int>(content.size()),
        "direct GetOffsetForPosition must clamp out-of-range lines to the document end");
    Expect(
        lsp::GetOffsetForPosition(lsPosition(2, 99), content) == static_cast<int>(content.size()),
        "direct GetOffsetForPosition must clamp out-of-range characters to the document end");
}

void TestFileIoUtilities()
{
    std::string const temp_path = "/tmp/lspcpp-utils-file-io-test.txt";
    std::remove(temp_path.c_str());

    Expect(!lsp::FileExists(temp_path), "FileExists must return false for missing files");

    AbsolutePath const missing_path = AbsolutePath::FromNormalized("/tmp/lspcpp-utils-missing-file.txt");
    std::remove(missing_path.path().c_str());
    auto const missing_content = lsp::ReadContent(missing_path);
    Expect(!missing_content.has_value(), "ReadContent must return nullopt for missing files");

    Expect(lsp::WriteToFile(temp_path, "hello\nworld"), "WriteToFile must create and write a temp file");
    Expect(lsp::FileExists(temp_path), "FileExists must return true after writing a temp file");

    AbsolutePath const written_path = AbsolutePath::FromNormalized(temp_path);
    auto const read_back = lsp::ReadContent(written_path);
    Expect(read_back.has_value(), "ReadContent must read back written temp file content");
    Expect(read_back.value() == "hello\nworld", "ReadContent must preserve written temp file bytes");

    Expect(lsp::WriteToFile(temp_path, ""), "WriteToFile must succeed when truncating to an empty file");
    auto const empty_read = lsp::ReadContent(written_path);
    Expect(empty_read.has_value(), "ReadContent must succeed for an empty file");
    Expect(empty_read.value().empty(), "ReadContent must return empty content for an empty file");

    Expect(
        !lsp::WriteToFile("/tmp", "cannot-write-to-directory"),
        "WriteToFile must fail when the target path is not a writable file");

    std::remove(temp_path.c_str());
    Expect(!lsp::FileExists(temp_path), "FileExists must return false after deleting the temp file");
}

void TestWideStringConversionUtilities()
{
    Expect(lsp::ws2s(lsp::s2ws("ascii")) == "ascii", "ws2s/s2ws must round-trip ASCII text");
    Expect(
        lsp::ws2s(lsp::s2ws("caf\xc3\xa9")) == "caf\xc3\xa9",
        "ws2s/s2ws must round-trip UTF-8 multi-byte text");
    Expect(
        lsp::ws2s(lsp::s2ws("emoji \xf0\x9f\x98\x80")) == "emoji \xf0\x9f\x98\x80",
        "ws2s/s2ws must round-trip four-byte UTF-8 sequences");
}

void TestMiscStringUtilities()
{
    std::vector<std::string> const split = lsp::SplitString("a|b|c", "|");
    Expect(split.size() == 3, "SplitString must split on every delimiter occurrence");
    Expect(split[0] == "a" && split[1] == "b" && split[2] == "c", "SplitString must preserve segment order");

    std::vector<std::string> const multi_char_split = lsp::SplitString("a::b::c", "::");
    Expect(multi_char_split.size() == 3, "SplitString must split multi-character delimiters into segments");
    Expect(
        multi_char_split[0] == "a" && multi_char_split[1] == "b" && multi_char_split[2] == "c",
        "SplitString must consume the full multi-character delimiter");

    std::vector<std::string> const no_delimiter = lsp::SplitString("single", "::");
    Expect(no_delimiter.size() == 1, "SplitString must return one segment when the delimiter is absent");
    Expect(no_delimiter[0] == "single", "SplitString must preserve the original string when no delimiter matches");

    std::vector<std::string> const empty_delimiter = lsp::SplitString("single", "");
    Expect(empty_delimiter.size() == 1, "SplitString must not loop on empty delimiters");
    Expect(empty_delimiter[0] == "single", "SplitString empty delimiter must preserve the source string");

    Expect(
        lsp::EscapeFileName("foo/bar.c") == "foo@bar.c",
        "EscapeFileName must replace path separators with @");
    Expect(
        lsp::EscapeFileName("C:\\tmp\\file.c") == "C@@tmp@file.c",
        "EscapeFileName must replace backslashes and drive colons");
    Expect(
        lsp::EscapeFileName("dir/") == "dir",
        "EscapeFileName must drop a trailing slash before escaping");

    lsPosition const char_pos = lsp::CharPos("a\nbx", 'x', 0);
    Expect(char_pos.line == 1 && char_pos.character == 1, "CharPos must locate characters across line breaks");

    Expect(lsp::UpdateToRnNewlines("a\nb") == "a\r\nb", "UpdateToRnNewlines must convert lone LF newlines");
    Expect(
        lsp::UpdateToRnNewlines("a\r\nb") == "a\r\nb",
        "UpdateToRnNewlines must leave existing CRLF sequences unchanged");
    Expect(
        lsp::UpdateToRnNewlines("a\n\nb") == "a\r\n\r\nb",
        "UpdateToRnNewlines must convert consecutive LF-only newlines");

    Expect(lsp::FormatMicroseconds(0) == "0.00ms", "FormatMicroseconds must format zero microseconds");
    Expect(lsp::FormatMicroseconds(1500) == "1.50ms", "FormatMicroseconds must keep two fractional digits");
    Expect(lsp::FormatMicroseconds(123456) == "123.45ms", "FormatMicroseconds must scale larger values to milliseconds");

    std::string const temp_path = "/tmp/lspcpp-utils-read-lines-test.txt";
    std::remove(temp_path.c_str());
    Expect(lsp::WriteToFile(temp_path, "first\nsecond"), "WriteToFile must prepare ReadLinesWithEnding fixture");

    AbsolutePath const lines_path = AbsolutePath::FromNormalized(temp_path);
    Expect(lsp::FileExists(temp_path), "ReadLinesWithEnding fixture file must exist on disk");
    auto const raw_content = lsp::ReadContent(lines_path);
    Expect(raw_content.has_value(), "ReadContent must read the ReadLinesWithEnding fixture");
    Expect(
        raw_content.value() == "first\nsecond",
        "ReadContent must preserve the ReadLinesWithEnding fixture bytes");

    std::vector<std::string> const lines = lsp::ReadLinesWithEnding(lines_path);
    Expect(lines.size() == 3, "ReadLinesWithEnding must return a trailing empty line after the final unterminated line");
    Expect(lines[0] == "first\n", "ReadLinesWithEnding must preserve LF line endings");
    Expect(lines[1] == "second", "ReadLinesWithEnding must return the final line without a trailing newline");
    Expect(lines[2].empty(), "ReadLinesWithEnding must append an empty line once EOF is reached");

    std::string const crlf_path = "/tmp/lspcpp-utils-read-lines-crlf-test.txt";
    std::remove(crlf_path.c_str());
    Expect(lsp::WriteToFile(crlf_path, "crlf\r\n"), "WriteToFile must prepare CRLF ReadLinesWithEnding fixture");
    std::vector<std::string> const crlf_lines =
        lsp::ReadLinesWithEnding(AbsolutePath::FromNormalized(crlf_path));
    Expect(crlf_lines.size() == 2, "ReadLinesWithEnding must append a trailing empty line after CRLF content");
    Expect(crlf_lines[0] == "crlf\r\n", "ReadLinesWithEnding must preserve CRLF line endings");
    Expect(crlf_lines[1].empty(), "ReadLinesWithEnding must append an empty line once EOF is reached");

    std::remove(crlf_path.c_str());

    std::remove(temp_path.c_str());
}
} // namespace

int main(int argc, char** argv)
{
    test::InitTestFilter(argc, argv);
RUN_TEST(TestEnqueueAndDequeueBasic);
    RUN_TEST(TestTryDequeueOnEmptyReturnsNullopt);
    RUN_TEST(TestEnqueueAllAndTryDequeueSome);
    RUN_TEST(TestDequeueAllReturnsEverything);
    RUN_TEST(TestPriorityQueueOrdering);
    RUN_TEST(TestTryDequeuePreferenceAndPriorityBoundary);
    RUN_TEST(TestBlockingDequeueWaitsForProducer);
    RUN_TEST(TestMultiQueueWaiterWakesOnEnqueue);
    RUN_TEST(TestMultiQueueWaiterInterruptReturnsTrue);
    RUN_TEST(TestMultiQueueWaiterWakesOnSecondQueue);
    RUN_TEST(TestThreadedQueueIterateVisitsPriorityThenNormalItems);
    RUN_TEST(TestMultiQueueWaiterWaitUntilTimesOutWhenQueuesEmpty);
    RUN_TEST(TestMultiQueueWaiterWaitUntilReturnsImmediatelyWhenQueueHasState);
    RUN_TEST(TestContextDeriveAndGet);
    RUN_TEST(TestWithContextRestoresOnDestruction);
    RUN_TEST(TestScopeExitRunsOnDestruction);
    RUN_TEST(TestScopeExitReleasePreventsExecution);
    RUN_TEST(TestToStringIntId);
    RUN_TEST(TestToStringStringId);
    RUN_TEST(TestToStringNoneId);
    RUN_TEST(TestPublicLogAndStreamHelpers);
    RUN_TEST(TestStandardIStreamInterruptStopsFurtherReads);
    RUN_TEST(TestStandardIStreamReadSomeBulkReadsMultipleBytes);
    RUN_TEST(TestStdinStreamInterruptStopsReads);
#ifndef _WIN32
    RUN_TEST(TestStdinIStreamReadsAvailableBytes);
    RUN_TEST(TestStdinIStreamGetAndReadConsumePipeData);
    RUN_TEST(TestStdinIStreamBufferedSmallReadsPreserveOrder);
    RUN_TEST(TestStdinIStreamMixedBufferedAndBulkReads);
    RUN_TEST(TestStdinIStreamStressMixedReads);
    RUN_TEST(TestStdinIStreamPseudoRandomLengthStress);
    RUN_TEST(TestStdinIStreamEofOnClosedPipe);
    RUN_TEST(TestStdinIStreamRejectsInvalidFileDescriptor);
    RUN_TEST(TestStdinIStreamInterruptWhileBlockedInSelect);
    RUN_TEST(TestStdinIStreamInterruptClearAndResume);
    RUN_TEST(TestStdinIStreamRepeatedInterruptWhileBlocked);
#endif
    RUN_TEST(TestStderrLogWritesMessages);
    RUN_TEST(TestLanguageSessionFacadeKeepsEndpointAccessible);
    RUN_TEST(TestWorkingFilesRangeChangeUsesCachedLineOffsets);
    RUN_TEST(TestWorkingFilesMaintainsLineOffsetIndex);
    RUN_TEST(TestAsciiPositionOffsetRoundTrip);
    RUN_TEST(TestStringAndPathUtilities);
    RUN_TEST(TestUtf8PositionOffsetUtilities);
    RUN_TEST(TestDirectGetOffsetForPositionCrlfAndBounds);
    RUN_TEST(TestFileIoUtilities);
    RUN_TEST(TestWideStringConversionUtilities);
    RUN_TEST(TestMiscStringUtilities);

    return test::Failures() == 0 ? 0 : 1;
}
