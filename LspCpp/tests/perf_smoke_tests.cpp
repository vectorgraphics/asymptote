#include "LibLsp/JsonRpc/RemoteEndPoint.h"
#include "LibLsp/JsonRpc/StreamMessageProducer.h"
#include "LibLsp/lsp/ProtocolJsonHandler.h"
#include "LibLsp/lsp/general/exit.h"
#include "LibLsp/lsp/general/initialize.h"
#include "LibLsp/lsp/utils.h"
#include "LibLsp/lsp/working_files.h"
#include "test_helpers.h"

#include <atomic>
#include <chrono>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace
{
using test::CollectingIssueHandler;
using test::DummyLog;
using test::Expect;
using test::MakeLspFrame;
using test::StringOStream;

int& PerfWarnings()
{
    static int warnings = 0;
    return warnings;
}

class Timer
{
public:
    Timer() : start_(std::chrono::steady_clock::now())
    {
    }

    double elapsedMs() const
    {
        auto const elapsed = std::chrono::steady_clock::now() - start_;
        return std::chrono::duration<double, std::milli>(elapsed).count();
    }

private:
    std::chrono::steady_clock::time_point start_;
};

void EmitMetric(std::string const& name, int count, double elapsed_ms)
{
    double const throughput = elapsed_ms > 0.0 ? (static_cast<double>(count) * 1000.0 / elapsed_ms) : 0.0;
    std::cout << "PERF_METRIC name=" << name << " count=" << count << " elapsed_ms=" << elapsed_ms
              << " throughput_per_sec=" << throughput << std::endl;
}

void WarnIf(bool condition, std::string const& name, std::string const& reason, double value)
{
    if (!condition)
    {
        return;
    }
    ++PerfWarnings();
    std::cout << "PERF_WARN name=" << name << " reason=\"" << reason << "\" value=" << value << std::endl;
}

bool WaitForPerfOutputContaining(std::shared_ptr<StringOStream> const& output_stream, std::string const& needle)
{
    for (int i = 0; i < 200; ++i)
    {
        if (output_stream->snapshot().find(needle) != std::string::npos)
        {
            return true;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    return false;
}

bool WaitForCount(std::atomic<int> const& value, int expected)
{
    for (int i = 0; i < 200; ++i)
    {
        if (value.load(std::memory_order_relaxed) >= expected)
        {
            return true;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    return false;
}

void PerfStreamSmallFrames()
{
    // Trend smoke for the buffered header path: many tiny frames should parse
    // quickly and report throughput without gating normal CTest.
    int const count = 10000;
    std::string input_data;
    input_data.reserve(static_cast<size_t>(count) * 80);
    for (int i = 0; i < count; ++i)
    {
        input_data += MakeLspFrame(std::string(R"({"jsonrpc":"2.0","method":"small","seq":)") + std::to_string(i) + "}");
    }

    std::istringstream input_storage(std::move(input_data));
    auto input = lsp::make_istream(input_storage);
    CollectingIssueHandler issues;
    LSPStreamMessageProducer producer(issues, input);

    int delivered = 0;
    Timer timer;
    producer.listen(
        [&](std::string&&)
        {
            ++delivered;
        });
    double const elapsed_ms = timer.elapsedMs();

    Expect(delivered == count, "perf smoke stream.small_frames must deliver every frame");
    EmitMetric("stream.small_frames", count, elapsed_ms);
    WarnIf(elapsed_ms > 1500.0, "stream.small_frames", "elapsed_ms exceeded smoke threshold", elapsed_ms);
}

void PerfStreamLargeBodies()
{
    // Trend smoke for large body handling: reports throughput while correctness
    // stays guarded by the hard-fail stream tests.
    int const count = 256;
    std::string body = R"({"jsonrpc":"2.0","method":"large","payload":")";
    body.append(8192, 'x');
    body += "\"}";

    std::string input_data;
    input_data.reserve((body.size() + 64) * static_cast<size_t>(count));
    for (int i = 0; i < count; ++i)
    {
        input_data += MakeLspFrame(body);
    }

    std::istringstream input_storage(std::move(input_data));
    auto input = lsp::make_istream(input_storage);
    CollectingIssueHandler issues;
    LSPStreamMessageProducer producer(issues, input);

    int delivered = 0;
    Timer timer;
    producer.listen(
        [&](std::string&&)
        {
            ++delivered;
        });
    double const elapsed_ms = timer.elapsedMs();

    Expect(delivered == count, "perf smoke stream.large_bodies must deliver every frame");
    EmitMetric("stream.large_bodies", count, elapsed_ms);
    WarnIf(elapsed_ms > 2000.0, "stream.large_bodies", "elapsed_ms exceeded smoke threshold", elapsed_ms);
}

void PerfRemoteEndpointDispatch()
{
    // Measures end-to-end producer-to-worker dispatch for a burst of requests;
    // slow runs emit PERF_WARN instead of failing by default.
    DummyLog log;
    auto json_handler = std::make_shared<lsp::ProtocolJsonHandler>();
    auto endpoint = std::make_shared<GenericEndpoint>(log);

    int const count = 512;
    std::string input_data;
    input_data.reserve(static_cast<size_t>(count) * 90);
    for (int i = 0; i < count; ++i)
    {
        input_data += MakeLspFrame(
            std::string(R"({"jsonrpc":"2.0","id":)") + std::to_string(10000 + i) +
            R"(,"method":"initialize","params":{}})");
    }

    std::istringstream input_storage(std::move(input_data));
    auto input_stream = lsp::make_istream(input_storage);
    auto output_stream = std::make_shared<StringOStream>();

    std::atomic<int> handled {0};
    RemoteEndPoint point(json_handler, endpoint, log, lsp::JSONStreamStyle::Standard, 4);
    point.registerHandler(
        [&](td_initialize::request const& req) -> lsp::ResponseOrError<td_initialize::response>
        {
            handled.fetch_add(1, std::memory_order_relaxed);
            td_initialize::response rsp;
            rsp.id = req.id;
            return rsp;
        });

    Timer timer;
    point.startProcessingMessages(input_stream, output_stream);
    bool const handled_every_request = WaitForCount(handled, count);
    bool const got_last_response =
        WaitForPerfOutputContaining(output_stream, "\"id\":" + std::to_string(10000 + count - 1));
    double const elapsed_ms = timer.elapsedMs();
    point.stop();

    Expect(got_last_response, "perf smoke remote.dispatch must receive the last response");
    Expect(handled_every_request, "perf smoke remote.dispatch must handle every request");
    EmitMetric("remote.dispatch", count, elapsed_ms);
    WarnIf(elapsed_ms > 2500.0, "remote.dispatch", "elapsed_ms exceeded smoke threshold", elapsed_ms);
}

double RunRemoteEndpointLargeJsonNotifications(int workers)
{
    DummyLog log;
    auto json_handler = std::make_shared<lsp::ProtocolJsonHandler>();
    auto endpoint = std::make_shared<GenericEndpoint>(log);

    int const count = 256;
    std::string body = R"({"jsonrpc":"2.0","method":"exit","params":{"payload":")";
    body.append(32768, 'x');
    body += R"("}})";

    std::string input_data;
    input_data.reserve((body.size() + 64) * static_cast<size_t>(count));
    for (int i = 0; i < count; ++i)
    {
        input_data += MakeLspFrame(body);
    }

    std::istringstream input_storage(std::move(input_data));
    auto input_stream = lsp::make_istream(input_storage);
    auto output_stream = std::make_shared<StringOStream>();

    std::atomic<int> handled {0};
    RemoteEndPoint point(json_handler, endpoint, log, lsp::JSONStreamStyle::Standard, static_cast<uint8_t>(workers));
    point.registerHandler(
        [&](Notify_Exit::notify const&)
        {
            handled.fetch_add(1, std::memory_order_relaxed);
        });

    Timer timer;
    point.startProcessingMessages(input_stream, output_stream);
    bool const handled_every_notification = WaitForCount(handled, count);
    double const elapsed_ms = timer.elapsedMs();
    point.stop();

    Expect(handled_every_notification, "perf smoke remote.large_json_notifications must handle every notification");
    EmitMetric("remote.large_json_notifications.workers" + std::to_string(workers), count, elapsed_ms);
    return elapsed_ms;
}

void PerfRemoteEndpointLargeJsonNotifications()
{
    // Exercises the full RemoteEndPoint path for large notification bodies,
    // comparing one worker against a parsing pool to track the expected parallelism.
    double const one_worker_ms = RunRemoteEndpointLargeJsonNotifications(1);
    double const four_workers_ms = RunRemoteEndpointLargeJsonNotifications(4);
    double const speedup = four_workers_ms > 0.0 ? one_worker_ms / four_workers_ms : 0.0;
    std::cout << "PERF_COMPARE name=remote.large_json_notifications speedup=" << speedup << "x"
              << " workers1_ms=" << one_worker_ms << " workers4_ms=" << four_workers_ms << std::endl;
    WarnIf(
        four_workers_ms > one_worker_ms * 1.5,
        "remote.large_json_notifications",
        "four-worker parsing slower than one-worker baseline",
        four_workers_ms);
}

void PerfWorkingFilesRangeEdits()
{
    // Measures repeated tail edits on a large file to watch line-index rebuild
    // cost without using a flaky default wall-clock assertion.
    WorkingFiles files;
    AbsolutePath path("/tmp/lspcpp-working-files-perf-smoke.cpp");

    std::string text;
    for (int i = 0; i < 20000; ++i)
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
    Expect(file->LineOffsetCountForTest() == 20001, "perf smoke WorkingFiles must build line offset index");

    int const edits = 1000;
    Timer timer;
    for (int i = 0; i < edits; ++i)
    {
        lsTextDocumentDidChangeParams change;
        change.textDocument.uri = lsDocumentUri(path);
        change.textDocument.version = optional<int>(i + 2);

        lsTextDocumentContentChangeEvent edit;
        edit.range = lsRange(lsPosition(19999, 5), lsPosition(19999, 9));
        edit.text = "tail";
        change.contentChanges.push_back(edit);
        files.OnChange(change);
    }
    double const elapsed_ms = timer.elapsedMs();

    Expect(file->LineOffsetCountForTest() == 20001, "perf smoke WorkingFiles must keep line offset index rebuilt");
    EmitMetric("working_files.range_edits", edits, elapsed_ms);
    WarnIf(elapsed_ms > 3000.0, "working_files.range_edits", "elapsed_ms exceeded smoke threshold", elapsed_ms);
}

void PerfWorkingFilesOffsetLookupComparison()
{
    // Micro-benchmark: same tail position as ranged edit start/end, comparing
    // utils::GetOffsetForPosition (scan from BOF) vs WorkingFile line_offsets.
    int const line_count = 20000;
    int const iterations = 1000;
    int const lookups_per_iteration = 2;

    std::string text;
    text.reserve(static_cast<size_t>(line_count) * 16);
    for (int i = 0; i < line_count; ++i)
    {
        text += "line ";
        text += std::to_string(i);
        text += "\n";
    }

    WorkingFiles files;
    AbsolutePath path("/tmp/lspcpp-working-files-offset-perf.cpp");
    lsTextDocumentItem open;
    open.uri = lsDocumentUri(path);
    open.languageId = "cpp";
    open.version = 1;
    open.text = text;
    auto file = files.OnOpen(open);

    lsPosition const start_pos(19999, 5);
    lsPosition const end_pos(19999, 9);

    int const uncached_start = lsp::GetOffsetForPosition(start_pos, text);
    int const uncached_end = lsp::GetOffsetForPosition(end_pos, text);
    int const cached_start = file->GetOffsetForPosition(start_pos);
    int const cached_end = file->GetOffsetForPosition(end_pos);
    Expect(uncached_start == cached_start, "perf smoke cached start offset must match uncached");
    Expect(uncached_end == cached_end, "perf smoke cached end offset must match uncached");

    int const total_lookups = iterations * lookups_per_iteration;

    Timer uncached_timer;
    for (int i = 0; i < iterations; ++i)
    {
        (void)lsp::GetOffsetForPosition(start_pos, text);
        (void)lsp::GetOffsetForPosition(end_pos, text);
    }
    double const uncached_ms = uncached_timer.elapsedMs();

    Timer cached_timer;
    for (int i = 0; i < iterations; ++i)
    {
        (void)file->GetOffsetForPosition(start_pos);
        (void)file->GetOffsetForPosition(end_pos);
    }
    double const cached_ms = cached_timer.elapsedMs();

    EmitMetric("working_files.offset_lookup.uncached", total_lookups, uncached_ms);
    EmitMetric("working_files.offset_lookup.cached", total_lookups, cached_ms);

    double const speedup = cached_ms > 0.0 ? uncached_ms / cached_ms : 0.0;
    std::cout << "PERF_COMPARE name=working_files.offset_lookup speedup=" << speedup << "x"
              << " uncached_ms=" << uncached_ms << " cached_ms=" << cached_ms << std::endl;

    Expect(speedup >= 1.0, "perf smoke cached offset lookup must not be slower than uncached scan");
    WarnIf(speedup < 5.0, "working_files.offset_lookup", "cached speedup below 5x smoke threshold", speedup);
}
} // namespace

int main(int argc, char** argv)
{
    test::InitTestFilter(argc, argv);
RUN_TEST(PerfStreamSmallFrames);
    RUN_TEST(PerfStreamLargeBodies);
    RUN_TEST(PerfRemoteEndpointDispatch);
    RUN_TEST(PerfRemoteEndpointLargeJsonNotifications);
    RUN_TEST(PerfWorkingFilesRangeEdits);
    RUN_TEST(PerfWorkingFilesOffsetLookupComparison);

#ifdef LSPCPP_PERF_WARNINGS_AS_ERRORS
    if (PerfWarnings() != 0)
    {
        return 1;
    }
#endif
    return test::Failures() == 0 ? 0 : 1;
}
