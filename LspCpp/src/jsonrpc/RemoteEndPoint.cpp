#include "LibLsp/JsonRpc/MessageJsonHandler.h"
#include "LibLsp/JsonRpc/Endpoint.h"
#include "LibLsp/JsonRpc/message.h"
#include "LibLsp/JsonRpc/RemoteEndPoint.h"
#include <future>
#include <functional>
#include "LibLsp/JsonRpc/Cancellation.h"
#include "LibLsp/JsonRpc/RequestCancellation.h"
#include "LibLsp/JsonRpc/StreamMessageProducer.h"
#include "LibLsp/JsonRpc/NotificationInMessage.h"
#include "LibLsp/JsonRpc/lsResponseMessage.h"
#include "LibLsp/JsonRpc/Condition.h"
#include "LibLsp/JsonRpc/Context.h"
#include "rapidjson/error/en.h"
#include "LibLsp/JsonRpc/json.h"
#include "LibLsp/JsonRpc/ScopeExit.h"
#include "LibLsp/JsonRpc/stream.h"
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <map>
#include <optional>
#include <set>
#include <thread>
#include <vector>
#ifdef LSPCPP_USE_STANDALONE_ASIO
#include <asio/thread_pool.hpp>
#include <asio/post.hpp>
#else
#include <boost/asio/thread_pool.hpp>
#include <boost/asio/post.hpp>
namespace asio = boost::asio;
#endif

#include "LibLsp/JsonRpc/GCThreadContext.h"

namespace lsp
{

// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===//

static Key<std::shared_ptr<lsp::detail::SessionOutputState>> g_sessionOutputStateKey;
} // namespace lsp

using namespace lsp;
using PendingResponseHandler = std::function<bool(std::unique_ptr<LspMessage>&)>;

class PendingRequestInfo
{
public:
    PendingRequestInfo(std::string const& md, PendingResponseHandler const& callback, bool deferred_callback);
    PendingRequestInfo(std::string const& md);
    PendingRequestInfo()
    {
    }
    std::string method;
    PendingResponseHandler futureInfo;
    bool complete(std::unique_ptr<LspMessage>& msg)
    {
        if (completed.exchange(true, std::memory_order_acq_rel))
        {
            return true;
        }
        return futureInfo ? futureInfo(msg) : false;
    }
    bool shouldDeferCallback() const
    {
        return deferred_callback;
    }

private:
    std::atomic<bool> completed {false};
    bool deferred_callback = true;
};

PendingRequestInfo::PendingRequestInfo(
    std::string const& _md,
    PendingResponseHandler const& callback,
    bool deferred
)
    : method(_md), futureInfo(callback), deferred_callback(deferred)
{
}

PendingRequestInfo::PendingRequestInfo(std::string const& md) : method(md)
{
}

struct RemoteEndPoint::ParsedMessage
{
    bool ok = false;
    bool outgoing = false;
    LspMessage::Kind kind = LspMessage::NOTIFICATION_MESSAGE;
    std::unique_ptr<LspMessage> message;
    std::shared_ptr<PendingRequestInfo> pending;
};

struct RequestCancellationRegistry
{
    using CancelKey = std::pair<lsRequestId, unsigned>;

    std::mutex mutex;
    std::map<lsRequestId, std::pair<Canceler, /*Cookie*/ unsigned>> requestCancelers;
    std::map<lsRequestId, uint64_t> pendingCancelRequests;
    std::deque<std::pair<lsRequestId, uint64_t>> pendingCancelOrder;
    std::map<lsRequestId, uint64_t> seenRequestIds;
    std::deque<std::pair<lsRequestId, uint64_t>> seenRequestOrder;
    std::map<CancelKey, size_t> retainedRequestCancelers;

    void recordPendingCancel(lsRequestId const& id, uint64_t sequence, size_t limit)
    {
        pendingCancelRequests[id] = sequence;
        pendingCancelOrder.emplace_back(id, sequence);
        prunePendingCancels(limit);
    }

    void recordSeenRequest(lsRequestId const& id, uint64_t sequence, size_t limit)
    {
        seenRequestIds[id] = sequence;
        seenRequestOrder.emplace_back(id, sequence);
        pruneSeenRequests(limit);
    }

    void prunePendingCancels(size_t limit)
    {
        if (limit == 0)
        {
            return;
        }
        while (pendingCancelRequests.size() > limit && !pendingCancelOrder.empty())
        {
            auto const oldest = pendingCancelOrder.front();
            pendingCancelOrder.pop_front();
            auto const it = pendingCancelRequests.find(oldest.first);
            if (it != pendingCancelRequests.end() && it->second == oldest.second)
            {
                pendingCancelRequests.erase(it);
            }
        }
    }

    void pruneSeenRequests(size_t limit)
    {
        if (limit == 0)
        {
            return;
        }
        while (seenRequestIds.size() > limit && !seenRequestOrder.empty())
        {
            auto const oldest = seenRequestOrder.front();
            seenRequestOrder.pop_front();
            auto const it = seenRequestIds.find(oldest.first);
            if (it != seenRequestIds.end() && it->second == oldest.second)
            {
                seenRequestIds.erase(it);
            }
        }
    }
};

struct RemoteEndPoint::Data
{
    explicit Data(
        lsp::JSONStreamStyle style,
        uint8_t workers,
        RemoteEndPointLimits configured_limits,
        lsp::Log& _log,
        RemoteEndPoint* owner
    )
        : max_workers(workers),
          limits(configured_limits),
          m_id(0),
          next_request_cookie(0),
          owner(owner),
          log(_log)
    {
        if (style == lsp::JSONStreamStyle::Standard)
        {
            message_producer = (new LSPStreamMessageProducer(*owner));
        }
        else
        {
            message_producer = (new DelimitedStreamMessageProducer(*owner));
        }
        message_producer->setMaxFrameSize(limits.max_frame_size);
        message_producer->setOverloadHandler([this](std::string const& message) { return handleOverload(message); });
    }
    ~Data()
    {
        stopOrderedDispatcher();
        stopAsyncCompletionLoop();
        stopWorkerPools();
        parse_pool.reset();
        handler_pool.reset();
        delete message_producer;
    }
    uint8_t max_workers;
    RemoteEndPointLimits limits;
    std::atomic<int> m_id;
    std::shared_ptr<asio::thread_pool> parse_pool;
    std::shared_ptr<asio::thread_pool> handler_pool;
    // Method calls may be cancelled by ID, so keep track of their state.
    // Async handlers may outlive the endpoint, so cancellation bookkeeping is
    // kept in a shared registry that retained async completions can release
    // safely after RemoteEndPoint destruction.
    std::shared_ptr<RequestCancellationRegistry> cancellation_registry {std::make_shared<RequestCancellationRegistry>()};
    std::shared_ptr<lsp::detail::SessionOutputState> session_output_state {
        std::make_shared<lsp::detail::SessionOutputState>()};

    std::atomic<unsigned> next_request_cookie; // To disambiguate reused IDs, see below.
    std::atomic<uint64_t> next_message_sequence {0};
    uint64_t nextMessageSequence()
    {
        return next_message_sequence.fetch_add(1, std::memory_order_relaxed);
    }

    bool handleOverload(std::string const& message)
    {
        std::string info = "RemoteEndPoint overload: ";
        info += message;
        log.log(Log::Level::WARNING, info);
        if (limits.overload_policy == RemoteEndPointOverloadPolicy::DropNewest)
        {
            return true;
        }
        owner->stop();
        return false;
    }

    bool acquireParseQueueSlot(uint64_t sequence);

    void releaseParseQueueSlot()
    {
        parse_queue_size.fetch_sub(1, std::memory_order_relaxed);
    }

    void onCancel(Notify_Cancellation::notify* notify, uint64_t sequence)
    {
        std::lock_guard<std::mutex> lock(cancellation_registry->mutex);
        auto const it = cancellation_registry->requestCancelers.find(notify->params.id);
        if (it != cancellation_registry->requestCancelers.end())
        {
            it->second.first(); // Invoke the canceler.
            return;
        }
        auto pending = cancellation_registry->pendingCancelRequests.find(notify->params.id);
        if (pending == cancellation_registry->pendingCancelRequests.end() || pending->second < sequence)
        {
            cancellation_registry->recordPendingCancel(
                notify->params.id,
                sequence,
                limits.max_pending_cancel_requests
            );
        }
    }

    // We run cancelable requests in a context that does two things:
    //  - allows cancellation using requestCancelers[ID]
    //  - cleans up the entry in requestCancelers when it's no longer needed
    // If a client reuses an ID, the last wins and the first cannot be canceled.
    Context cancelableRequestContext(lsRequestId id, uint64_t sequence)
    {
        auto task = cancelableTask(
            id,
            /*Reason=*/static_cast<int>(lsErrorCodes::RequestCancelled)
        );
        unsigned cookie;
        bool should_cancel = false;
        Canceler canceler_to_invoke;
        auto registry = cancellation_registry;
        {
            std::lock_guard<std::mutex> lock(registry->mutex);
            cookie = next_request_cookie.fetch_add(1, std::memory_order_relaxed);
            registry->requestCancelers[id] = {std::move(task.second), cookie};
            auto const pending = registry->pendingCancelRequests.find(id);
            if (pending != registry->pendingCancelRequests.end())
            {
                // If this id was already used, a pending cancel with an older
                // sequence belongs to a completed request and must not cancel
                // the reused id.
                should_cancel = registry->seenRequestIds.find(id) == registry->seenRequestIds.end()
                    || pending->second > sequence;
                registry->pendingCancelRequests.erase(pending);
            }
            registry->recordSeenRequest(id, sequence, limits.max_seen_request_ids);
            if (should_cancel)
            {
                canceler_to_invoke = registry->requestCancelers[id].first;
            }
        }
        if (canceler_to_invoke)
        {
            canceler_to_invoke();
        }
        // When the request ends, we can clean up the entry we just added.
        // The cookie lets us check that it hasn't been overwritten due to ID
        // reuse.
        return task.first.derive(lsp::make_scope_exit(
            [registry, id, cookie]
            {
                std::lock_guard<std::mutex> lock(registry->mutex);
                RequestCancellationRegistry::CancelKey key {id, cookie};
                if (registry->retainedRequestCancelers.find(key) != registry->retainedRequestCancelers.end())
                {
                    return;
                }

                auto const& it = registry->requestCancelers.find(id);
                if (it != registry->requestCancelers.end() && it->second.second == cookie)
                {
                    registry->requestCancelers.erase(it);
                }
            }
        ));
    }

    std::shared_ptr<void> retainRequestCancellation(lsRequestId const& id)
    {
        auto registry = cancellation_registry;
        RequestCancellationRegistry::CancelKey key;
        {
            std::lock_guard<std::mutex> lock(registry->mutex);
            auto const it = registry->requestCancelers.find(id);
            if (it == registry->requestCancelers.end())
            {
                return {};
            }
            key = {id, it->second.second};
            ++registry->retainedRequestCancelers[key];
        }

        return std::shared_ptr<void>(
            new char(0),
            [registry, key](void* ptr)
            {
                delete static_cast<char*>(ptr);
                std::lock_guard<std::mutex> lock(registry->mutex);
                auto retained = registry->retainedRequestCancelers.find(key);
                if (retained != registry->retainedRequestCancelers.end())
                {
                    if (--retained->second != 0)
                    {
                        return;
                    }
                    registry->retainedRequestCancelers.erase(retained);
                }

                auto request = registry->requestCancelers.find(key.first);
                if (request != registry->requestCancelers.end() && request->second.second == key.second)
                {
                    registry->requestCancelers.erase(request);
                }
            }
        );
    }

    std::map<lsRequestId, std::shared_ptr<PendingRequestInfo>> _client_request_futures;
    StreamMessageProducer* message_producer;
    std::atomic<bool> quit {};
    RemoteEndPoint* owner;
    lsp::Log& log;
    std::shared_ptr<lsp::istream> input;
    std::shared_ptr<lsp::ostream> output;

    mutable std::mutex lifecycle_mutex;
    bool working = false;
    uint64_t next_session_generation = 0;

    std::mutex m_requestInfo;
    std::mutex async_completion_mutex;
    std::condition_variable async_completion_cv;
    std::deque<std::function<bool()>> async_completions;
    std::thread async_completion_thread;
    bool async_completion_stop = false;
    std::atomic<size_t> parse_queue_size {0};

    struct ParkedRequest
    {
        uint64_t gate = 0;
        uint64_t sequence = 0;
        std::unique_ptr<LspMessage> message;
        std::shared_ptr<lsp::detail::SessionOutputState> response_state;
        std::shared_ptr<PendingRequestInfo> pending;
    };

    struct DispatchItem
    {
        uint64_t sequence = 0;
        std::unique_ptr<LspMessage> message;
        std::shared_ptr<lsp::detail::SessionOutputState> response_state;
        std::shared_ptr<PendingRequestInfo> pending;
    };

    std::mutex ordered_mutex;
    std::condition_variable ordered_cv;
    std::deque<DispatchItem> notification_queue;
    std::deque<ParkedRequest> parked_requests;
    std::thread notification_thread;
    bool notification_stop = false;
    bool has_last_ordered_notification = false;
    uint64_t last_ordered_notification_seq = 0;
    bool has_completed_notification = false;
    uint64_t completed_notification_seq = 0;

    struct SequencedParsedMessage
    {
        SequencedParsedMessage(uint64_t sequence, RemoteEndPoint::ParsedMessage&& parsed)
            : sequence(sequence), parsed(std::move(parsed))
        {
        }

        uint64_t sequence = 0;
        RemoteEndPoint::ParsedMessage parsed;
    };

    std::mutex route_mutex;
    std::mutex route_dispatch_mutex;
    std::map<uint64_t, RemoteEndPoint::ParsedMessage> reorder_buffer;
    uint64_t next_route_sequence = 0;
    std::set<std::string> concurrent_notifications;

    void resetMessageRouting()
    {
        std::lock_guard<std::mutex> dispatch_lock(route_dispatch_mutex);
        std::lock_guard<std::mutex> lock(route_mutex);
        reorder_buffer.clear();
        next_route_sequence = next_message_sequence.load(std::memory_order_relaxed);
    }

    void clearMessageRouting()
    {
        std::lock_guard<std::mutex> dispatch_lock(route_dispatch_mutex);
        std::lock_guard<std::mutex> lock(route_mutex);
        reorder_buffer.clear();
    }

    void submitParsedMessage(
        uint64_t sequence,
        RemoteEndPoint::ParsedMessage&& parsed,
        bool bypass_reorder_limit = false
    )
    {
        std::vector<SequencedParsedMessage> ready;
        bool overloaded = false;
        std::string overload_message;
        std::unique_lock<std::mutex> dispatch_lock(route_dispatch_mutex);
        {
            std::lock_guard<std::mutex> lock(route_mutex);
            if (quit.load(std::memory_order_relaxed))
            {
                return;
            }

            if (sequence == next_route_sequence)
            {
                ready.emplace_back(next_route_sequence, std::move(parsed));
                ++next_route_sequence;
            }
            else
            {
                if (!bypass_reorder_limit && limits.max_reorder_buffer_size != 0 &&
                    reorder_buffer.size() >= limits.max_reorder_buffer_size)
                {
                    overload_message = "reorder buffer size would exceed configured maximum ";
                    overload_message += std::to_string(limits.max_reorder_buffer_size);
                    overload_message += ".";
                    if (limits.overload_policy == RemoteEndPointOverloadPolicy::DropNewest)
                    {
                        parsed = RemoteEndPoint::ParsedMessage();
                        bypass_reorder_limit = true;
                    }
                    else
                    {
                        overloaded = true;
                    }
                }
                if (!overloaded)
                {
                    reorder_buffer.emplace(sequence, std::move(parsed));
                }
            }

            if (!overloaded)
            {
                for (;;)
                {
                    auto it = reorder_buffer.find(next_route_sequence);
                    if (it == reorder_buffer.end())
                    {
                        break;
                    }
                    ready.emplace_back(next_route_sequence, std::move(it->second));
                    reorder_buffer.erase(it);
                    ++next_route_sequence;
                }
            }
        }

        if (overloaded)
        {
            dispatch_lock.unlock();
            handleOverload(overload_message);
            return;
        }

        for (auto& item : ready)
        {
            owner->routeParsedIncoming(std::move(item.parsed), item.sequence);
        }
    }

    void postParseTask(std::string&& content, uint64_t sequence)
    {
        if (quit.load(std::memory_order_relaxed) || !parse_pool)
        {
            return;
        }
        if (!acquireParseQueueSlot(sequence))
        {
            return;
        }

        asio::post(
            *parse_pool,
            [owner = owner, content = std::move(content), sequence]() mutable
            {
#ifdef LSPCPP_USEGC
                GCThreadContext gcContext;
#endif
                auto release_parse_slot = lsp::make_scope_exit(
                    [owner]
                    {
                        if (owner->d_ptr)
                        {
                            owner->d_ptr->releaseParseQueueSlot();
                        }
                    }
                );
                auto parsed = owner->parseAndClassify(content);
                if (owner->d_ptr)
                {
                    owner->d_ptr->submitParsedMessage(sequence, std::move(parsed));
                }
            }
        );
    }

    void allowConcurrentNotification(std::string const& method)
    {
        std::lock_guard<std::mutex> lock(route_mutex);
        concurrent_notifications.insert(method);
    }

    bool allowsConcurrentNotification(char const* method)
    {
        if (method == nullptr)
        {
            return false;
        }
        std::lock_guard<std::mutex> lock(route_mutex);
        return concurrent_notifications.find(method) != concurrent_notifications.end();
    }

    bool notificationGateSatisfied(uint64_t gate) const
    {
        return has_completed_notification && completed_notification_seq >= gate;
    }

    void postToWorker(
        std::unique_ptr<LspMessage> msg,
        uint64_t sequence,
        std::shared_ptr<lsp::detail::SessionOutputState> response_state,
        std::shared_ptr<PendingRequestInfo> pending = {}
    )
    {
        if (!msg || quit.load(std::memory_order_relaxed) || !handler_pool)
        {
            return;
        }
        asio::post(
            *handler_pool,
            [owner = owner,
             msg = std::move(msg),
             sequence,
             response_state = std::move(response_state),
             pending = std::move(pending)]() mutable
            {
#ifdef LSPCPP_USEGC
                GCThreadContext gcContext;
#endif
                owner->mainLoopCatching(std::move(msg), sequence, std::move(response_state), std::move(pending));
            }
        );
    }

    void completePendingResponse(std::unique_ptr<LspMessage> msg, std::shared_ptr<PendingRequestInfo> pending)
    {
        if (!msg || !pending)
        {
            return;
        }
        auto const id = static_cast<ResponseInMessage*>(msg.get())->id;
        if (!removeRequestInfo(id, pending))
        {
            return;
        }

        if (!pending->shouldDeferCallback())
        {
            try
            {
                pending->complete(msg);
            }
            catch (...)
            {
            }
            return;
        }

        if (quit.load(std::memory_order_relaxed) || !handler_pool)
        {
            return;
        }
        asio::post(
            *handler_pool,
            [pending = std::move(pending), msg = std::move(msg)]() mutable
            {
                try
                {
                    pending->complete(msg);
                }
                catch (...)
                {
                }
            }
        );
    }

    void releaseParkedRequestsLocked(std::vector<ParkedRequest>& ready)
    {
        auto it = parked_requests.begin();
        while (it != parked_requests.end())
        {
            if (notificationGateSatisfied(it->gate))
            {
                ready.emplace_back(std::move(*it));
                it = parked_requests.erase(it);
            }
            else
            {
                ++it;
            }
        }
    }

    void postReadyParkedRequests(std::vector<ParkedRequest>& ready)
    {
        for (auto& request : ready)
        {
            postToWorker(
                std::move(request.message),
                request.sequence,
                std::move(request.response_state),
                std::move(request.pending)
            );
        }
        ready.clear();
    }

    void startOrderedDispatcher()
    {
        if (notification_thread.joinable() && notification_thread.get_id() != std::this_thread::get_id())
        {
            notification_thread.join();
        }
        std::lock_guard<std::mutex> lock(ordered_mutex);
        notification_stop = false;
        has_last_ordered_notification = false;
        last_ordered_notification_seq = 0;
        has_completed_notification = false;
        completed_notification_seq = 0;
        notification_queue.clear();
        parked_requests.clear();
        if (notification_thread.joinable())
        {
            return;
        }
        notification_thread = std::thread([this] { notificationLoop(); });
    }

    void stopOrderedDispatcher()
    {
        {
            std::lock_guard<std::mutex> lock(ordered_mutex);
            notification_stop = true;
            notification_queue.clear();
            parked_requests.clear();
        }
        ordered_cv.notify_all();
        if (notification_thread.joinable())
        {
            if (notification_thread.get_id() == std::this_thread::get_id())
            {
                notification_thread.detach();
                return;
            }
            else
            {
                notification_thread.join();
            }
        }
    }

    void enqueueNotification(
        std::unique_ptr<LspMessage> msg,
        uint64_t sequence,
        std::shared_ptr<lsp::detail::SessionOutputState> response_state
    )
    {
        if (!msg || quit.load(std::memory_order_relaxed))
        {
            return;
        }
        bool overloaded = false;
        std::string overload_message;
        {
            std::lock_guard<std::mutex> lock(ordered_mutex);
            if (notification_stop || quit.load(std::memory_order_relaxed))
            {
                return;
            }
            if (limits.max_notification_queue_size != 0 &&
                notification_queue.size() >= limits.max_notification_queue_size)
            {
                overload_message = "notification queue size would exceed configured maximum ";
                overload_message += std::to_string(limits.max_notification_queue_size);
                overload_message += ".";
                overloaded = true;
            }
            else
            {
                has_last_ordered_notification = true;
                last_ordered_notification_seq = sequence;
                DispatchItem item;
                item.sequence = sequence;
                item.message = std::move(msg);
                item.response_state = std::move(response_state);
                notification_queue.emplace_back(std::move(item));
            }
        }
        if (overloaded)
        {
            handleOverload(overload_message);
            return;
        }
        ordered_cv.notify_one();
    }

    void enqueueRequest(
        std::unique_ptr<LspMessage> msg,
        uint64_t sequence,
        std::shared_ptr<lsp::detail::SessionOutputState> response_state
    )
    {
        if (!msg || quit.load(std::memory_order_relaxed))
        {
            return;
        }

        bool post_now = true;
        bool overloaded = false;
        std::string overload_message;
        {
            std::lock_guard<std::mutex> lock(ordered_mutex);
            if (!notification_stop && has_last_ordered_notification &&
                !notificationGateSatisfied(last_ordered_notification_seq))
            {
                post_now = false;
                if (limits.max_parked_request_queue_size != 0 &&
                    parked_requests.size() >= limits.max_parked_request_queue_size)
                {
                    overload_message = "parked request queue size would exceed configured maximum ";
                    overload_message += std::to_string(limits.max_parked_request_queue_size);
                    overload_message += ".";
                    overloaded = true;
                }
                else
                {
                    ParkedRequest parked;
                    parked.gate = last_ordered_notification_seq;
                    parked.sequence = sequence;
                    parked.message = std::move(msg);
                    parked.response_state = std::move(response_state);
                    parked_requests.emplace_back(std::move(parked));
                }
            }
        }

        if (overloaded)
        {
            handleOverload(overload_message);
            return;
        }
        if (post_now)
        {
            postToWorker(std::move(msg), sequence, std::move(response_state));
        }
    }

    void notificationLoop()
    {
        for (;;)
        {
            DispatchItem item;
            {
                std::unique_lock<std::mutex> lock(ordered_mutex);
                ordered_cv.wait(lock, [&] { return notification_stop || !notification_queue.empty(); });
                if (notification_stop && notification_queue.empty())
                {
                    return;
                }
                item = std::move(notification_queue.front());
                notification_queue.pop_front();
            }

            try
            {
                owner->mainLoopCatching(
                    std::move(item.message),
                    item.sequence,
                    std::move(item.response_state),
                    std::move(item.pending)
                );
            }
            catch (...)
            {
            }

            std::vector<ParkedRequest> ready;
            {
                std::lock_guard<std::mutex> lock(ordered_mutex);
                has_completed_notification = true;
                completed_notification_seq = item.sequence;
                releaseParkedRequestsLocked(ready);
            }
            postReadyParkedRequests(ready);
        }
    }

    std::shared_ptr<PendingRequestInfo> pendingRequest(
        RequestInMessage& info,
        PendingResponseHandler&& handler,
        bool deferred_callback
    )
    {
        std::lock_guard<std::mutex> lock(m_requestInfo);
        if (limits.max_pending_outgoing_requests != 0 &&
            _client_request_futures.size() >= limits.max_pending_outgoing_requests)
        {
            return {};
        }
        if (!info.id.has_value())
        {
            auto id = getNextRequestId();
            info.id.set(id);
        }
        else
        {
            if (_client_request_futures.find(info.id) != _client_request_futures.end())
            {
                return {};
            }
        }
        auto pending = std::make_shared<PendingRequestInfo>(info.method, handler, deferred_callback);
        _client_request_futures[info.id] = pending;
        return pending;
    }
    std::shared_ptr<PendingRequestInfo> const getRequestInfo(lsRequestId const& _id)
    {
        std::lock_guard<std::mutex> lock(m_requestInfo);
        auto findIt = _client_request_futures.find(_id);
        if (findIt != _client_request_futures.end())
        {
            return findIt->second;
        }
        return nullptr;
    }

    void removeRequestInfo(lsRequestId const& _id)
    {
        std::lock_guard<std::mutex> lock(m_requestInfo);
        auto findIt = _client_request_futures.find(_id);
        if (findIt != _client_request_futures.end())
        {
            _client_request_futures.erase(findIt);
        }
    }
    bool removeRequestInfo(lsRequestId const& id, std::shared_ptr<PendingRequestInfo> const& expected)
    {
        std::lock_guard<std::mutex> lock(m_requestInfo);
        auto findIt = _client_request_futures.find(id);
        if (findIt == _client_request_futures.end() || findIt->second != expected)
        {
            return false;
        }
        _client_request_futures.erase(findIt);
        return true;
    }
    void failPendingRequests()
    {
        std::vector<std::pair<lsRequestId, std::shared_ptr<PendingRequestInfo>>> pending;
        {
            std::lock_guard<std::mutex> lock(m_requestInfo);
            pending.reserve(_client_request_futures.size());
            for (auto& it : _client_request_futures)
            {
                pending.emplace_back(it.first, it.second);
            }
            _client_request_futures.clear();
        }

        for (auto& it : pending)
        {
            std::unique_ptr<LspMessage> msg(new Rsp_Error());
            auto error = static_cast<Rsp_Error*>(msg.get());
            error->id = it.first;
            error->error.code = lsErrorCodes::InternalError;
            error->error.message = "Remote endpoint stopped.";
            try
            {
                it.second->complete(msg);
            }
            catch (...)
            {
            }
        }
    }
    void asyncCompletionLoop()
    {
        // Number of consecutive completions that were still pending. Once it
        // reaches the queue size, a full pass made no progress and the loop
        // must back off instead of busy-polling the same unready futures.
        size_t pending_streak = 0;
        for (;;)
        {
            std::function<bool()> completion;
            {
                std::unique_lock<std::mutex> lock(async_completion_mutex);
                async_completion_cv.wait(
                    lock, [&] { return async_completion_stop || !async_completions.empty(); }
                );
                if (async_completion_stop)
                {
                    return;
                }
                if (pending_streak >= async_completions.size())
                {
                    size_t const size_before_wait = async_completions.size();
                    async_completion_cv.wait_for(
                        lock,
                        std::chrono::milliseconds(10),
                        [&] { return async_completion_stop || async_completions.size() > size_before_wait; }
                    );
                    if (async_completion_stop)
                    {
                        return;
                    }
                    pending_streak = 0;
                }
                completion = std::move(async_completions.front());
                async_completions.pop_front();
            }

            bool completed = true;
            try
            {
                completed = completion();
            }
            catch (...)
            {
                completed = true;
            }

            if (completed)
            {
                pending_streak = 0;
                continue;
            }

            std::lock_guard<std::mutex> lock(async_completion_mutex);
            if (async_completion_stop)
            {
                return;
            }
            async_completions.emplace_back(std::move(completion));
            ++pending_streak;
        }
    }
    void startAsyncCompletionLoop()
    {
        std::lock_guard<std::mutex> lock(async_completion_mutex);
        async_completion_stop = false;
        if (async_completion_thread.joinable())
        {
            return;
        }
        async_completion_thread = std::thread([this] { asyncCompletionLoop(); });
    }
    void stopAsyncCompletionLoop()
    {
        {
            std::lock_guard<std::mutex> lock(async_completion_mutex);
            async_completion_stop = true;
            async_completions.clear();
        }
        async_completion_cv.notify_all();
        if (async_completion_thread.joinable())
        {
            async_completion_thread.join();
        }
    }
    void postAsyncCompletion(std::function<bool()>&& completion)
    {
        {
            std::lock_guard<std::mutex> lock(async_completion_mutex);
            if (async_completion_stop || quit.load(std::memory_order_relaxed))
            {
                return;
            }
            async_completions.emplace_back(std::move(completion));
        }
        async_completion_cv.notify_one();
    }
    void clear()
    {
        quit.store(true, std::memory_order_relaxed);
        clearMessageRouting();
        if (message_producer)
        {
            message_producer->keepRunning.store(false, std::memory_order_relaxed);
        }
        if (input)
        {
            input->interrupt();
        }
        if (session_output_state)
        {
            session_output_state->active.store(false, std::memory_order_relaxed);
            if (session_output_state->send_mutex)
            {
                std::lock_guard<std::mutex> lock(*session_output_state->send_mutex);
                session_output_state->output.reset();
            }
            else
            {
                session_output_state->output.reset();
            }
        }
        std::vector<Canceler> cancelers;
        {
            std::lock_guard<std::mutex> lock(cancellation_registry->mutex);
            for (auto& entry : cancellation_registry->requestCancelers)
            {
                cancelers.push_back(entry.second.first);
            }
            cancellation_registry->requestCancelers.clear();
            cancellation_registry->retainedRequestCancelers.clear();
            cancellation_registry->pendingCancelRequests.clear();
            cancellation_registry->pendingCancelOrder.clear();
            cancellation_registry->seenRequestIds.clear();
            cancellation_registry->seenRequestOrder.clear();
        }
        for (auto& canceler : cancelers)
        {
            canceler();
        }
        failPendingRequests();
        stopOrderedDispatcher();
        stopAsyncCompletionLoop();
        stopWorkerPools();
    }

    int getNextRequestId()
    {
        return m_id.fetch_add(1, std::memory_order_relaxed);
    }

    void stopWorkerPools()
    {
        if (parse_pool)
        {
            parse_pool->stop();
        }
        if (handler_pool)
        {
            handler_pool->stop();
        }
    }
};

bool RemoteEndPoint::Data::acquireParseQueueSlot(uint64_t sequence)
{
    size_t current = parse_queue_size.load(std::memory_order_relaxed);
    for (;;)
    {
        if (limits.max_parse_queue_size != 0 && current >= limits.max_parse_queue_size)
        {
            std::string info = "parse queue size would exceed configured maximum ";
            info += std::to_string(limits.max_parse_queue_size);
            info += ".";
            if (handleOverload(info))
            {
                submitParsedMessage(sequence, RemoteEndPoint::ParsedMessage(), true);
            }
            return false;
        }
        if (parse_queue_size.compare_exchange_weak(
                current,
                current + 1,
                std::memory_order_relaxed,
                std::memory_order_relaxed
            ))
        {
            return true;
        }
    }
}

namespace
{
void WriterMsg(std::shared_ptr<lsp::ostream>& output, LspMessage& msg)
{
    auto const& s = msg.ToJson();
    auto const value = std::string("Content-Length: ") + std::to_string(s.size()) + "\r\n\r\n" + s;
    output->write(value);
    output->flush();
}

bool isResponseMessage(JsonReader& visitor)
{

    if (!visitor.HasMember("id"))
    {
        return false;
    }

    if (!visitor.HasMember("result") && !visitor.HasMember("error"))
    {
        return false;
    }

    return true;
}

bool isRequestMessage(JsonReader& visitor)
{
    if (!visitor.HasMember("method"))
    {
        return false;
    }
    if (!visitor["method"]->IsString())
    {
        return false;
    }
    if (!visitor.HasMember("id"))
    {
        return false;
    }
    return true;
}
bool isNotificationMessage(JsonReader& visitor)
{
    if (!visitor.HasMember("method"))
    {
        return false;
    }
    if (!visitor["method"]->IsString())
    {
        return false;
    }
    if (visitor.HasMember("id"))
    {
        return false;
    }
    return true;
}

std::unique_ptr<LspMessage> makeErrorMessage(
    lsRequestId const& id,
    lsErrorCodes code,
    std::string const& message
)
{
    std::unique_ptr<LspMessage> msg(new Rsp_Error());
    auto* error = static_cast<Rsp_Error*>(msg.get());
    error->id = id;
    error->error.code = code;
    error->error.message = message;
    return msg;
}
} // namespace

CancelMonitor RemoteEndPoint::getCancelMonitor(lsRequestId const& id)
{
    auto monitor = getCancelledMonitor(id);
    if (monitor.has_value())
    {
        return monitor.value();
    }
    return [] { return 0; };
}

std::shared_ptr<lsp::detail::SessionOutputState> RemoteEndPoint::getSessionOutputState() const
{
    if (auto const* state = Context::current().get(g_sessionOutputStateKey))
    {
        return *state;
    }
    return d_ptr->session_output_state;
}

std::shared_ptr<void> RemoteEndPoint::retainRequestCancellation(lsRequestId const& id)
{
    return d_ptr->retainRequestCancellation(id);
}

void RemoteEndPoint::postAsyncCompletion(std::function<bool()>&& completion)
{
    d_ptr->postAsyncCompletion(std::move(completion));
}

void RemoteEndPoint::sendSessionMessage(std::shared_ptr<lsp::detail::SessionOutputState> const& state, LspMessage& msg)
{
    if (!state || !state->active.load(std::memory_order_relaxed) || !state->send_mutex)
    {
        return;
    }

    std::lock_guard<std::mutex> lock(*state->send_mutex);
    if (!state->active.load(std::memory_order_relaxed) || !state->output || state->output->bad())
    {
        return;
    }
    WriterMsg(state->output, msg);
}

RemoteEndPoint::RemoteEndPoint(
    std::shared_ptr<MessageJsonHandler> const& json_handler, std::shared_ptr<Endpoint> const& localEndPoint,
    lsp::Log& _log, lsp::JSONStreamStyle style, uint8_t max_workers
)
    : RemoteEndPoint(json_handler, localEndPoint, _log, style, max_workers, RemoteEndPointLimits())
{
}

RemoteEndPoint::RemoteEndPoint(
    std::shared_ptr<MessageJsonHandler> const& json_handler,
    std::shared_ptr<Endpoint> const& localEndPoint,
    lsp::Log& _log,
    lsp::JSONStreamStyle style,
    uint8_t max_workers,
    RemoteEndPointLimits limits
)
    : d_ptr(new Data(style, max_workers, limits, _log, this)), jsonHandler(json_handler), local_endpoint(localEndPoint)
{
    jsonHandler->SetNotificationJsonHandler(
        Notify_Cancellation::notify::kMethodInfo,
        [](Reader& visitor) { return Notify_Cancellation::notify::ReflectReader(visitor); }
    );
    d_ptr->session_output_state->send_mutex = m_sendMutex;

    d_ptr->quit.store(false, std::memory_order_relaxed);
}

RemoteEndPoint::~RemoteEndPoint()
{
    stop();
    delete d_ptr;
    d_ptr = nullptr;
}

RemoteEndPoint::ParsedMessage RemoteEndPoint::parseAndClassify(std::string const& content)
{
    ParsedMessage result;
    rapidjson::Document document;
    document.Parse(content.c_str(), content.length());
    if (document.HasParseError())
    {
        std::string info = "lsp msg format error:";
        rapidjson::GetParseErrorFunc GetParseError = rapidjson::GetParseError_En; // or whatever
        info += GetParseError(document.GetParseError());
        info += "\n";
        info += "ErrorContext offset:\n";
        info += content.substr(document.GetErrorOffset());
        d_ptr->log.log(Log::Level::SEVERE, info);

        return result;
    }

    JsonReader visitor {&document};
    if (!visitor.HasMember("jsonrpc") || !visitor["jsonrpc"]->IsString() ||
        std::string(visitor["jsonrpc"]->GetString()) != "2.0")
    {
        std::string reason;
        reason = "Reason:Bad or missing jsonrpc version\n";
        reason += "content:\n" + content;
        d_ptr->log.log(Log::Level::SEVERE, reason);
        return result;
    }
    LspMessage::Kind _kind = LspMessage::NOTIFICATION_MESSAGE;
    lsRequestId current_id;
    bool has_current_id = false;
    try
    {
        if (isRequestMessage(visitor))
        {
            _kind = LspMessage::REQUEST_MESSAGE;
            ReflectMember(visitor, "id", current_id);
            has_current_id = current_id.has_value();
            auto const method = visitor["method"]->GetString();
            auto msg = jsonHandler->parseRequstMessage(method, visitor);
            if (msg)
            {
                result.ok = true;
                result.kind = LspMessage::REQUEST_MESSAGE;
                result.message = std::move(msg);
                return result;
            }
            else
            {
                std::string info = "Unknown support request message when consumer message:\n";
                info += content;
                d_ptr->log.log(Log::Level::WARNING, info);
                if (has_current_id)
                {
                    result.ok = true;
                    result.outgoing = true;
                    result.kind = LspMessage::RESPONCE_MESSAGE;
                    result.message = makeErrorMessage(
                        current_id,
                        lsErrorCodes::MethodNotFound,
                        "Method not found: " + method
                    );
                }
                return result;
            }
        }
        else if (isResponseMessage(visitor))
        {
            _kind = LspMessage::RESPONCE_MESSAGE;
            lsRequestId id;
            ReflectMember(visitor, "id", id);
            current_id = id;
            has_current_id = id.has_value();

            auto msgInfo = d_ptr->getRequestInfo(id);
            if (!msgInfo)
            {
                std::string info = "Unknown response message :\n";
                info += content;
                d_ptr->log.log(Log::Level::INFO, info);
                result.ok = true;
                result.kind = LspMessage::RESPONCE_MESSAGE;
                return result;
            }
            else
            {
                auto msg = jsonHandler->parseResponseMessage(msgInfo->method, visitor);
                if (msg)
                {
                    result.ok = true;
                    result.kind = LspMessage::RESPONCE_MESSAGE;
                    result.message = std::move(msg);
                    result.pending = msgInfo;
                    return result;
                }
                else
                {
                    std::string info = "Unknown response message :\n";
                    info += content;
                    d_ptr->log.log(Log::Level::SEVERE, info);
                    result.ok = true;
                    result.kind = LspMessage::RESPONCE_MESSAGE;
                    result.pending = msgInfo;
                    result.message = makeErrorMessage(
                        id,
                        lsErrorCodes::InternalError,
                        "Failed to parse response message."
                    );
                    return result;
                }
            }
        }
        else if (isNotificationMessage(visitor))
        {
            auto const method = visitor["method"]->GetString();
            auto msg = jsonHandler->parseNotificationMessage(method, visitor);
            if (!msg)
            {
                std::string info = "Unknown notification message :\n";
                info += content;
                d_ptr->log.log(Log::Level::SEVERE, info);
                return result;
            }
            result.ok = true;
            result.kind = LspMessage::NOTIFICATION_MESSAGE;
            result.message = std::move(msg);
            return result;
        }
        else
        {
            std::string info = "Unknown lsp message when consumer message:\n";
            info += content;
            d_ptr->log.log(Log::Level::WARNING, info);
            return result;
        }
    }
    catch (std::exception& e)
    {

        std::string info = "Exception  when process ";
        if (_kind == LspMessage::REQUEST_MESSAGE)
        {
            info += "request";
        }
        else if (_kind == LspMessage::RESPONCE_MESSAGE)
        {
            info += "response";
        }
        else
        {
            info += "notification";
        }
        info += " message:\n";
        info += e.what();
        std::string reason = "Reason:" + info + "\n";
        reason += "content:\n" + content;
        d_ptr->log.log(Log::Level::SEVERE, reason);
        if (_kind == LspMessage::REQUEST_MESSAGE && has_current_id)
        {
            result.ok = true;
            result.outgoing = true;
            result.kind = LspMessage::RESPONCE_MESSAGE;
            result.message = makeErrorMessage(current_id, lsErrorCodes::InvalidParams, e.what());
        }
        else if (_kind == LspMessage::RESPONCE_MESSAGE && has_current_id)
        {
            auto msgInfo = d_ptr->getRequestInfo(current_id);
            if (msgInfo)
            {
                result.ok = true;
                result.kind = LspMessage::RESPONCE_MESSAGE;
                result.pending = msgInfo;
                result.message = makeErrorMessage(
                    current_id,
                    lsErrorCodes::InternalError,
                    "Failed to parse response message."
                );
            }
        }
        return result;
    }
    return result;
}

void RemoteEndPoint::mainLoopCatching(
    std::unique_ptr<LspMessage> msg,
    uint64_t sequence,
        std::shared_ptr<lsp::detail::SessionOutputState> response_state,
    PendingRequestToken pending
)
{
    if (!msg)
    {
        return;
    }
    auto const kind = msg->GetKid();
    try
    {
        mainLoop(std::move(msg), sequence, std::move(response_state), std::move(pending));
    }
    catch (std::exception& e)
    {
        std::string info = "Exception  when process ";
        if (kind == LspMessage::REQUEST_MESSAGE)
        {
            info += "request";
        }
        else if (kind == LspMessage::RESPONCE_MESSAGE)
        {
            info += "response";
        }
        else
        {
            info += "notification";
        }
        info += " message:\n";
        info += e.what();
        std::string reason = "Reason:" + info + "\n";
        d_ptr->log.log(Log::Level::SEVERE, reason);
    }
}

void RemoteEndPoint::routeIncoming(std::string&& content, uint64_t sequence) const
{
    d_ptr->postParseTask(std::move(content), sequence);
}

void RemoteEndPoint::routeParsedIncoming(ParsedMessage&& parsed, uint64_t sequence) const
{
    if (!parsed.ok || !parsed.message || d_ptr->quit.load(std::memory_order_relaxed))
    {
        return;
    }

    auto response_state = d_ptr->session_output_state;
    if (parsed.outgoing)
    {
        sendSessionMessage(response_state, *parsed.message);
        return;
    }

    if (parsed.kind == LspMessage::NOTIFICATION_MESSAGE)
    {
        if (strcmp(Notify_Cancellation::notify::kMethodInfo, parsed.message->GetMethodType()) == 0)
        {
            d_ptr->onCancel(static_cast<Notify_Cancellation::notify*>(parsed.message.get()), sequence);
        }
        else if (d_ptr->allowsConcurrentNotification(parsed.message->GetMethodType()))
        {
            d_ptr->postToWorker(std::move(parsed.message), sequence, response_state);
        }
        else
        {
            d_ptr->enqueueNotification(std::move(parsed.message), sequence, response_state);
        }
    }
    else if (parsed.kind == LspMessage::REQUEST_MESSAGE)
    {
        d_ptr->enqueueRequest(std::move(parsed.message), sequence, response_state);
    }
    else if (parsed.kind == LspMessage::RESPONCE_MESSAGE && parsed.pending)
    {
        d_ptr->completePendingResponse(std::move(parsed.message), std::move(parsed.pending));
    }
    else
    {
        d_ptr->postToWorker(std::move(parsed.message), sequence, response_state, std::move(parsed.pending));
    }
}

bool RemoteEndPoint::allowConcurrentNotification(std::string const& method)
{
    if (!canRegisterBeforeStart("allowConcurrentNotification"))
    {
        return false;
    }
    d_ptr->allowConcurrentNotification(method);
    return true;
}

bool RemoteEndPoint::canRegisterBeforeStart(char const* operation)
{
    std::lock_guard<std::mutex> lock(d_ptr->lifecycle_mutex);
    if (!d_ptr->working)
    {
        return true;
    }
    std::string message = operation ? operation : "registration";
    message += " must be called before startProcessingMessages().";
    d_ptr->log.log(Log::Level::WARNING, message);
    assert(false && "RemoteEndPoint registration must happen before startProcessingMessages()");
    return false;
}

bool RemoteEndPoint::internalSendRequest(RequestInMessage& info, GenericResponseHandler handler)
{
    return internalSendRequestWithPolicy(info, std::move(handler), PendingCompletionPolicy::DeferredCallback);
}

bool RemoteEndPoint::internalSendRequestWithPolicy(
    RequestInMessage& info,
    GenericResponseHandler handler,
    PendingCompletionPolicy policy
)
{
    std::lock_guard<std::mutex> lock(*m_sendMutex);
    auto response_state = d_ptr->session_output_state;
    if (!response_state || !response_state->active.load(std::memory_order_relaxed) || !response_state->output ||
        response_state->output->bad())
    {
        std::string desc = "Output isn't good any more:\n";
        d_ptr->log.log(Log::Level::WARNING, desc);
        return false;
    }
    PendingResponseHandler pending_handler = [handler = std::move(handler)](std::unique_ptr<LspMessage>& msg) mutable
    { return handler(std::move(msg)); };
    auto pending = d_ptr->pendingRequest(
        info,
        std::move(pending_handler),
        policy == PendingCompletionPolicy::DeferredCallback
    );
    if (!pending)
    {
        std::string desc = "Duplicate id  which of request:";
        desc += info.ToJson();
        desc += "\n";
        d_ptr->log.log(Log::Level::WARNING, desc);
        return false;
    }
    try
    {
        WriterMsg(response_state->output, info);
    }
    catch (...)
    {
        d_ptr->removeRequestInfo(info.id, pending);
        throw;
    }
    return true;
}

int RemoteEndPoint::getNextRequestId()
{
    return d_ptr->getNextRequestId();
}
bool RemoteEndPoint::cancelRequest(lsRequestId const& id)
{
    if (!isWorking())
    {
        return false;
    }
    auto msgInfo = d_ptr->getRequestInfo(id);
    if (msgInfo)
    {
        Notify_Cancellation::notify cancel_notify;
        cancel_notify.params.id = id;
        send(cancel_notify);
        return true;
    }
    return false;
}

void RemoteEndPoint::removeRequestInfo(lsRequestId const& id)
{
    d_ptr->removeRequestInfo(id);
}

std::unique_ptr<LspMessage> RemoteEndPoint::internalWaitResponse(RequestInMessage& request, unsigned time_out)
{
    auto eventFuture = std::make_shared<Condition<LspMessage>>();
    if (!internalSendRequestWithPolicy(
        request,
        [=](std::unique_ptr<LspMessage> data)
        {
            eventFuture->notify(std::move(data));
            return true;
        },
        PendingCompletionPolicy::FastStateOnly
    ))
    {
        return {};
    }
    auto response = eventFuture->wait(time_out);
    if (!response)
    {
        d_ptr->removeRequestInfo(request.id);
    }
    return response;
}

void RemoteEndPoint::mainLoop(
    std::unique_ptr<LspMessage> msg,
    uint64_t sequence,
    std::shared_ptr<lsp::detail::SessionOutputState> response_state,
    PendingRequestToken pending
)
{
    if (d_ptr->quit.load(std::memory_order_relaxed))
    {
        return;
    }
    WithContext WithResponseState(Context::current().derive(g_sessionOutputStateKey, response_state));
    auto const _kind = msg->GetKid();
    if (_kind == LspMessage::REQUEST_MESSAGE)
    {
        auto req = static_cast<RequestInMessage*>(msg.get());
        auto const request_id = req->id;
        auto const method = std::string(req->GetMethodType());
        // Calls can be canceled by the client. Add cancellation context.
        WithContext WithCancel(d_ptr->cancelableRequestContext(request_id, sequence));
        if (!local_endpoint->onRequest(std::move(msg)))
        {
            Rsp_Error error;
            error.id = request_id;
            error.error.code = lsErrorCodes::MethodNotFound;
            error.error.message = "Method not found: " + method;
            send(error);
        }
    }

    else if (_kind == LspMessage::RESPONCE_MESSAGE)
    {
        auto const id = static_cast<ResponseInMessage*>(msg.get())->id;
        auto msgInfo = pending ? std::static_pointer_cast<PendingRequestInfo>(pending) : d_ptr->getRequestInfo(id);
        if (!msgInfo)
        {
            auto const _method_desc = msg->GetMethodType();
            local_endpoint->onResponse(_method_desc, std::move(msg));
        }
        else
        {
            d_ptr->completePendingResponse(std::move(msg), msgInfo);
        }
    }
    else if (_kind == LspMessage::NOTIFICATION_MESSAGE)
    {
        if (strcmp(Notify_Cancellation::notify::kMethodInfo, msg->GetMethodType()) == 0)
        {
            d_ptr->onCancel(static_cast<Notify_Cancellation::notify*>(msg.get()), sequence);
        }
        else
        {
            local_endpoint->notify(std::move(msg));
        }
    }
    else
    {
        std::string info = "Unknown lsp message  when process  message  in mainLoop:\n";
        d_ptr->log.log(Log::Level::WARNING, info);
    }
}

void RemoteEndPoint::handle(std::vector<MessageIssue>&& issue)
{
    for (auto& it : issue)
    {
        d_ptr->log.log(it.code, it.text);
    }
}

void RemoteEndPoint::handle(MessageIssue&& issue)
{
    d_ptr->log.log(issue.code, issue.text);
}

void RemoteEndPoint::startProcessingMessages(std::shared_ptr<lsp::istream> r, std::shared_ptr<lsp::ostream> w)
{
    std::lock_guard<std::mutex> lifecycle_lock(d_ptr->lifecycle_mutex);
    if (d_ptr->working)
    {
        d_ptr->log.log(Log::Level::WARNING, "RemoteEndPoint is already processing messages.");
        return;
    }
#if defined(LSPCPP_USEGC)
    // Initialize Boehm GC on the starting thread before worker pools register.
    GCThreadContext::ensureInitialized();
#endif
    d_ptr->working = true;
    d_ptr->quit.store(false, std::memory_order_relaxed);
    d_ptr->input = r;
    d_ptr->output = w;
    d_ptr->session_output_state = std::make_shared<lsp::detail::SessionOutputState>();
    d_ptr->session_output_state->output = w;
    d_ptr->session_output_state->send_mutex = m_sendMutex;
    d_ptr->session_output_state->generation = ++d_ptr->next_session_generation;
    d_ptr->startAsyncCompletionLoop();
    d_ptr->message_producer->bind(r);
    auto const worker_count = d_ptr->max_workers == 0 ? 1 : d_ptr->max_workers;
    d_ptr->parse_pool = std::make_shared<asio::thread_pool>(worker_count);
    d_ptr->handler_pool = std::make_shared<asio::thread_pool>(worker_count);
    d_ptr->startOrderedDispatcher();
    d_ptr->resetMessageRouting();
    message_producer_thread_ = std::make_shared<std::thread>(
        [&]()
        {
            d_ptr->message_producer->listen(
                [&](std::string&& content)
                {
                    auto const sequence = d_ptr->nextMessageSequence();
                    routeIncoming(std::move(content), sequence);
                }
            );
        }
    );
}

void RemoteEndPoint::stop()
{
    if (!d_ptr)
    {
        return;
    }
    std::lock_guard<std::mutex> lifecycle_lock(d_ptr->lifecycle_mutex);
    if (!d_ptr->working && !message_producer_thread_)
    {
        return;
    }
    d_ptr->clear();
    if (message_producer_thread_ && message_producer_thread_->joinable())
    {
        if (message_producer_thread_->get_id() == std::this_thread::get_id())
        {
            message_producer_thread_->detach();
        }
        else
        {
            message_producer_thread_->join();
        }
    }
    message_producer_thread_ = nullptr;
    d_ptr->working = false;
}

void RemoteEndPoint::sendMsg(LspMessage& msg)
{
    sendSessionMessage(getSessionOutputState(), msg);
}

bool RemoteEndPoint::isWorking() const
{
    std::lock_guard<std::mutex> lock(d_ptr->lifecycle_mutex);
    return d_ptr->working;
}
