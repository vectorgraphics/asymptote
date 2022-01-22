#include "LibLsp/JsonRpc/MessageJsonHandler.h"
#include "LibLsp/JsonRpc/Endpoint.h"
#include "LibLsp/JsonRpc/message.h"
#include "LibLsp/JsonRpc/RemoteEndPoint.h"
#include <future>
#include "LibLsp/JsonRpc/Cancellation.h"
#include "LibLsp/JsonRpc/StreamMessageProducer.h"
#include "LibLsp/JsonRpc/NotificationInMessage.h"
#include "LibLsp/JsonRpc/lsResponseMessage.h"
#include "LibLsp/JsonRpc/Condition.h"
#include "LibLsp/JsonRpc/Context.h"
#include "rapidjson/error/en.h"
#include "LibLsp/JsonRpc/json.h"
#include "LibLsp/JsonRpc/GCThreadContext.h"
#include "LibLsp/JsonRpc/ScopeExit.h"
#include "LibLsp/JsonRpc/stream.h"

#define BOOST_BIND_GLOBAL_PLACEHOLDERS
#include "boost/threadpool.hpp"
#include <atomic>
namespace lsp {

// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===//

// Cancellation mechanism for long-running tasks.
//
// This manages interactions between:
//
// 1. Client code that starts some long-running work, and maybe cancels later.
//
//   std::pair<Context, Canceler> Task = cancelableTask();
//   {
//     WithContext Cancelable(std::move(Task.first));
//     Expected
//     deepThoughtAsync([](int answer){ errs() << answer; });
//   }
//   // ...some time later...
//   if (User.fellAsleep())
//     Task.second();
//
//  (This example has an asynchronous computation, but synchronous examples
//  work similarly - the Canceler should be invoked from another thread).
//
// 2. Library code that executes long-running work, and can exit early if the
//   result is not needed.
//
//   void deepThoughtAsync(std::function<void(int)> Callback) {
//     runAsync([Callback]{
//       int A = ponder(6);
//       if (getCancelledMonitor())
//         return;
//       int B = ponder(9);
//       if (getCancelledMonitor())
//         return;
//       Callback(A * B);
//     });
//   }
//
//   (A real example may invoke the callback with an error on cancellation,
//   the CancelledError is provided for this purpose).
//
// Cancellation has some caveats:
//   - the work will only stop when/if the library code next checks for it.
//     Code outside clangd such as Sema will not do this.
//   - it's inherently racy: client code must be prepared to accept results
//     even after requesting cancellation.
//   - it's Context-based, so async work must be dispatched to threads in
//     ways that preserve the context. (Like runAsync() or TUScheduler).
//

	/// A canceller requests cancellation of a task, when called.
	/// Calling it again has no effect.
	using Canceler = std::function<void()>;

	// We don't want a cancelable scope to "shadow" an enclosing one.
	struct CancelState {
		std::shared_ptr<std::atomic<int>> cancelled;
		const CancelState* parent = nullptr;
		lsRequestId id;
	};
	static Key<CancelState> g_stateKey;

	/// Defines a new task whose cancellation may be requested.
	/// The returned Context defines the scope of the task.
	/// When the context is active, getCancelledMonitor() is 0 until the Canceler is
	/// invoked, and equal to Reason afterwards.
	/// Conventionally, Reason may be the LSP error code to return.
	std::pair<Context, Canceler> cancelableTask(const lsRequestId& id,int reason = 1){
		assert(reason != 0 && "Can't detect cancellation if Reason is zero");
		CancelState state;
		state.id = id;
		state.cancelled = std::make_shared<std::atomic<int>>();
		state.parent = Context::current().get(g_stateKey);
		return {
			Context::current().derive(g_stateKey, state),
			[reason, cancelled(state.cancelled)] { *cancelled = reason; },
		};
	}
	/// If the current context is within a cancelled task, returns the reason.
/// (If the context is within multiple nested tasks, true if any are cancelled).
/// Always zero if there is no active cancelable task.
/// This isn't free (context lookup) - don't call it in a tight loop.
	boost::optional<CancelMonitor> getCancelledMonitor(const lsRequestId& id, const Context& ctx = Context::current()){
		for (const CancelState* state = ctx.get(g_stateKey); state != nullptr;
			state = state->parent)
		{
			if (id != state->id)continue;
			const std::shared_ptr<std::atomic<int> > cancelled = state->cancelled;
			std::function<int()> temp = [=]{
				return cancelled->load();
			};
			return std::move(temp);
		}

		return {};
	}
} // namespace lsp

using namespace  lsp;
class PendingRequestInfo
{
	using   RequestCallBack = std::function< bool(std::unique_ptr<LspMessage>) >;
public:
	PendingRequestInfo(const std::string& md,
		const RequestCallBack& callback);
	PendingRequestInfo(const std::string& md);
	PendingRequestInfo() {}
	std::string method;
	RequestCallBack futureInfo;
};

PendingRequestInfo::PendingRequestInfo(const std::string& _md,
	const	RequestCallBack& callback) : method(_md),
	futureInfo(callback)
{
}

PendingRequestInfo::PendingRequestInfo(const std::string& md) : method(md)
{
}
struct RemoteEndPoint::Data
{
	explicit Data(lsp::Log& _log , RemoteEndPoint* owner)
          : m_id(0), next_request_cookie(0), message_producer(new StreamMessageProducer(*owner)), log(_log)
	{

	}
	~Data()
	{
	   delete	message_producer;
	}
	std::atomic<unsigned> m_id;
	boost::threadpool::pool tp;
	// Method calls may be cancelled by ID, so keep track of their state.
 // This needs a mutex: handlers may finish on a different thread, and that's
 // when we clean up entries in the map.
	mutable std::mutex request_cancelers_mutex;

	std::map< lsRequestId, std::pair<Canceler, /*Cookie*/ unsigned> > requestCancelers;

	std::atomic<unsigned>  next_request_cookie; // To disambiguate reused IDs, see below.
	void onCancel(Notify_Cancellation::notify* notify) {
		std::lock_guard<std::mutex> Lock(request_cancelers_mutex);
		const auto it = requestCancelers.find(notify->params.id);
		if (it != requestCancelers.end())
			it->second.first(); // Invoke the canceler.
	}

	// We run cancelable requests in a context that does two things:
	//  - allows cancellation using requestCancelers[ID]
	//  - cleans up the entry in requestCancelers when it's no longer needed
	// If a client reuses an ID, the last wins and the first cannot be canceled.
	Context cancelableRequestContext(lsRequestId id) {
		auto task = cancelableTask(id,
			/*Reason=*/static_cast<int>(lsErrorCodes::RequestCancelled));
		unsigned cookie;
		{
			std::lock_guard<std::mutex> Lock(request_cancelers_mutex);
			cookie = next_request_cookie.fetch_add(1, std::memory_order_relaxed);
			requestCancelers[id] = { std::move(task.second), cookie };
		}
		// When the request ends, we can clean up the entry we just added.
		// The cookie lets us check that it hasn't been overwritten due to ID
		// reuse.
		return task.first.derive(lsp::make_scope_exit([this, id, cookie] {
			std::lock_guard<std::mutex> lock(request_cancelers_mutex);
			const auto& it = requestCancelers.find(id);
			if (it != requestCancelers.end() && it->second.second == cookie)
				requestCancelers.erase(it);
			}));
	}

	std::map <lsRequestId, std::shared_ptr<PendingRequestInfo>>  _client_request_futures;
	StreamMessageProducer* message_producer;
	std::atomic<bool> quit{};
	lsp::Log& log;
	std::shared_ptr<lsp::istream>  input;
	std::shared_ptr<lsp::ostream>  output;

	void pendingRequest(RequestInMessage& info, GenericResponseHandler&& handler)
	{
		auto id = m_id.fetch_add(1, std::memory_order_relaxed);
		info.id.set(id);
		std::lock_guard<std::mutex> lock(m_requsetInfo);
		_client_request_futures[info.id] = std::make_shared<PendingRequestInfo>(info.method, handler);

	}
	const std::shared_ptr<const PendingRequestInfo> getRequestInfo(const lsRequestId& _id)
	{
		std::lock_guard<std::mutex> lock(m_requsetInfo);
		auto findIt = _client_request_futures.find(_id);
		if (findIt != _client_request_futures.end())
		{
			return findIt->second;
		}
		return  nullptr;
	}

	std::mutex m_requsetInfo;
	void removeRequestInfo(const lsRequestId& _id)
	{
		std::lock_guard<std::mutex> lock(m_requsetInfo);
		auto findIt = _client_request_futures.find(_id);
		if (findIt != _client_request_futures.end())
		{
			_client_request_futures.erase(findIt);
		}
	}
	void clear()
	{
		{
			std::lock_guard<std::mutex> lock(m_requsetInfo);
			_client_request_futures.clear();

		}
		tp.clear();
		quit.store(true, std::memory_order_relaxed);
	}
};

namespace
{
void WriterMsg(std::shared_ptr<lsp::ostream>&  output, LspMessage& msg)
{
	const auto& s = msg.ToJson();
	const auto value =
		std::string("Content-Length: ") + std::to_string(s.size()) + "\r\n\r\n" + s;
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
}

CancelMonitor RemoteEndPoint::getCancelMonitor(const lsRequestId& id)
{
	auto  monitor =  getCancelledMonitor(id);
	if(monitor.has_value())
	{
		return  monitor.value();
	}
	return [] {
		return 0;
	};

}

RemoteEndPoint::RemoteEndPoint(
	const std::shared_ptr < MessageJsonHandler >& json_handler,const std::shared_ptr < Endpoint>& localEndPoint, lsp::Log& _log, uint8_t max_workers):
    d_ptr(new Data(_log,this)),jsonHandler(json_handler), local_endpoint(localEndPoint)
{
	jsonHandler->method2notification[Notify_Cancellation::notify::kMethodInfo] = [](Reader& visitor)
	{
		return Notify_Cancellation::notify::ReflectReader(visitor);
	};

	d_ptr->quit.store(false, std::memory_order_relaxed);
	d_ptr->tp.size_controller().resize(max_workers);
}

RemoteEndPoint::~RemoteEndPoint()
{
	delete d_ptr;
	d_ptr->quit.store(true, std::memory_order_relaxed);
}

bool RemoteEndPoint::dispatch(const std::string& content)
{
		rapidjson::Document document;
		document.Parse(content.c_str(), content.length());
		if (document.HasParseError())
		{
			std::string info ="lsp msg format error:";
			rapidjson::GetParseErrorFunc GetParseError = rapidjson::GetParseError_En; // or whatever
			info+= GetParseError(document.GetParseError());
			info += "\n";
			info += "ErrorContext offset:\n";
			info += content.substr(document.GetErrorOffset());
			d_ptr->log.log(Log::Level::SEVERE, info);

			return false;
		}

		JsonReader visitor{ &document };
		if (!visitor.HasMember("jsonrpc") ||
			std::string(visitor["jsonrpc"]->GetString()) != "2.0")
		{
			std::string reason;
			reason = "Reason:Bad or missing jsonrpc version\n";
			reason += "content:\n" + content;
			d_ptr->log.log(Log::Level::SEVERE, reason);
			return  false;

		}
		LspMessage::Kind _kind = LspMessage::NOTIFICATION_MESSAGE;
		try {
			if (isRequestMessage(visitor))
			{
				_kind = LspMessage::REQUEST_MESSAGE;
				auto msg = jsonHandler->parseRequstMessage(visitor["method"]->GetString(), visitor);
				if (msg) {
					mainLoop(std::move(msg));
				}
				else {
					std::string info = "Unknown support request message when consumer message:\n";
					info += content;
					d_ptr->log.log(Log::Level::WARNING, info);
					return false;
				}
			}
			else if (isResponseMessage(visitor))
			{
				_kind = LspMessage::RESPONCE_MESSAGE;
				lsRequestId id;
				ReflectMember(visitor, "id", id);

				auto msgInfo = d_ptr->getRequestInfo(id);
				if (!msgInfo)
				{
					std::pair<std::string, std::unique_ptr<LspMessage>> result;
					auto b = jsonHandler->resovleResponseMessage(visitor, result);
					if (b)
					{
						result.second->SetMethodType(result.first.c_str());
						mainLoop(std::move(result.second));
					}
					else
					{
						std::string info = "Unknown response message :\n";
						info += content;
						d_ptr->log.log(Log::Level::INFO, info);
					}
				}
				else
				{

					auto msg = jsonHandler->parseResponseMessage(msgInfo->method, visitor);
					if (msg) {
						mainLoop(std::move(msg));
					}
					else
					{
						std::string info = "Unknown response message :\n";
						info += content;
						d_ptr->log.log(Log::Level::SEVERE, info);
						return  false;
					}

				}
			}
			else if (isNotificationMessage(visitor))
			{
				auto msg = jsonHandler->parseNotificationMessage(visitor["method"]->GetString(), visitor);
				if (!msg)
				{
					std::string info = "Unknown notification message :\n";
					info += content;
					d_ptr->log.log(Log::Level::SEVERE, info);
					return  false;
				}
				mainLoop(std::move(msg));
			}
			else
			{
				std::string info = "Unknown lsp message when consumer message:\n";
				info += content;
				d_ptr->log.log(Log::Level::WARNING, info);
				return false;
			}
		}
		catch (std::exception& e)
		{

			std::string info = "Exception  when process ";
			if(_kind==LspMessage::REQUEST_MESSAGE)
			{
				info += "request";
			}
			if (_kind == LspMessage::RESPONCE_MESSAGE)
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
			return false;
		}
	return  true;
}



void RemoteEndPoint::internalSendRequest( RequestInMessage& info, GenericResponseHandler handler)
{
	std::lock_guard<std::mutex> lock(m_sendMutex);
	if (!d_ptr->output || d_ptr->output->bad())
	{
		std::string desc = "Output isn't good any more:\n";
		d_ptr->log.log(Log::Level::INFO, desc);
		return ;
	}
	d_ptr->pendingRequest(info, std::move(handler));
	WriterMsg(d_ptr->output, info);
}


std::unique_ptr<LspMessage> RemoteEndPoint::internalWaitResponse(RequestInMessage& request, unsigned time_out)
{
	auto  eventFuture = std::make_shared< Condition< LspMessage > >();
	internalSendRequest(request, [=](std::unique_ptr<LspMessage> data)
	{
		eventFuture->notify(std::move(data));
		return  true;
	});
	return   eventFuture->wait(time_out);
}

void RemoteEndPoint::mainLoop(std::unique_ptr<LspMessage>msg)
{
	if(d_ptr->quit.load(std::memory_order_relaxed))
	{
		return;
	}
	const auto _kind = msg->GetKid();
	if (_kind == LspMessage::REQUEST_MESSAGE)
	{
		auto req = static_cast<RequestInMessage*>(msg.get());
		// Calls can be canceled by the client. Add cancellation context.
		WithContext WithCancel(d_ptr->cancelableRequestContext(req->id));
		local_endpoint->onRequest(std::move(msg));
	}

	else if (_kind == LspMessage::RESPONCE_MESSAGE)
	{
		auto response = static_cast<ResponseInMessage*>(msg.get());
		auto msgInfo = d_ptr->getRequestInfo(response->id);
		if (!msgInfo)
		{
			const auto _method_desc = msg->GetMethodType();
			local_endpoint->onResponse(_method_desc, std::move(msg));
		}
		else
		{
			bool needLocal = true;
			if (msgInfo->futureInfo)
			{
				if (msgInfo->futureInfo(std::move(msg)))
				{
					needLocal = false;
				}
			}
			if (needLocal)
			{
				local_endpoint->onResponse(msgInfo->method, std::move(msg));
			}
			d_ptr->removeRequestInfo(response->id);
		}
	}
	else if (_kind == LspMessage::NOTIFICATION_MESSAGE)
	{
		if (strcmp(Notify_Cancellation::notify::kMethodInfo, msg->GetMethodType())==0)
		{
			d_ptr->onCancel(static_cast<Notify_Cancellation::notify*>(msg.get()));
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
	for(auto& it : issue)
	{
		d_ptr->log.log(it.code, it.text);
	}
}

void RemoteEndPoint::handle(MessageIssue&& issue)
{
	d_ptr->log.log(issue.code, issue.text);
}


void RemoteEndPoint::startProcessingMessages(std::shared_ptr<lsp::istream> r,
	std::shared_ptr<lsp::ostream> w)
{
	d_ptr->quit.store(false, std::memory_order_relaxed);
	d_ptr->input = r;
	d_ptr->output = w;
	d_ptr->message_producer->bind(r);
	message_producer_thread_ = std::make_shared<std::thread>([&]()
   {
		d_ptr->message_producer->listen([&](std::string&& content){
			const auto temp = std::make_shared<std::string>(std::move(content));
				d_ptr->tp.schedule([this, temp]{
#ifdef USEGC
                        GCThreadContext gcContext;
#endif

						dispatch(*temp);
				});
		});
	});
}

void RemoteEndPoint::Stop()
{
	if(message_producer_thread_ && message_producer_thread_->joinable())
	{
		message_producer_thread_->detach();
	}
	d_ptr->clear();

}

void RemoteEndPoint::sendMsg( LspMessage& msg)
{

	std::lock_guard<std::mutex> lock(m_sendMutex);
	if (!d_ptr->output || d_ptr->output->bad())
	{
		std::string info = "Output isn't good any more:\n";
		d_ptr->log.log(Log::Level::INFO, info);
		return;
	}
	WriterMsg(d_ptr->output, msg);

}
