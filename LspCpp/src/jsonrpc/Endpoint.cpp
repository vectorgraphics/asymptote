#include "LibLsp/JsonRpc/Endpoint.h"
#include "LibLsp/JsonRpc/message.h"


bool GenericEndpoint::notify(std::unique_ptr<LspMessage> msg)
{
	auto findIt = method2notification.find(msg->GetMethodType());

	if (findIt != method2notification.end())
	{
		return  findIt->second(std::move(msg));
	}
	std::string info = "can't find method2notification for notification:\n" + msg->ToJson() + "\n";
	log.log(lsp::Log::Level::SEVERE, info);
	return false;
}

bool GenericEndpoint::onResponse(const std::string& method, std::unique_ptr<LspMessage>msg)
{
	auto findIt = method2response.find(method);

	if (findIt != method2response.end())
	{
		return  findIt->second(std::move(msg));
	}
	
	std::string info = "can't find method2response for response:\n" + msg->ToJson() + "\n";
	log.log(lsp::Log::Level::SEVERE, info);
	
	return false;
}



bool GenericEndpoint::onRequest(std::unique_ptr<LspMessage> request)
{
	auto findIt = method2request.find(request->GetMethodType());

	if (findIt != method2request.end())
	{
		return  findIt->second(std::move(request));
	}
	std::string info = "can't find method2request for request:\n" + request->ToJson() + "\n";
	log.log(lsp::Log::Level::SEVERE, info);
	return false;
}
