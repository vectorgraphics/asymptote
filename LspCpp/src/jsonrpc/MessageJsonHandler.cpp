#include "LibLsp/JsonRpc/MessageJsonHandler.h"
#include <string>
#include <rapidjson/document.h>



std::unique_ptr<LspMessage> MessageJsonHandler::parseResponseMessage(const std::string& method, Reader& r)
{
	auto findIt = method2response.find(method);
	
	if( findIt != method2response.end())
	{
		return  findIt->second(r);
	}
	return nullptr;
}

std::unique_ptr<LspMessage> MessageJsonHandler::parseRequstMessage(const std::string& method, Reader&r)
{
	auto findIt = method2request.find(method);

	if (findIt != method2request.end())
	{
		return  findIt->second(r);
	}
	return nullptr;
}

bool MessageJsonHandler::resovleResponseMessage(Reader&r, std::pair<std::string, std::unique_ptr<LspMessage>>& result)
{
	for (auto& handler : method2response)
	{
		try
		{
			auto msg =  handler.second(r);
			result.first = handler.first;
			result.second = std::move(msg);
			return true;
		}
		catch (...)
		{

		}
	}
	return false;
}

std::unique_ptr<LspMessage> MessageJsonHandler::parseNotificationMessage(const std::string& method, Reader& r)
{
	auto findIt = method2notification.find(method);

	if (findIt != method2notification.end())
	{
		return  findIt->second(r);
	}
	return nullptr;
}
