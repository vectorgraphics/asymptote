
#include <deque>
#include "sct.h"
#include "SCTConfig.h"
#include <rapidjson/document.h>
#include <rapidjson/istreamwrapper.h>
#include <fstream>
#include <boost/filesystem.hpp>
#include "LibLsp/lsp/general/exit.h"
#include "LibLsp/lsp/general/initialized.h"
#include "LibLsp/lsp/windows/MessageNotify.h"
#include "LibLsp/lsp/language/language.h"
#include "LibLsp/JsonRpc/Condition.h"
#include "LibLsp/lsp/workspace/execute_command.h"
#include "LibLsp/JsonRpc/json.h"

namespace lsp {
	class Log;
}

using namespace std;
using lsp::Log;


//
//MethodType sct_DownLoadCapFile::request::kMethodInfo = "sct/download_cap";
//
//MethodType sct_Transmit::request::kMethodInfo = "sct/transmit";
//
//MethodType sct_Connect::request::kMethodInfo = "sct/connect";
//
//MethodType sct_Disconnect::request::kMethodInfo = "sct/disconnect";
//
//MethodType sct_InstalllApplet::request::kMethodInfo = "sct/install_applet";
//
//MethodType sct_gp_auth::request::kMethodInfo = "sct/gp_auth";
//
//MethodType sct_SetWindowsPos::request::kMethodInfo = "sct/set_windows_pos";
//
//MethodType sct_SetWindowsVisible::request::kMethodInfo = "sct/set_windows_visible";
//
//
//MethodType sct_NotifyJcvmOutput::request::kMethodInfo = "sct/notify_jcvm_output";
//
//MethodType sct_Launch::request::kMethodInfo = "sct/launch";
//
//MethodType sct_SetProtocol::request::kMethodInfo = "sct/set_protocol";
//
//MethodType sct_CheckBeforeLaunch::request::kMethodInfo = "sct/check_before_launch";
//
//MethodType sct_GetCardInfo::request::kMethodInfo = "sct/get_card_info";
//
//MethodType sct_NotifyDisconnect::request::kMethodInfo = "sct/notify_disconnect";
//MethodType sct_TerminateLaunch::request::kMethodInfo = "sct/terminate_launch";
//MethodType sct_initialize::request::kMethodInfo = "sct/initialize";


 SCTConfig* SCTConfig::newInstance(const string& file_path, string& error)
{
 	 if(!boost::filesystem::exists(file_path))
 	 {
		 error = "file no exists.";
		 return nullptr;
 	 }
	 using namespace rapidjson;
	 using namespace std;
	 std::unique_ptr<SCTConfig>  sct = std::make_unique<SCTConfig>();
	 try
	 {
		 std::wifstream ifs(file_path);
		 WIStreamWrapper isw(ifs);

		 Document d;
		 d.ParseStream(isw);
		 if(!d.HasParseError())
		 {
			 JsonReader reader{ &d };
			 Reflect(reader, *sct.get());
		 }
	 }
 	catch (std::exception& e)
 	{
		string  temp = "Reflect failed. exception info:";
			
		temp +=	e.what();
		error = temp;
		sct.get()->broken = true;
		sct.get()->error = temp;	
 	}
	return sct.release();
}


SmartCardTool::SmartCardTool():  m_jdwpPort(0), m_curProtocol(SctProtocol::T01), log(nullptr)
{
	m_ipAddr = "127.0.0.1";
}

void AddNotifyJsonRpcMethod(sct::ProtocolJsonHandler& handler)
{
	handler.method2notification[Notify_Exit::notify::kMethodInfo] = [](Reader& visitor)
	{
		return Notify_Exit::notify::ReflectReader(visitor);
	};
	handler.method2notification[Notify_InitializedNotification::notify::kMethodInfo] = [](Reader& visitor)
	{
		return Notify_InitializedNotification::notify::ReflectReader(visitor);
	};

	
	handler.method2notification[Notify_LogMessage::notify::kMethodInfo] = [](Reader& visitor)
	{
		return Notify_LogMessage::notify::ReflectReader(visitor);
	};
	handler.method2notification[Notify_ShowMessage::notify::kMethodInfo] = [](Reader& visitor)
	{
		return Notify_ShowMessage::notify::ReflectReader(visitor);
	};
	
	handler.method2notification[Notify_sendNotification::notify::kMethodInfo] = [](Reader& visitor)
	{
		return Notify_sendNotification::notify::ReflectReader(visitor);
	};
	
	handler.method2notification[lang_actionableNotification::notify::kMethodInfo] = [](Reader& visitor)
	{
		return lang_actionableNotification::notify::ReflectReader(visitor);
	};
	handler.method2notification[lang_progressReport::notify::kMethodInfo] = [](Reader& visitor)
	{
		return lang_progressReport::notify::ReflectReader(visitor);
	};
	

	handler.method2notification[sct_NotifyJcvmOutput::notify::kMethodInfo] = [](Reader& visitor)
	{
		return sct_NotifyJcvmOutput::notify::ReflectReader(visitor);
	};
	handler.method2notification[sct_NotifyDisconnect::notify::kMethodInfo] = [](Reader& visitor)
	{
		return sct_NotifyDisconnect::notify::ReflectReader(visitor);
	};
 	
}

sct::ProtocolJsonHandler::ProtocolJsonHandler()
{
	AddNotifyJsonRpcMethod(*this);

	method2response[sct_DownLoadCapFile::request::kMethodInfo ] = [](Reader& visitor)
	{
		if (visitor.HasMember("error"))
			return 	Rsp_Error::ReflectReader(visitor);

		return sct_DownLoadCapFile::response::ReflectReader(visitor);
	};
	method2response[sct_Connect::request::kMethodInfo] = [](Reader& visitor)
	{
		if (visitor.HasMember("error"))
			return 	Rsp_Error::ReflectReader(visitor);

		return sct_Connect::response::ReflectReader(visitor);
	};
	method2response[sct_SetProtocol::request::kMethodInfo] = [](Reader& visitor)
	{
		if (visitor.HasMember("error"))
			return 	Rsp_Error::ReflectReader(visitor);

		return sct_SetProtocol::response::ReflectReader(visitor);
	};
	method2response[sct_gp_auth::request::kMethodInfo] = [](Reader& visitor)
	{
		if (visitor.HasMember("error"))
			return 	Rsp_Error::ReflectReader(visitor);

		return sct_gp_auth::response::ReflectReader(visitor);
	};
	method2response[sct_InstalllApplet::request::kMethodInfo] = [](Reader& visitor)
	{
		if (visitor.HasMember("error"))
			return 	Rsp_Error::ReflectReader(visitor);

		return sct_InstalllApplet::response::ReflectReader(visitor);
	};
	method2response[sct_Transmit::request::kMethodInfo] = [](Reader& visitor)
	{
		if (visitor.HasMember("error"))
			return 	Rsp_Error::ReflectReader(visitor);

		return sct_Transmit::response::ReflectReader(visitor);
	};
 	
	method2response[sct_GetCardInfo::request::kMethodInfo] = [](Reader& visitor)
	{
		if (visitor.HasMember("error"))
			return 	Rsp_Error::ReflectReader(visitor);

		return sct_GetCardInfo::response::ReflectReader(visitor);
	};
 	
	method2response[sct_Launch::request::kMethodInfo] = [](Reader& visitor)
	{
		if (visitor.HasMember("error"))
			return 	Rsp_Error::ReflectReader(visitor);

		return sct_Launch::response::ReflectReader(visitor);
	};
	method2response[sct_CheckBeforeLaunch::request::kMethodInfo] = [](Reader& visitor)
	{
		if (visitor.HasMember("error"))
			return 	Rsp_Error::ReflectReader(visitor);

		return sct_CheckBeforeLaunch::response::ReflectReader(visitor);
	};
 	

	method2response[sct_initialize::request::kMethodInfo] = [](Reader& visitor)
	{
		if (visitor.HasMember("error"))
			return 	Rsp_Error::ReflectReader(visitor);

		return sct_initialize::response::ReflectReader(visitor);
	};
	
}
bool SmartCardTool::check_sct_alive()
{
	if (sct)
	{
		return true;
	}
 	if(log)
 	{
		wstring strPrompt = L"sct is not alvie.";
		log->log(Log::Level::SEVERE, strPrompt);
 	}
	return false;
}

bool SmartCardTool::initialize(int processId, int version)
{

 	if(!check_sct_alive())
 	{
		return false;
 	}
	sct_initialize::request request;
	request.params.processId = processId;
	request.params.version = version;
	
	auto msg = sct->waitResponse(request, 100000);
	
	if (!msg)
	{
		return false;
	}
	
	if (msg->is_error)
	{
		auto error = &msg->error;
		log->error( error->error.ToString());
		return false;
	}
	auto  result = &msg->response;
	_lsServerCapabilities.swap(result->result.capabilities);
	return true;
}



SmartCardTool::~SmartCardTool()
{


}

bool SmartCardTool::GetCardInfo(CardInfoType type_, std::vector<unsigned char>& out)
{
	if (!check_sct_alive())
	{
		return false;
	}
	sct_GetCardInfo::request request;
	request.params.type_ = type_;
 	
	auto  data = sct->waitResponse(request, 40000);

	if (!data)
	{
		if (log)
		{
			wstring strPrompt = L"GetCardInfo request timeout.";
			log->log(Log::Level::SEVERE, strPrompt);
		}
		return false;
	}

	if (data->is_error)
	{
		if (log)
		{
			string strPrompt = "GetCardInfo request error." + data->error.ToJson();
			log->log(Log::Level::SEVERE, strPrompt);
		}
		return false;
	}
	auto rsp = &data->response;
	if (rsp->result.state)
	{
		out.swap(rsp->result.data.value());
		return  true;
	}
	
	if (log)
	{
		string strPrompt = "GetCardInfo failed. Reason:";
		strPrompt += rsp->result.info.value();
		log->log(Log::Level::SEVERE, strPrompt);
	}
	return false;
}



bool SmartCardTool::Launch(bool for_debug)
{
	if (!check_sct_alive())
	{
		return false;
	}
	sct_Launch::request request;
 	if(for_debug)
 	{
		request.params.launch_for_what = LaunchParam::LAUNCH_FOR_DEBUG;
 	}
	else
	{
		request.params.launch_for_what = LaunchParam::LAUNCH_FOR_RUN;
	}
	auto  data = sct->waitResponse(request, 100000);

	if (!data)
	{
		if (log)
		{
			wstring strPrompt = L"Launch request timeout.";
			log->log(Log::Level::SEVERE, strPrompt);
		}
		return false;
	}
	
	if (data->is_error)
	{
		if (log)
		{
			string strPrompt = "Launch request error." + data->error.ToJson();

			log->log(Log::Level::SEVERE, strPrompt);
		}
		return false;
	}
	auto rsp = &data->response;
	if (rsp->result.state)
	{
		if (log)
		{
			log->log(Log::Level::INFO, L"Launch successfully");
		}
		if(rsp->result.info)
		{
			m_ipAddr.swap(rsp->result.info.value().host);
			m_jdwpPort = rsp->result.info.value().jdwp_port;

		}
	}
	else
	{
		if (log)
		{
			string strPrompt = "Launch failed. Reason:";
			strPrompt += rsp->result.error.value();
			log->log(Log::Level::SEVERE, strPrompt);
		}
		return false;
	}
	return true;
}

void SmartCardTool::TerminateLaunch()
{
	if (!check_sct_alive())
	{
		return ;
	}
	sct_TerminateLaunch::notify notify;
	sct->send(notify);
	connect_state = false;
	return  ;
}

void SmartCardTool::show_message(lsMessageType type_,
	const std::string& msg)
{
	if (!check_sct_alive())
	{
		return ;
	}
	Notify_ShowMessage::notify notify;
	notify.params.type = type_;
	notify.params.message = msg;
	sct->send(notify);
}

bool SmartCardTool::CheckBeforeLaunch()
{
	if (!check_sct_alive())
	{
		return false;
	}
	sct_CheckBeforeLaunch::request request;
	auto  data = sct->waitResponse(request, 50000);

	if (!data)
	{
		wstring strPrompt = L"CheckBeforeLaunch request timeout.";
		if(log)
		{
			log->log(Log::Level::SEVERE, strPrompt);
		}

		return false;
	}
	
	if (data->is_error)
	{
		string strPrompt = "CheckBeforeLaunch request error." + data->error.ToJson();
		log->log(Log::Level::SEVERE, strPrompt);
		return false;
	}
	auto rsp = &data->response;
	if (rsp->result.state)
	{
		return true;
	}

	
	string strPrompt = "Check Before Launch JCVM failed. Reason:";
	strPrompt += rsp->result.info.value();
	log->log(Log::Level::SEVERE, strPrompt);
	return false;

}


bool SmartCardTool::Connect(SctProtocol protocol)
{
	if (!check_sct_alive())
	{
		return false;
	}
	sct_Connect::request request;

	request.params.protocol = protocol;

	auto  data = sct->waitResponse(request, 40000);

	if (!data)
	{
		wstring strPrompt = L"Connect request timeout.";
		log->log(Log::Level::SEVERE, strPrompt);
		return false;
	}
	
	if (data->is_error)
	{
		string strPrompt = "Connect request error." + data->error.ToJson();
		log->log(Log::Level::SEVERE, strPrompt);
		return false;
	}
	auto rsp =&data->response;
	if (rsp->result.state)
	{
		connect_state = true;
		return true;
	}
	else
	{
		string strPrompt = "Connect failed. Reason:";
		strPrompt += rsp->result.info.value();
		log->log(Log::Level::SEVERE, strPrompt);
		return false;
	}

}




void  SmartCardTool::DisConnect()
{
	if (!check_sct_alive())
	{
		return ;
	}
	sct_Disconnect::notify notify;
	sct->send(notify);
	connect_state = false;
	return ;
}

bool SmartCardTool::DownLoadCapFile(const string& strCapFileName)
{
	if (!check_sct_alive())
	{
		return false;
	}
	sct_DownLoadCapFile::request request;
	request.params.uri.raw_uri_ = make_file_scheme_uri(strCapFileName);
	
	auto  data = sct->waitResponse(request, 40000);
	
	if(!data)
	{
		wstring strPrompt = L"DownLoadCapFile request timeout.";
		log->log(Log::Level::SEVERE,strPrompt);
		return false;
	}

	if(data->is_error)
	{
		string strPrompt = "DownLoadCapFile request error." + data->error.ToJson();
		log->log(Log::Level::SEVERE, strPrompt);
		return false;
	}
	auto rsp = &data->response;
	if(rsp->result.state)
	{
		string strPrompt = "DownLoadCapFile successfully";
		if(rsp->result.info)
		{
			strPrompt = rsp->result.info.value();
		}
		log->log(Log::Level::INFO, strPrompt);
		return true;
	}
	else
	{
		string strPrompt = "DownLoadCapFile failed. Reason:";
		strPrompt += rsp->result.info.value();
		log->log(Log::Level::SEVERE, strPrompt);
		return false;
	}

}

void SmartCardTool::SetWindowsVisible(SetWindowVisibleParams& params)
{
	if (!check_sct_alive())
	{
		return ;
	}
	sct_SetWindowsVisible::notify notify;
	notify.params.swap(params);
	sct->send(notify);

}

void SmartCardTool::SetWindowPos(SetWindowPosParams& params)
{
	sct_SetWindowsPos::notify notify;
	notify.params.swap(params);
	sct->send(notify);
}

bool SmartCardTool::SetProtocol(SctProtocol protocol)
{
	if (!check_sct_alive())
	{
		return false;
	}
	sct_SetProtocol::request request;
 	
	request.params.protocol = protocol;
 	
	auto  data = sct->waitResponse(request, 40000);

	if (!data)
	{
		wstring strPrompt = L"SetProtocol request timeout.";
		log->log(Log::Level::SEVERE, strPrompt);
		return false;
	}
	
	if (data->is_error)
	{
		string strPrompt = "SetProtocol request error." + data->error.ToJson();
		log->log(Log::Level::SEVERE, strPrompt);
		return false;
	}
	auto rsp = &data->response;
	if (rsp->result.state)
	{
		m_curProtocol = protocol;
		return true;
	}
	else
	{
		string strPrompt = "SetProtocol failed. Reason:";
		strPrompt += rsp->result.info.value();
		log->log(Log::Level::SEVERE, strPrompt);
		return false;
	}
	
}

bool SmartCardTool::GpAuth()
{

	if (!check_sct_alive())
	{
		return false;
	}
	sct_gp_auth::request request;

	auto  data = sct->waitResponse(request, 100000);

	if (!data)
	{
		wstring strPrompt = L"gp_auth request timeout.";
		log->log(Log::Level::SEVERE, strPrompt);
		return false;
	}

	if (data->is_error)
	{
		string strPrompt = "gp_auth request error." + data->error.ToJson();
		log->log(Log::Level::SEVERE, strPrompt);
		return false;
	}
	auto rsp = &data->response;
	if (rsp->result.state)
	{
		string strPrompt = "gp_auth request successfully.";
		if (rsp->result.info)
			strPrompt = rsp->result.info.value();
		
		log->log(Log::Level::INFO, strPrompt);
		return true;
	}
	else
	{
		string strPrompt = "gp_auth failed. Reason:";
		strPrompt += rsp->result.info.value();
		log->log(Log::Level::SEVERE, strPrompt);
		return false;
	}

}

bool SmartCardTool::InstallApplet(InstallAppletParams& params)
{
	if (!check_sct_alive())
	{
		return false;
	}
	sct_InstalllApplet::request request;
	request.params.swap(params);
	auto  data = sct->waitResponse(request);

	if (!data)
	{
		wstring strPrompt = L"Install Applet request timeout.";
		log->log(Log::Level::SEVERE, strPrompt);
		return false;
	}
	
	if (data->is_error)
	{
		string strPrompt = "Install Applet request error." + data->error.ToJson();
		log->log(Log::Level::SEVERE, strPrompt);
		return false;
	}
	auto rsp = &data->response;
	if (rsp->result.state)
	{
		string strPrompt = "Install Applet successfully.";
		if (rsp->result.info)
			strPrompt = rsp->result.info.value();
		log->log(Log::Level::INFO, strPrompt);
		return true;
	}
	else
	{
		string strPrompt = "Install Applet failed. Reason:";
		strPrompt += rsp->result.info.value();
		log->log(Log::Level::SEVERE, strPrompt);
		return false;
	}
}


bool SmartCardTool::Transmit(const std::vector<unsigned char>& cmdApdu, std::vector<unsigned char>& rspApdu)
{

	if (!check_sct_alive())
	{
		return false;
	}
	sct_Transmit::request request;
	request.params.command = cmdApdu;
	
	auto  data = sct->waitResponse(request);

	if (!data)
	{
		wstring strPrompt = L"Transmit request timeout.";
		log->log(Log::Level::SEVERE, strPrompt);
		return false;
	}

	if (data->is_error)
	{
		string strPrompt = "Transmit request error." + data->error.ToJson();
		log->log(Log::Level::SEVERE, strPrompt);
		return false;
	}
	auto rsp = &data->response;
	if (rsp->result.state)
	{
		rspApdu.swap(rsp->result.data.value());
		//log->log(Log::Level::INFO, L"Transmit successfully");
		return true;
	}
	else
	{
		string strPrompt = "Transmit failed. Reason:";
		strPrompt += rsp->result.info.value();
		log->log(Log::Level::SEVERE, strPrompt);
		return false;
	}
	

}
