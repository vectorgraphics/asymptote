#pragma once
#include "LibLsp/JsonRpc/RequestInMessage.h"
#include "LibLsp/lsp/lsDocumentUri.h"
#include "LibLsp/lsp/lsAny.h"
#include "LibLsp/JsonRpc/NotificationInMessage.h"

struct DownLoadCapFileParams
{
	lsDocumentUri uri;
	MAKE_SWAP_METHOD(DownLoadCapFileParams, uri);
};
MAKE_REFLECT_STRUCT(DownLoadCapFileParams, uri);

struct NormalActionResult
{
	bool state = false;
	boost::optional<std::vector<uint8_t>> data;
	boost::optional<std::string> info;
	MAKE_SWAP_METHOD(NormalActionResult, state, data, info);
};
MAKE_REFLECT_STRUCT(NormalActionResult, data, state, info)

DEFINE_REQUEST_RESPONSE_TYPE(sct_DownLoadCapFile, DownLoadCapFileParams, NormalActionResult, "sct/download_cap");




enum class SctProtocol :uint8_t
{
	T01 = 0, T0 = 1, T1 = 2,
};
MAKE_REFLECT_TYPE_PROXY(SctProtocol);

struct ConnectParams
{
	
	SctProtocol protocol= SctProtocol::T01;
	boost::optional<std::string> reader;
	boost::optional<lsp::Any> data;
	MAKE_SWAP_METHOD(ConnectParams, reader, protocol,data);
};
MAKE_REFLECT_STRUCT(ConnectParams, reader, protocol, data);
DEFINE_REQUEST_RESPONSE_TYPE(sct_Connect, ConnectParams, NormalActionResult, "sct/connect");



struct SetProtocolParams
{

	SctProtocol protocol = SctProtocol::T01;

};
MAKE_REFLECT_STRUCT(SetProtocolParams, protocol);

DEFINE_REQUEST_RESPONSE_TYPE(sct_SetProtocol, SetProtocolParams, NormalActionResult, "sct/set_protocol");

struct GPAuthParams
{
	boost::optional < std::string>  scp;
	boost::optional < std::string > key;
	boost::optional < lsp::Any >   option;
	MAKE_SWAP_METHOD(GPAuthParams, key, scp, option);
};
MAKE_REFLECT_STRUCT(GPAuthParams, key, scp, option);
DEFINE_REQUEST_RESPONSE_TYPE(sct_gp_auth, GPAuthParams, NormalActionResult ,"sct/gp_auth");



struct InstallAppletParams
{
	std::vector<uint8_t> package_aid;
	std::vector<uint8_t> applet_aid;
	boost::optional < std::vector<uint8_t>> instance_aid;
	boost::optional<std::vector<uint8_t>>  authority;
	boost::optional<std::vector<uint8_t>>  parameters;
	MAKE_SWAP_METHOD(InstallAppletParams, package_aid, applet_aid, instance_aid, authority, parameters);
};
MAKE_REFLECT_STRUCT(InstallAppletParams, package_aid, applet_aid, instance_aid, authority, parameters);
DEFINE_REQUEST_RESPONSE_TYPE(sct_InstalllApplet, InstallAppletParams, NormalActionResult, "sct/install_applet");


struct TransmitParams
{
	std::vector<unsigned char> command;
	MAKE_SWAP_METHOD(TransmitParams, command);
};
MAKE_REFLECT_STRUCT(TransmitParams, command);

DEFINE_REQUEST_RESPONSE_TYPE(sct_Transmit, TransmitParams, NormalActionResult,"sct/transmit");

DEFINE_NOTIFICATION_TYPE(sct_Disconnect,JsonNull, "sct/disconnect")


struct SetWindowPosParams{
	int X = 0;
	int Y = 0;
	int cx = 100;
	int cy = 100;
	
	MAKE_SWAP_METHOD(SetWindowPosParams, X, Y, cx, cy);
};
MAKE_REFLECT_STRUCT(SetWindowPosParams, X, Y, cx, cy);
DEFINE_NOTIFICATION_TYPE(sct_SetWindowsPos, SetWindowPosParams, "sct/set_windows_pos")

struct SetWindowVisibleParams
{
	static const int  HIDE = 0;
	static const int  MINSIZE = 1;
	static const int  MAXSIZE = 2;
	static const int  NORMAL = 3;
	int state = NORMAL;
	MAKE_SWAP_METHOD(SetWindowVisibleParams, state);
};
MAKE_REFLECT_STRUCT(SetWindowVisibleParams, state);
DEFINE_NOTIFICATION_TYPE(sct_SetWindowsVisible, SetWindowVisibleParams, "sct/set_windows_visible")



enum  CardInfoType:uint32_t
{
	ATR_TYPE = 0,
	ATS_TYPE = 1,
};
MAKE_REFLECT_TYPE_PROXY(CardInfoType);


struct  GetCardInfoParams
{
	CardInfoType type_;
};
MAKE_REFLECT_STRUCT(GetCardInfoParams, type_);

DEFINE_REQUEST_RESPONSE_TYPE(sct_GetCardInfo, GetCardInfoParams, NormalActionResult, "sct/get_card_info");


struct JdwpInfo
{
	std::string host="127.0.0.1";
	uint32_t jdwp_port = 9075;

};

MAKE_REFLECT_STRUCT(JdwpInfo, host, jdwp_port);

struct  LaunchResult
{
	bool state;
	boost::optional<JdwpInfo> info;
	boost::optional<std::string> error;
	MAKE_SWAP_METHOD(LaunchResult, state, info, error);
};
MAKE_REFLECT_STRUCT(LaunchResult, state, info, error);


struct JcvmOutputParams
{
	std::string  data;
	MAKE_SWAP_METHOD(JcvmOutputParams, data);
};
MAKE_REFLECT_STRUCT(JcvmOutputParams, data);

DEFINE_NOTIFICATION_TYPE(sct_NotifyJcvmOutput, JcvmOutputParams,"sct/notify_jcvm_output");


struct LaunchParam
{
	enum
	{
		LAUNCH_FOR_DEBUG = 0,
		LAUNCH_FOR_RUN = 1
	};
	boost::optional<uint32_t> launch_for_what;
	
};
MAKE_REFLECT_STRUCT(LaunchParam, launch_for_what);


DEFINE_REQUEST_RESPONSE_TYPE(sct_Launch, LaunchParam, LaunchResult , "sct/launch");


DEFINE_REQUEST_RESPONSE_TYPE(sct_CheckBeforeLaunch, JsonNull, NormalActionResult, "sct/check_before_launch");



DEFINE_NOTIFICATION_TYPE(sct_NotifyDisconnect, JsonNull,"sct/notify_disconnect");


DEFINE_NOTIFICATION_TYPE(sct_TerminateLaunch, JsonNull, "sct/terminate_launch");




struct sctInitializeParams {
	// The process Id of the parent process that started
	// the server. Is null if the process has not been started by another process.
	// If the parent process is not alive then the server should exit (see exit
	// notification) its process.
	boost::optional<int> processId;

	// User provided initialization options.
	boost::optional<lsp::Any> initializationOptions;
	boost::optional<int> version;

};
MAKE_REFLECT_STRUCT(sctInitializeParams,processId,initializationOptions, version);

struct sctServerCapabilities {
	bool gp_auth = false;
	bool gp_key = false;
	boost::optional<int> version;
	MAKE_SWAP_METHOD(sctServerCapabilities, gp_auth, gp_key, version);
};
MAKE_REFLECT_STRUCT(sctServerCapabilities, gp_auth, gp_key, version);


struct stcInitializeResult
{
	sctServerCapabilities   capabilities;
};
MAKE_REFLECT_STRUCT(stcInitializeResult, capabilities);

DEFINE_REQUEST_RESPONSE_TYPE(sct_initialize, sctInitializeParams, stcInitializeResult, "sct/initialize");
