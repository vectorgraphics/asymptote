#pragma once

#include <LibLsp/JsonRpc/RemoteEndPoint.h>
#include <LibLsp/JsonRpc/Endpoint.h>
#include <LibLsp/lsp/ProtocolJsonHandler.h>
#include "protocol.h"
enum class lsMessageType;
enum CardInfoType : unsigned;
enum class SctProtocol : unsigned char;
struct InstallAppletParams;


using namespace std;

class ModeState;
namespace sct
{
	class ProtocolJsonHandler : public MessageJsonHandler
	{
	public:
		ProtocolJsonHandler();
	};

}
class SmartCardTool 
{
	//IP �����ֽ���
	std::string m_ipAddr;
	volatile uint16_t m_jdwpPort;

	
	SctProtocol m_curProtocol;
public:
	enum JCIDE_VERSION
	{
		V5_KIND = 5
	};
	SmartCardTool();
	~SmartCardTool();

	
	bool GetCardInfo(CardInfoType type_,std::vector<unsigned char>&);
	bool Launch(bool for_debug = false);

	void TerminateLaunch();
	void show_message(lsMessageType type_,const std::string& msg);
	bool CheckBeforeLaunch();
	
	 string GetIpAddr(){return m_ipAddr;}
	
	 uint16_t GetJdwpPort(){return m_jdwpPort;}
	


	 void SetWindowsVisible(SetWindowVisibleParams&);
	 void SetWindowPos(SetWindowPosParams&);
	
	 SctProtocol GetProtocol(){return m_curProtocol;}


	bool SetProtocol(SctProtocol protocol = SctProtocol::T01);

	bool GpAuth();


	bool InstallApplet(InstallAppletParams&);;
	
	bool DownLoadCapFile(const string& strCapFileName);



	bool IsConnected() const {
	 
		return connect_state;
	}
	
	string GetErrorString() { return {}; };
	
	bool Connect(SctProtocol protocol = SctProtocol::T01);
	
	void DisConnect();




	bool Transmit(const std::vector<unsigned char>& cmdApdu, std::vector<unsigned char>& rspApdu) ;
	

	lsp::Log* log;
	volatile bool connect_state = false;

	std::shared_ptr<RemoteEndPoint> sct;


	bool initialize(int processId,int version);
	const sctServerCapabilities& getServerCapabilities()const
	{
		return _lsServerCapabilities;
	}
private:
	sctServerCapabilities _lsServerCapabilities;
	bool check_sct_alive();
};
