CXX=g++

CFLAGS = -Wall
INCLUDES=-I. -ILibLsp/lsp/extention/jdtls/ -ILibLsp/JsonRpc/ -ILibLsp/JsonRpc/lsp/extention/jdtls \
	-Ithird_party/threadpool -Ithird_party/utfcpp/source -Ithird_party/rapidjson/include
CXXFLAGS = -std=c++14
OPTFLAGS = -O3

ALL_CXXFLAGS = $(CFLAGS) $(CXXFLAGS) $(CPPFLAGS) $(INCLUDES) $(OPTFLAGS)

NETWORKS_DETAIL = $(addprefix detail/, uri_advance_parts.o \
	uri_normalize.o uri_parse.o uri_parse_authority.o uri_resolve.o)
NETWORK_FILES = $(addprefix uri/, uri.o uri_builder.o uri_errors.o $(NETWORKS_DETAIL))
LSP_FILES = extention/sct/sct.o general/initialize.o lsp.o lsp_diagnostic.o \
	ProtocolJsonHandler.o textDocument/textDocument.o Markup/Markup.o ParentProcessWatcher.o \
	utils.o working_files.o
JSONRPC_FILES = TcpServer.o threaded_queue.o WebSocketServer.o RemoteEndPoint.o \
	Endpoint.o message.o MessageJsonHandler.o serializer.o StreamMessageProducer.o \
	Context.o GCThreadContext.o

OFILES = $(addprefix ./network/,$(NETWORK_FILES)) \
	$(addprefix ./LibLsp/lsp/, $(LSP_FILES)) \
	$(addprefix ./LibLsp/JsonRpc/, $(JSONRPC_FILES))

HEADERS = $(shell find ./LibLsp ./network -regex ".*\.\(h\|hpp\)")

default: liblspcpp.a headers.tar.gz

liblspcpp.a: $(OFILES)
	ar -r $@ $^

headers.tar.gz: $(HEADERS) macro_map.h
	tar -czf $@ $^

%.o: %.cpp
	$(CXX) $(ALL_CXXFLAGS) $< -c -o $@

clean:
	find ./ -name *.o | xargs rm -rf
	rm -rf *.a *.tar.gz
