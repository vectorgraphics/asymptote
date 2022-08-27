
#include "LibLsp/JsonRpc/StreamMessageProducer.h"
#include <cassert>

#include "LibLsp/JsonRpc/stream.h"


bool StartsWith(std::string value, std::string start);
bool StartsWith(std::string value, std::string start) {
        if (start.size() > value.size())
                return false;
        return std::equal(start.begin(), start.end(), value.begin());
}

using  namespace std;
namespace
{
        string JSONRPC_VERSION = "2.0";
        string CONTENT_LENGTH_HEADER = "Content-Length";
        string CONTENT_TYPE_HEADER = "Content-Type";
        string JSON_MIME_TYPE = "application/json";
        string CRLF = "\r\n";

}

  void StreamMessageProducer::parseHeader(std::string& line, StreamMessageProducer::Headers& headers)
  {
          int sepIndex = line.find(':');
          if (sepIndex >= 0) {
                  auto key = line.substr(0, sepIndex);
              if(key == CONTENT_LENGTH_HEADER)
              {
                        headers.contentLength = atoi(line.substr(sepIndex + 1).data());
              }
                  else if(key == CONTENT_TYPE_HEADER)
                  {
                          int charsetIndex = line.find("charset=");
                          if (charsetIndex >= 0)
                                  headers.charset = line.substr(charsetIndex + 8);
                  }
          }
  }


void StreamMessageProducer::listen(MessageConsumer callBack)
{
        if(!input)
                return;

        keepRunning = true;
        bool newLine = false;
        Headers headers;
        string headerBuilder ;
        string debugBuilder ;
        // Read the content length. It is terminated by the "\r\n" sequence.
        while (keepRunning)
        {
                if(input->bad())
                {
                        std::string info = "Input stream is bad.";
                        auto what = input->what();
                        if (what.size())
                        {
                                info += "Reason:";
                                info += input->what();
                        }
                        MessageIssue issue(info, lsp::Log::Level::SEVERE);
                        issueHandler.handle(std::move(issue));
                        return;
                }
                if(input->fail())
                {
                        std::string info = "Input fail.";
                        auto what = input->what();
                        if(what.size())
                        {
                                info += "Reason:";
                                info += input->what();
                        }
                        MessageIssue issue(info, lsp::Log::Level::WARNING);
                        issueHandler.handle(std::move(issue));
                        if(input->need_to_clear_the_state())
                                input->clear();
                        else
                        {
                                return;
                        }
                }
                int c = input->get();
                if (c == EOF) {
                        // End of input stream has been reached
                        keepRunning = false;
                }
                else
                {

                        debugBuilder.push_back((char)c);
                        if (c == '\n')
                        {
                                if (newLine) {
                                        // Two consecutive newlines have been read, which signals the start of the message content
                                        if (headers.contentLength <= 0)
                                        {
                                                string info = "Unexpected token:" + debugBuilder;
                                                info = +"  (expected Content-Length: sequence);";
                                                 MessageIssue issue(info, lsp::Log::Level::WARNING);
                                                 issueHandler.handle(std::move(issue));
                                        }
                                        else {
                                                bool result = handleMessage(headers,callBack);
                                                if (!result)
                                                        keepRunning = false;
                                                newLine = false;
                                        }
                                        headers.clear();
                                        debugBuilder.clear();
                                }
                                else if (!headerBuilder.empty()) {
                                        // A single newline ends a header line
                                        parseHeader(headerBuilder, headers);
                                        headerBuilder.clear();
                                }
                                newLine = true;
                        }
                        else if (c != '\r') {
                                // Add the input to the current header line

                                headerBuilder.push_back((char)c);
                                newLine = false;
                        }
                }
        }

}

void StreamMessageProducer::bind(std::shared_ptr<lsp::istream>_in)
{
        input = _in;
}

bool StreamMessageProducer::handleMessage(Headers& headers ,MessageConsumer callBack)
{
                         // Read content.
        auto content_length = headers.contentLength;
         std::string content(content_length,0);
         auto data = &content[0];
         input->read(data, content_length);
         if (input->bad())
         {
                 std::string info = "Input stream is bad.";
                 auto what = input->what();
                 if (!what.empty())
                 {
                         info += "Reason:";
                         info += input->what();
                 }
                 MessageIssue issue(info, lsp::Log::Level::SEVERE);
                 issueHandler.handle(std::move(issue));
                 return false;
         }

         if (input->eof())
         {
                 MessageIssue issue("No more input when reading content body", lsp::Log::Level::INFO);
                 issueHandler.handle(std::move(issue));
                 return false;
         }
         if (input->fail())
         {
                 std::string info = "Input fail.";
                 auto what = input->what();
                 if (!what.empty())
                 {
                         info += "Reason:";
                         info += input->what();
                 }
                 MessageIssue issue(info, lsp::Log::Level::WARNING);
                 issueHandler.handle(std::move(issue));
                 if (input->need_to_clear_the_state())
                         input->clear();
                 else
                 {
                         return false;
                 }
         }

         callBack(std::move(content));

        return true;
}

