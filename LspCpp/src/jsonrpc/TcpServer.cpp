//
// server.cpp

#include "LibLsp/JsonRpc/TcpServer.h"
#include <signal.h>
#include <utility>
#include <boost/bind/bind.hpp>

#include "LibLsp/JsonRpc/MessageIssue.h"
#include "LibLsp/JsonRpc/stream.h"


namespace lsp {
        struct tcp_connect_session;


                class tcp_stream_wrapper :public istream, public ostream
            {
            public:
                    tcp_stream_wrapper(tcp_connect_session& _w);

                tcp_connect_session& session;
                std::atomic<bool> quit{};
                std::shared_ptr < MultiQueueWaiter> request_waiter;
                ThreadedQueue< char > on_request;
            std::string error_message;


                bool fail() override
                {
                    return  bad();
                }



                bool eof() override
                {
                    return  bad();
                }
                bool good() override
                {
                    return  !bad();
                }
                tcp_stream_wrapper& read(char* str, std::streamsize count)
                  override
                {
                auto some = on_request.TryDequeueSome(static_cast<size_t>( count ));
                memcpy(str,some.data(),some.size());
                for (std::streamsize i = some.size(); i < count; ++i)
                {
                    str[i] = static_cast<char>(get());
                }

                    return *this;
                }
                int get() override
                {
                    return on_request.Dequeue();
                }

                bool bad() override;

                tcp_stream_wrapper& write(const std::string& c) override;

                tcp_stream_wrapper& write(std::streamsize _s) override;

                tcp_stream_wrapper& flush() override
                {
                    return *this;
                }
                void reset_state()
                {
                    return;
                }

                    void clear() override
                {

                }

                    std::string what() override;
                    bool need_to_clear_the_state() override
                    {
                return false;
                    }
            };
            struct tcp_connect_session:std::enable_shared_from_this<tcp_connect_session>
            {
            /// Buffer for incoming data.
            std::array<unsigned char, 8192> buffer_;
            boost::asio::ip::tcp::socket socket_;
            /// Strand to ensure the connection's handlers are not called concurrently.
            boost::asio::io_context::strand strand_;
            std::shared_ptr<tcp_stream_wrapper>  proxy_;
            explicit tcp_connect_session(boost::asio::io_context& io_context, boost::asio::ip::tcp::socket&& _socket)
                    : socket_(std::move(_socket)), strand_(io_context), proxy_(new tcp_stream_wrapper(*this))
            {
                do_read();
            }
            void do_write(const char* data, size_t size)
            {
                socket_.async_write_some(boost::asio::buffer(data, size),
                                         boost::asio::bind_executor(strand_,[this](boost::system::error_code ec, std::size_t n)
                                         {
                                             if (!ec)
                                             {
                                                 return;
                                             }
                                             proxy_->error_message = ec.message();

                                         }));
            }
            void do_read()
            {
                socket_.async_read_some(boost::asio::buffer(buffer_),
            boost::asio::bind_executor(strand_,
                [this](boost::system::error_code ec, size_t bytes_transferred)
                {
                    if (!ec)
                    {
                        std::vector<char> elements(buffer_.data(), buffer_.data() + bytes_transferred);
                        proxy_->on_request.EnqueueAll(std::move(elements), false);
                        do_read();
                        return;
                    }
                    proxy_->error_message = ec.message();

                }));
            }
            };

        tcp_stream_wrapper::tcp_stream_wrapper(tcp_connect_session& _w): session(_w)
        {
        }

        bool tcp_stream_wrapper::bad()
    {
        return !session.socket_.is_open();
    }

        tcp_stream_wrapper& tcp_stream_wrapper::write(const std::string& c)
        {
            size_t off = 0;
            for(;off < c.size();){
                if(off + 1024 < c.size()){
                    session.do_write(c.data()+off,1024);
                    off += 1024;
                }else{
                    session.do_write(c.data()+off,c.size() - off);
                    break;
                }
            }
            return *this;
        }

    tcp_stream_wrapper& tcp_stream_wrapper::write(std::streamsize _s)
    {
        auto s = std::to_string(_s);
        session.do_write(s.data(),s.size());
        return *this;
    }

        std::string tcp_stream_wrapper::what()
        {
        if (error_message.size())
            return error_message;

       if(! session.socket_.is_open())
       {
           return  "Socket is not open.";
       }
                return {};
        }

    struct TcpServer::Data
    {
        Data(
            lsp::Log& log, uint32_t _max_workers) :
                    acceptor_(io_context_), _log(log)
            {
            }

            ~Data()
            {

            }
        /// The io_context used to perform asynchronous operations.
        boost::asio::io_context io_context_;

        std::shared_ptr<boost::asio::io_service::work> work;

        std::shared_ptr<tcp_connect_session> _connect_session;
        /// Acceptor used to listen for incoming connections.
        boost::asio::ip::tcp::acceptor acceptor_;

        lsp::Log& _log;

    };

            TcpServer::~TcpServer()
            {
            delete d_ptr;
            }

        TcpServer::TcpServer(const std::string& address, const std::string& port,
            std::shared_ptr < MessageJsonHandler> json_handler,
            std::shared_ptr < Endpoint> localEndPoint, lsp::Log& log, uint32_t _max_workers)
            : point(json_handler, localEndPoint, log,lsp::JSONStreamStyle::Standard, _max_workers),d_ptr(new Data( log, _max_workers))

        {

            d_ptr->work = std::make_shared<boost::asio::io_service::work>(d_ptr->io_context_);

            // Open the acceptor with the option to reuse the address (i.e. SO_REUSEADDR).
            boost::asio::ip::tcp::resolver resolver(d_ptr->io_context_);
            boost::asio::ip::tcp::endpoint endpoint =
                *resolver.resolve(address, port).begin();
            d_ptr->acceptor_.open(endpoint.protocol());
            d_ptr->acceptor_.set_option(boost::asio::ip::tcp::acceptor::reuse_address(true));
            try
            {
                d_ptr->acceptor_.bind(endpoint);
            }
            catch (boost::system::system_error & e)
            {
                std::string temp = "Socket Server  bind failed.";
                d_ptr->_log.log(lsp::Log::Level::INFO , temp + e.what());
                return;
            }
            d_ptr->acceptor_.listen();

            do_accept();
            std::string desc = "Socket TcpServer " + address + " " + port + " start.";
            d_ptr->_log.log(lsp::Log::Level::INFO, desc);
        }

        void TcpServer::run()
        {
            // The io_context::run() call will block until all asynchronous operations
            // have finished. While the TcpServer is running, there is always at least one
            // asynchronous operation outstanding: the asynchronous accept call waiting
            // for new incoming connections.
            d_ptr->io_context_.run();

        }

        void TcpServer::stop()
        {
            try
            {
                if(d_ptr->work)
                    d_ptr->work.reset();

                do_stop();
            }
            catch (...)
            {
            }
        }

        void TcpServer::do_accept()
        {
            d_ptr->acceptor_.async_accept(
                [this](boost::system::error_code ec, boost::asio::ip::tcp::socket socket)
                {
                    // Check whether the TcpServer was stopped by a signal before this
                    // completion handler had a chance to run.
                    if (!d_ptr->acceptor_.is_open())
                    {
                        return;
                    }

                    if (!ec)
                    {
                        if(d_ptr->_connect_session)
                        {
                                if(d_ptr->_connect_session->socket_.is_open())
                                {
                                std::string desc = "Disconnect previous client " + d_ptr->_connect_session->socket_.local_endpoint().address().to_string();
                                d_ptr->_log.log(lsp::Log::Level::INFO, desc);
                                d_ptr->_connect_session->socket_.close();
                                }

                            point.stop();
                        }
                        auto local_point = socket.local_endpoint();

                        std::string desc = ("New client " + local_point.address().to_string() + " connect.");
                        d_ptr->_log.log(lsp::Log::Level::INFO, desc);
                        d_ptr->_connect_session = std::make_shared<tcp_connect_session>(d_ptr->io_context_,std::move(socket));

                        point.startProcessingMessages(d_ptr->_connect_session->proxy_, d_ptr->_connect_session->proxy_);
                        do_accept();
                    }
                });
        }

        void TcpServer::do_stop()
        {
            d_ptr->acceptor_.close();

            point.stop();

        }

    } // namespace
