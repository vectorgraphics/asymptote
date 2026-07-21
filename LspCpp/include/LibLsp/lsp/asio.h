// Copyright (c) Chemical Language Foundation 2025.

#pragma once

#ifdef LSPCPP_USE_STANDALONE_ASIO

#include <asio.hpp>

typedef asio::error_code asio_error_code;

typedef asio::system_error asio_system_error;

#else

#include <boost/asio.hpp>

namespace asio = boost::asio;

typedef boost::system::error_code asio_error_code;

typedef boost::system::system_error asio_system_error;

#endif