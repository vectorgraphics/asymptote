#pragma once

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#ifdef HAVE_LSP
#include <boost/optional.hpp>
#include <boost/none.hpp>
using boost::optional;
#define nullopt boost::none
using boost::make_optional;
#else
#include "optional.hpp"
using nonstd::optional;
using nonstd::nullopt;
#define boost nonstd
#endif
