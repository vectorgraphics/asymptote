#if __cplusplus < 201703L
#ifndef boost
#include <boost/optional.hpp>
using boost::optional;
#endif
#else
using std::optional;
#endif
