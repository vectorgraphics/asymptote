//  Exception implementation file  -------------------------------------------//

//  Copyright © 2002 Beman Dawes
//  Copyright © 2001 Dietmar Kühl 
//  Use, modification, and distribution is subject to the Boost Software
//  License, Version 1.0. (See accompanying file LICENSE_1_0.txt or copy
//  at http://www.boost.org/LICENSE_1_0.txt)

//  See library home page at http://www.boost.org/libs/filesystem

//----------------------------------------------------------------------------//

// define BOOST_FILESYSTEM_SOURCE so that <boost/filesystem/config.hpp> knows
// the library is being built (possibly exporting rather than importing code)
#define BOOST_FILESYSTEM_SOURCE 

#include <boost/filesystem/config.hpp>
#include <boost/filesystem/exception.hpp>

namespace fs = boost::filesystem;

#include <cstring> // SGI MIPSpro compilers need this
#include <string>

# ifdef BOOST_NO_STDC_NAMESPACE
    namespace std { using ::strerror; }
# endif

// BOOST_POSIX or BOOST_WINDOWS specify which API to use.
# if !defined( BOOST_WINDOWS ) && !defined( BOOST_POSIX )
#   if defined(_WIN32) || defined(__WIN32__) || defined(WIN32) || defined(__CYGWIN__)
#     define BOOST_WINDOWS
#   else
#     define BOOST_POSIX
#   endif
# endif

# if defined( BOOST_WINDOWS )
#   include "windows.h"
# else
#   include <errno.h> // for POSIX error codes
# endif

#include <boost/config/abi_prefix.hpp> // must be the last header

//----------------------------------------------------------------------------//

namespace
{
# ifdef BOOST_WINDOWS
  std::string system_message( int sys_err_code )
  {
    std::string str;
    LPVOID lpMsgBuf;
    ::FormatMessageA( 
        FORMAT_MESSAGE_ALLOCATE_BUFFER | 
        FORMAT_MESSAGE_FROM_SYSTEM | 
        FORMAT_MESSAGE_IGNORE_INSERTS,
        NULL,
        sys_err_code,
        MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), // Default language
        (LPSTR) &lpMsgBuf,
        0,
        NULL 
    );
    str += static_cast<LPCSTR>(lpMsgBuf);
    ::LocalFree( lpMsgBuf ); // free the buffer
    while ( str.size()
      && (str[str.size()-1] == '\n' || str[str.size()-1] == '\r') )
        str.erase( str.size()-1 );
    return str;
  }
# else
  std::string system_message( int )
  {
    std::string str;
    str += std::strerror( errno );
    return str;
  }
# endif

  struct ec_xlate { int sys_ec; fs::error_code ec; };
  const ec_xlate ec_table[] =
  {
#     ifdef BOOST_WINDOWS
    { ERROR_ACCESS_DENIED, fs::security_error },
    { ERROR_INVALID_ACCESS, fs::security_error },
    { ERROR_SHARING_VIOLATION, fs::security_error },
    { ERROR_LOCK_VIOLATION, fs::security_error },
    { ERROR_LOCKED, fs::security_error },
    { ERROR_NOACCESS, fs::security_error },
    { ERROR_WRITE_PROTECT, fs::read_only_error },
    { ERROR_NOT_READY, fs::io_error },
    { ERROR_SEEK, fs::io_error },
    { ERROR_READ_FAULT, fs::io_error },
    { ERROR_WRITE_FAULT, fs::io_error },
    { ERROR_CANTOPEN, fs::io_error },
    { ERROR_CANTREAD, fs::io_error },
    { ERROR_CANTWRITE, fs::io_error },
    { ERROR_DIRECTORY, fs::path_error },
    { ERROR_INVALID_NAME, fs::path_error },
    { ERROR_FILE_NOT_FOUND, fs::not_found_error },
    { ERROR_PATH_NOT_FOUND, fs::not_found_error },
    { ERROR_DEV_NOT_EXIST, fs::not_found_error },
    { ERROR_DEVICE_IN_USE, fs::busy_error },
    { ERROR_OPEN_FILES, fs::busy_error },
    { ERROR_BUSY_DRIVE, fs::busy_error },
    { ERROR_BUSY, fs::busy_error },
    { ERROR_FILE_EXISTS, fs::already_exists_error },
    { ERROR_ALREADY_EXISTS, fs::already_exists_error },
    { ERROR_DIR_NOT_EMPTY, fs::not_empty_error },
    { ERROR_HANDLE_DISK_FULL, fs::out_of_space_error },
    { ERROR_DISK_FULL, fs::out_of_space_error },
    { ERROR_OUTOFMEMORY, fs::out_of_memory_error },
    { ERROR_NOT_ENOUGH_MEMORY, fs::out_of_memory_error },
    { ERROR_TOO_MANY_OPEN_FILES, fs::out_of_resource_error }
#     else
    { EACCES, fs::security_error },
    { EROFS, fs::read_only_error },
    { EIO, fs::io_error },
    { ENAMETOOLONG, fs::path_error },
    { ENOENT, fs::not_found_error },
    { ENOTDIR, fs::not_directory_error },
    { EAGAIN, fs::busy_error },
    { EBUSY, fs::busy_error },
    { ETXTBSY, fs::busy_error },
    { EEXIST, fs::already_exists_error },
    { ENOTEMPTY, fs::not_empty_error },
    { EISDIR, fs::is_directory_error },
    { ENOSPC, fs::out_of_space_error },
    { ENOMEM, fs::out_of_memory_error },
    { EMFILE, fs::out_of_resource_error }
#     endif
  };

  fs::error_code lookup_error( int sys_err_code )
  {
    for ( const ec_xlate * cur = &ec_table[0];
      cur != ec_table
        + sizeof(ec_table)/sizeof(ec_xlate); ++cur )
    {
      if ( sys_err_code == cur->sys_ec ) return cur->ec;
    }
    return fs::system_error; // general system error code
  }

  // These helper functions work for POSIX and Windows. For systems where
  // path->native_file_string() != path->native_directory_string(), more
  // care would be required to get the right form for the function involved.

  std::string other_error_prep(
    const std::string & who,
    const std::string & message )
  {
    return who + ": " + message;
  }

  std::string other_error_prep(
    const std::string & who,
    const fs::path & path1,
    const std::string & message )
  {
    return who + ": \"" + path1.native_file_string() + "\": " + message;
  }

  std::string system_error_prep(
    const std::string & who,
    const fs::path & path1,
    int sys_err_code )
  {
    return who + ": \"" + path1.native_file_string() + "\": "
      + system_message( sys_err_code );
  }

  std::string system_error_prep(
    const std::string & who,
    const fs::path & path1,
    const fs::path & path2,
    int sys_err_code )
  {
    return who + ": \"" + path1.native_file_string()
      + "\", \"" + path2.native_file_string() + "\": "
      + system_message( sys_err_code );
  }

  const fs::path empty_path;
  const std::string empty_string;
} // unnamed namespace

namespace boost
{
  namespace filesystem
  {
//  filesystem_error m_imp class  --------------------------------------------//
//  see www.boost.org/more/error_handling.html for implementation rationale

    class filesystem_error::m_imp
    {
    public:
      std::string     m_who;
      path            m_path1;
      path            m_path2;
      std::string     m_what;
    };


//  filesystem_error implementation  -----------------------------------------//

    filesystem_error::filesystem_error(
      const std::string & who,
      const std::string & message )
      : m_sys_err(0), m_err(other_error)
    {
      try
      {
        m_imp_ptr.reset( new m_imp );
        m_imp_ptr->m_who = who;
        m_imp_ptr->m_what = other_error_prep( who, message );
      }
      catch (...) { m_imp_ptr.reset(); }
    }
 
    filesystem_error::filesystem_error(
      const std::string & who,
      const path & path1,
      const std::string & message,
      error_code ec )
      : m_sys_err(0), m_err(ec)
    {
      try
      {
        m_imp_ptr.reset( new m_imp );
        m_imp_ptr->m_who = who;
        m_imp_ptr->m_what = other_error_prep( who, path1, message );
        m_imp_ptr->m_path1 = path1;
      }
      catch (...) { m_imp_ptr.reset(); }
    }
 
    filesystem_error::filesystem_error(
      const std::string & who,
      const path & path1,
      int sys_err_code )
      : m_sys_err(sys_err_code), m_err(lookup_error(sys_err_code))
    {
      try
      {
        m_imp_ptr.reset( new m_imp );
        m_imp_ptr->m_who = who;
        m_imp_ptr->m_what = system_error_prep( who, path1, sys_err_code );
        m_imp_ptr->m_path1 = path1;
      }
      catch (...) { m_imp_ptr.reset(); }
    }

    filesystem_error::filesystem_error(
      const std::string & who,
      const path & path1,
      const path & path2,
      int sys_err_code )
      : m_sys_err(sys_err_code), m_err(lookup_error(sys_err_code))
    {
      try
      {
        m_imp_ptr.reset( new m_imp );
        m_imp_ptr->m_who = who;
        m_imp_ptr->m_what = system_error_prep( who, path1, path2, sys_err_code );
        m_imp_ptr->m_path1 = path1;
        m_imp_ptr->m_path2 = path2;
      }
      catch (...) { m_imp_ptr.reset(); }
    }

    filesystem_error::~filesystem_error() throw()
    {
    }

    const std::string & filesystem_error::who() const
    {
      return m_imp_ptr.get() == 0 ? empty_string : m_imp_ptr->m_who;
    }

    const path & filesystem_error::path1() const
    {
      return m_imp_ptr.get() == 0 ? empty_path : m_imp_ptr->m_path1;
    }

    const path & filesystem_error::path2() const
    {
      return m_imp_ptr.get() == 0 ? empty_path : m_imp_ptr->m_path2;
    }

    const char * filesystem_error::what() const throw()
    {
      return m_imp_ptr.get() == 0 ? empty_string.c_str()
                                  : m_imp_ptr->m_what.c_str();
    }

    namespace detail
    {
      BOOST_FILESYSTEM_DECL int system_error_code() // artifact of POSIX and WINDOWS error reporting
      {
  #   ifdef BOOST_WINDOWS
        return ::GetLastError();
  #   else
        return errno; // GCC 3.1 won't accept ::errno
  #   endif
      }
    } // namespace detail
  } // namespace filesystem
} // namespace boost
 
