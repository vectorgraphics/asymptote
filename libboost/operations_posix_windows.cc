//  directory_posix_windows.cpp  ---------------------------------------------//

//  Copyright © 2002 Beman Dawes
//  Copyright © 2001 Dietmar Kühl
//  Use, modification, and distribution is subject to the Boost Software
//  License, Version 1.0. (See accompanying file LICENSE_1_0.txt or copy
//  at http://www.boost.org/LICENSE_1_0.txt)

//  See library home page at http://www.boost.org/libs/filesystem

//----------------------------------------------------------------------------//


//  The point of this implementation is to prove the interface.  There is no
//  claim the implementation is efficient, follows good coding practice, etc.


//----------------------------------------------------------------------------// 

// define BOOST_FILESYSTEM_SOURCE so that <boost/filesystem/config.hpp> knows
// the library is being built (possibly exporting rather than importing code)
#define BOOST_FILESYSTEM_SOURCE 

#include <boost/filesystem/config.hpp>
#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/exception.hpp>
#include <boost/scoped_array.hpp>
#include <boost/throw_exception.hpp>
#include <boost/detail/workaround.hpp>
#include <cstring>

#ifdef BOOST_NO_STDC_NAMESPACE
namespace std { using ::strcmp; using ::remove; using ::rename; }
#endif

namespace fs = boost::filesystem;

// BOOST_POSIX or BOOST_WINDOWS specify which API to use.
# if !defined( BOOST_WINDOWS ) && !defined( BOOST_POSIX )
#   if defined(_WIN32) || defined(__WIN32__) || defined(WIN32) || defined(__CYGWIN__)
#     define BOOST_WINDOWS
#   else
#     define BOOST_POSIX
#   endif
# endif

# if defined(BOOST_WINDOWS)
#   include "windows.h"
#   if defined(__BORLANDC__) || defined(__MWERKS__)
#     if defined(__BORLANDC__)
        using std::time_t;
#     endif
#     include "utime.h"
#   else
#     include "sys/utime.h"
#   endif

// For Windows, the xxxA form of various function names is used to avoid
// inadvertently getting wide forms of the functions. (The undecorated
// forms are actually macros, so can misfire if the user has various
// other macros defined. There was a bug report of this happening.)

# else // BOOST_POSIX
#   define _FILE_OFFSET_BITS 64
      // at worst, this define may do nothing,
      // but that won't matter on 64-bit systems or on 32-bit systems which
      // don't have files larger than can be represented by a traditional
      // POSIX/UNIX off_t type. OTOH, defining them should kick
      // in 64-bit off_t's (and thus st_size) on 32-bit systems that support
      // the Large File Support (LFS) interface, such as Linux, Solaris, and IRIX.

#   include <sys/types.h>
#   include "dirent.h"
#   include "unistd.h"
#   include "fcntl.h"
#   include "utime.h"

#ifdef __DECCXX_LIBCXX_RH70
extern "C" int lstat(const char *file_name, struct stat *buf);
#endif

# endif

#include <sys/stat.h>  // even on Windows some functions use stat()
#include <string>
#include <cstdio>      // for remove, rename
#include <cerrno>
#include <cassert>

#include <boost/config/abi_prefix.hpp> // must be the last header

//  helpers  -----------------------------------------------------------------//

namespace
{
#ifdef BOOST_POSIX

# define BOOST_HANDLE DIR *
# define BOOST_INVALID_HANDLE_VALUE 0
# define BOOST_SYSTEM_DIRECTORY_TYPE struct dirent *

  inline const char *  find_first_file( const char * dir,
    BOOST_HANDLE & handle, BOOST_SYSTEM_DIRECTORY_TYPE & )
  // Returns: 0 if error, otherwise name
  {
    const char * dummy_first_name = ".";
    return ( (handle = ::opendir( dir ))
      == BOOST_INVALID_HANDLE_VALUE ) ? 0 : dummy_first_name;
  }  

  inline void find_close( BOOST_HANDLE handle )
  {
    assert( handle != BOOST_INVALID_HANDLE_VALUE );
    ::closedir( handle );
  }

  inline const char * find_next_file(
    BOOST_HANDLE handle, const fs::path & ph, BOOST_SYSTEM_DIRECTORY_TYPE & )
  // Returns: if EOF 0, otherwise name
  // Throws: if system reports error
  {

//  TODO: use readdir_r() if available, so code is multi-thread safe.
//  Fly-in-ointment: not all platforms support readdir_r().

    struct dirent * dp;
    errno = 0;
    if ( (dp = ::readdir( handle )) == 0 )
    {
      if ( errno != 0 )
      {
        boost::throw_exception(
          fs::filesystem_error(
            "boost::filesystem::directory_iterator::operator++",
            ph, errno ) );
      }
      else { return 0; } // end reached
    }
    return dp->d_name;
  }
#else // BOOST_WINDOWS

# define BOOST_HANDLE HANDLE
# define BOOST_INVALID_HANDLE_VALUE INVALID_HANDLE_VALUE
# define BOOST_SYSTEM_DIRECTORY_TYPE WIN32_FIND_DATAA

  inline const char *  find_first_file( const char * dir,
    BOOST_HANDLE & handle, BOOST_SYSTEM_DIRECTORY_TYPE & data )
  // Returns: 0 if error, otherwise name
  {
//    std::cout << "find_first_file " << dir << std::endl;
    std::string dirpath( std::string(dir) + "/*" );
    return ( (handle = ::FindFirstFileA( dirpath.c_str(), &data ))
      == BOOST_INVALID_HANDLE_VALUE ) ? 0 : data.cFileName;
  }  

  inline void find_close( BOOST_HANDLE handle )
  {
//    std::cout << "find_close" << std::endl;
    assert( handle != BOOST_INVALID_HANDLE_VALUE );
    ::FindClose( handle );
  }

  inline const char * find_next_file(
    BOOST_HANDLE handle, const fs::path & ph,
    BOOST_SYSTEM_DIRECTORY_TYPE & data )
  // Returns: 0 if EOF, otherwise name
  // Throws: if system reports error
  {
    if ( ::FindNextFileA( handle, &data ) == 0 )
    {
      if ( ::GetLastError() != ERROR_NO_MORE_FILES )
      {
        boost::throw_exception( fs::filesystem_error(
          "boost::filesystem::directory_iterator::operator++",
          ph.branch_path(), fs::detail::system_error_code() ) );
      }
      else { return 0; } // end reached
     }
    return data.cFileName;
  }

#endif

  
  fs::directory_iterator end_itr;

  bool is_empty_directory( const fs::path & dir_path )
  {
    return fs::directory_iterator(dir_path) == end_itr;
  }

  unsigned long remove_all_aux( const fs::path & ph )
  {
    unsigned long count = 1;
    if ( !fs::symbolic_link_exists( ph ) // don't recurse symbolic links
      && fs::is_directory( ph ) )
    {
      for ( fs::directory_iterator itr( ph );
            itr != end_itr; ++itr )
      {
        count += remove_all_aux( *itr );
      }
    }
    fs::remove( ph );
    return count;
  }

} // unnamed namespace

namespace boost
{
  namespace filesystem
  {
    namespace detail
    {

//  dir_itr_imp  -------------------------------------------------------------// 

      class dir_itr_imp
      {
      public:
        path              entry_path;
        BOOST_HANDLE      handle;

        ~dir_itr_imp()
        {
          if ( handle != BOOST_INVALID_HANDLE_VALUE ) find_close( handle );
        }
      };

//  dot_or_dot_dot  ----------------------------------------------------------//

      inline bool dot_or_dot_dot( const char * name )
      {
# if !BOOST_WORKAROUND( __BORLANDC__, BOOST_TESTED_AT(0x0564) )
        return std::strcmp( name, "." ) == 0
            || std::strcmp( name, ".." ) == 0;
# else
        // Borland workaround for failure of intrinsics to be placed in
        // namespace std with certain combinations of compiler options.
        // To ensure test coverage, the workaround is applied to all
        // configurations, regardless of option settings.
        return name[0]=='.'
          && (name[1]=='\0' || (name[1]=='.' && name[2]=='\0'));
# endif
      }

//  directory_iterator implementation  ---------------------------------------//

      BOOST_FILESYSTEM_DECL void dir_itr_init( dir_itr_imp_ptr & m_imp,
                                               const path & dir_path )
      {
        m_imp.reset( new dir_itr_imp );
        BOOST_SYSTEM_DIRECTORY_TYPE scratch;
        const char * name = 0;  // initialization quiets compiler warnings
        if ( dir_path.empty() )
        m_imp->handle = BOOST_INVALID_HANDLE_VALUE;
        else
          name = find_first_file( dir_path.native_directory_string().c_str(),
            m_imp->handle, scratch );  // sets handle

        if ( m_imp->handle != BOOST_INVALID_HANDLE_VALUE )
        {
          m_imp->entry_path = dir_path;
          // append name, except ignore "." or ".."
          if ( !dot_or_dot_dot( name ) )
          { 
            m_imp->entry_path.m_path_append( name, no_check );
          }
          else
          {
            m_imp->entry_path.m_path_append( "dummy", no_check );
            dir_itr_increment( m_imp );
          }
        }
        else
        {
          boost::throw_exception( filesystem_error(  
            "boost::filesystem::directory_iterator constructor",
            dir_path, fs::detail::system_error_code() ) );
        }  
      }

      BOOST_FILESYSTEM_DECL path & dir_itr_dereference(
        const dir_itr_imp_ptr & m_imp )
      {
        assert( m_imp.get() ); // fails if dereference end iterator
        return m_imp->entry_path;
      }

      BOOST_FILESYSTEM_DECL void dir_itr_increment( dir_itr_imp_ptr & m_imp )
      {
        assert( m_imp.get() ); // fails on increment end iterator
        assert( m_imp->handle != BOOST_INVALID_HANDLE_VALUE ); // reality check

        BOOST_SYSTEM_DIRECTORY_TYPE scratch;
        const char * name;

        while ( (name = find_next_file( m_imp->handle,
          m_imp->entry_path, scratch )) != 0 )
        {
          // append name, except ignore "." or ".."
          if ( !dot_or_dot_dot( name ) )
          {
            m_imp->entry_path.m_replace_leaf( name );
            return;
          }
        }
        m_imp.reset(); // make base() the end iterator
      }
    } // namespace detail

//  free functions  ----------------------------------------------------------//

    BOOST_FILESYSTEM_DECL bool exists( const path & ph )
    {
#   ifdef BOOST_POSIX
      struct stat path_stat;
      if(::stat( ph.string().c_str(), &path_stat ) != 0)
      {
         if((errno == ENOENT) || (errno == ENOTDIR))
            return false;  // stat failed because the path does not exist
         // for any other error we assume the file does exist and fall through,
         // this may not be the best policy though...  (JM 20040330)
      }
      return true;
#   else
      if(::GetFileAttributesA( ph.string().c_str() ) == 0xFFFFFFFF)
      {
         UINT err = ::GetLastError();
         if((err == ERROR_FILE_NOT_FOUND) || (err == ERROR_INVALID_PARAMETER) || (err == ERROR_PATH_NOT_FOUND) || (err == ERROR_INVALID_NAME))
            return false; // GetFileAttributes failed because the path does not exist
         // for any other error we assume the file does exist and fall through,
         // this may not be the best policy though...  (JM 20040330)
         return true;
      }
      return true;
#   endif
    }

    BOOST_FILESYSTEM_DECL bool possible_large_file_size_support()
    {
#   ifdef BOOST_POSIX
      struct stat lcl_stat;
      return sizeof( lcl_stat.st_size ) > 4;
#   else
      return true;
#   endif
    }

    // suggested by Walter Landry
    BOOST_FILESYSTEM_DECL bool symbolic_link_exists( const path & ph )
    {
#   ifdef BOOST_POSIX
      struct stat path_stat;
      return ::lstat( ph.native_file_string().c_str(), &path_stat ) == 0
        && S_ISLNK( path_stat.st_mode );
#   else
      return false; // Windows has no O/S concept of symbolic links
                    // (.lnk files are an application feature, not
                    // a Windows operating system feature)
#   endif
    }

    BOOST_FILESYSTEM_DECL bool is_directory( const path & ph )
    {
#   ifdef BOOST_POSIX
      struct stat path_stat;
      if ( ::stat( ph.native_directory_string().c_str(), &path_stat ) != 0 )
        boost::throw_exception( filesystem_error(
          "boost::filesystem::is_directory",
          ph, fs::detail::system_error_code() ) );
      return S_ISDIR( path_stat.st_mode );
#   else
      DWORD attributes = ::GetFileAttributesA( ph.native_directory_string().c_str() );
      if ( attributes == 0xFFFFFFFF )
        boost::throw_exception( filesystem_error(
          "boost::filesystem::is_directory",
          ph, fs::detail::system_error_code() ) );
      return (attributes & FILE_ATTRIBUTE_DIRECTORY) != 0;
#   endif
    }

    BOOST_FILESYSTEM_DECL bool _is_empty( const path & ph )
    {
#   ifdef BOOST_POSIX
      struct stat path_stat;
      if ( ::stat( ph.string().c_str(), &path_stat ) != 0 )
        boost::throw_exception( filesystem_error(
          "boost::filesystem::is_empty",
          ph, fs::detail::system_error_code() ) );
      
      return S_ISDIR( path_stat.st_mode )
        ? is_empty_directory( ph )
        : path_stat.st_size == 0;
#   else
      WIN32_FILE_ATTRIBUTE_DATA fad;
      if ( !::GetFileAttributesExA( ph.string().c_str(),
        ::GetFileExInfoStandard, &fad ) )
        boost::throw_exception( filesystem_error(
          "boost::filesystem::is_empty",
          ph, fs::detail::system_error_code() ) );
      
      return ( fad.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY )
        ? is_empty_directory( ph )
        :( !fad.nFileSizeHigh && !fad.nFileSizeLow );
#   endif
    }

# ifdef BOOST_WINDOWS
    // Thanks to Jeremy Maitin-Shepard for much help and for permission to
    // base the implementation on portions of his file-equivalence-win32.cpp
    // experimental code.
    struct handle_wrapper
    {
      BOOST_HANDLE handle;
      handle_wrapper( BOOST_HANDLE h )
        : handle(h) {}
      ~handle_wrapper()
      {
        if ( handle != BOOST_INVALID_HANDLE_VALUE )
          ::CloseHandle(handle);
      }
    };
# endif

    BOOST_FILESYSTEM_DECL bool equivalent( const path & ph1, const path & ph2 )
    {
#   ifdef BOOST_POSIX
      struct stat s1;
      int s1_result = ::stat( ph1.string().c_str(), &s1 );
      // save error code in case we have to throw
      int error1 = (s1_result != 0 ? fs::detail::system_error_code() : 0);
      struct stat s2;
      int s2_result = ::stat( ph2.string().c_str(), &s2 );
      if ( s1_result != 0
        || s2_result != 0 )
      {
        if ( s1_result == 0 || s2_result == 0 ) return false;
        assert( s1_result != 0 && s2_result != 0 );
        boost::throw_exception( filesystem_error(
          "boost::filesystem::equivalent",
          ph1, error1 ) );
      }
      // at this point, both stats are known to be valid
      return s1.st_dev == s2.st_dev
          && s1.st_ino == s2.st_ino
          // According to the POSIX stat specs, "The st_ino and st_dev fields
          // taken together uniquely identify the file within the system."
          // Just to be sure, size and mod time are also checked.
          && s1.st_size == s2.st_size
          && s1.st_mtime == s2.st_mtime;
#   else
      // Note well: Physical location on external media is part of the
      // equivalence criteria. If there are no open handles, physical location
      // can change due to defragmentation or other relocations. Thus handles
      // must be held open until location information for both paths has
      // been retrieved.
      handle_wrapper p1(
        ::CreateFileA(
            ph1.string().c_str(),
            0,
            FILE_SHARE_DELETE | FILE_SHARE_READ | FILE_SHARE_WRITE,
            0,
            OPEN_EXISTING,
            FILE_FLAG_BACKUP_SEMANTICS,
            0 ) );
      int error1; // save error code in case we have to throw
      if ( p1.handle == BOOST_INVALID_HANDLE_VALUE )
        error1 = fs::detail::system_error_code();
      handle_wrapper p2(
        ::CreateFileA(
            ph2.string().c_str(),
            0,
            FILE_SHARE_DELETE | FILE_SHARE_READ | FILE_SHARE_WRITE,
            0,
            OPEN_EXISTING,
            FILE_FLAG_BACKUP_SEMANTICS,
            0 ) );
      if ( p1.handle == BOOST_INVALID_HANDLE_VALUE
        || p2.handle == BOOST_INVALID_HANDLE_VALUE )
      {
        if ( p1.handle != BOOST_INVALID_HANDLE_VALUE
          || p2.handle != BOOST_INVALID_HANDLE_VALUE ) return false;
        assert( p1.handle == BOOST_INVALID_HANDLE_VALUE
          && p2.handle == BOOST_INVALID_HANDLE_VALUE );
        boost::throw_exception( filesystem_error(
          "boost::filesystem::equivalent",
          ph1, error1 ) );
      }
      // at this point, both handles are known to be valid
      BY_HANDLE_FILE_INFORMATION info1, info2;
      if ( !::GetFileInformationByHandle( p1.handle, &info1 ) )
          boost::throw_exception( filesystem_error(
            "boost::filesystem::equivalent",
            ph1, fs::detail::system_error_code() ) );
      if ( !::GetFileInformationByHandle( p2.handle, &info2 ) )
          boost::throw_exception( filesystem_error(
            "boost::filesystem::equivalent",
            ph2, fs::detail::system_error_code() ) );
      // In theory, volume serial numbers are sufficient to distinguish between
      // devices, but in practice VSN's are sometimes duplicated, so device id
      // is also checked. Device id's alone aren't sufficient because network
      // share devices on different machines will have the same id. Furthermore,
      // cheap floppy disks often have 0 VSN's and are mounted on the same
      // lettered drive across networks, so last write time and file size is
      // checked to distinguish that case as far as is possible.
      if ( info1.dwVolumeSerialNumber != info2.dwVolumeSerialNumber
        || info1.nFileIndexHigh != info2.nFileIndexHigh
        || info1.nFileIndexLow != info2.nFileIndexLow
        || info1.nFileSizeHigh != info2.nFileSizeHigh
        || info1.nFileSizeLow != info2.nFileSizeLow ) return false;
      struct stat s1;
      if ( ::stat( ph1.string().c_str(), &s1 ) != 0 )
        boost::throw_exception( filesystem_error(
          "boost::filesystem::equivalent",
          ph1, fs::detail::system_error_code() ) );
      struct stat s2;
      if ( ::stat( ph2.string().c_str(), &s2 ) != 0 )
        boost::throw_exception( filesystem_error(
          "boost::filesystem::equivalent",
          ph2, fs::detail::system_error_code() ) );
      return s1.st_dev == s2.st_dev;
#   endif
    }


    BOOST_FILESYSTEM_DECL boost::intmax_t file_size( const path & ph )
    {
#   ifdef BOOST_POSIX
      struct stat path_stat;
      if ( ::stat( ph.string().c_str(), &path_stat ) != 0 )
        boost::throw_exception( filesystem_error(
          "boost::filesystem::file_size",
          ph, fs::detail::system_error_code() ) );
      if ( S_ISDIR( path_stat.st_mode ) )
        boost::throw_exception( filesystem_error(
          "boost::filesystem::file_size",
          ph, "invalid: is a directory",
          is_directory_error ) ); 
      return static_cast<boost::intmax_t>(path_stat.st_size);
#   else
      // by now, intmax_t is 64-bits on all Windows compilers
      WIN32_FILE_ATTRIBUTE_DATA fad;
      if ( !::GetFileAttributesExA( ph.string().c_str(),
        ::GetFileExInfoStandard, &fad ) )
        boost::throw_exception( filesystem_error(
          "boost::filesystem::file_size",
          ph, fs::detail::system_error_code() ) );
      if ( (fad.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) !=0 )
        boost::throw_exception( filesystem_error(
          "boost::filesystem::file_size",
          ph, "invalid: is a directory",
          is_directory_error ) ); 
      return (static_cast<boost::intmax_t>(fad.nFileSizeHigh)
          << (sizeof(fad.nFileSizeLow)*8))
        + fad.nFileSizeLow;
#   endif
    }

    BOOST_FILESYSTEM_DECL std::time_t last_write_time( const path & ph )
    {
      // Works for both Windows and POSIX
      struct stat path_stat;
      if ( ::stat( ph.string().c_str(), &path_stat ) != 0 )
        boost::throw_exception( filesystem_error(
          "boost::filesystem::last_write_time",
          ph, fs::detail::system_error_code() ) );
      return path_stat.st_mtime;
    }

    BOOST_FILESYSTEM_DECL void last_write_time( const path & ph, const std::time_t new_time )
    {
      // Works for both Windows and POSIX
      struct stat path_stat;
      if ( ::stat( ph.string().c_str(), &path_stat ) != 0 )
        boost::throw_exception( filesystem_error(
          "boost::filesystem::last_write_time",
          ph, fs::detail::system_error_code() ) );
      ::utimbuf buf;
      buf.actime = path_stat.st_atime; // utime() updates access time too:-(
      buf.modtime = new_time;
      if ( ::utime( ph.string().c_str(), &buf ) != 0 )
        boost::throw_exception( filesystem_error(
          "boost::filesystem::last_write_time",
          ph, fs::detail::system_error_code() ) );
    }

    BOOST_FILESYSTEM_DECL bool create_directory( const path & dir_path )
    {
#   ifdef BOOST_POSIX
      if ( ::mkdir( dir_path.native_directory_string().c_str(),
        S_IRWXU|S_IRWXG|S_IRWXO ) == 0 ) return true;
      if ( errno != EEXIST ) 
#   else
      if ( ::CreateDirectoryA( dir_path.native_directory_string().c_str(), 0 ) )
        return true;
      if ( ::GetLastError() != ERROR_ALREADY_EXISTS )
#   endif
        boost::throw_exception( filesystem_error(
          "boost::filesystem::create_directory",
          dir_path, fs::detail::system_error_code() ) );
      if ( !is_directory( dir_path ) )
        boost::throw_exception( filesystem_error(
          "boost::filesystem::create_directory",
          dir_path, "path exists and is not a directory", not_directory_error ) );
      return false;
    }

    BOOST_FILESYSTEM_DECL bool remove( const path & ph )
    {
      if ( exists( ph )
#   ifdef BOOST_POSIX
        || symbolic_link_exists( ph ) ) // handle dangling symbolic links
      {
        // note that the POSIX behavior for symbolic links is what we want;
        // the link rather than what it points to is deleted
        if ( std::remove( ph.string().c_str() ) != 0 )
        {
          int error = fs::detail::system_error_code();
          // POSIX says "If the directory is not an empty directory, rmdir()
          // shall fail and set errno to EEXIST or ENOTEMPTY."
          // Linux uses ENOTEMPTY, Solaris uses EEXIST.
          if ( error == EEXIST ) error = ENOTEMPTY;
          boost::throw_exception( filesystem_error(
            "boost::filesystem::remove", ph, error ) );
        }
#   else
      )
      {
        if ( is_directory( ph ) )
        {
          if ( !::RemoveDirectoryA( ph.string().c_str() ) )
            boost::throw_exception( filesystem_error(
              "boost::filesystem::remove",
              ph, fs::detail::system_error_code() ) );
        }
        else
        {
          if ( !::DeleteFileA( ph.string().c_str() ) )
            boost::throw_exception( filesystem_error(
              "boost::filesystem::remove",
              ph, fs::detail::system_error_code() ) );
        }
#   endif
        return true;
      }
      return false;
    }

    BOOST_FILESYSTEM_DECL unsigned long remove_all( const path & ph )
    {
      return exists( ph )|| symbolic_link_exists( ph )
        ? remove_all_aux( ph ) : 0;
    }

    BOOST_FILESYSTEM_DECL void rename( const path & old_path,
                 const path & new_path )
    {
#   ifdef BOOST_POSIX
      if ( exists( new_path ) // POSIX is too permissive so must check
        || std::rename( old_path.string().c_str(), new_path.string().c_str() ) != 0 )
#   else
      if ( !::MoveFileA( old_path.string().c_str(), new_path.string().c_str() ) )
#   endif
        boost::throw_exception( filesystem_error(
          "boost::filesystem::rename",
          old_path, new_path, fs::detail::system_error_code() ) );
    }

    BOOST_FILESYSTEM_DECL void copy_file( const path & from_file_ph,
                    const path & to_file_ph )
    {
#   ifdef BOOST_POSIX
      // TODO: Ask POSIX experts if this is the best way to copy a file

      const std::size_t buf_sz = 32768;
      boost::scoped_array<char> buf( new char [buf_sz] );
      int infile=0, outfile=0;  // init quiets compiler warning
      struct stat from_stat;

      if ( ::stat( from_file_ph.string().c_str(), &from_stat ) != 0
        || (infile = ::open( from_file_ph.string().c_str(),
                             O_RDONLY )) < 0
        || (outfile = ::open( to_file_ph.string().c_str(),
                              O_WRONLY | O_CREAT | O_EXCL,
                              from_stat.st_mode )) < 0 )
      {
        if ( infile != 0 ) ::close( infile );
        boost::throw_exception( filesystem_error(
          "boost::filesystem::copy_file",
          from_file_ph, to_file_ph, fs::detail::system_error_code() ) );
      }

      ssize_t sz;
      while ( (sz = ::read( infile, buf.get(), buf_sz )) > 0
        && (sz = ::write( outfile, buf.get(), sz )) > 0 ) {}

      ::close( infile );
      ::close( outfile );

      if ( sz != 0 )
#   else
      if ( !::CopyFileA( from_file_ph.string().c_str(),
                      to_file_ph.string().c_str(), /*fail_if_exists=*/true ) )
#   endif
        boost::throw_exception( filesystem_error(
          "boost::filesystem::copy_file",
          from_file_ph, to_file_ph, fs::detail::system_error_code() ) );
    }

    BOOST_FILESYSTEM_DECL path current_path()
    {
#   ifdef BOOST_POSIX
      for ( long path_max = 32;; path_max *=2 ) // loop 'til buffer large enough
      {
        boost::scoped_array<char>
          buf( new char[static_cast<std::size_t>(path_max)] );
        if ( ::getcwd( buf.get(), static_cast<std::size_t>(path_max) ) == 0 )
        {
          if ( errno != ERANGE )
            boost::throw_exception(
              filesystem_error( "boost::filesystem::current_path", path(),
                fs::detail::system_error_code() ) );
        }
        else return path( buf.get(), native );
      }
      BOOST_UNREACHABLE_RETURN(0)
#   else
      DWORD sz;
      if ( (sz = ::GetCurrentDirectoryA( 0, static_cast<char*>(0) )) == 0 )
        boost::throw_exception(
          filesystem_error( "boost::filesystem::current_path",
            "size is 0" ) );
      boost::scoped_array<char> buf( new char[sz] );
      if ( ::GetCurrentDirectoryA( sz, buf.get() ) == 0 )
        boost::throw_exception(
          filesystem_error( "boost::filesystem::current_path", path(),
            fs::detail::system_error_code() ) );
      return path( buf.get(), native );
#   endif
    }

    BOOST_FILESYSTEM_DECL const path & initial_path()
    {
      static path init_path;
      if ( init_path.empty() ) init_path = current_path();
      return init_path;
    }

    BOOST_FILESYSTEM_DECL path system_complete( const path & ph )
    {
#   ifdef BOOST_WINDOWS
      if ( ph.empty() ) return ph;
      char buf[MAX_PATH];
      char * pfn;
      std::size_t len = ::GetFullPathNameA( ph.string().c_str(),
                                            sizeof(buf) , buf, &pfn );
      if ( !len )
        { boost::throw_exception(
            filesystem_error( "boost::filesystem::system_complete",
              ph, "size is 0" ) ); }
      buf[len] = '\0';
      return path( buf, native );
#   else
      return (ph.empty() || ph.is_complete())
        ? ph : current_path() / ph;
#   endif
    }
    
    BOOST_FILESYSTEM_DECL path complete( const path & ph, const path & base )
    {
      assert( base.is_complete()
        && (ph.is_complete() || !ph.has_root_name()) ); // precondition
#   ifdef BOOST_WINDOWS
      if (ph.empty() || ph.is_complete()) return ph;
      if ( !ph.has_root_name() )
        return ph.has_root_directory()
          ? path( base.root_name(), native ) / ph
          : base / ph;
      return base / ph;
#   else
      return (ph.empty() || ph.is_complete()) ? ph : base / ph;
#   endif
    }
  } // namespace filesystem
} // namespace boost

