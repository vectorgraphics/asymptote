//  path implementation  -----------------------------------------------------//

//  © Copyright Beman Dawes 2002-2003
//  Use, modification, and distribution is subject to the Boost Software
//  License, Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

//  See library home page at http://www.boost.org/libs/filesystem


//****************************************************************************//

//  WARNING: This code was hacked time and time again as different library 
//  designs were tried.  Thus portions may be residue from the earlier
//  experiments, and be totally stupid or wrong in light of the final library
//  specifications.  The code needs to be reviewed line-by-line to elmininate
//  such problems.

//****************************************************************************//

// define BOOST_FILESYSTEM_SOURCE so that <boost/filesystem/config.hpp> knows
// the library is being built (possibly exporting rather than importing code)
#define BOOST_FILESYSTEM_SOURCE 

#include <boost/filesystem/path.hpp>
#include <boost/filesystem/exception.hpp>

// BOOST_POSIX or BOOST_WINDOWS specify which API to use.
# if !defined( BOOST_WINDOWS ) && !defined( BOOST_POSIX )
#   if defined(_WIN32) || defined(__WIN32__) || defined(WIN32) || defined(__CYGWIN__)
#     define BOOST_WINDOWS
#   else
#     define BOOST_POSIX
#   endif
# endif


namespace fs = boost::filesystem;

#include <boost/config.hpp>
#ifdef BOOST_NO_STDC_NAMESPACE
  namespace std { using ::strlen; }
#endif

#include <boost/throw_exception.hpp>
#include <cstring>  // SGI MIPSpro compilers need this
#include <vector>
#include <algorithm> // for lexicographical_compare
#include <cassert>

#include <boost/config/abi_prefix.hpp> // must be the last header

//  helpers  -----------------------------------------------------------------// 

namespace
{
  // POSIX & Windows cases: "", "/", "/foo", "foo", "foo/bar"
  // Windows only cases: "c:", "c:/", "c:foo", "c:/foo",
  //                     "prn:", "//share", "//share/", "//share/foo"

  std::string::size_type leaf_pos( const std::string & str,
    std::string::size_type end_pos ) // end_pos is past-the-end position
  // return 0 if str itself is leaf (or empty) 
  {
    if ( end_pos && str[end_pos-1] == '/' ) return end_pos-1;
    
    std::string::size_type pos( str.find_last_of( '/', end_pos-1 ) );
#   ifdef BOOST_WINDOWS
    if ( pos == std::string::npos ) pos = str.find_last_of( ':', end_pos-2 );
#   endif

    return ( pos == std::string::npos // path itself must be a leaf (or empty)
#     ifdef BOOST_WINDOWS
      || (pos == 1 && str[0] == '/') // or share
#     endif
      ) ? 0 // so leaf is entire string
        : pos + 1; // or starts after delimiter
  }

  void first_name( const std::string & src, std::string & target )
  {
    target = ""; // VC++ 6.0 doesn't have string::clear()
    std::string::const_iterator itr( src.begin() );

#   ifdef BOOST_WINDOWS
    // deal with //share
    if ( src.size() >= 2 && src[0] == '/' && src[1] == '/' )
      { target = "//"; itr += 2; }
#   endif

    while ( itr != src.end()
#     ifdef BOOST_WINDOWS
      && *itr != ':'
#     endif
      && *itr != '/' ) { target += *itr++; }

    if ( itr == src.end() ) return;

#   ifdef BOOST_WINDOWS
    if ( *itr == ':' )
    {
      target += *itr++;
      return;
    }
#   endif

    // *itr is '/'
    if ( itr == src.begin() ) { target += '/'; }
    return;
  }

  const char invalid_chars[] =
    "\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0A\x0B\x0C\x0D\x0E\x0F"
    "\x10\x11\x12\x13\x14\x15\x16\x17\x18\x19\x1A\x1B\x1C\x1D\x1E\x1F"
    "<>:\"/\\|";
  // note that the terminating '\0' is part of the string - thus the size below
  // is sizeof(invalid_chars) rather than sizeof(invalid_chars)-1.  I 
  const std::string windows_invalid_chars( invalid_chars, sizeof(invalid_chars) );

  const std::string valid_posix(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789._-" );

  fs::path::name_check default_check = fs::portable_name;
  bool safe_to_write_check = true; // write-once-before-read allowed

} // unnamed namespace

//----------------------------------------------------------------------------// 

namespace boost
{
  namespace filesystem
  {
    //  name_check functions  ----------------------------------------------//

#   ifdef BOOST_WINDOWS
    BOOST_FILESYSTEM_DECL bool native( const std::string & name )
    {
      return windows_name( name );
    }
#   else
    BOOST_FILESYSTEM_DECL bool native( const std::string & )
    {
      return true;
    }
#   endif

    BOOST_FILESYSTEM_DECL bool no_check( const std::string & ) { return true; }

    BOOST_FILESYSTEM_DECL bool portable_posix_name( const std::string & name )
    {
      return name.size() != 0
        && name.find_first_not_of( valid_posix ) == std::string::npos;     
    }

    BOOST_FILESYSTEM_DECL bool windows_name( const std::string & name )
    {
      return name.size() != 0
        && name.find_first_of( windows_invalid_chars ) == std::string::npos
        && *(name.end()-1) != ' '
        && (*(name.end()-1) != '.'
          || name.length() == 1 || name == "..");
    }

    BOOST_FILESYSTEM_DECL bool portable_name( const std::string & name )
    {
      return
        name.size() == 0
        || name == "."
        || name == ".."
        || (windows_name( name )
        && portable_posix_name( name )
        && name[0] != '.' && name[0] != '-');
    }

    BOOST_FILESYSTEM_DECL bool portable_directory_name( const std::string & name )
    {
      return
        name == "."
        || name == ".."
        || (portable_name( name )
          && name.find('.') == std::string::npos);
    }

    BOOST_FILESYSTEM_DECL bool portable_file_name( const std::string & name )
    {
      std::string::size_type pos;
      return
         name == "."
        || name == ".."
        || (portable_name( name )
          && ( (pos = name.find( '.' )) == std::string::npos
            || (name.find( '.', pos+1 )== std::string::npos
              && (pos + 5) > name.length() )))
        ;
    }


//  path implementation  -----------------------------------------------------//

    path::path( const std::string & src )
    {
      m_path_append( src, default_name_check() );
    }

    path::path( const char * src )
    {
      assert( src != 0 );
      m_path_append( src, default_name_check() );
    }

    path::path( const std::string & src, name_check checker )
    {
      m_path_append( src, checker );
    }

    path::path( const char * src, name_check checker )
    {
      assert( src != 0 );
      m_path_append( src, checker );
    }

    path & path::operator /=( const path & rhs )
    {
      m_path_append( rhs.m_path, no_check );
      return *this;
    }

    void path::m_path_append( const std::string & src, name_check checker )
    {
      // convert backslash to forward slash if checker==native 
      // allow system-specific-root if checker==no_check || checker==native

      assert( checker );
      assert( src.size() == std::strlen( src.c_str() ) ); // no embedded 0

      if ( src.size() == 0 ) return;

      std::string::const_iterator itr( src.begin() );

      // [root-filesystem]
#     ifdef BOOST_WINDOWS
      if ( (checker == no_check || checker == native) && src.size() >= 2 )
      {
        // drive or device
        if ( src[1] == ':' || src[src.size()-1] == ':' )
        {        
          for ( ; *itr != ':'; ++itr ) m_path += *itr;
          m_path += ':';
          ++itr;
        }

        // share
        else if ( (*itr == '/' || (*itr == '\\' && checker == native))
          && (*(itr+1) == '/' || (*(itr+1) == '\\' && checker == native)) )
        {
          m_path += "//";
          for ( itr += 2;
                itr != src.end() && *itr != '/' && *itr != '\\';
                ++itr ) m_path += *itr;
        }
      }
#     endif

      // root directory [ "/" ]
      if ( itr != src.end() && (*itr == '/'
#         ifdef BOOST_WINDOWS
          || (*itr == '\\' && checker == native)
#         endif
          ) )
      {
        ++itr;
        if ( m_path.size() == 0
#         ifdef BOOST_WINDOWS
          || m_path[m_path.size()-1] == ':' // drive or device
          || (  // share
             m_path.size() > 2
             && m_path[0] == '/'
             && m_path[1] == '/'
             && m_path.find( '/', 2 ) == std::string::npos
             )
#         endif
          ) m_path += '/';
      }

      // element { "/" element } [ "/" ]
      while ( itr != src.end() )
      {
        if ( m_path == "." ) m_path = "";

        // directory-placeholder
        if ( *itr == '.' && ((itr+1) == src.end() || *(itr+1) == '/'
#         ifdef BOOST_WINDOWS
          || *(itr+1) == '\\'
#         endif
          ) )
        {
          if ( empty() ) m_path += '.';
          ++itr;
        }

        // parent-directory or name
        else
        {
          // append '/' if needed
          if ( !empty()
#         ifdef BOOST_WINDOWS
            && *(m_path.end()-1) != ':'
#         endif
            && *(m_path.end()-1) != '/' )
              m_path += '/';

          // parent-directory
          if ( *itr == '.'
            && (itr+1) != src.end() && *(itr+1) == '.'
            && ((itr+2) == src.end() || *(itr+2) == '/'
#           ifdef BOOST_WINDOWS
            || *(itr+2) == '\\'
#           endif
           ) )
          {
            m_path += "..";
            ++itr;
            ++itr;
          } // parent-directory

          // name
          else
          {
            std::string name;
            do
              { name += *itr; }
            while ( ++itr != src.end() && *itr != '/'
#             ifdef BOOST_WINDOWS
              && (*itr != '\\' || checker != native)
#             endif
              );

            if ( !checker( name ) )
            {
              boost::throw_exception( filesystem_error(
                "boost::filesystem::path",
                "invalid name \"" + name + "\" in path: \"" + src + "\"" ) );
            }

            m_path += name;
          }
        } // parent-directory or name

        // end or "/"
        if ( itr != src.end() )
        {
          if ( *itr == '/'
#         ifdef BOOST_WINDOWS
          || (*itr == '\\' && checker == native)
#         endif
          ) ++itr;
          else 
            boost::throw_exception( filesystem_error(
              "boost::filesystem::path",
              "invalid path syntax: \"" + src + "\"" ) );
        }

      } // while more elements

      // special case: remove one or more leading "/.."

      std::string::size_type pos = 0, sz = m_path.size();

#     ifdef BOOST_WINDOWS
      if ( sz > 2 && m_path[pos] != '/' && m_path[pos+1] == ':' ) // drive
        { pos += 2;  sz  -= 2; }
#     endif

      while ( sz >= 3 && m_path[pos] == '/'
         && m_path[pos+1] == '.' && m_path[pos+2] == '.'
         && (sz == 3 || m_path[pos+3] == '/') )
      {
        m_path.erase( pos+1, 3 ); // "..[/]"
        sz -= 3; // on last, 3 should be 2; that doesn't matter
      }
    }

// path conversion functions  ------------------------------------------------//

    std::string path::native_file_string() const
    {
#   ifdef BOOST_WINDOWS
      std::string s( m_path );
      for ( std::string::iterator itr( s.begin() );
        itr != s.end(); ++itr )
        if ( *itr == '/' ) *itr = '\\';
      return s;
#   else
      return m_path;
#   endif
    }

    std::string path::native_directory_string() const
      { return native_file_string(); }

// path modification functions -----------------------------------------------//

      path & path::normalize()
      {
        if ( m_path.empty() ) return *this;
        std::string::size_type end, beg(0), start(0);

#       ifdef BOOST_WINDOWS
          if ( m_path.size() > 2
            && m_path[0] != '/' && m_path[1] == ':' ) start = 2; // drive
#       endif

        while ( (beg=m_path.find( "/..", beg )) != std::string::npos )
        {
          end = beg + 3;
          if ( (beg == 1 && m_path[0] == '.')
            || (beg == 2 && m_path[0] == '.' && m_path[1] == '.')
            || (beg > 2 && m_path[beg-3] == '/'
                        && m_path[beg-2] == '.' && m_path[beg-1] == '.') )
          {
            beg = end;
            continue;
          }
          if ( end < m_path.size() )
          {
            if ( m_path[end] == '/' ) ++end;
            else { beg = end; continue; } // name starts with ..
          }

          // end is one past end of substr to be erased; now set beg
          while ( beg > start && m_path[--beg] != '/' ) {}
          if ( m_path[beg] == '/') ++beg;
          m_path.erase( beg, end-beg );
          if ( beg ) --beg;
        }

        if ( m_path.empty() ) m_path = ".";
        else
        { // remove trailing '/' if not root directory
          std::string::size_type sz = m_path.size();

#       ifdef BOOST_WINDOWS
          if ( start ) sz  -= 2; // drive
#       endif

          if ( sz > 1 && m_path[m_path.size()-1] == '/' )
            { m_path.erase( m_path.size()-1 ); }
        }
        return *this;
      }

 // path decomposition functions  ---------------------------------------------//

    path::iterator path::begin() const
    {
      iterator itr;
      itr.m_path_ptr = this;
      first_name( m_path, itr.m_name );
      itr.m_pos = 0;
      return itr;
    }

    void path::m_replace_leaf( const char * new_leaf )
    {
      m_path.erase( leaf_pos( m_path, m_path.size() ) );
      m_path += new_leaf;
    }

    std::string path::leaf() const
    {
      return m_path.substr( leaf_pos( m_path, m_path.size() ) );
    }

    namespace detail
    {
      inline bool is_absolute_root( const std::string & s,
        std::string::size_type len )
      {
        return
          len && s[len-1] == '/'
          &&
          (
            len == 1 // "/"
#       ifdef BOOST_WINDOWS
            || ( len > 1
                 && ( s[len-2] == ':' // drive or device
                   || ( s[0] == '/'   // share
                     && s[1] == '/'
                     && s.find( '/', 2 ) == len-1
                      )
                    )
                )
#       endif
          );
      }
    }

    path path::branch_path() const
    {
      std::string::size_type end_pos( leaf_pos( m_path, m_path.size() ) );

      // skip a '/' unless it is a root directory
      if ( end_pos && m_path[end_pos-1] == '/'
        && !detail::is_absolute_root( m_path, end_pos ) ) --end_pos;
      return path( m_path.substr( 0, end_pos ), no_check );
    }

    path path::relative_path() const
    {
      std::string::size_type pos( 0 );
      if ( m_path.size() && m_path[0] == '/' )
      { pos = 1;
#     ifdef BOOST_WINDOWS
        if ( m_path.size()>1 && m_path[1] == '/' ) // share
        {
          if ( (pos = m_path.find( '/', 2 )) != std::string::npos ) ++pos;
          else return path();
        }
      }
      else if ( (pos = m_path.find( ':' )) == std::string::npos ) pos = 0;
      else // has ':'
      {
        if ( ++pos < m_path.size() && m_path[pos] == '/' ) ++pos;
#     endif
      }
      return path( m_path.substr( pos ), no_check );
    }

    std::string path::root_name() const
    {
#   ifdef BOOST_WINDOWS
      std::string::size_type pos( m_path.find( ':' ) );
      if ( pos != std::string::npos ) return m_path.substr( 0, pos+1 );
      if ( m_path.size() > 2 && m_path[0] == '/' && m_path[1] == '/' )
      {
        pos = m_path.find( '/', 2 );
        return m_path.substr( 0, pos );
      }
#   endif
      return std::string();
    }

    std::string path::root_directory() const
    {
      return std::string(
        ( m_path.size() && m_path[0] == '/' )  // covers both "/" and "//share"
#       ifdef BOOST_WINDOWS
          || ( m_path.size() > 2
               && m_path[1] == ':'
               && m_path[2] == '/' )  // "c:/"
#       endif
               ? "/" : "" );
    }

    path path::root_path() const
    {
      return path(
#   ifdef BOOST_WINDOWS
        root_name(), no_check ) /= root_directory();
#   else
        root_directory() );
#   endif
    }

// path query functions  -----------------------------------------------------//

    bool path::is_complete() const
    {
#   ifdef BOOST_WINDOWS
      return m_path.size() > 2
        && ( (m_path[1] == ':' && m_path[2] == '/') // "c:/"
          || (m_path[0] == '/' && m_path[1] == '/') // "//share"
          || m_path[m_path.size()-1] == ':' );
#   else
      return m_path.size() && m_path[0] == '/';
#   endif
    }

    bool path::has_root_path() const
    {
      return ( m_path.size() 
               && m_path[0] == '/' )  // covers both "/" and "//share"
#            ifdef BOOST_WINDOWS
               || ( m_path.size() > 1 && m_path[1] == ':' ) // "c:" and "c:/"
               || ( m_path.size() > 3
                    && m_path[m_path.size()-1] == ':' ) // "device:"
#            endif
               ;
    }

    bool path::has_root_name() const
    {
#   ifdef BOOST_WINDOWS
      return m_path.size() > 1
        && ( m_path[1] == ':' // "c:"
          || m_path[m_path.size()-1] == ':' // "prn:"
          || (m_path[0] == '/' && m_path[1] == '/') // "//share"
           );
#   else
      return false;
#   endif
    }

    bool path::has_root_directory() const
    {
      return ( m_path.size() 
               && m_path[0] == '/' )  // covers both "/" and "//share"
#            ifdef BOOST_WINDOWS
               || ( m_path.size() > 2
                    && m_path[1] == ':' && m_path[2] == '/' ) // "c:/"
#            endif
               ;
    }

    bool path::has_relative_path() const { return !relative_path().empty(); }
    bool path::has_branch_path() const { return !branch_path().empty(); }

    //  default name_check mechanism  ----------------------------------------//

    bool path::default_name_check_writable()
    {
      return safe_to_write_check;
    }

    void path::default_name_check( name_check new_check )
    {
      assert( new_check );
      if ( !safe_to_write_check )
        boost::throw_exception(
          filesystem_error( "boost::filesystem::default_name_check",
                            "default name check already set" ));
      default_check = new_check;
      safe_to_write_check = false;
    }

    path::name_check path::default_name_check()
    {
      safe_to_write_check = false;
      return default_check;
    }

    //  path operator<  ------------------------------------------------------//
    bool path::operator<( const path & that ) const
    {
      return std::lexicographical_compare(
        begin(), end(), that.begin(), that.end() );
    }


// path::iterator implementation  --------------------------------------------// 

    void path::iterator::increment()
    {
      assert( m_pos < m_path_ptr->m_path.size() ); // detect increment past end
      m_pos += m_name.size();
      if ( m_pos == m_path_ptr->m_path.size() )
      {
        m_name = "";  // not strictly required, but might aid debugging
        return;
      }
      if ( m_path_ptr->m_path[m_pos] == '/' )
      {
#       ifdef BOOST_WINDOWS
        if ( m_name[m_name.size()-1] == ':' // drive or device
          || (m_name[0] == '/' && m_name[1] == '/') ) // share
        {
          m_name = "/";
          return;
        }
#       endif
        ++m_pos;
      }
      std::string::size_type end_pos( m_path_ptr->m_path.find( '/', m_pos ) );
      if ( end_pos == std::string::npos )
        end_pos = m_path_ptr->m_path.size();
      m_name = m_path_ptr->m_path.substr( m_pos, end_pos - m_pos );
    }

    void path::iterator::decrement()
    {                                                                                
      assert( m_pos ); // detect decrement of begin
      std::string::size_type end_pos( m_pos );

      // skip a '/' unless it is a root directory
      if ( m_path_ptr->m_path[end_pos-1] == '/'
        && !detail::is_absolute_root( m_path_ptr->m_path, end_pos ) ) --end_pos;
      m_pos = leaf_pos( m_path_ptr->m_path, end_pos );
      m_name = m_path_ptr->m_path.substr( m_pos, end_pos - m_pos );
    }
  } // namespace filesystem
} // namespace boost
