//  libs/filesystem/src/convenience.cpp  -------------------------------------//

//  © Copyright Beman Dawes, 2002
//  © Copyright Vladimir Prus, 2002
//  Use, modification, and distribution is subject to the Boost Software
//  License, Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

//  See library home page at http://www.boost.org/libs/filesystem

//----------------------------------------------------------------------------//

// define BOOST_FILESYSTEM_SOURCE so that <boost/filesystem/config.hpp> knows
// the library is being built (possibly exporting rather than importing code)
#define BOOST_FILESYSTEM_SOURCE 

#include <boost/filesystem/convenience.hpp>
#include <boost/filesystem/exception.hpp>
#include <boost/throw_exception.hpp>

#include <boost/config/abi_prefix.hpp> // must be the last header

namespace boost
{
  namespace filesystem
  {

//  create_directories (contributed by Vladimir Prus)  -----------------------//

     BOOST_FILESYSTEM_DECL bool create_directories(const path& ph)
     {
         if (ph.empty() || exists(ph))
         {
           if ( !ph.empty() && !is_directory(ph) )
               boost::throw_exception( filesystem_error(
                 "boost::filesystem::create_directories",
                 ph, "path exists and is not a directory",
                 not_directory_error ) );
           return false;
         }

         // First create branch, by calling ourself recursively
         create_directories(ph.branch_path());
         // Now that parent's path exists, create the directory
         create_directory(ph);
         return true;
     }

    BOOST_FILESYSTEM_DECL std::string extension(const path& ph)
    {
      std::string leaf = ph.leaf();

      std::string::size_type n = leaf.rfind('.');
      if (n != std::string::npos)
        return leaf.substr(n);
      else
        return std::string();
    }

    BOOST_FILESYSTEM_DECL std::string basename(const path& ph)
    {
      std::string leaf = ph.leaf();

      std::string::size_type n = leaf.rfind('.');
      return leaf.substr(0, n);
    }

    BOOST_FILESYSTEM_DECL path change_extension(const path& ph, const std::string& new_extension)
    {
      return ph.branch_path() / (basename(ph) + new_extension);
    }


  } // namespace filesystem
} // namespace boost
