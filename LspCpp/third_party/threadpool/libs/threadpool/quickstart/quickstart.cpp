/*! \file
 * \brief Quick start example.
 *
 * This is a very simple example which can be used to configure the threadpool environment on your system. 
 *
 * Copyright (c) 2005-2006 Philipp Henkel
 *
 * Distributed under the Boost Software License, Version 1.0. (See
 * accompanying file LICENSE_1_0.txt or copy at
 * http://www.boost.org/LICENSE_1_0.txt)
 *
 * http://threadpool.sourceforge.net
 *
 */

#include <iostream>

#include <boost/threadpool.hpp>

using namespace std;
using namespace boost::threadpool;

// Some example tasks
void first_task()
{
   cout << "first task is running\n" ;
}

void second_task()
{
   cout << "second task is running\n" ;
}

int main(int argc,char *argv[])
{
   // Create fifo thread pool container with two threads.
   pool tp(2);
   
   // Add some tasks to the pool.
   tp.schedule(&first_task);
   tp.schedule(&second_task);   
  
   //  Wait until all tasks are finished.
   tp.wait();
   
   // Now all tasks are finished!	
   return(0);
}


