/*! \file
* \brief threadpool tutorial.
*
* This file contains a tutorial for the threadpool library. 
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
#include <sstream>
#include <boost/thread/mutex.hpp>
#include <boost/bind.hpp>

#include <boost/threadpool.hpp>

using namespace std;
using namespace boost::threadpool;


//
// Helpers
boost::mutex m_io_monitor;

void print(string text)
{
  boost::mutex::scoped_lock lock(m_io_monitor);
  cout << text;
}

template<typename T>
string to_string(T const & value)
{
  ostringstream ost;
  ost << value;
  ost.flush();
  return ost.str();
}



//
// An example task functions
void task_1()
{
  print("  task_1()\n");
}

void task_2()
{
  print("  task_2()\n");
}

void task_3()
{
  print("  task_3()\n");
}

int task_4()
{
  print("  task_4()\n");
  return 4;
}

void task_with_parameter(int value)
{
  print("  task_with_parameter(" + to_string(value) + ")\n");
}

int loops = 0;
bool looped_task()
{
  print("  looped_task()\n");
  return ++loops < 5; 
}


int task_int()
{
  print("  task_int()\n");
  return 23;
}


void fifo_pool_test()
{
    pool tp;
    
    tp.schedule(&task_1);
    tp.schedule(boost::bind(task_with_parameter, 4));

    if(!tp.empty())
    {
      tp.clear();  // remove all tasks -> no output in this test
    }

    size_t active_threads   = tp.active();
    size_t pending_threads  = tp.pending();
    size_t total_threads    = tp.size();
    
    size_t dummy = active_threads + pending_threads + total_threads;
    dummy++;

    tp.size_controller().resize(5);
    tp.wait();
}

void lifo_pool_test()
{
    lifo_pool tp;
    tp.size_controller().resize(0);
    schedule(tp, &task_1);
    tp.size_controller().resize(10);
    tp.wait();
}

void prio_pool_test()
{
    prio_pool tp(2);
    schedule(tp, prio_task_func(1, &task_1));
    schedule(tp, prio_task_func(10,&task_2));
}


void future_test()
{
    fifo_pool tp(5);
    future<int> fut = schedule(tp, &task_4);
    int res = fut();
}


int main (int , char * const []) 
{
  fifo_pool_test();
  lifo_pool_test();
  prio_pool_test();
  future_test();
  return 0;
}
