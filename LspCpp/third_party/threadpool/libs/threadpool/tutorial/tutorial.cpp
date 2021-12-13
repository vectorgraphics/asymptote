/*! \file
* \brief threadpool tutorial.
*
* This file contains a tutorial for the threadpool library. 
*
* Copyright (c) 2005-2007 Philipp Henkel
*
* Distributed under the Boost Software License, Version 1.0. (See
* accompanying file LICENSE_1_0.txt or copy at
* http://www.boost.org/LICENSE_1_0.txt)
*
* http://threadpool.sourceforge.net
*
*/

//#define _CRTDBG_MAP_ALLOC
#include <stdlib.h>

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
  throw 5;
}

void task_2()
{
  print("  task_2()\n");
  throw 5;
}

void task_3()
{
  print("  task_3()\n");
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


int task_int_23()
{
  print("  task_int_23()\n");
  return 23;
}

int task_int_1()
{
  print("  task_int_1()\n");
  return 1;
}


class CTest
{
  pool m_pool;
public:
  CTest()
    : m_pool(pool(1000))
  {
  }
};


//
// A demonstration of the thread_pool class
int main (int , char * const []) 
{
  print("\nWelcome to the threadpool tutorial!\n");

  print("\n**************************************\n");
  print("Section 1: Quick Start\n");
  
  //void func()
  {	
    print("  Create a new thread pool\n");
    pool tp(2); // tp is handle to the pool

    // Add tasks
    tp.schedule(&task_1);
    tp.schedule(&task_2);
    tp.schedule(&task_3);
    tp.schedule(boost::bind(task_with_parameter, 4));

    // The pool handle tp is allocated on stack and will 
    // be destructed if it gets out of scope. Before the 
    // pool is destroyed it waits for its tasks. 
    // That means the current thread of execution is 
    // blocked at the end of the function 
    // (until all tasks are processed).
    // while (&tp){int i = 3; ++i;}
  }	 

  { // Section Futures
    print("\n**************************************\n");
    print("Section 1: Futures\n");
    
  //typedef thread_pool<task_func, fifo_scheduler, static_size, empty_controller, wait_for_all_tasks> test_pool;

    pool tp;

//    tp.resize(0);
    tp.pending();
//    tp.clear();
    boost::xtime t;
    tp.wait(t);
    bool test = tp.empty();
    if(test) 
    {
      test = false;
    }

    tp.size_controller().resize(2);

    //test_pool::size_controller_type controller = tp.size_controller();
//    controller.resize(5);

    schedule(tp, &task_int_1);
    future<int> res = schedule(tp, &task_int_23);
    future<int> res2 = schedule(tp, &task_int_1);

    res.wait();
    int value = res.get() + res2.get();

    res.cancel();
    res.is_cancelled();
value ++;

//thread_pool<boost::function0<int>, fifo_scheduler> test2332;

//TODO runnable comile test
  }



  {	// Section 2
    print("\n**************************************\n");
    print("Section 2: Controlling scheduling\n");

    // Create a lifo_pool: last task in, first task out
    lifo_pool tp(0);

    print("  Add tasks (using the pool's schedule function)\n");	
    schedule(tp, &task_1);
    schedule(tp, &task_2);
    schedule(tp, &task_3);

    // tp.wait();  This would be a deadlock as there are no threads which process the tasks.

    print("  Add some threads ...\n");	
    tp.size_controller().resize(5);

    print("  Wait until all tasks are finished ...\n");
    tp.wait();
    print("  Tasks finished!\n");	
  }	



  {	// Section 3:
    print("\n**************************************\n");
    print("Section 3: Prioritized Tasks\n");

    prio_pool tp(0);


    print("  Add prioritized tasks ...\n");	
    schedule(tp, prio_task_func(1, &task_1));
    schedule(tp, prio_task_func(10,&task_2));
    schedule(tp, prio_task_func(5,&task_3));

    // Tasks are ordered according to their priority: task_2, task_4, task_3, task_1

    print("  Thread added\n");	
    tp.size_controller().resize(10);

    print("  Wait until all tasks are finished ...\n");
    tp.wait();
    print("  Tasks finished!\n");	
  }		


/* */
  {	// Section 5:
    print("\n**************************************\n");
    print("Section 5: Advanced thread pool instantiation\n");
    // Create the pool directly
/*
TODO
boost::shared_ptr<fifo_pool> tp = fifo_pool::create_pool(5);			

    print("  Add tasks ...\n");
    tp->schedule(&task_1);
    tp->schedule(&task_2);
    tp->schedule(&task_3);
    tp->schedule(looped_task_func(&looped_task, 1500));

    print("  Wait until all tasks are finished ...\n");
    tp->wait();
*/
  
    print("  Tasks finished!\n");
  			
  }			


  print("\n**************************************\n");
  print("Tutorial finished!\n");



  {	// Section Compile Tests
    print("\n**************************************\n");
    print("Section Compile Tests\n");


    fifo_pool tp;
    tp.size_controller().resize(0);
    tp.empty(); 
  }

  return 0;
}
