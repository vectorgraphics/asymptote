/*! \file
 * \brief Mergesort example.
 *
 * This example shows how to use the threadpool library. 
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


#include <boost/threadpool.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/smart_ptr.hpp>
#include <iostream>
#include <sstream>
#include <algorithm>



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

template<class T>
string to_string(const T& value)
{
  ostringstream ost;
  ost << value;
  ost.flush();
  return ost.str();
}

unsigned long get_ms_diff(boost::xtime& start, boost::xtime& end)
{
  boost::xtime::xtime_sec_t start_ms = start.sec * 1000	+ start.nsec/1000000; 
  boost::xtime::xtime_sec_t end_ms = end.sec * 1000	+ end.nsec/1000000; 
  return static_cast<unsigned long>(end_ms - start_ms);
}

class image
{
public:
  image() : m_content(0)	{}
  image(int content) : m_content(content)	{}

  image(const image& src)
  {
    m_content = src.m_content;
  }

  bool operator<(const image& l) const
  {
    {	// simulate time needed for image comparision
      boost::xtime xt;
      boost::xtime_get(&xt, boost::TIME_UTC);
      int duration = 1+(m_content % 4);
      xt.nsec += 250 * 1000 * duration;	
      boost::thread::sleep(xt); 
	    print(".");
    }	
    return m_content < l.m_content;
  }

protected:
  int m_content;	// represents image data in this example
};


template<class T>
class merge_job
{
public:
  merge_job(boost::shared_array<T> data, unsigned int position, unsigned int length) 
    : m_data(data)
    , m_position(position)
    , m_length(length) 
  {
    print("merge job created : " + to_string(m_position) +", "+ to_string(m_length) +"\n");
  }

  void run()
  {	
    print("merge job running :   " + to_string(m_position) +", "+ to_string(m_length) +"\n");

    T* begin = m_data.get();
    std::advance(begin, m_position);

    T* mid = m_data.get();
    std::advance(mid, m_position + m_length/2);

    T* end = m_data.get();
    std::advance(end, m_position + m_length);

    std::inplace_merge(begin, mid, end);

    print("\nmerge job finished:     "  + to_string(m_position) +", "+ to_string(m_length) +"\n");
  }

protected:
  boost::shared_array<T> m_data;
  unsigned int m_position;
  unsigned int m_length;
};




//
// A demonstration of the thread_pool class
int main (int argc, char * const argv[]) 
{
  print("MAIN: construct thread pool\n");

		

  boost::xtime start;
  boost::xtime_get(&start, boost::TIME_UTC);

  int exponent = 7;
  int data_len = 1 << exponent;  // = pow(2, exponent) 

  print("MAIN: sort array with "+ to_string(data_len) + " elements.\n");

  boost::shared_array<image> data(new image[data_len]);

  // fill array with arbitrary values (not sorted ascendingly)
  for(int i = 0; i < data_len; i++)
  {
    data[i] = image((data_len - i - 1) % 23);
  }


  /***************************/
  /* Standard implementation */
  /***************************/

  pool tp;
  tp.size_controller().resize(5);	

// merge data array
  for(int step = 1; step <= exponent; step++)
  {
    print("\nMAIN: merge step "+ to_string(step)+"\n");

    // divide array into partitions
    int partition_size = 1 << step;
    for(int partition = 0; partition < data_len/partition_size; partition++)
    {
      // sort partition
      boost::shared_ptr<merge_job<image> > job(new merge_job<image>(data, partition*partition_size, partition_size));
      //tp->schedule(prio_task_func(5, boost::bind(&merge_job<image>::run, job)));
      schedule(tp, boost::bind(&merge_job<image>::run, job));
     // schedule(tp, job);
    }
    tp.wait();	// wait until all partitions are sorted
  } 

  boost::xtime end;
  boost::xtime_get(&end, boost::TIME_UTC);				

  print("\nMAIN: duration " + to_string(get_ms_diff(start, end)) + " ms \n");

  print("\nMAIN: check if array is sorted... \n");

  // check if array is sorted ascendingly 
  bool ascending = true;
  for(int i = 0; i < data_len-1; i++)
  {
    if(data[i+1] < data[i])
    {
      ascending = false;
    }
  }

  if(ascending)
  {
    print("\nMAIN: array is sorted\n");
  }
  else
  {
    print("\nMAIN: array is NOT sorted!\n");
  }

  return 0;
}
