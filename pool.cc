/****
 * pool.cc
 * Tom Prince 2004/07/15
 *
 * Memory tracking utilities.
 *****/

#include <deque>
#include "pool.h"

namespace memory {

bool cmp(poolitem l, poolitem r)
{
  return l.ptr < r.ptr;
}

typedef std::deque<poolitem> pool_t;
pool_t thePool;

void free()
{
  for(pool_t::iterator p = thePool.begin();
      p != thePool.end(); ++p) {
    p->free();
  }
  pool_t().swap(thePool);
}

void insert(poolitem p)
{
  thePool.push_back(p);
}

void erase(poolitem)
{}

} // namespace pool

