/****
 * pool.cc
 * Tom Prince 2004/07/15
 *
 * Memory tracking utilities.
 *****/

#include <set>
#include "pool.h"

namespace mempool {

bool cmp(poolitem l, poolitem r)
{
  return l.ptr < r.ptr;
}

typedef std::set<poolitem,bool(*)(poolitem,poolitem)> pool_t;
pool_t thePool(cmp);

void free()
{
  for(pool_t::iterator p = thePool.begin();
      p != thePool.end(); ++p) {
    p->free();
  }
  pool_t(cmp).swap(thePool);
}

void insert(poolitem p)
{
  thePool.insert(p);
}

void erase(poolitem p)
{
  pool_t::iterator it = thePool.find(p);
  it->free();
  thePool.erase(it);
}

} // namespace pool

