/**
 * @file ThreadSafeQueue.h
 * @author Supakorn "Jamie" Rassameemasmuang (jamievlin [at] outlook.com)
 * @brief A Thread safe queue for sending messages between threads
 */

#pragma once

#include <queue>
#include "common.h"

#if defined(HAVE_PTHREAD)

template<typename T>
class ThreadSafeQueue
{
public:
  ThreadSafeQueue()
  {
    if (pthread_mutex_init(&this->_lockMutex, nullptr) != 0)
    {
      throw std::runtime_error("Cannot initialize mutex");
    }
  }

  ~ThreadSafeQueue() noexcept
  {
    pthread_mutex_destroy(&this->_lockMutex);
  }

  ThreadSafeQueue(ThreadSafeQueue const& other) :
    ThreadSafeQueue()
  {
    pthread_mutex_lock(&other._lockMutex);
    _internalQueue = other._internalQueue;
    pthread_mutex_unlock(&other._lockMutex);
  }

  ThreadSafeQueue(ThreadSafeQueue&& other) noexcept : ThreadSafeQueue()
  {
    pthread_mutex_lock(&other._lockMutex);
    _internalQueue= std::move(other._internalQueue);
    pthread_mutex_unlock(&other._lockMutex);
  }

  ThreadSafeQueue& operator= (ThreadSafeQueue const& other) = delete;
  ThreadSafeQueue& operator= (ThreadSafeQueue&& other) noexcept = delete;

  
  void enqueue(T const& item)
  {
    pthread_mutex_lock(&_lockMutex);
    _internalQueue.push(item);
    pthread_mutex_unlock(&_lockMutex);
  }

  optional<T> dequeue()
  {
    optional<T> ret = nullopt;
    pthread_mutex_lock(&_lockMutex);

    if (!_internalQueue.empty())
    {
      ret=make_optional(std::move(_internalQueue.front()));
      _internalQueue.pop();
    }

    pthread_mutex_unlock(&_lockMutex);
    return ret;
  }

  
  private:
  std::queue<T> _internalQueue;
  pthread_mutex_t _lockMutex;
};

#else
// no thread; calls are already serialized
template<typename T>
class ThreadSafeQueue
{
public:
  ThreadSafeQueue() = default;
  ~ThreadSafeQueue() = default;
  ThreadSafeQueue(ThreadSafeQueue const& other) = default;
  ThreadSafeQueue(ThreadSafeQueue&& other) noexcept = default; 
  ThreadSafeQueue& operator=(ThreadSafeQueue const& other)= delete;
  ThreadSafeQueue& operator=(ThreadSafeQueue&& other) noexcept= delete;

  void enqueue(T const& item)
  {
    _internalQueue.push(item);
  }

  optional<T> dequeue()
  {
    if (_internalQueue.empty())
    {
      return nullopt;
    }

    auto value= make_optional(std::move(_internalQueue.front()));
    _internalQueue.pop();

    return value;
  }
private:
  std::queue<T> _internalQueue;
};
#endif