/**
 * @file ThreadSafeQueue.h
 * @author Supakorn "Jamie" Rassameemasmuang (jamievlin [at] outlook.com)
 * @brief A Thread safe queue for sending messages between threads
 */

#pragma once

#include <queue>
#include <mutex>
#include "common.h"

#if defined(HAVE_PTHREAD)

template<typename T>
class ThreadSafeQueue
{
public:
  ThreadSafeQueue() = default;
  ~ThreadSafeQueue() noexcept = default;

  ThreadSafeQueue(ThreadSafeQueue const& other)
  {
    std::lock_guard<std::mutex> lock(other._lockMutex);
    _internalQueue = other._internalQueue;
  }

  ThreadSafeQueue(ThreadSafeQueue&& other) noexcept
  {
    std::lock_guard<std::mutex> lock(other._lockMutex);
    _internalQueue = std::move(other._internalQueue);
  }

  ThreadSafeQueue& operator= (ThreadSafeQueue const& other) = delete;
  ThreadSafeQueue& operator= (ThreadSafeQueue&& other) noexcept = delete;


  void enqueue(T const& item)
  {
    std::lock_guard<std::mutex> lock(_lockMutex);
    _internalQueue.push(item);
  }

  optional<T> dequeue()
  {
    std::lock_guard<std::mutex> lock(_lockMutex);
    if (_internalQueue.empty())
      return nullopt;

    auto value = make_optional(std::move(_internalQueue.front()));
    _internalQueue.pop();
    return value;
  }


  private:
  std::queue<T> _internalQueue;
  mutable std::mutex _lockMutex;
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
