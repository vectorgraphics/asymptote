/**
 * @file ThreadSafeQueue.cc
 * @author Supakorn "Jamie" Rassameemasmuang (jamievlin [at] outlook.com)
 * @brief Test for thread safe queue
 */

#include "asycxxtests/common.h"
#include "ThreadSafeQueue.h"

#define TESTING_FILE_NAME ThreadSafeQueueTest

#if !defined(HAVE_PTHREAD)
#pragma message("pthreads not enabled; this test will not do anything much")
#endif
TEST(TESTING_FILE_NAME, TestEmptyQueue)
{
  ThreadSafeQueue<uint32_t> tsq;

  ASSERT_FALSE(tsq.dequeue().has_value());
}

TEST(TESTING_FILE_NAME, TestEnqueueAndDequeue)
{
  ThreadSafeQueue<uint32_t> tsq;
  tsq.enqueue(1);
  tsq.enqueue(5);

  auto const val1 = tsq.dequeue();
  auto const val2 = tsq.dequeue();

  ASSERT_TRUE(val1.has_value());
  ASSERT_EQ(*val1, 1);
  ASSERT_TRUE(val2.has_value());
  ASSERT_EQ(*val2, 5);

  ASSERT_FALSE(tsq.dequeue().has_value());
}

class ThreadSafeQueueMultithreadedTests
  : public testing::Test
{
protected:
  void SetUp() override
  {
    pthread_cond_init(&cv, nullptr);
    pthread_mutex_init(&cvLock, nullptr);
  }

  void TearDown() override
  {
    pthread_cond_destroy(&cv);
    pthread_mutex_destroy(&cvLock);
  }

  pthread_mutex_t cvLock;
  pthread_cond_t cv;
};

struct TSQMtEnqueueArgs
{
  ThreadSafeQueue<uint32_t>* queue;
  pthread_mutex_t* cvLock;
  pthread_cond_t* cv;
  uint32_t* checkVal;
  void* retPtr;
};

TEST_F(ThreadSafeQueueMultithreadedTests, TestMultithreadedEnqueue)
{
  ThreadSafeQueue<uint32_t> tsq;

  uint32_t checkVal = 0;
  pthread_t consumerThread;
  uint32_t dummy = 500;
  TSQMtEnqueueArgs args{
    &tsq,
    &cvLock,
    &cv,
    &checkVal,
    &dummy
  };

  pthread_create(&consumerThread, nullptr,
    [](void* threadArgsRaw) -> void*
    {
      auto threadArgs = reinterpret_cast<TSQMtEnqueueArgs*>(threadArgsRaw);

      pthread_mutex_lock(threadArgs->cvLock);
      while (*threadArgs->checkVal < 1) pthread_cond_wait(threadArgs->cv, threadArgs->cvLock);
      pthread_mutex_unlock(threadArgs->cvLock);
      auto const val = threadArgs->queue->dequeue();

      if (!(val.has_value() && *val == 1))
      {
        return nullptr;
      }

      pthread_mutex_lock(threadArgs->cvLock);
      while (*threadArgs->checkVal < 2) pthread_cond_wait(threadArgs->cv, threadArgs->cvLock);
      pthread_mutex_unlock(threadArgs->cvLock);
      auto const val2= threadArgs->queue->dequeue();

      if (!(val2.has_value() && *val2 == 2))
      {
        return nullptr;
      }

      if (threadArgs->queue->dequeue().has_value())
      {
        return nullptr;
      }

      // use return value to check if values are successful or not.
      // if null, fail
      return threadArgs->retPtr;
    },
    &args);

  tsq.enqueue(1);
  pthread_mutex_lock(&cvLock);
  checkVal += 1;
  pthread_cond_signal(&cv);
  pthread_mutex_unlock(&cvLock);

  tsq.enqueue(2);
  pthread_mutex_lock(&cvLock);
  checkVal+= 1;
  pthread_cond_signal(&cv);
  pthread_mutex_unlock(&cvLock);

  uint32_t* retValue;
  pthread_join(consumerThread, reinterpret_cast<void**>(&retValue));
  ASSERT_EQ(retValue, &dummy);
}
