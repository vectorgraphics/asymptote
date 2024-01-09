#pragma once
#include <vk_mem_alloc.h>
#include <stdexcept>

namespace vma
{
namespace cxx
{
class UniqueBuffer
{
public:
  [[maybe_unused]]
  UniqueBuffer() = default;
  UniqueBuffer(VmaAllocator const& allocator, VmaAllocation const& alloc, VkBuffer const& buf);
  ~UniqueBuffer();

  UniqueBuffer(UniqueBuffer const&) = delete;
  UniqueBuffer& operator=(UniqueBuffer const&) = delete;

  UniqueBuffer(UniqueBuffer&& other) noexcept;
  UniqueBuffer& operator=(UniqueBuffer&& other) noexcept;

  [[nodiscard]]
  VkBuffer getBuffer() const;

  [[nodiscard]]
  VmaAllocation getAllocation() const;

  [[nodiscard]]
  VmaAllocator getAllocator() const;

private:
  VmaAllocation _allocation = VK_NULL_HANDLE;
  VkBuffer _buffer = VK_NULL_HANDLE;
  VmaAllocator _allocator = VK_NULL_HANDLE;
};

class MemoryMapperLock
{
public:
  MemoryMapperLock(UniqueBuffer const& buffer);
  ~MemoryMapperLock();

  MemoryMapperLock(MemoryMapperLock const&) = delete;
  MemoryMapperLock& operator=(MemoryMapperLock const&) = delete;

  MemoryMapperLock(MemoryMapperLock&& other) = delete;
  MemoryMapperLock& operator=(MemoryMapperLock&& other) = delete;

  template<typename T=uint32_t>
  [[nodiscard]]
  T* getCopyPtr() const
  {
    return reinterpret_cast<T*>(copyPtr);
  }
private:
  UniqueBuffer const* sourceBuffer;
  void* copyPtr=nullptr;
};


class UniqueAllocator
{
public:
  [[maybe_unused]]
  UniqueAllocator() = default;
  UniqueAllocator(VmaAllocatorCreateInfo const& vmaAllocatorCreateInfo);
  ~UniqueAllocator();

  // no copy operations
  UniqueAllocator(UniqueAllocator const&) = delete;
  UniqueAllocator& operator=(UniqueAllocator const&) = delete;

  UniqueAllocator(UniqueAllocator&& other) noexcept;
  UniqueAllocator& operator=(UniqueAllocator&& other) noexcept;

  [[nodiscard]]
  VmaAllocator getAllocator() const;

  [[nodiscard]]
  UniqueBuffer createBuffer(VkBufferCreateInfo const& bufferCreateInfo, VmaAllocationCreateInfo const& allocInfo);

private:
  VmaAllocator _allocator = VK_NULL_HANDLE;
};

//

}
}
