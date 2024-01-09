#include "vma_cxx.h"

#include <utility>
#include <stdexcept>

namespace vma
{
namespace cxx
{

UniqueAllocator::UniqueAllocator(VmaAllocatorCreateInfo const& vmaAllocatorCreateInfo)
{
  if (vmaCreateAllocator(&vmaAllocatorCreateInfo, &_allocator) != VK_SUCCESS)
  {
    throw std::runtime_error("Cannot create Vulkan memory allocator");
  }
}

UniqueAllocator::~UniqueAllocator()
{
  if (_allocator != VK_NULL_HANDLE)
  {
    vmaDestroyAllocator(_allocator);
  }
}

UniqueAllocator::UniqueAllocator(UniqueAllocator&& other) noexcept
  : _allocator(std::exchange(other._allocator, VK_NULL_HANDLE))
{
}

UniqueAllocator& UniqueAllocator::operator=(UniqueAllocator&& other) noexcept
{
  if (this != &other)
  {
    std::swap(_allocator, other._allocator);
  }
  return *this;
}

VmaAllocator UniqueAllocator::getAllocator() const
{
  return _allocator;
}

UniqueBuffer
UniqueAllocator::createBuffer(VkBufferCreateInfo const& bufferCreateInfo, VmaAllocationCreateInfo const& allocInfo)
{
  VkBuffer buf;
  VmaAllocation alloc;
  if (vmaCreateBuffer(_allocator, &bufferCreateInfo, &allocInfo, &buf, &alloc, nullptr) != VK_SUCCESS)
  {
    throw std::runtime_error("Cannot create Vulkan memory buffer");
  }

  return {_allocator, alloc, buf};
}
UniqueImage UniqueAllocator::createImage(
  VkImageCreateInfo const& imgCreateInfo,
  VmaAllocationCreateInfo const& allocInfo
)
{
  VkImage resultImage;
  VmaAllocation resultAllocation;

  if (vmaCreateImage(_allocator, &imgCreateInfo, &allocInfo, &resultImage, &resultAllocation, nullptr) != VK_SUCCESS)
  {
    throw std::runtime_error("Cannot create Vulkan image");
  }

  return {_allocator, resultImage, resultAllocation};
}

// unique buffer

UniqueBuffer::UniqueBuffer(
        VmaAllocator const& allocator,
        VmaAllocation const& alloc,
        VkBuffer const& buf
        )
    : _allocator(allocator), _buffer(buf), _allocation(alloc)
{

}
UniqueBuffer::~UniqueBuffer()
{
  if (_allocation != VK_NULL_HANDLE)
  {
    vmaDestroyBuffer(_allocator, _buffer, _allocation);
  }
}
UniqueBuffer::UniqueBuffer(UniqueBuffer&& other) noexcept
    : _allocator(std::exchange(other._allocator, VK_NULL_HANDLE)),
      _buffer(std::exchange(other._buffer, VK_NULL_HANDLE)),
      _allocation(std::exchange(other._allocation, VK_NULL_HANDLE))
{

}
UniqueBuffer& UniqueBuffer::operator=(UniqueBuffer&& other) noexcept
{
  if (this != &other)
  {
    std::swap(_allocator, other._allocator);
    std::swap(_buffer, other._buffer);
    std::swap(_allocation, other._allocation);
  }
  return *this;
}

VkBuffer UniqueBuffer::getBuffer() const
{
  return _buffer;
}
VmaAllocation UniqueBuffer::getAllocation() const
{
  return _allocation;
}
VmaAllocator UniqueBuffer::getAllocator() const
{
  return _allocator;
}


MemoryMapperLock::MemoryMapperLock(UniqueBuffer const& buffer) : sourceBuffer(&buffer)
{
  if (vmaMapMemory(buffer.getAllocator(), buffer.getAllocation(), &copyPtr) != VK_SUCCESS)
  {
    throw std::runtime_error("Cannot map memory");
  }
}
MemoryMapperLock::~MemoryMapperLock()
{
  if (copyPtr != nullptr)
  {
    vmaUnmapMemory(sourceBuffer->getAllocator(), sourceBuffer->getAllocation());
  }
}
UniqueImage::UniqueImage(VmaAllocator const& allocator, VkImage const& image, VmaAllocation const& allocation)
    : _allocator(allocator), _image(image), _allocation(allocation)
{
  
}
UniqueImage::~UniqueImage()
{
  if (_allocation != VK_NULL_HANDLE)
  {
    vmaDestroyImage(_allocator, _image, _allocation);
  }
}
UniqueImage::UniqueImage(UniqueImage&& other) noexcept
    : _allocator(std::exchange(other._allocator, VK_NULL_HANDLE)), _image(std::exchange(other._image, VK_NULL_HANDLE)),
      _allocation(std::exchange(other._allocation, VK_NULL_HANDLE))
{
  
}
UniqueImage& UniqueImage::operator=(UniqueImage&& other) noexcept
{
  if (this != &other)
  {
    std::swap(_allocator, other._allocator);
    std::swap(_image, other._image);
    std::swap(_allocation, other._allocation);
  }
  return *this;
}
VkImage UniqueImage::getImage() const
{
  return _image;
}
}// namespace cxx
}
