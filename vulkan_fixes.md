# Vulkan GPU Hang Fix Documentation

## Problem Analysis

The error `[mvk-error] VK_TIMEOUT: Lost VkDevice after MTLCommandBuffer execution failed` occurs when:
- View=true (interactive mode enabled)
- Opaque=false (transparency rendering enabled) 
- fxaa=false (post-processing disabled)
- GPUcompress=false (CPU-based compression)

## Root Cause

The issue is caused by:

1. **GPU Memory Pressure**: Large transparency fragment counts overwhelm the GPU
2. **Insufficient Synchronization**: Command buffer submission without proper pacing
3. **Timeout Handling**: Inadequate retry mechanisms for swapchain operations
4. **Resource Management**: No dynamic batching based on GPU capabilities

## Fixes Applied

### 1. Enhanced Transparency Batching
- Reduced max fragments per batch from 100k to 50k (25k for problematic configs)
- Added dynamic batch sizing based on configuration
- Improved synchronization between batches with `device->waitIdle()`

### 2. Improved Swapchain Acquisition
- Added retry logic with progressive timeouts
- Reduced maximum timeout from 10s to 1s for swapchain operations
- Added small delays between retry attempts

### 3. GPU Overload Prevention
- Added configuration-aware fragment limits
- Added verbose logging for debugging transparency issues
- Enhanced error handling and recovery

### 4. Memory Usage Optimization
- Added GPU memory pressure detection using fragment count heuristic
- Implemented conservative rendering when transparency is enabled without compression
- Fixed compilation issues with memory detection logic

## Testing Recommendations

After applying these fixes, test with:

```bash
# Basic test case that was failing
./asy -dir base -V teapot0

# Verify different configurations
./asy -dir base -V -render=0 teapot0  # Disable rendering
./asy -dir base -V -fxaa=true teapot0  # Enable FXAA
./asy -dir base -V -GPUcompress=true teapot0  # Enable GPU compression
```

## Configuration Guidelines

For systems experiencing GPU hangs:

1. **Enable FXAA**: Reduces fragment processing load
2. **Enable GPU compression**: Offloads work to compute shaders
3. **Reduce fragment count**: Use simpler transparency models
4. **Enable verbose mode**: Monitor batch rendering progress