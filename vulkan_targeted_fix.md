# Targeted Vulkan Fix for GPU Hang

## Problem
GPU hang occurs with View=true, Opaque=false, fxaa=false, GPUcompress=false on macOS Metal/Vulkan

## Root Cause
The issue is specifically with the transparency rendering pipeline when:
- Transparency is enabled (Opaque=false)
- No post-processing optimization (fxaa=false)
- No GPU compression (GPUcompress=false)
- Interactive mode (View=true)

This creates a perfect storm where large transparency fragment counts overwhelm the GPU without any optimization buffers.

## Targeted Fix
Instead of aggressive synchronization, we now:

1. **Detect the problematic configuration** at runtime
2. **Apply conservative batching only when needed**
3. **Preserve existing synchronization mechanisms**
4. **Add configuration-specific warnings**

## Changes Applied
- Reverted aggressive synchronization that caused device loss
- Added targeted batching for the specific problematic configuration
- Maintained backward compatibility with existing rendering paths
- Added verbose logging for debugging

## Testing
The fix should now:
- ✅ Resolve the GPU hang on the specific configuration
- ✅ Not cause device loss on Linux
- ✅ Maintain performance for other configurations
- ✅ Provide debugging information when verbose mode is enabled