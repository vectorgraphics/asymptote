#pragma once

#ifdef HAVE_VULKAN
#include <glslang/SPIRV/GlslangToSpv.h>

inline TBuiltInResource getShaderResources()
{
  TBuiltInResource res;

  res.maxLights = 32;
	res.maxClipPlanes = 6;
	res.maxTextureUnits = 32;
	res.maxTextureCoords = 32;
	res.maxVertexAttribs = 64;
	res.maxVertexUniformComponents = 4096;
	res.maxVaryingFloats = 64;
	res.maxVertexTextureImageUnits = 32;
	res.maxCombinedTextureImageUnits = 80;
	res.maxTextureImageUnits = 32;
	res.maxFragmentUniformComponents = 4096;
	res.maxDrawBuffers = 32;
	res.maxVertexUniformVectors = 128;
	res.maxVaryingVectors = 8;
	res.maxFragmentUniformVectors = 16;
	res.maxVertexOutputVectors = 16;
	res.maxFragmentInputVectors = 15;
	res.minProgramTexelOffset = -8;
	res.maxProgramTexelOffset = 7;
	res.maxClipDistances = 8;
	res.maxComputeWorkGroupCountX = 65535;
	res.maxComputeWorkGroupCountY = 65535;
	res.maxComputeWorkGroupCountZ = 65535;
	res.maxComputeWorkGroupSizeX = 1024;
	res.maxComputeWorkGroupSizeY = 1024;
	res.maxComputeWorkGroupSizeZ = 64;
	res.maxComputeUniformComponents = 1024;
	res.maxComputeTextureImageUnits = 16;
	res.maxComputeImageUniforms = 8;
	res.maxComputeAtomicCounters = 8;
	res.maxComputeAtomicCounterBuffers = 1;
	res.maxVaryingComponents = 60;
	res.maxVertexOutputComponents = 64;
	res.maxGeometryInputComponents = 64;
	res.maxGeometryOutputComponents = 128;
	res.maxFragmentInputComponents = 128;
	res.maxImageUnits = 8;
	res.maxCombinedImageUnitsAndFragmentOutputs = 8;
	res.maxCombinedShaderOutputResources = 8;
	res.maxImageSamples = 0;
	res.maxVertexImageUniforms = 0;
	res.maxTessControlImageUniforms = 0;
	res.maxTessEvaluationImageUniforms = 0;
	res.maxGeometryImageUniforms = 0;
	res.maxFragmentImageUniforms = 8;
	res.maxCombinedImageUniforms = 8;
	res.maxGeometryTextureImageUnits = 16;
	res.maxGeometryOutputVertices = 256;
	res.maxGeometryTotalOutputComponents = 1024;
	res.maxGeometryUniformComponents = 1024;
	res.maxGeometryVaryingComponents = 64;
	res.maxTessControlInputComponents = 128;
	res.maxTessControlOutputComponents = 128;
	res.maxTessControlTextureImageUnits = 16;
	res.maxTessControlUniformComponents = 1024;
	res.maxTessControlTotalOutputComponents = 4096;
	res.maxTessEvaluationInputComponents = 128;
	res.maxTessEvaluationOutputComponents = 128;
	res.maxTessEvaluationTextureImageUnits = 16;
	res.maxTessEvaluationUniformComponents = 1024;
	res.maxTessPatchComponents = 120;
	res.maxPatchVertices = 32;
	res.maxTessGenLevel = 64;
	res.maxViewports = 16;
	res.maxVertexAtomicCounters = 0;
	res.maxTessControlAtomicCounters = 0;
	res.maxTessEvaluationAtomicCounters = 0;
	res.maxGeometryAtomicCounters = 0;
	res.maxFragmentAtomicCounters = 8;
	res.maxCombinedAtomicCounters = 8;
	res.maxAtomicCounterBindings = 1;
	res.maxVertexAtomicCounterBuffers = 0;
	res.maxTessControlAtomicCounterBuffers = 0;
	res.maxTessEvaluationAtomicCounterBuffers = 0;
	res.maxGeometryAtomicCounterBuffers = 0;
	res.maxFragmentAtomicCounterBuffers = 1;
	res.maxCombinedAtomicCounterBuffers = 1;
	res.maxAtomicCounterBufferSize = 16384;
	res.maxTransformFeedbackBuffers = 4;
	res.maxTransformFeedbackInterleavedComponents = 64;
	res.maxCullDistances = 8;
	res.maxCombinedClipAndCullDistances = 8;
	res.maxSamples = 64;
	res.maxMeshOutputVerticesNV = 256;
	res.maxMeshOutputPrimitivesNV = 512;
	res.maxMeshWorkGroupSizeX_NV = 32;
	res.maxMeshWorkGroupSizeY_NV = 1;
	res.maxMeshWorkGroupSizeZ_NV = 1;
	res.maxTaskWorkGroupSizeX_NV = 32;
	res.maxTaskWorkGroupSizeY_NV = 1;
	res.maxTaskWorkGroupSizeZ_NV = 1;
	res.maxMeshViewCountNV = 4;
	res.limits.nonInductiveForLoops = 1;
	res.limits.whileLoops = 1;
	res.limits.doWhileLoops = 1;
	res.limits.generalUniformIndexing = 1;
	res.limits.generalAttributeMatrixVectorIndexing = 1;
	res.limits.generalVaryingIndexing = 1;
	res.limits.generalSamplerIndexing = 1;
	res.limits.generalVariableIndexing = 1;
	res.limits.generalConstantMatrixVectorIndexing = 1;

  return res;
}

#endif
