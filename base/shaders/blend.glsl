
layout(binding = 3, std430) buffer CountBuffer
{
  uint maxSize;
  uint count[];
};

layout(binding = 4, std430) buffer OffsetBuffer
{
  uint maxDepth;
  uint offset[];
};

layout(binding = 5, std430) buffer FragmentBuffer
{
  vec4 fragment[];
};

layout(binding = 6, std430) buffer DepthBuffer
{
  float depth[];
};

layout(binding = 7, std430) buffer OpaqueBuffer
{
  vec4 opaqueColor[];
};

layout(binding = 8, std430) buffer OpaqueDepthBuffer
{
  float opaqueDepth[];
};

#ifdef GPUCOMPRESS
layout(binding=9, std430) buffer indexBuffer
{
  uint index[];
};
#define INDEX(pixel) index[pixel]
#define COUNT(pixel) index[pixel]
#else
#define INDEX(pixel) pixel
#define COUNT(pixel) count[pixel]
#endif

layout(push_constant) uniform PushConstants
{
  uvec4 constants;
  // constants[0] = nlights
  // constants[1] = width
  vec4 background;
} push;

layout(location = 0) out vec4 outColor;

vec4 blend(vec4 outColor, vec4 color)
{
  return mix(outColor,color,color.a);
}

const float gamma=2.2;
const float invGamma=1.0/gamma;

/**
 * @brief Converts linear color (measuring photon count) to srgb (what our brain thinks
 * is the brightness
 * example linearToPerceptual(vec3(0.5)) is approximately vec3(0.729)
 */
vec3 linearToPerceptual(vec3 inColor)
{
  // an actual 0.5 brightness (half amount of photons) would
  // look brighter than what our eyes think is "half" light
  return pow(inColor, vec3(invGamma));
}

void main()
{
  uint pixel=uint(gl_FragCoord.y)*push.constants[1]+uint(gl_FragCoord.x);
  float OpaqueDepth=opaqueDepth[pixel];
  uint element=INDEX(pixel);

#ifdef GPUCOMPRESS
  if(element == 0u) {
   if(OpaqueDepth != 0.0)
      opaqueDepth[pixel]=0.0;
    discard;
  }
#endif

  uint listIndex=offset[element];
  uint size=offset[element+1u]-listIndex;

#ifndef GPUCOMPRESS
  if(size == 0u) {
    if(OpaqueDepth != 0.0)
      opaqueDepth[pixel]=0.0;
    discard;
  }
#endif

  outColor=OpaqueDepth != 0.0 ? opaqueColor[pixel] : push.background;

  uint k=0u;
  if(OpaqueDepth != 0.0)
    while(k < size && depth[listIndex+k] >= OpaqueDepth)
      ++k;

  uint n=size-k;

  // Sort the fragments with respect to descending depth
  if(n <= ARRAYSIZE) {
    if(n == 1)
      outColor=blend(outColor,fragment[listIndex+k]);
    else if(n > 0) {
      struct element {
        uint index;
        float depth;
      };

      element E[ARRAYSIZE];
      E[0]=element(k,depth[listIndex+k]);
      uint i=1u;
      while(++k < size) {
        float d=depth[listIndex+k];
        if(OpaqueDepth != 0.0) {
          while(k < size && d >= OpaqueDepth) {
            ++k;
            d=depth[listIndex+k];
          }
          if(k == size) break;
        }
        uint j=i;
        while(j > 0u && d > E[j-1u].depth) {
          E[j]=E[j-1u];
          --j;
        }
        E[j]=element(k,d);
        ++i;
      }


      for(uint j=0u; j < i; ++j)
        outColor=blend(outColor,fragment[listIndex+E[j].index]);
    }

    if(OpaqueDepth != 0.0)
      opaqueDepth[pixel]=0.0;
  } else {
    atomicMax(maxDepth,size);
    maxSize=maxDepth;
    for(uint i=k+1u; i < size; i++) {
      vec4 temp=fragment[listIndex+i];
      float d=depth[listIndex+i];
      uint j=i;
      while(j > 0u && d > depth[listIndex+j-1u]) {
        fragment[listIndex+j]=fragment[listIndex+j-1u];
        depth[listIndex+j]=depth[listIndex+j-1u];
        --j;
      }
      fragment[listIndex+j]=temp;
      depth[listIndex+j]=d;
    }

    uint stop=listIndex+size;
    if(OpaqueDepth == 0.0)
      for(uint i=listIndex+k; i < stop; i++)
        outColor=blend(outColor,fragment[i]);
    else {
      for(uint i=listIndex+k; i < stop; i++) {
        if(depth[i] < OpaqueDepth)
          outColor=blend(outColor,fragment[i]);
      }
      opaqueDepth[pixel]=0.0;
    }
  }

  // if FXAA is enabled, convert it to perceptual since FXAA needs it
  // otherwise, if OUTPUT_AS_SRGB is enabled, also convert it to perceptual
#if defined(ENABLE_FXAA) || defined(OUTPUT_AS_SRGB)
  // before we pass to post-processing stage, convert the color into
  // perceptual (sRGB) first since we blended the colors using linear values
  vec3 perceptualColor=linearToPerceptual(outColor.rgb);
  outColor = vec4(perceptualColor, outColor.a);
#endif

  COUNT(pixel)=0u;
}
