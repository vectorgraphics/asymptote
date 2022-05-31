layout(local_size_x=localSize) in;

layout(binding=0) uniform atomic_uint elements;

layout(binding=0, std430) buffer offsetBuffer
{
  uint maxDepth;
  uint offset[];
};

layout(binding=3, std430) buffer globalSumBuffer
{
  uint globalSum[];
};

layout(binding=8, std430) buffer feedbackBuffer
{
  uint maxSize;
  uint fragments;
};

uniform uint final;

// avoid bank conflicts and coalesce global memory accesses
shared uint groupSum[localSize+1u];
shared uint shuffle[localSize*(blockSize+1u)];

void main(void)
{
  uint id=gl_LocalInvocationID.x;
  uint shuffleOffset=id/blockSize+id;
  const uint stride=localSize/blockSize+localSize;
  for(uint i=0; i < blockSize; i++)
    shuffle[shuffleOffset+i*stride]=globalSum[id+i*localSize];

  barrier();

  uint Offset=id*blockSize+id;
  uint stop=Offset+blockSize;
  uint sum=shuffle[Offset];
  for(uint i=Offset+1u; i < stop; ++i)
    shuffle[i]=sum += shuffle[i];

  if(id == 0u)
    groupSum[0u]=0u;
  groupSum[id+1u]=sum;
  barrier();

  // Apply Hillis-Steele algorithm over all sums in workgroup
  for(uint shift=1u; shift < localSize; shift *= 2u) {
    uint read;
    if(shift <= id) read=groupSum[id]+groupSum[id-shift];
    barrier();
    if(shift <= id) groupSum[id]=read;
    barrier();
  }

  for(uint i=0u; i < blockSize; i++)
    globalSum[id+i*localSize]=
      shuffle[shuffleOffset+i*stride]+groupSum[(i*localSize+id)/blockSize];

  if(id == final % localSize) {
    maxSize=maxDepth;
    maxDepth=0u;
    fragments=globalSum[final];
  }
}
