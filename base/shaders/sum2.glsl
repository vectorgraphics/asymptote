layout(local_size_x=localSize) in;

const uint groupSize=localSize*blockSize;

uniform uint blockSize2;
uniform uint workGroups; // Number of workgroups in sum1 and sum3 shaders

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

shared uint groupSum[localSize];

void main(void)
{
  uint id=gl_LocalInvocationID.x;
  uint localSum[groupSize];

  uint dataOffset=blockSize2*id;
  uint sum=0u;
  for(uint i=0u; i < blockSize2; i++)
    localSum[i]=sum += globalSum[dataOffset+i];

  groupSum[id]=sum;
  barrier();

  // Apply Hillis-Steele algorithm over all sums in work group
  for(uint shift=1u; shift < localSize; shift *= 2u) {
    uint read;
    if(shift <= id) read=groupSum[id]+groupSum[id-shift];
    barrier();
    if(shift <= id) groupSum[id]=read;
    barrier();
  }

  // shift work group sums and store
  uint shift=id > 0u ? groupSum[id-1u] : 0u;

  for(uint i=0; i < blockSize2; i++)
    globalSum[dataOffset+i]=localSum[i]+shift;

  if(id == workGroups % localSize) {
    maxSize=maxDepth;
    maxDepth=0u;
    fragments=globalSum[workGroups];
  }
}
