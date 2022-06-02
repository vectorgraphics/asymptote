layout(local_size_x=localSize) in;

const uint BLOCKSIZE=2048u;

uniform uint workgroups;

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

// Return x divided by y rounded up to the nearest integer.
uint ceilquotient(uint x, uint y)
{
  return (x+y-1)/y;
}

void main(void)
{
  uint id=gl_LocalInvocationID.x;
  uint localSum[BLOCKSIZE];

  uint blocksize=ceilquotient(workgroups,localSize);

  uint dataOffset=blocksize*id;
  uint sum=0u;
  for(uint i=0u; i < blocksize; i++)
    localSum[i]=sum += globalSum[dataOffset+i];

  groupSum[id]=sum;
  barrier();

  // Apply Hillis-Steele algorithm over all sums in workgroup
  for(uint shift=1u; shift < localSize; shift *= 2u) {
    uint read;
    if(shift <= id) read=groupSum[id]+groupSum[id-shift];
    barrier();
    if(shift <= id) groupSum[id]=read;
    barrier();
  }

  // shift workgroup sums and store
  uint shift=id > 0u ? groupSum[id-1u] : 0u;

  for(uint i=0; i < blocksize; i++)
    globalSum[dataOffset+i]=localSum[i]+shift;

  if(id == workgroups % localSize) {
    maxSize=maxDepth;
    maxDepth=0u;
    fragments=globalSum[workgroups];
  }
}
