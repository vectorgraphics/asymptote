layout(local_size_x=LOCAL_SIZE_X) in;

layout(binding=2, std430) buffer localSumBuffer
{
  uint localSum[];
};

layout(binding=3, std430) buffer globalSumBuffer
{
  uint globalSum[];
};

layout(binding=7, std430) buffer opaqueDepthBuffer
{
  uint maxSize;
  float opaqueDepth[];
};

shared uint groupSum[gl_WorkGroupSize.x+1u];

void main(void)
{
  uint id=gl_GlobalInvocationID.x;
  uint row=LOCAL_SIZE_X*id;

  uint cache[LOCAL_SIZE_X];

  uint sum;
  cache[0]=sum=localSum[row];
  for(uint i=1u; i < LOCAL_SIZE_X; ++i)
    cache[i]=sum += localSum[row+i];

  uint index=gl_LocalInvocationID.x;
  if(index == 0u)
   groupSum[0u]=0u;

  groupSum[index+1u]=sum;

  barrier();

  // Hillis and Steele over all sums in workgroup
  for(uint shift=1u; shift < gl_WorkGroupSize.x; shift *= 2u) {
    uint read=index < shift ? groupSum[index] :
      groupSum[index]+groupSum[index-shift];
    barrier();
    groupSum[index]=read;
    barrier();
  }

  uint shift=groupSum[index];

  for(uint i=0u; i < LOCAL_SIZE_X; ++i)
    localSum[row+i]=cache[i]+shift;

  if(index+1u == LOCAL_SIZE_X) {
    if(gl_WorkGroupID.x == 0u) {
      globalSum[0]=maxSize;
      globalSum[gl_WorkGroupID.x+1u]=sum+shift;
    }
    globalSum[gl_WorkGroupID.x+1u]=sum+shift;
  }
}
