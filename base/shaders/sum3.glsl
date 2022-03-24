layout(local_size_x=LOCAL_SIZE_X) in;

uniform uint offset2;
uniform uint final;

layout(binding=2, std430) buffer localSumBuffer
{
  uint localSum[];
};

layout(binding=3, std430) buffer globalSumBuffer
{
  uint maxCount;
  uint globalSum[];
};

layout(binding=8, std430) buffer indexBuffer
{
  uint maxSize;
  uint index[];
};

void main(void)
{
  uint id=gl_GlobalInvocationID.x;
  if(id == 0u)
    maxCount=maxSize;

  uint row=offset2+LOCAL_SIZE_X*id;
  uint stop=row+LOCAL_SIZE_X;

  uint Sum=localSum[row];
  for(uint i=row+1u; i < stop; ++i)
    localSum[i]=Sum += localSum[i];

  uint id1=id+1u;
  globalSum[id1]=id1 < gl_WorkGroupSize.x*gl_NumWorkGroups.x ?
    Sum : Sum+localSum[offset2-1u]+localSum[final];
}
