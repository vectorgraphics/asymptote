layout(local_size_x=LOCAL_SIZE_X) in;

uniform uint elements;

layout(binding=0, std430) buffer offsetBuffer
{
  uint offset[];
};

layout(binding=2, std430) buffer localSumBuffer
{
  uint localSum[];
};

shared uint groupSum[gl_WorkGroupSize.x+1u];

void main(void)
{
  uint id=gl_GlobalInvocationID.x;

  uint m=elements/(gl_WorkGroupSize.x*gl_NumWorkGroups.x);
  uint r=elements-m*gl_WorkGroupSize.x*gl_NumWorkGroups.x;
  uint row,stop;
  if(id < r) {
    row=m*id+id;
    stop=m+1u;
  } else {
    row=m*id+r;
    stop=m;
  }

  uint cache[4]; // must be bigger than m.

  uint sum;
  cache[0]=sum=offset[row];
  for(uint i=1u; i < stop; ++i)
    cache[i]=sum += offset[row+i];

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

  uint start=elements+row;
  for(uint i=0u; i < stop; ++i)
    offset[start+i]=cache[i]+shift;

  if(index+1u == LOCAL_SIZE_X)
    localSum[gl_WorkGroupID.x+1u]=sum+shift;
}
