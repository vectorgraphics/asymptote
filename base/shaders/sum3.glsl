layout(local_size_x=LOCAL_SIZE_X) in;

uniform uint offset2;

layout(binding=1, std430) buffer maxBuffer {
  uint maxSize;
};

layout(binding=2, std430) buffer localSumBuffer
{
  uint localSum[];
};

layout(binding=3, std430) buffer globalSumBuffer {
  uint maxCount;
  uint globalSum[];
};

void main(void)
{
  uint id=gl_GlobalInvocationID.x;
  if(id == 0u) {
    maxCount=maxSize;
    maxSize=0u;
  }

  uint row=offset2+LOCAL_SIZE_X*id;
  uint stop=row+LOCAL_SIZE_X;

  uint Sum=localSum[row];
  for(uint i=row+1u; i < stop; ++i) {
    Sum += localSum[i];
    localSum[i]=Sum;
  }

  globalSum[id+1u]=Sum;
}
