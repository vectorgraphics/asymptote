layout(local_size_x=LOCAL_SIZE_X) in;

uniform uint offset2;

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
  uint Sum=globalSum[id];

  uint row=offset2+LOCAL_SIZE_X*id+1u;
  uint stop=row+LOCAL_SIZE_X;

  for(uint i=row; i < stop; ++i) {
    Sum += localSum[i];
    localSum[i]=Sum;
  }
}
