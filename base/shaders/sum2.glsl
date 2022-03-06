layout(local_size_x=LOCAL_SIZE_X) in;

uniform uint offset2;

layout(binding=2, std430) buffer localSumBuffer
{
  uint localSum[];
};

void main(void)
{
  uint id=gl_GlobalInvocationID.x;

  uint row=LOCAL_SIZE_X*id;
  uint stop=row+LOCAL_SIZE_X;

  uint Sum=localSum[row];
  for(uint i=row+1u; i < stop; ++i) {
    Sum += localSum[i];
    localSum[i]=Sum;
  }

  localSum[offset2+id+1u]=Sum;
}
