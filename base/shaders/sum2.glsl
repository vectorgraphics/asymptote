layout(local_size_x=LOCAL_SIZE_X) in;

uniform uint offset2;

layout(binding=1, std430) buffer sumBuffer
{
  uint sum[];
};

void main(void)
{
  uint id=gl_GlobalInvocationID.x;

  uint row=LOCAL_SIZE_X*id;
  uint stop=row+LOCAL_SIZE_X;

  uint Sum=sum[row];
  for(uint i=row+1u; i < stop; ++i) {
    Sum += sum[i];
    sum[i]=Sum;
  }

  sum[offset2+id+1u]=Sum;
}
