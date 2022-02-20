layout(local_size_x=LOCAL_SIZE_X) in;

uniform uint m;

layout(binding=7, std430) buffer sumBuffer
{
  uint sum[];
};

layout(binding=8, std430) buffer sum2Buffer
{
  uint sum2[];
};

void main(void)
{
  uint id=gl_GlobalInvocationID.x;

  uint row=m*id;
  uint stop=row+m;

  uint Sum=sum[row];
  for(uint i=row+1u; i < stop; ++i) {
    Sum += sum[i];
    sum[i]=Sum;
  }

  sum2[id+1u]=Sum;
}
