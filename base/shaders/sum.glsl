layout(local_size_x=LOCAL_SIZE_X) in;

uniform uint m;

layout(binding=1, std430) buffer offsetBuffer
{
  uint offset[];
};

layout(binding=2, std430) buffer sumBuffer
{
  uint sum[];
};

void main(void)
{
  uint id=gl_GlobalInvocationID.x;

  uint row=m*id;
  uint stop=row+m;

  uint Sum=offset[row];
  for(uint i=row+1u; i < stop; ++i) {
    Sum += offset[i];
    offset[i]=Sum;
  }

  sum[id+1u]=Sum;
}
