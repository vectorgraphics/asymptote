layout(local_size_x=1) in;

uniform uint elements;

layout(binding=0) buffer sumBuffer
{
  uint sum[];
};

layout(binding=1) buffer offsetBuffer
{
  uint offset[];
};

uint ceilquotient(uint a, uint b)
{
  return (a+b-1u)/b;
}

void main(void)
{
  uint id=gl_GlobalInvocationID.x;

  uint m=ceilquotient(elements,gl_NumWorkGroups.x);
  uint row=m*id;
  uint col=min(m,elements-row);
  uint stop=row+col;

  uint Sum=offset[row];
  for(uint i=row+1u; i < stop; ++i)
    Sum += offset[i];

  sum[id]=Sum;
}
