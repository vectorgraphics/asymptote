layout(local_size_x=1) in;

uniform uint elements;

layout(binding=1, std430) buffer offsetBuffer
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
  uint stop=row+min(m,elements-row);

  uint Sum=offset[row];
  for(uint i=row+1u; i < stop; ++i) {
    Sum += offset[i];
    offset[i]=Sum;
  }
}
