layout(local_size_x=1) in;

uniform uint nElements;

layout(binding=1) buffer Data
{
  uint data[];
};

uint ceilquotient(uint a, uint b)
{
  return (a+b-1)/b;
}

void main(void)
{
  const uint id=gl_GlobalInvocationID.x;

  const uint m=ceilquotient(nElements,gl_NumWorkGroups.x);
  const uint row=m*id;
  const uint col=min(m,nElements-row);
  const uint stop=row+col-1;

  uint curr=data[row];
  for(uint i=row; i < stop; ++i) {
    uint next=data[i+1];
    curr += next;
    data[i+1]=curr;
  }
}
