layout(local_size_x=1) in;

uniform uint elements;

layout(binding=0, std430) buffer sumBuffer
{
  uint sum[];
};

layout(binding=1, std430) buffer offsetBuffer
{
  uint offset[];
};

void main(void)
{
  uint id=gl_GlobalInvocationID.x+1u;

  uint p=gl_NumWorkGroups.x+1u;
  uint m=elements/p;
  uint r=elements-m*p;
  uint row,stop;
  if(id < r) {
    row=m*id+id;
    stop=row+m+1u;
  } else {
    row=m*id+r;
    stop=row+m;
  }

  uint Sum=offset[row];
  for(uint i=row+1u; i < stop; ++i) {
    Sum += offset[i];
    offset[i]=Sum;
  }

  if(id == gl_NumWorkGroups.x)
    sum[0]=Sum; // Store fragment size in sum[0]
}
