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
  uint id=gl_GlobalInvocationID.x;

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
  if(id == 0) {
    for(uint i=1u; i < stop; ++i) {
      Sum += offset[i];
      offset[i]=Sum;
    }
  } else {
    for(uint i=row+1u; i < stop; ++i)
      Sum += offset[i];
  }

  sum[id]=Sum;
}
