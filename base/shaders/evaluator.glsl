layout(triangles, equal_spacing, ccw) in;

patch in int materialIndex;
out int MaterialIndex;

in vec3 worldControls[];

vec4 P[10];
out vec3 p[10];

in vec3 normal[];
out vec3 Normal;

out vec3 Parameter;

vec3 mixVec(vec3 v1, vec3 v2, vec3 v3)
{
return (gl_TessCoord.x * v1)
      + (gl_TessCoord.y * v2)
      + (gl_TessCoord.z * v3);
}

vec4 point(float u, float v, float w) {
  return w*w*(w*P[0]+3*(u*P[1]+v*P[2]))+u*u*(3*(w*P[3]+v*P[7])+u*P[6])+
            6*u*v*w*P[4]+v*v*(3*(w*P[5]+u*P[8])+v*P[9]);
}


void main()
{

 for(int i=0; i < 10; ++i) {
   P[i]=gl_in[i].gl_Position;
 }

 Parameter=vec3(gl_TessCoord[1],gl_TessCoord[2],gl_TessCoord[0]);
// Parameter=vec3(gl_TessCoord[0],gl_TessCoord[1],gl_TessCoord[2]);
 gl_Position=point(Parameter[0],Parameter[1],Parameter[2]);
    
//    gl_Position = gl_in[0].gl_Position*gl_TessCoord[0]+gl_in[6].gl_Position*gl_TessCoord[1]+gl_in[9].gl_Position*gl_TessCoord[2];

//   Normal=ComputeNormal(Parameter);
   for(int i=0; i < 10; ++i)
     p[i]=worldControls[i];
 
 Normal=gl_TessCoord[0]*normal[0]+gl_TessCoord[1]*normal[6]+gl_TessCoord[2]*normal[9];

    MaterialIndex=materialIndex;
}
