//#version 140
in vec3 Normal;

#ifdef EXPLICIT_COLOR
in vec4 Color; 
#endif

out vec4 outColor;

void main()
{
#ifdef EXPLICIT_COLOR
    outColor=Color;
#else
    outColor=vec4(Normal,1);
#endif

}
