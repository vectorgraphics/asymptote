//#version 140
in vec3 Normal;

#if EXPLICIT_COLOR==1
in vec4 Color; 
#endif

out vec4 outColor;

void main()
{
#if EXPLICIT_COLOR==1
    outColor=Color;
#else
    outColor=vec4(Normal,1);
#endif

}
