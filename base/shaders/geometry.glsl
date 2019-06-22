layout(triangles) in;
layout(triangle_strip, max_vertices = 3) out;

in vec3 vNormal[];

#ifdef EXPLICIT_COLOR
in vec4 vColor[];
#endif

flat in int vMaterialIndex[];

// out values to fragment

out vec3 fNormal;
flat out int materialIndex;
#ifdef EXPLICIT_COLOR
out vec4 Color;
#endif

vec3 barycentric[3] = vec3[](vec3(1,0,0), vec3(0,1,0), vec3(0,0,1));

out vec3 Barycentric;

void main()
{
    for (int i = 0; i < 3; ++i) {
        gl_Position = gl_in[i].gl_Position;
        fNormal = vNormal[i];
        materialIndex = vMaterialIndex[i];
        Barycentric = barycentric[i];

#ifdef EXPLICIT_COLOR
        Color = vColor[i];
#endif

        EmitVertex();
    }
    EndPrimitive();
}