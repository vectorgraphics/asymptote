layout(points) in;
layout(points, max_vertices = 1) out;

flat in int vMaterialIndex[];
flat out int materialIndex;

void main() {
    gl_Position = gl_in[0].gl_Position;
    materialIndex = vMaterialIndex[0];
    EmitVertex();

    EndPrimitive();
}