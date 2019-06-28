in vec2 Texcoord;
out vec4 outColor;
uniform sampler2D texFrameBuffer;

void main() {
    //outColor=vec4(1,0,0,1);
    outColor = texture(texFrameBuffer, Texcoord);
}