struct OpaqueFragment
{
    vec4 color;
    float depth;
};
layout(std430, binding=3) coherent buffer opaque {
    OpaqueFragment zbuffer[];
};

uniform sampler2D ColorTex;

uniform uint width;

out vec4 outColor;

void main(void)
{
  uint headIndex = uint(gl_FragCoord.y) * width + uint(gl_FragCoord.x);
  vec4 frontColor = texture(ColorTex, gl_FragCoord.xy);
  vec4 BackgroundColor = zbuffer[headIndex].color;
  outColor = frontColor + BackgroundColor * frontColor.a;
  outColor = vec4(1,0,0,1);
}
