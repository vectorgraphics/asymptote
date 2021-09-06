uniform sampler2D ColorTex;

struct OpaqueFragment
{
  vec4 color;
  float depth;
};
layout(binding=3) coherent buffer opaque {
  OpaqueFragment zbuffer[];
};

uniform uint width;
uniform uint height;

out vec4 outColor;

void main(void)
{
  uint headIndex = uint(gl_FragCoord.y) * width + uint(gl_FragCoord.x);
  outColor = texture(ColorTex, gl_FragCoord.xy/vec2(width,height));
  vec4 background = vec4(1);
  if (zbuffer[headIndex].depth != 0) background = zbuffer[headIndex].color;
  outColor.rgb = outColor.rgb + outColor.a * background.rgb;
}
