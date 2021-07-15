uniform sampler2D ColorTex;

uniform uint width;
uniform uint height;

out vec4 outColor;

void main(void)
{
  outColor = texture(ColorTex, gl_FragCoord.xy/vec2(width,height));
}
