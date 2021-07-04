uniform sampler2D BlendTex;

out vec4 outColor;

void main(void)
{
  outColor = texture(TempTex, gl_FragCoord.xy);
}
