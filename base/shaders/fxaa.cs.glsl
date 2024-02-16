/**
 * @file fxaa.cs.glsl
 * @author Supakorn "Jamie" Rassameemasmuang (jamievlin [at] outlook.com)
 * @brief FXAA post-processing shader
 */

layout(binding=0, rgba8)
uniform image2D inputImage;

layout(binding=1, rgba8)
writeonly uniform image2D outputImage;

layout(local_size_x=20, local_size_y=20, local_size_z=1) in;

void main()
{
  ivec2 coord = ivec2(gl_GlobalInvocationID.xy);
  ivec2 size=imageSize(outputImage);
  if ((coord.x >= size.x) || (coord.y >= size.y))
  {
    return;
  }
  vec3 pixelColor=imageLoad(inputImage,coord).rgb;
  // coordinate is valid, can process

}