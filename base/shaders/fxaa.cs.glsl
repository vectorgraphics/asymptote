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

const float gamma=2.2;
const float invGamma=1.0/gamma;

vec3 linearToPerceptual(vec3 inColor)
{
  // an actual 0.5 brightness (half amount of photons) would
  // look brighter than what our eyes think is "half" light
  return pow(inColor, vec3(invGamma));
}

void main()
{
  ivec2 coord = ivec2(gl_GlobalInvocationID.xy);
  ivec2 size=imageSize(outputImage);
  if ((coord.x >= size.x) || (coord.y >= size.y))
  {
    return;
  }

  // coordinate is valid

#ifdef ENABLE_FXAA
  // fxaa code here
#else
  // already in perceptual space, no need to do any further
  // gamma adjustment
  vec3 pixelColor=imageLoad(inputImage,coord).rgb;

  imageStore(
    outputImage,
    coord,
    vec4(pixelColor, 1)
  );
#endif

}
