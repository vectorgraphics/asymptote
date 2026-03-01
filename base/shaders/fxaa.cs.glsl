/**
 * @file fxaa.cs.glsl
 * @author Supakorn "Jamie" Rassameemasmuang (jamievlin [at] outlook.com)
 * @brief FXAA post-processing shader
 */

layout(binding=0)
uniform sampler2D inputImageSampler;

layout(binding=1, rgba8)
uniform image2D inputImage;

layout(binding=2, rgba8)
writeonly uniform image2D outputImage;

layout(local_size_x=20, local_size_y=20, local_size_z=1) in;

#ifndef OUTPUT_AS_SRGB
const float gamma=2.2;

/**
 * @brief Converts perceptual color (measuring what we think the brightness is)
 * to linear color (photon count)
 * example linearToPerceptual(vec3(0.729)) is approximately vec3(0.5)
 */
vec3 perceptualToLinear(vec3 inColor)
{
  // what we think is 0.5 brightness has much less photons than half
  // of the light's original photon count
  return pow(inColor, vec3(gamma));
}
#endif

// based on https://catlikecoding.com/unity/tutorials/advanced-rendering/fxaa/

// configuration values
const float FXAA_EDGE_THRESHOLD_MIN=1/16.0;
const float FXAA_EDGE_THRESHOLD=1/8.0;

const float FXAA_EDGE_STEPS[] = {1, 1.5, 2, 2, 2, 2, 2, 2, 2, 4};
const float FXAA_EDGE_GUESS_IF_NOT_ENDOFEDGE = 8.0;

/** Assumes inColor is in perceptual space */
float getLumi(vec3 inColor)
{
  // from nvidia's paper at
  // https://developer.download.nvidia.com/assets/gamedev/files/sdk/11/FXAA_WhitePaper.pdf
  return inColor.y * (0.587/0.299) + inColor.x;
}

vec3 getColorAtOffset(ivec2 coord, ivec2 offset, ivec2 size)
{
  return imageLoad(inputImage, clamp(coord + offset, ivec2(0,0), size-ivec2(1,1))).rgb;
}

// return color in perceptual space, no need to do any gamma correction after
vec3 fxaa(ivec2 coord, ivec2 size)
{
  // fxaa code here
  vec3 perceptualPixelColor=imageLoad(inputImage,coord).rgb;

  // step 1: luminance
  float lumi= getLumi(perceptualPixelColor);

  vec3 returnColor=perceptualPixelColor;

  // step 2: local contrast check

  vec3 colorUp= getColorAtOffset(coord, ivec2(0, -1), size);
  vec3 colorDown= getColorAtOffset(coord, ivec2(0, 1), size);
  vec3 colorLeft= getColorAtOffset(coord, ivec2(-1, 0), size);
  vec3 colorRight= getColorAtOffset(coord, ivec2(1, 0), size);

  float lumiUp= getLumi(colorUp);
  float lumiDown= getLumi(colorDown);
  float lumiLeft= getLumi(colorLeft);
  float lumiRight= getLumi(colorRight);

  float rangeMin=min(lumi,min(min(lumiUp,lumiDown),min(lumiLeft,lumiRight)));
  float rangeMax=max(lumi,max(max(lumiUp,lumiDown),max(lumiLeft,lumiRight)));
  float localContrast=rangeMax-rangeMin;

  if (localContrast < max(FXAA_EDGE_THRESHOLD_MIN, rangeMax * FXAA_EDGE_THRESHOLD))
  {
    return returnColor;
  }
  // return here to see pixels that will be processed

  // step 3: calculate pixel blend factor
  vec3 colorNw= getColorAtOffset(coord, ivec2(-1, -1), size);
  vec3 colorNe= getColorAtOffset(coord, ivec2(1, -1), size);
  vec3 colorSw= getColorAtOffset(coord, ivec2(-1, 1), size);
  vec3 colorSe= getColorAtOffset(coord, ivec2(1, 1), size);
  float lumiNw= getLumi(colorNw);
  float lumiNe= getLumi(colorNe);
  float lumiSw= getLumi(colorSw);
  float lumiSe= getLumi(colorSe);

  float pixelBlendFactorBase=
    (2*(lumiUp + lumiDown + lumiLeft + lumiRight) + (lumiNe + lumiNw + lumiSe + lumiSw)) / 12.0;
  float pixelBlendFactorBeforeSmoothing = clamp(abs(pixelBlendFactorBase-lumi) / localContrast,0,1);
  float pixelBlendFactorBeforeSquare=smoothstep(0, 1, pixelBlendFactorBeforeSmoothing);
  float pixelBlendFactor=pixelBlendFactorBeforeSquare*pixelBlendFactorBeforeSquare;

  // return here to see belnd factor

  // step 4: determine if edge is horizontal or vertical
  const vec3 vecCheckEdge=vec3(0.25,-0.5,0.25);
  const vec3 vecCheckMid=vec3(0.5,-1.0,0.5);

  float edgeVerticalFactor =
    abs(dot(vecCheckEdge,vec3(lumiNw,lumiUp,lumiNe)))
    + abs(dot(vecCheckMid,vec3(lumiLeft,lumi,lumiRight)))
    + abs(dot(vecCheckEdge,vec3(lumiSw,lumiDown,lumiSe)));

  float edgeHorizontalFactor =
    abs(dot(vecCheckEdge,vec3(lumiNw,lumiLeft,lumiSw)))
    + abs(dot(vecCheckMid,vec3(lumiUp,lumi,lumiDown)))
    + abs(dot(vecCheckEdge,vec3(lumiNe,lumiRight,lumiSe)));

  bool isHorizontalEdge=edgeHorizontalFactor >= edgeVerticalFactor;
  vec2 pixelOffsetBase=isHorizontalEdge ? vec2(0, -1) : vec2(1, 0);

  // return here to see if edge is horizontal or vertical

  // step 4.5: determining if blend is in positive or negative direction

  float pLumi=isHorizontalEdge ? lumiUp : lumiRight;
  float nLumi=isHorizontalEdge ? lumiDown : lumiLeft;
  float pLumiDiff=abs(pLumi-lumi);
  float nLumiDiff=abs(nLumi-lumi);
  bool isPositiveDirection=pLumiDiff >= nLumiDiff;
  float pixelDirection= isPositiveDirection ? 1 : -1;

  vec2 pixelOffset=pixelDirection * pixelOffsetBase;

  // step 5: process edge blending
  float oppositeLumi=isPositiveDirection ? pLumi : nLumi;
  float lumiDiff=isPositiveDirection ? pLumiDiff : nLumiDiff;

  vec2 middleCoord=vec2(coord)+vec2(0.5,0.5);
  vec2 edgeCoordBase=middleCoord+(0.5 * pixelOffset);
  vec2 edgeOffsetStep=isHorizontalEdge ? vec2(1,0) : vec2(0,-1);

  float edgeLumi=(lumi + oppositeLumi) * 0.5;
  float edgeLumiDeltaThreshold = 0.25 * lumiDiff;

  // for positive direction
  float positiveSteps=0;
  bool endOfPosEdge=false;
  float posLumiDelta=0;
  for (int i=0;(!endOfPosEdge) && (i<FXAA_EDGE_STEPS.length()); ++i)
  {
    positiveSteps += FXAA_EDGE_STEPS[i];
    posLumiDelta=getLumi(
      texture(inputImageSampler, edgeCoordBase + (positiveSteps * edgeOffsetStep)).rgb
    )-edgeLumi;
    endOfPosEdge=abs(posLumiDelta) >= edgeLumiDeltaThreshold;
  }

  positiveSteps += endOfPosEdge ? 0 : FXAA_EDGE_GUESS_IF_NOT_ENDOFEDGE;

  // for negative direction
  float negativeSteps=0;
  bool endOfNegEdge=false;
  float negLumiDelta=0;
  for (int i=0;(!endOfNegEdge) && (i<FXAA_EDGE_STEPS.length()); ++i)
  {
    negativeSteps += FXAA_EDGE_STEPS[i];
    negLumiDelta=getLumi(
      texture(inputImageSampler, edgeCoordBase - (negativeSteps * edgeOffsetStep)).rgb
    )-edgeLumi;
    endOfNegEdge=abs(negLumiDelta) >= edgeLumiDeltaThreshold;
  }

  negativeSteps += endOfNegEdge ? 0 : FXAA_EDGE_GUESS_IF_NOT_ENDOFEDGE;

  bool shortestDistanceIsPositive = positiveSteps <= negativeSteps;
  float shortestSteps = min(positiveSteps,negativeSteps);
  float shortestLumiDelta= shortestDistanceIsPositive ? posLumiDelta : negLumiDelta;

  // if shortest luminance delta is same sign as edge lumaniance delta,
  // we are on the edge that "goes away", and hence should skip
  // since we already handling the pixel on the other side
  // of the edge
  if (shortestLumiDelta * (lumi - edgeLumi) >= 0)
  {
    return returnColor;
  }

  float edgeBlendFactor = 0.5 - (float(shortestSteps) / (positiveSteps + negativeSteps));

  return texture(
    inputImageSampler,
    middleCoord + (max(pixelBlendFactor,edgeBlendFactor) * pixelOffset)
  ).rgb;
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
  vec3 outputColor=fxaa(coord,size);

#ifdef OUTPUT_AS_SRGB
  vec3 returnColor=outputColor;
#else
  vec3 returnColor=perceptualToLinear(outputColor);
#endif

  // final
  imageStore(
    outputImage,
    coord,
    vec4(returnColor, 1)
  );

}
