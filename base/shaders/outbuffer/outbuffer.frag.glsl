in vec2 Texcoord;
out vec4 outColor;

// now in gamma-mode for FXAA. 
uniform sampler2D texFrameBuffer;
uniform ivec2 screenResolution;
uniform ivec2 renderResolution;

#ifdef ENABLE_FXAA

// based on Timothy Lottes's FXAA paper
// and https://catlikecoding.com/unity/tutorials/advanced-rendering/fxaa/


#define GAMMA 2.2
#define INVERSE_GAMMA 1/2.2

// offsets
#define OFFSET_EAST ivec2(1,0)
#define OFFSET_WEST ivec2(-1,0)
#define OFFSET_NORTH ivec2(0,1)
#define OFFSET_SOUTH ivec2(0,-1)

// fxaa parameters
// "high-quality" by Lottes' paper

// edge thresholds
#define FXAA_MIN_EDGE_THRESHOLD 1/16
#define FXAA_EDGE_THRESHOLD 1/8

#define FXAA_BLEND_FACTOR 0.4
#define FXAA_SUBPIXEL_BLEND_FACTOR 1
#define FXAA_EDGE_BLEND_FACTOR 1
#define FXAA_GRADIENT_THRESHOLD_EDGE_SCALE 0.25

#define FXAA_EDGE_SEARCH_DISTANCE 14

// lumas
struct luminancedata {
    float N, E, S, W;
    float NW, NE, SW, SE;
    float M;
};

struct edgeData {
    bool isHorizontal;
    vec2 pixelstep;
    float oppositeGrad, oppositeLuma;
};

vec3 gamma2lin(vec3 invec) {
    return pow(invec, vec3(GAMMA));
}

float getluma(vec3 col) {
    // constant from Lottes' NVIDIA FXAA paper
    return col.g * (0.587/0.299) + col.r;
}

float saturate(float inflt) {
    return clamp(inflt, 0, 1);
}

float smoothen(float val) {
    // https://catlikecoding.com/unity/tutorials/advanced-rendering/fxaa/
    // based on smoothstep between (0,1), given by
    // x |-> (x^2*(3-2x))^2
    return pow(val * val * (3 - (2 * val)), 2);
}

luminancedata getneighborlumas(vec2 coord) {
    luminancedata data;
    data.M = getluma(texture(texFrameBuffer, coord).rgb);

    data.E = getluma(textureOffset(texFrameBuffer, coord, OFFSET_EAST).rgb);
    data.S = getluma(textureOffset(texFrameBuffer, coord, OFFSET_SOUTH).rgb);
    data.N = getluma(textureOffset(texFrameBuffer, coord, OFFSET_NORTH).rgb);
    data.W = getluma(textureOffset(texFrameBuffer, coord, OFFSET_WEST).rgb);

    data.NW = getluma(textureOffset(texFrameBuffer, coord, OFFSET_NORTH+OFFSET_WEST).rgb);
    data.SW = getluma(textureOffset(texFrameBuffer, coord, OFFSET_SOUTH+OFFSET_WEST).rgb);
    data.NE = getluma(textureOffset(texFrameBuffer, coord, OFFSET_NORTH+OFFSET_EAST).rgb);
    data.SE = getluma(textureOffset(texFrameBuffer, coord, OFFSET_SOUTH+OFFSET_EAST).rgb);

    return data;
}

float horizontaledgefactor(luminancedata luma) {
    float result = 2 * abs(luma.N + luma.S - (2 * luma.M));
    result += abs(luma.NW + luma.SW - (2 * luma.W));
    result += abs(luma.NE + luma.SE - (2 * luma.E));
    return result;
}

float verticaledgefactor(luminancedata luma) {
    float result = 2 * abs(luma.E + luma.W - (2 * luma.M));
    result += abs(luma.NE + luma.NW - (2 * luma.N));
    result += abs(luma.SE + luma.SW - (2 * luma.S));
    return result;
}

bool ishorizontaledge(luminancedata luma) {
    return horizontaledgefactor(luma) > verticaledgefactor(luma);
}

vec3 fxaa(vec2 coord) {
    // gamma correction
    vec3 rawColor = texture(texFrameBuffer, coord).rgb;
    luminancedata lumasdata = getneighborlumas(coord);
    // neighbor range checking
    float neighborlumas[4] = float[4](
        lumasdata.N, lumasdata.S, lumasdata.E, lumasdata.W
    );

    float minluma = lumasdata.M;
    float maxluma = lumasdata.M;

    for (int i = 0; i < 5; ++i) {
        if (neighborlumas[i] < minluma) {
            minluma = neighborlumas[i];
        }
        if (neighborlumas[i] > maxluma) {
            maxluma = neighborlumas[i];
        }
    }

    float range = maxluma - minluma;

    // ignore areas with low contrast. 
    if (range < max(FXAA_MIN_EDGE_THRESHOLD, maxluma * FXAA_EDGE_THRESHOLD)) {
        return rawColor;
    }

    // at this point, we are only processing pixels
    // with sufficient contrast

    // subpixel aliasing test & blend factor calculation
    float avgluma = 2 * (lumasdata.N + lumasdata.S + lumasdata.E + lumasdata.W);
    avgluma += (lumasdata.NW + lumasdata.SW + lumasdata.NE + lumasdata.SE);
    avgluma *= (1/12);
    float pixelcontrast = abs(avgluma - lumasdata.M);
    float blendfactor = smoothen(saturate(pixelcontrast / range));

    // edge detection
    edgeData localEdge;
    // calculating edge direction 
    localEdge.isHorizontal = ishorizontaledge(lumasdata);

    // positive or negative blending
    float pLuma = localEdge.isHorizontal ? lumasdata.N : lumasdata.E;
    float nLuma = localEdge.isHorizontal ? lumasdata.S : lumasdata.W;
    float pGrad = abs(lumasdata.M - pLuma);
    float nGrad = abs(lumasdata.M - nLuma);

    bool positiveGrad = pGrad >= nGrad;
    
    // get what's on the opposite side
    if (positiveGrad) {
        localEdge.oppositeGrad = nGrad;
        localEdge.oppositeLuma = nLuma;
    }

    float pixelYsize = 1.0/(renderResolution.y);
    float pixelXsize = 1.0/(renderResolution.x);
    
    // calculating the pixel step. 
    if (localEdge.isHorizontal) {
        localEdge.pixelstep = positiveGrad ? vec2(0, pixelYsize) : vec2(0, -pixelYsize);
    } else {
        localEdge.pixelstep = positiveGrad ? vec2(pixelXsize, 0) : vec2(-pixelXsize, 0);
    }

    float pixelblendfactor = FXAA_SUBPIXEL_BLEND_FACTOR * blendfactor;

    // edge walking
    vec2 currentcoords = coord + (0.5 * localEdge.pixelstep);
    vec2 edgestep = (localEdge.isHorizontal) ? vec2(pixelXsize, 0) : vec2(0, pixelYsize);

    float edgelumen = (lumasdata.M + localEdge.oppositeLuma) * 0.5;
    float gradientThreshold = FXAA_GRADIENT_THRESHOLD_EDGE_SCALE * localEdge.oppositeGrad;

    // get the edge walk distance
    bool pEnd = false;
    float pDelta;
    for (int j = 0; (j < FXAA_EDGE_SEARCH_DISTANCE) && (!pEnd); ++j) {
        currentcoords += edgestep;
        float positiveLuma = getluma(textureLod(texFrameBuffer, currentcoords, 0).rgb);
        pDelta = positiveLuma - edgelumen;
        pEnd = (abs(pDelta) >= gradientThreshold);
    }

    float edgeBlendFactor = 0;

    vec2 currentcoordsneg = coord + (0.5 * localEdge.pixelstep);
    bool nEnd = false;
    float nDelta;
    for (int j = 0; (j < FXAA_EDGE_SEARCH_DISTANCE) && (!nEnd); ++j) {
        currentcoordsneg -= edgestep;
        float negativeLuma = getluma(textureLod(texFrameBuffer, currentcoordsneg, 0).rgb);
        nDelta = negativeLuma - edgelumen;
        nEnd = (abs(nDelta) >= gradientThreshold);
    }

    float pDistancediff = (localEdge.isHorizontal) ? 
        (currentcoords.x - coord.x) : (currentcoords.y - coord.y);
    float nDistancediff = (localEdge.isHorizontal) ? 
        (coord.x - currentcoordsneg.x) : (coord.y - currentcoordsneg.y);

    // getting now the final edge blend factor
    float minDistance;
    bool deltaSign;
    if (pDistancediff <= nDistancediff) {
        minDistance = pDistancediff;
        deltaSign = (pDelta >= 0);
    } else {
        minDistance = nDistancediff;
        deltaSign = (nDelta >= 0);
    }

    if (deltaSign == (lumasdata.M < edgelumen)) {
        edgeBlendFactor = FXAA_EDGE_BLEND_FACTOR * (0.5 - minDistance / (pDistancediff + nDistancediff));
    }

    float finalBlendFactor = max(edgeBlendFactor, blendfactor);
    // 
    return textureLod(texFrameBuffer, coord + (FXAA_BLEND_FACTOR * finalBlendFactor * localEdge.pixelstep), 0).rgb;
}
#endif

void main() {

#ifdef ENABLE_FXAA
    outColor = vec4(gamma2lin(fxaa(Texcoord)) , 1);
#else
    outColor = texture(texFrameBuffer, Texcoord);
#endif
}