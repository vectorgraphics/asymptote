in vec2 Texcoord;
out vec4 outColor;

// now in gamma-mode for FXAA. 
uniform sampler2D texFrameBuffer;
uniform ivec2 screenResolution;
uniform ivec2 renderResolution;

uniform int forceNoPostProcessAA;

vec2 getOffset(float x, float y) {
    float offsetx = x * (1.0 / screenResolution.x);
    float offsety = y * (1.0 / screenResolution.y);
    return vec2(offsetx,offsety);
}

int positivesign(bool val) {
    return val ? 1 : -1;
}

#ifdef ENABLE_FXAA

// based on Timothy Lottes's FXAA paper
// and https://catlikecoding.com/unity/tutorials/advanced-rendering/fxaa/

#define GAMMA 2.2
#define INVERSE_GAMMA 1/2.2

// offsets
#define OFFSET_EAST getOffset(1,0)
#define OFFSET_WEST getOffset(-1,0)
#define OFFSET_NORTH getOffset(0,1)
#define OFFSET_SOUTH getOffset(0,-1)

// fxaa parameters
// "high-quality" by Lottes' paper

// edge thresholds
#define FXAA_MIN_EDGE_THRESHOLD 1.0/16
#define FXAA_EDGE_THRESHOLD 1.0/8

#define FXAA_BLEND_FACTOR 1
#define FXAA_SUBPIXEL_BLEND_FACTOR 1
#define FXAA_EDGE_BLEND_FACTOR 1
#define FXAA_GRADIENT_THRESHOLD_EDGE_SCALE 0.25

#define FXAA_EDGE_SEARCH_DISTANCE 10

#define FXAA_FINAL_EDGE_GUESS 8

// debug flags
// #define FXAA_CONTRAST_CHECK_DEBUG
// #define FXAA_FINAL_EDGE_GUESS_DEBUG
// #define FXAA_EDGE_BLENDING_DEBUG
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

    data.E = getluma(texture(texFrameBuffer, coord + OFFSET_EAST).rgb);
    data.S = getluma(texture(texFrameBuffer, coord + OFFSET_SOUTH).rgb);
    data.N = getluma(texture(texFrameBuffer, coord + OFFSET_NORTH).rgb);
    data.W = getluma(texture(texFrameBuffer, coord + OFFSET_WEST).rgb);

    data.NW = getluma(texture(texFrameBuffer, coord + OFFSET_NORTH+OFFSET_WEST).rgb);
    data.SW = getluma(texture(texFrameBuffer, coord + OFFSET_SOUTH+OFFSET_WEST).rgb);
    data.NE = getluma(texture(texFrameBuffer, coord + OFFSET_NORTH+OFFSET_EAST).rgb);
    data.SE = getluma(texture(texFrameBuffer, coord + OFFSET_SOUTH+OFFSET_EAST).rgb);

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

    for (int i = 0; i < 4; ++i) {
        if (neighborlumas[i] < minluma) {
            minluma = neighborlumas[i];
        }
        if (neighborlumas[i] > maxluma) {
            maxluma = neighborlumas[i];
        }
    }

    float range = maxluma - minluma;

    // ignore areas with low contrast. 
    bool contrastCheckFail = range < max(FXAA_MIN_EDGE_THRESHOLD, maxluma * FXAA_EDGE_THRESHOLD);
    if (contrastCheckFail) {
        return rawColor;
    }
#ifdef FXAA_CONTRAST_CHECK_DEBUG
    else {
        return vec3(1,0,0);
    }
#endif

    // at this point, we are only processing pixels
    // with sufficient contrast

    // subpixel aliasing test & blend factor calculation
    float avgluma = (lumasdata.N + lumasdata.S + lumasdata.E + lumasdata.W);
    avgluma += (lumasdata.NW + lumasdata.SW + lumasdata.NE + lumasdata.SE);
    avgluma *= (1.0/8);
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
    } else {
        localEdge.oppositeGrad = pGrad;
        localEdge.oppositeLuma = pLuma;
    }

    
    // calculating the pixel step. 
    if (localEdge.isHorizontal) {
        localEdge.pixelstep = positivesign(positiveGrad) * getOffset(0, 1);
    } else {
        localEdge.pixelstep = positivesign(positiveGrad) * getOffset(1, 0);
    }

    float pixelblendfactor = FXAA_SUBPIXEL_BLEND_FACTOR * blendfactor;

    // edge walking
    vec2 currentcoords = coord + (0.5 * localEdge.pixelstep);
    vec2 edgestep = (localEdge.isHorizontal) ?  getOffset(1, 0) : getOffset(0, 1);

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

    if (!pEnd) {
        currentcoords += edgestep * FXAA_FINAL_EDGE_GUESS;
#ifdef FXAA_FINAL_EDGE_GUESS_DEBUG
        return vec3(0,1,0);
#endif

    }
    if (!nEnd) {
        currentcoordsneg -= edgestep * FXAA_FINAL_EDGE_GUESS;
#ifdef FXAA_FINAL_EDGE_GUESS_DEBUG
        return vec3(0,1,0);
#endif
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
        edgeBlendFactor = FXAA_EDGE_BLEND_FACTOR * (0.5 - (minDistance / (pDistancediff + nDistancediff)));
#ifdef FXAA_EDGE_BLENDING_DEBUG
        return vec3(0,1,0);
#endif
    } else {
        edgeBlendFactor = 0;
#ifdef FXAA_EDGE_BLENDING_DEBUG
        return vec3(1,0,0);
#endif
    }

    float finalBlendFactor = max(edgeBlendFactor, blendfactor);

    return textureLod(texFrameBuffer, coord + (FXAA_BLEND_FACTOR * finalBlendFactor * localEdge.pixelstep), 0).rgb;
}
#endif

void main() {

#ifdef ENABLE_FXAA
if (forceNoPostProcessAA == 0) {
    outColor = vec4(gamma2lin(fxaa(Texcoord)) , 1);
} else {
#endif
    outColor = texture(texFrameBuffer, Texcoord);

#ifdef ENABLE_FXAA
    }  
#endif


}