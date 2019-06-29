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

#define FXAA_SUBPIXEL_BLEND_FACTOR 0.45

// lu
struct luminancedata {
    float N, E, S, W;
    float NW, NE, SW, SE;
    float M;
};

vec3 gamma2lin(vec3 invec) {
    return pow(invec, vec3(INVERSE_GAMMA));
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

    // calculating edge direction 
    bool ishorizontal = ishorizontaledge(lumasdata);

    // positive or negative blending
    float pLuma = ishorizontal ? lumasdata.N : lumasdata.E;
    float nLuma = ishorizontal ? lumasdata.S : lumasdata.W;

    float pGrad = abs(lumasdata.M - pLuma);
    float nGrad = abs(lumasdata.M - nLuma);

    vec2 pixelstep = vec2(0);

    if (ishorizontal) {
        float basestep = FXAA_SUBPIXEL_BLEND_FACTOR * blendfactor/(screenResolution.y);
        pixelstep = (pGrad >= nGrad) ? vec2(0, basestep) : vec2(0, -basestep);
    } else {
        float basestep = FXAA_SUBPIXEL_BLEND_FACTOR * blendfactor/(screenResolution.x);
        pixelstep = (pGrad >= nGrad) ? vec2(basestep, 0) : vec2(-basestep, 0);
    }

    return textureLod(texFrameBuffer, coord + pixelstep, 0).rgb;
}
#endif

void main() {

#ifdef ENABLE_FXAA
    outColor = vec4(gamma2lin(fxaa(Texcoord)) , 1);
#else
    outColor = texture(texFrameBuffer, Texcoord);
#endif
}