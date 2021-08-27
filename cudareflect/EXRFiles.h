#pragma once

#include "common.h"
#include <cuda_runtime.h>

class EXRFile
{
public:
    EXRFile(std::string const& input);

    float const* getData() const
    {
        return flt;
    }

    ~EXRFile()
    {
        free(flt);
    }

    int getWidth() const
    {
        return width;
    }

    int getHeight() const
    {
        return height;
    }

    float4 getPixel4(size_t const& x, size_t const& y)
    {
        size_t base = 4 * (y * width + x);
        return make_float4(flt[base], flt[base + 1], flt[base + 2], flt[base + 3]);
    }

    float3 getPixel3(size_t const& x, size_t const& y)
    {
        size_t base = 4 * (y * width + x);
        return make_float3(flt[base], flt[base + 1], flt[base + 2]);
    }

private:
    int width, height;
    float* flt;
}; 

class OEXRFile
{
public:
    OEXRFile(std::vector<float3> const& dat, int width, int height, int compressionType=TINYEXR_COMPRESSIONTYPE_PIZ);
    OEXRFile(std::vector<float2> const& dat, int width, int height, int compressionType=TINYEXR_COMPRESSIONTYPE_PIZ);
    void write(std::string const& filename);
    ~OEXRFile() = default;

protected:
    void initChannelInfo();
    void initHeader();
private:
    int width, height;
    int compressionType;

    std::vector<EXRChannelInfo> infos;
    EXRHeader hd;

    std::vector<int> pixelType;
    std::vector<int> reqPixelType;

    std::vector<float> r, g, b, a;

};