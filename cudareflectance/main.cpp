/**
* @file main.cpp
* @author Supakorn "Jamie" Rassameemasmuang <jamievlin@outlook.com>
* Program for loading image and writing the irradiated image.
*/

#include <iostream>
#include <iomanip>
#include <vector>
#include <memory>
#include <array>
#include <string>
#include <cmath>

#define TINYEXR_USE_THREAD 1
#include <tinyexr/tinyexr.h>

#include "kernel.h"
#include "ReflectanceMapper.cuh"

class EXRFile
{
public:
    EXRFile(std::string const& input)
    {
        char const* err = nullptr;
        if (LoadEXR(&flt, &width, &height, input.c_str(), &err) != TINYEXR_SUCCESS)
        {
            if (err)
            {
                std::cerr << "TinyEXR ERROR: " << err << std::endl;
                FreeEXRErrorMessage(err);
                exit(1);
            }
        }
    }

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

    float4 getPixel(int x, int y)
    {
        int base = 4 * (y * width + x);
        return make_float4(flt[base], flt[base + 1], flt[base + 2], flt[base + 3]);
    }

private:
    int width, height;
    float* flt;
};

class OEXRFile
{
public:
    OEXRFile(std::vector<float3> const& dat, int width, int height) :
        width(std::move(width)), height(std::move(height)),
        infos(3)
    {
        for (float3 const& col : dat)
        {
            r.push_back(col.x);
            g.push_back(col.y);
            b.push_back(col.z);
            //a.push_back(col.w);
        }

        for (int i = 0; i < 3; ++i)
        {
            pixelType.push_back(TINYEXR_PIXELTYPE_FLOAT);
            reqPixelType.push_back(TINYEXR_PIXELTYPE_FLOAT);
        }

        initChannelInfo();
        initHeader();
    }

    void initChannelInfo()
    {
        infos.resize(4);
        // strcpy(infos[0].name, "A");
        strcpy(infos[0].name, "B");
        strcpy(infos[1].name, "G");
        strcpy(infos[2].name, "R");

        for (auto& info : infos)
        {
            info.name[1] = '\0';
        }
    }

    void initHeader()
    {
        InitEXRHeader(&hd);
        hd.num_channels = 3;
        hd.channels = infos.data();
        hd.pixel_types = pixelType.data();
        hd.requested_pixel_types = reqPixelType.data();
    }

    void write(std::string const& filename)
    {
        EXRImage im;
        InitEXRImage(&im);
        im.num_channels = 3;
        im.width = width;
        im.height = height;

        std::array<float*, 3> arr { b.data(), g.data(), r.data() };

        im.images = reinterpret_cast<unsigned char**>(arr.data());

        char const* err = nullptr;
        if (SaveEXRImageToFile(&im, &hd, filename.c_str(), &err) != TINYEXR_SUCCESS)
        {
            std::cerr << "TinyEXR ERROR: " << err << std::endl;
            FreeEXRErrorMessage(err);
            exit(1);
        }
    }

    ~OEXRFile()
    {
    }

private:
    int width, height;

    std::vector<EXRChannelInfo> infos;
    EXRHeader hd;

    std::vector<int> pixelType;
    std::vector<int> reqPixelType;

    std::vector<float> r, g, b, a;

};

int main(int argc, char* argv[])
{
    EXRFile im(argv[1]);

    std::cout << "Loaded file " << argv[1] << std::endl;
    std::vector<float4> im_proc;
    size_t width = im.getWidth();
    size_t height = im.getHeight();

    for (size_t i = 0; i < height; ++i)
    {
        for (size_t j = 0; j < width; ++j)
        {
            // index is i*height+j <-> (i,j)
            im_proc.emplace_back(im.getPixel(j,i));
        }
        // std::cout << "pushed row " << i << " into array" << std::endl;
    }
    size_t sz = static_cast<size_t>(width*height);
    std::vector<float3> out_proc(sz);

    std::cout << "finished converting to float3" << std::endl;


    if (argc >= 4 && std::string(argv[3]) == "refl")
    {
        std::cout << "Mapping reflectance map..." << std::endl;
        map_reflectance_ker(im_proc.data(), out_proc.data(), width, height, 0.15);
    }
    else
    {
        std::cout << "Irradiating image..." << std::endl;
        irradiate_ker(im_proc.data(), out_proc.data(), width, height);
    }


    std::cout << "copying data back" << std::endl;
    std::cout << "writing to: " << argv[2] << std::endl;

    OEXRFile ox(out_proc, width, height);
    ox.write(argv[2]);
}