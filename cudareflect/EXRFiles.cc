#include "EXRFiles.h"

EXRFile::EXRFile(std::string const& input)
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

OEXRFile::OEXRFile(std::vector<float3> const& dat, int width, int height, int compressionType) :
    width(std::move(width)), height(std::move(height)),
    compressionType(compressionType), infos(3)
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

OEXRFile::OEXRFile(std::vector<float2> const& dat, int width, int height, int compressionType) :
    width(std::move(width)), height(std::move(height)),
    compressionType(compressionType), infos(3)
{
    for (float2 const& col : dat)
    {
        r.push_back(col.x);
        g.push_back(col.y);
        b.push_back(0);
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

void OEXRFile::initChannelInfo()
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

void OEXRFile::initHeader()
{
    InitEXRHeader(&hd);
    hd.num_channels = 3;
    hd.channels = infos.data();
    hd.pixel_types = pixelType.data();
    hd.requested_pixel_types = reqPixelType.data();
    hd.compression_type = compressionType;
}

void OEXRFile::write(std::string const& filename)
{
    EXRImage im;
    InitEXRImage(&im);
    im.num_channels = 3;
    im.width = width;
    im.height = height;

    std::array<float*, 3> arr{ b.data(), g.data(), r.data() };

    im.images = reinterpret_cast<unsigned char**>(arr.data());

    char const* err = nullptr;
    if (SaveEXRImageToFile(&im, &hd, filename.c_str(), &err) != TINYEXR_SUCCESS)
    {
        std::cerr << "TinyEXR ERROR: " << err << std::endl;
        FreeEXRErrorMessage(err);
        exit(1);
    }
}
