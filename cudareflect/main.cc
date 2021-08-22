/**
* @file main.cpp
* @author Supakorn "Jamie" Rassameemasmuang <jamievlin@outlook.com>
* Program for loading image and writing the irradiated image.
*/

#include "common.h"

#ifdef _WIN32
// use vcpkg getopt package for this
#undef _UNICODE
#include <getopt.h>
#else
#include <unistd.h>
#endif

#include "kernel.h"
#include "ReflectanceMapper.cuh"
#include "EXRFiles.h"

std::string const ARG_HELP = "./reflectance [mode] in_file out_file";

struct Args
{
    char mode = 0;
    char const* file_in = nullptr;
    char const* file_out_prefix = nullptr;

    bool validate()
    {
        return mode != 0 && file_in && file_out_prefix;
    }
};

Args parseArgs(int argc, char* argv[])
{
    Args arg;
    int c;
    while ((c = getopt(argc, argv, "irof:p:")) != -1)
    {
        switch (c)
        {
        case 'i':
            arg.mode = 'i';
            break;
        case 'r':
            arg.mode = 'r';
            break;
        case 'o':
            arg.mode = 'o';
            break;
        case 'f':
            arg.file_in = optarg;
            break;
        case 'p':
            arg.file_out_prefix = optarg;
            break;
        case '?':
            std::cerr << ARG_HELP << std::endl;
            exit(0);
            break;
        default:
            std::cerr << ARG_HELP << std::endl;
            exit(1);
        }
    }

    if (!arg.validate())
    {
        std::cerr << ARG_HELP << std::endl;
        exit(1);
    }
    return arg;
}

int main(int argc, char* argv[])
{
    Args args = parseArgs(argc, argv);

    EXRFile im(args.file_in);

    std::cout << "Loaded file " << argv[1] << std::endl;
    std::vector<float4> im_proc;
    size_t width = im.getWidth();
    size_t height = im.getHeight();

    for (size_t i = 0; i < height; ++i)
    {
        for (size_t j = 0; j < width; ++j)
        {
            // index is i*height+j <-> (i,j)
            im_proc.emplace_back(im.getPixel4(j, i));
        }
        // std::cout << "pushed row " << i << " into array" << std::endl;
    }
    size_t sz = static_cast<size_t>(width*height);

    std::cout << "finished converting to float3" << std::endl;

    if (argc >= 4 && std::string(argv[3]) == "intg")
    {
        int res = 200;
        std::vector<float2> out_proc(res * res);

        generate_brdf_integrate_lut_ker(res, res, out_proc.data());
        OEXRFile ox(out_proc, res, res);
        ox.write(argv[2]);
    }
    else
    {
        std::vector<float3> out_proc(sz);
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

}
