/**
* @file main.cpp
* @author Supakorn "Jamie" Rassameemasmuang <jamievlin@outlook.com>
* Program for loading image and writing the irradiated image.
*/

#include "common.h"
#include <fstream>

#ifdef _WIN32
// use vcpkg getopt package for this
#undef _UNICODE
#include <getopt.h>
#include <Windows.h>
#else
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#include "kernel.h"
#include "ReflectanceMapper.cuh"
#include "EXRFiles.h"

std::string const ARG_HELP = "./reflectance mode -f in_file -p out_file_prefix [-d directory]";
std::string const DEFAULT_DIRECTORY = ".";

struct Args
{
    char mode = 0;
    char const* file_in = nullptr;
    char const* file_out_prefix = nullptr;
    char const* directory = nullptr;

    bool validate()
    {
        if (mode == 0)
            return false;

        if ((mode == 'a' || mode == 'i' || mode == 'r') && !file_in)
        {
            return false;
        }
        return file_out_prefix != nullptr;
    }
};

Args parseArgs(int argc, char* argv[])
{
    Args arg;
    int c;
    while ((c = getopt(argc, argv, "airof:p:d:")) != -1)
    {
        switch (c)
        {
        case 'a':
            arg.mode = 'a';
            break;
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
        case 'd':
            arg.directory = optarg;
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

struct image_t
{
    float4* im;
    int width, height;

    int sz() const { return width * height; }

    image_t(float4* im, int width, int height) :
        im(im), width(std::move(width)), height(std::move(height)) {}
};

void irradiate_im(image_t& im, std::string const& prefix)
{
    std::vector<float3> out_proc(im.sz());
    std::stringstream out_name;
    out_name << prefix;
    std::cout << "Irradiating image..." << std::endl;
    irradiate_ker(im.im, out_proc.data(), im.width, im.height);
    out_name << "_diffuse.exr";

    std::string out_name_str(std::move(out_name).str());
    OEXRFile ox(out_proc, im.width, im.height);

    std::cout << "copying data back" << std::endl;
    std::cout << "writing to: " << out_name_str << std::endl;
    ox.write(out_name_str);
}

std::string generate_refl_file(std::string const& prefix, float const& step, int const& i)
{
    std::stringstream out_name;
    out_name << prefix << "_refl_" << std::fixed <<
        std::setprecision(3) << step << "_" << i << ".exr";
    return out_name.str();
}

void map_refl_im(image_t& im, std::string const& prefix, float const& step, int const& i)
{
    float roughness = step * i;
    std::vector<float3> out_proc(im.sz());

    std::string out_name_str = generate_refl_file(prefix, step, i);
    std::cout << "Mapping reflectance map..." << std::endl;
    map_reflectance_ker(im.im, out_proc.data(), im.width, im.height, roughness);
    OEXRFile ox(out_proc, im.width, im.height);
    std::cout << "copying data back" << std::endl;
    std::cout << "writing to: " << out_name_str << std::endl;
    ox.write(out_name_str);
}

std::string const INVALID_FILE_ATTRIB = "Intermediate directories do not exist";

void make_dir(std::string const& directory)
{
    // different implementation for windows vs linux
#ifdef _WIN32
    DWORD ret = CreateDirectoryA(directory.c_str(), nullptr);
    if (ret == 0 && GetLastError() != ERROR_ALREADY_EXISTS)
    {
        std::cerr << INVALID_FILE_ATTRIB << std::endl;
        exit(1);
    }
#else
    struct stat st;
    if (stat(directory.c_str(), &st) != -1)
    {
        mkdir(directory.c_str(), 0700);
    }
#endif
}

void copy_file(std::string const& in, std::string const& out)
{
    std::ifstream ifs(in, std::ios::binary);
    std::ofstream ofs(out, std::ios::binary);

    ofs << ifs.rdbuf();

}

void generate_brdf_refl(
    int res, std::string const& outdir, std::string const& outname="refl.exr")
{
    std::vector<float2> out_proc(res * res);
    std::string finalName = outdir + "/" + outname;
    std::cout << "generating Fresnel/Roughness/cos_v data" << std::endl;
    std::cout << "writing to " << finalName << std::endl;
    generate_brdf_integrate_lut_ker(res, res, out_proc.data());
    OEXRFile ox(out_proc, res, res);
    ox.write(finalName);
}

int main(int argc, char* argv[])
{
    Args args = parseArgs(argc, argv);
    std::vector<float4> im_proc;
    int width = 0;
    int height = 0;

    if (args.file_in)
    {
        std::cout << "Loaded file " << args.file_in << std::endl;
        EXRFile im(args.file_in);
        width = im.getWidth();
        height = im.getHeight();

        for (int i = 0; i < height; ++i)
        {
            for (int j = 0; j < width; ++j)
            {
                // index is i*height+j <-> (i,j)
                im_proc.emplace_back(im.getPixel4(j, i));
            }
            // std::cout << "pushed row " << i << " into array" << std::endl;
        }
        std::cout << "finished converting to float3" << std::endl;
    }

    std::stringstream outss;
    if (args.directory)
    {
      std::cout << args.directory << std::endl;
        make_dir(args.directory);
    }
    else
    {
        args.directory = ".";
    }
    outss << args.directory << "/" << args.file_out_prefix;


    std::string outprefix(outss.str());

    image_t imt(im_proc.data(), width, height);

    if (args.mode == 'o')
    {
        generate_brdf_refl(200, args.directory);
    }
    else
    {
        if (args.mode == 'r')
        {
            size_t count = 10;
            float step = 1.0f / count;
            for (int i = 1; i <= count; ++i)
            {
                map_refl_im(imt, outprefix, step, i);
            }
        }
        else if (args.mode == 'i')
        {
            irradiate_im(imt, outprefix);
        }
        else if (args.mode == 'a')
        {
            generate_brdf_refl(200, args.directory);
            irradiate_im(imt, outprefix);
            // reflectance
            size_t count = 10;
            float step = 1.0f / count;
            
            // for 0 roughness, use the original image.
            copy_file(args.file_in, generate_refl_file(outprefix, step, 0));
            for (int i = 1; i <= count; ++i)
            {
                map_refl_im(imt, outprefix, step, i);
            }
        }
    }
}
