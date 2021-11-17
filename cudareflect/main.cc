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

std::string const ARG_HELP = "reflect -i|-b|-r|-a -f input [-d directory]";
std::string const DEFAULT_DIRECTORY = ".";
size_t const MIN_WIDTH = 2;
size_t const MIN_HEIGHT = 2;


struct Args
{
  char mode = 0;
  bool webgl = false;
  char const* file_in = nullptr;
  char const* directory = nullptr;

  size_t count;

  bool validate()
  {
    if (mode == 0)
      return false;

    if ((mode == 'a' || mode == 'b' || mode == 'i' || mode == 'r') && !file_in)
      {
        return false;
      }

    return true;
  }
};

Args parseArgs(int argc, char* argv[])
{
  Args arg;
  int c;
  while ((c = getopt(argc, argv, "abd:f:hir")) != -1)
    {
      switch (c)
        {
          case 'a':
            arg.mode = 'a'; // all
            break;
          case 'i':
            arg.mode = 'i'; // irradiance image diffuse.exr
            break;
          case 'b':
            arg.mode = 'b'; // brdf image refl.exr
            break;
          case 'r':
            arg.mode = 'r'; // reflectance images reflN.exr
            break;
/*
  case 'c':
  {
  std::stringstream ss;
  ss << optarg;
  ss >> arg.count;
  }
  break;
*/
          case 'f':
            arg.file_in = optarg;
            break;
          case 'd':
            arg.directory = optarg;
            break;
          case 'h':
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
  out_name << "diffuse.exr";

  std::string out_name_str(std::move(out_name).str());
  OEXRFile ox(out_proc, im.width, im.height);

  std::cout << "copying data back" << std::endl;
  std::cout << "writing to: " << out_name_str << std::endl;
  ox.write(out_name_str);
}

std::string generate_refl_file(std::string const& prefix, int const& i, std::string const suffix="")
{
  std::stringstream out_name;
  out_name << prefix << "refl" << i << suffix << ".exr";
  return out_name.str();
}

void map_refl_im(image_t& im, std::string const& prefix, float const& step, int const& i, std::pair<size_t, size_t> const& outSize, bool halve=false)
{
  float roughness = step * i;
  auto [outWidth, outHeight] = outSize;
  std::vector<float3> out_proc(outWidth * outHeight);

  std::string out_name_str = generate_refl_file(prefix, i,
                                                halve ? "w" : "");
  std::cout << "Mapping reflectance map..." << std::endl;
  map_reflectance_ker(im.im, out_proc.data(), im.width, im.height, roughness, outWidth, outHeight);
  OEXRFile ox(out_proc, outWidth, outHeight);
  std::cout << "copying data back" << std::endl;
  std::cout << "writing to: " << out_name_str << std::endl;
  ox.write(out_name_str);
}

void map_refl_im(image_t& im, std::string const& prefix, float const& step, int const& i)
{
  map_refl_im(im, prefix, step, i, std::pair<size_t, size_t>(im.width, im.height));
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
  if (stat(directory.c_str(), &st) == -1)
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
      std::cout << "Image dimensions: " << width << "x" << height << std::endl;
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
  outss << args.directory << "/";


  std::string outprefix(outss.str());

  image_t imt(im_proc.data(), width, height);

  if (args.mode == 'i' || args.mode == 'a')
    {
      irradiate_im(imt, outprefix);
    }
  if (args.mode == 'b' || args.mode == 'a')
    {
      generate_brdf_refl(200, args.directory);
    }
  if (args.mode == 'r' || args.mode == 'a')
    {
      copy_file(args.file_in, generate_refl_file(outprefix, 0));

      for(size_t halve=0; halve < 2; ++halve) {
        size_t count=halve ? 8 : 10;
        float step = 1.0f / count;

        unsigned int out_width = imt.width;
        unsigned int out_height = imt.height;

        for (size_t i = 1; i <= count; ++i)
          {
            if (halve && out_width >= MIN_WIDTH && out_height >= MIN_HEIGHT)
              {
                out_width = out_width >> 1;
                out_height = out_height >> 1; // halve
              }
            map_refl_im(imt, outprefix, step, i, std::pair<size_t,size_t>(out_width, out_height), halve);
          }
      }
    }
}
