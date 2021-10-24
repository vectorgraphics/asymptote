#ifndef __TOGL_SHADERSPROC
#define __TOGL_SHADERSPROC

#define GLEW_NO_GLU

#ifdef __MSDOS__
#define GLEW_STATIC
#define _WIN32
#endif

#include "GL/glew.h"

#ifdef __MSDOS__
#undef _WIN32
#include <GL/wglew.h>
#include <GL/wglext.h>
#endif

#include <string>

typedef std::pair<std::string, int> ShaderfileModePair;

GLuint compileAndLinkShader(
  std::vector<ShaderfileModePair> const& shaders,
  std::vector<std::string> const& defineflags, bool compute=false);

GLuint createShaders(const std::string& src, int shaderType,
                     std::string const& filename, bool compute=false);

GLuint createShaderFile(std::string file, int shaderType,
                        std::vector<std::string> const& constflags,
                        bool compute=false);

enum attrib {positionAttrib=0,normalAttrib,materialAttrib,colorAttrib,
             widthAttrib};

#endif
