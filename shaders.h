#ifndef __TOGL_SHADERSPROC
#define __TOGL_SHADERSPROC

#define GLEW_NO_GLU

#include "GL/glew.h"

#ifdef __MSDOS__
#define GLEW_STATIC
#include<windows.h>
#include <GL/wglew.h>
#include <GL/wglext.h>
#endif

#include <string>

typedef std::pair<std::string, int> ShaderfileModePair;

GLuint compileAndLinkShader(
  std::vector<ShaderfileModePair> const& shaders, 
  size_t NLights, size_t NMaterials,
  std::vector<std::string> const& defineflags);

GLuint createShaders(GLchar const *src, int shaderType,
                     std::string const& filename);

GLuint createShaderFile(std::string file, int shaderType, size_t Nlights,
                        size_t Nmaterials,
                        std::vector<std::string> const& constflags);
#endif
