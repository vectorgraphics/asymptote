#ifndef __TOGL_SHADERSPROC
#define __TOGL_SHADERSPROC

#define GLEW_NO_GLU

#ifdef __APPLE__
#include <OpenGL/glew.h>
#else
#include <GL/glew.h>
#endif

#include <string>
#include <unordered_set>

GLuint createShaders(GLchar const *src, int shaderType);
GLuint createShaderFile(std::string file, int shaderType,std::unordered_set<std::string> compilerFlags={});
#endif
