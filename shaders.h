#ifndef __TOGL_SHADERSPROC
#define __TOGL_SHADERSPROC

#include <GL/gl.h>
#include <string>

GLuint createShaders(GLchar const *src, int shaderType);
GLuint createShaderFile(std::string file, int shaderType);
#endif