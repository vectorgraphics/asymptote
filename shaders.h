#ifndef __TOGL_SHADERSPROC
#define __TOGL_SHADERSPROC

#define GLEW_STATIC
#define GLEW_NO_GLU

#ifdef __APPLE__
#include <OpenGL/glew.h>
#else
#include <GL/glew.h>
#ifdef __MSDOS__
#include<windows.h>
#include <GL/wglew.h>
#include <GL/wglext.h>
#endif
#endif

#include <string>
#include <unordered_set>

GLuint createShaders(GLchar const *src, int shaderType);
GLuint createShaderFile(std::string file, int shaderType, size_t Nlights,
                        size_t Nmaterials,  bool explicitcolor=false);
#endif
