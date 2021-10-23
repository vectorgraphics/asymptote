// shader handling
// Author: Supakorn "Jamie" Rassameemasmuang

#include "common.h"

#ifdef HAVE_GL

#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <iostream>

#include "shaders.h"

GLuint compileAndLinkShader(std::vector<ShaderfileModePair> const& shaders,
                            std::vector<std::string> const& defineflags,
  bool compute)
{
  GLuint shader = glCreateProgram();
  std::vector<GLuint> compiledShaders;

  size_t n=shaders.size();
  for(size_t i=0; i < n; ++i) {
    GLint newshader=createShaderFile(shaders[i].first,shaders[i].second,
                                     defineflags,compute);
    glAttachShader(shader,newshader);
    compiledShaders.push_back(newshader);
  }

  glBindAttribLocation(shader,positionAttrib,"position");
  glBindAttribLocation(shader,normalAttrib,"normal");
  glBindAttribLocation(shader,materialAttrib,"material");
  glBindAttribLocation(shader,colorAttrib,"color");
  glBindAttribLocation(shader,widthAttrib,"width");

  glLinkProgram(shader);

  for(size_t i=0; i < n; ++i) {
    glDetachShader(shader,compiledShaders[i]);
    glDeleteShader(compiledShaders[i]);
  }

  return shader;
}

GLuint createShaders(GLchar const* src, int shaderType,
                     std::string const& filename, bool compute)
{
  GLuint shader=glCreateShader(shaderType);
  glShaderSource(shader, 1, &src, NULL);
  glCompileShader(shader);

  GLint status;
  glGetShaderiv(shader, GL_COMPILE_STATUS, &status);

  if(status != GL_TRUE) {
    GLint length;

    glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &length);

    std::vector<GLchar> msg(length);

    glGetShaderInfoLog(shader, length, &length, msg.data());

    size_t n=msg.size();
    for(size_t i=0; i < n; ++i) {
      std::cerr << msg[i];
    }

    std::cerr << std::endl << "GL Compile error" << std::endl;
    std::cerr << src << std::endl;
    exit(-1);
  }
  return shader;
}

GLuint createShaderFile(std::string file, int shaderType,
                        std::vector<std::string> const& defineflags,
                        bool compute)
{
  std::ifstream shaderFile;
  shaderFile.open(file.c_str());
  std::stringstream shaderSrc;

#ifdef __APPLE__
#define GLSL_VERSION "410"
#else
#ifdef HAVE_SSBO
#define GLSL_VERSION "150"
#else
#define GLSL_VERSION "130"
#endif
#endif

  shaderSrc << "#version " << GLSL_VERSION << "\n";
#ifndef __APPLE__
  shaderSrc << "#extension GL_ARB_uniform_buffer_object : enable" << "\n";
#ifdef HAVE_SSBO
  shaderSrc << "#extension GL_ARB_shader_storage_buffer_object : enable" << "\n";
  if(compute)
    shaderSrc << "#extension GL_ARB_compute_shader : enable" << "\n";
#endif
#endif

  size_t n=defineflags.size();
  for(size_t i=0; i < n; ++i)
    shaderSrc << "#define " << defineflags[i] << "\n";

  if(shaderFile) {
    shaderSrc << shaderFile.rdbuf();
    shaderFile.close();
  } else {
    std::cerr << "Cannot read from shader file " << file << std::endl;
    exit(-1);
  }

  return createShaders(shaderSrc.str().data(),shaderType,file,compute);
}
#endif
