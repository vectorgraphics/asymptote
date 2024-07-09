#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#ifdef __MSDOS__
#define _WIN32
#endif

#ifndef GLEW_INCLUDE
#include <GL/glew.h>
#else
#include GLEW_INCLUDE
#endif

#ifdef HAVE_LIBGL
#ifdef HAVE_LIBOSMESA
#define GLEW_OSMESA
#define APIENTRY
#endif

#include "GL/glew.c"
#endif /* HAVE_LIBGL */
