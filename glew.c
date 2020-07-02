#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#ifdef __MSDOS__
#define _WIN32
#endif

#ifdef HAVE_LIBGL

#include <GL/glew.h>

#ifdef HAVE_LIBOSMESA
#define GLEW_OSMESA
#  define GLAPI extern
#  include <GL/osmesa.h>
#define OSMesaGetProcAddress getProcAddress
#  undef GLAPI
#  include <GL/glxew.h>
#endif

int offscreen=0;

typedef void (*GLXextFuncPtr)(void);

GLXextFuncPtr getProcAddress(const char* name);

#include "GL/glew.c"

#undef OSMesaGetProcAddress

GLXextFuncPtr getProcAddress(const char* name) {
  return offscreen ? OSMesaGetProcAddress((const char *) name) :
    (*glXGetProcAddressARB)((const GLubyte *) name);
}

#endif /* HAVE_LIBGL */
