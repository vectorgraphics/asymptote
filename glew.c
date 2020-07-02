#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#ifdef __MSDOS__
#define _WIN32
#endif

int offscreen=0;

#ifdef HAVE_LIBGL

#include <GL/glew.h>

#ifdef HAVE_LIBOSMESA
#define GLEW_OSMESA
#  define GLAPI extern
#  include <GL/osmesa.h>
#  undef GLAPI
#define OSMesaGetProcAddress getProcAddress
#  include <GL/glxew.h>

typedef void (*GLXextFuncPtr)(void);

GLXextFuncPtr getProcAddress(const char* name);
#endif

#include "GL/glew.c"

#ifdef HAVE_LIBOSMESA
#undef OSMesaGetProcAddress

GLXextFuncPtr getProcAddress(const char* name) {
  return offscreen ? OSMesaGetProcAddress((const char *) name) :
    (*glXGetProcAddressARB)((const GLubyte *) name);
}
#endif

#endif /* HAVE_LIBGL */
