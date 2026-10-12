/**
 * @file openglshim.cc
 * C-exported entry point for the OpenGL renderer shared library.
 * This is compiled into libasyopengl.so alongside glrender.o and GLTextures.o.
 */

#include "glrender.h"

extern "C" {

/**
 * Create and return a new AsyGLRender instance.
 * The caller (rendererloader.cc) receives this as a void* and casts it
 * back to the appropriate type.  Returns NULL on failure.
 */
void *createAsyGLRender()
{
#ifdef HAVE_GL
    try {
        return new camp::AsyGLRender();
    } catch (...) {
        return nullptr;
    }
#else
    return nullptr;
#endif
}

} // extern "C"
