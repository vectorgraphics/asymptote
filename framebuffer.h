/**
 * @file framebuffer.cc
 * @author Supakorn "Jamie" Rassameemasmuang <rassamee@ualberta.ca>
 * 
 * Renders FrameBuffer
 */
#include "common.h"
#include <GL/glew.h>

#define TEXTURE_NUMBER 2

namespace outFrameBuffer {
    typedef std::pair<int, int> intpair;
    typedef std::pair<GLuint, GLuint> GLuintPair;
    typedef std::pair<GLuint, GLuintPair> GLuintTriple;

    /**
     * Creates a new Frame Buffer
     * 
     * @param height Height of the final image
     * @param width Width of the final image
     * @param resfactor Resolution scale of width and height.
     * 
     * @returns A triple of (fbo, texture color, depth stencil buffer)
     */
    GLuintTriple createFrameBuffer(uint width, uint height, uint textureunit=TEXTURE_NUMBER);

    GLuintTriple createFrameBufferMultiSample(uint width, uint height, uint numSaples);
    /**
     * Creates the output VAO of the final drawing.
     * 
     * @param outputShader Output Shader to bind attribuites to.
     * 
     * @returns The output VAO
     */
    GLuint createoutputVAO(GLuint outputShader);

    /**
     * Renders the buffer.
     * 
     */
    void renderBuffer(GLuint outputShader, GLuint vao, GLuint textureFboTarget,
        intpair renderRes, intpair screenRes, bool disablePostProcessAA,
        uint textureNumber=TEXTURE_NUMBER);

    void deleteFrameBuffer(GLuint fbo, GLuint texture, GLuint stencil);
}