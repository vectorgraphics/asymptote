/**
 * @file framebuffer.cc
 * @author Supakorn "Jamie" Rassameemasmuang <rassamee@ualberta.ca>
 * @date June 27, 2019
 * 
 * Renders FrameBuffer
 */
#include "framebuffer.h"
#include <vector>

namespace outFrameBuffer {
    // returns (fbo, texture color buffer, depth stencil)
    GLuintTriple createFrameBuffer(uint width, uint height, uint textureunit) {
        GLuint fbo, texcolbuffer, rboDepthStencil;
        glGenFramebuffers(1, &fbo);
        glBindFramebuffer(GL_FRAMEBUFFER, fbo);

        glActiveTexture(GL_TEXTURE0+textureunit);
        glGenTextures(1, &texcolbuffer);
        glBindTexture(GL_TEXTURE_2D, texcolbuffer);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB12, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, nullptr);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texcolbuffer, 0);

        glGenerateMipmap(GL_TEXTURE_2D);

        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, 0);


        glGenRenderbuffers(1, &rboDepthStencil);
        glBindRenderbuffer(GL_RENDERBUFFER, rboDepthStencil);
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, width, height);
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, rboDepthStencil);

        glBindTexture(GL_TEXTURE_2D, 0);
        return GLuintTriple(fbo, GLuintPair(texcolbuffer, rboDepthStencil));
    }

    GLuintTriple createFrameBufferMultiSample(uint width, uint height, uint numSamples) {
        GLuint fbo, tcb, rds;
        glGenFramebuffers(1, &fbo);
        glBindFramebuffer(GL_FRAMEBUFFER, fbo);

        glGenTextures(1, &tcb);
        glBindTexture(GL_TEXTURE_2D_MULTISAMPLE, tcb);
        glTexImage2DMultisample(GL_TEXTURE_2D_MULTISAMPLE, numSamples, GL_RGB12, width, height, true);
        glBindTexture(GL_TEXTURE_2D_MULTISAMPLE, 0);

        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D_MULTISAMPLE, tcb, 0);

        glGenRenderbuffers(1, &rds);
        glBindRenderbuffer(GL_RENDERBUFFER, rds);
        glRenderbufferStorageMultisample(GL_RENDERBUFFER, numSamples, GL_DEPTH24_STENCIL8, width, height);
        glBindRenderbuffer(GL_RENDERBUFFER, 0);

        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, rds);

        glBindTexture(GL_TEXTURE_2D_MULTISAMPLE, 0);  

        return GLuintTriple(fbo, GLuintPair(tcb, rds));
    }


    // returns vao of the object.
    GLuint createoutputVAO(GLuint outputShader) {
        std::vector<float> scrverts = {
                -1, -1, 0, 0, 
                1, -1, 1 ,0, 
                1, 1, 1 ,1, 

                -1, -1, 0, 0, 
                -1, 1, 0 ,1, 
                1, 1, 1 ,1,
                };
        GLuint vao, vbo;
        glGenVertexArrays(1, &vao);
        glBindVertexArray(vao);

        glGenBuffers(1,&vbo);
        glBindBuffer(GL_ARRAY_BUFFER,vbo);
        glBufferData(GL_ARRAY_BUFFER,scrverts.size()*sizeof(float),scrverts.data(),GL_STATIC_DRAW);

        GLint fbPosAttrib = glGetAttribLocation(outputShader, "position");
        GLint fbTextCoordAttrib = glGetAttribLocation(outputShader, "texcoord");

        glVertexAttribPointer(fbPosAttrib, 2, GL_FLOAT, GL_FALSE, 4*sizeof(float), 0);
        glVertexAttribPointer(fbTextCoordAttrib, 2, GL_FLOAT, GL_FALSE, 4*sizeof(float), (void*)(2*sizeof(float)));

        glEnableVertexAttribArray(fbPosAttrib);
        glEnableVertexAttribArray(fbTextCoordAttrib);

        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glBindVertexArray(0);

        return vao;
    }

    void renderBuffer(GLuint outputShader, GLuint vao, GLuint textureFboTarget,
        intpair renderRes, intpair screenRes, bool disablePostProcessAA, uint textureNumber) {

        glUseProgram(outputShader);
        glBindVertexArray(vao);

        GLuint finalTextureNumber = GL_TEXTURE0 + textureNumber;

        glActiveTexture(finalTextureNumber);
        glBindTexture(GL_TEXTURE_2D, textureFboTarget);
        glGenerateMipmap(GL_TEXTURE_2D);

        GLint screenResUnif=glGetUniformLocation(outputShader, "screenResolution");
        GLint renderResUnif=glGetUniformLocation(outputShader, "renderResolution");
        glUniform2i(screenResUnif, screenRes.first, screenRes.second);
        glUniform2i(renderResUnif, renderRes.first , renderRes.second);

        GLint fbTextureUni = glGetUniformLocation(outputShader, "texFrameBuffer");
        glUniform1i(fbTextureUni, textureNumber);

        glUniform1i(glGetUniformLocation(outputShader, "forceNoPostProcessAA"),
            disablePostProcessAA ? 1 : 0);

        glDisable(GL_DEPTH_TEST);
        glDrawArrays(GL_TRIANGLES,0,6);

        glBindVertexArray(0);
        glUseProgram(0);
        glEnable(GL_DEPTH_TEST);
    }

    void deleteFrameBuffer(GLuint fbo, GLuint texture, GLuint stencil) {
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glDeleteTextures(1, &texture);
        glDeleteRenderbuffers(1, &stencil);

        glDeleteFramebuffers(1, &fbo);
    }
}