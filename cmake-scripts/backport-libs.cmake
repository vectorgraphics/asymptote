add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/backports/optional)

list(APPEND ASY_STATIC_LIBARIES OptionalBackport)

if (ENABLE_OPENGL)
    if (ENABLE_GL_OFFSCREEN_RENDERING)
        set(GLEW_ENABLE_OFFSCREEN 1)
    endif()
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/backports/glew)
    # using glew within the repo
    list(APPEND ASY_STATIC_LIBARIES GLEW)

    list(APPEND ASYMPTOTE_INCLUDES $<TARGET_PROPERTY:GLEW,INCLUDE_DIRECTORIES>)
endif()
