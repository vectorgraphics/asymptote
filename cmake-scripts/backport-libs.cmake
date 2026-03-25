add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/backports/optional)

list(APPEND ASY_STATIC_LIBRARIES OptionalBackport)

if (ENABLE_OPENGL)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/backports/glew)
    # using glew within the repo
    list(APPEND ASY_STATIC_LIBRARIES GLEW)

    list(APPEND ASYMPTOTE_INCLUDES $<TARGET_PROPERTY:GLEW,INCLUDE_DIRECTORIES>)
endif()
