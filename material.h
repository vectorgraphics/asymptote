#ifndef MATERIAL_STRUCT
#define MATERIAL_STRUCT
#ifdef HAVE_GL

#include <glm/glm.hpp>

namespace camp {
    struct Material {
public:
  glm::vec4 diffuse, specular, ambient, emission;
  float shininess; 

  Material() {}

  Material(glm::vec4 diff, glm::vec4 spec, glm::vec4 amb, glm::vec4 emissive, float shininess):
    diffuse(diff), specular(spec), ambient(amb), emission(emissive), shininess(shininess) {}

  Material(Material const& other):
    diffuse(other.diffuse), specular(other.specular), ambient(other.ambient), emission(other.emission), shininess(other.shininess) {}
  ~Material() {}

  Material& operator=(Material const& other)
  {
    if(&other!=this) {
      diffuse=other.diffuse;
      specular=other.specular;
      ambient=other.ambient;
      emission=other.emission;
      shininess=other.shininess;
    }
    return *this; 
  }
}; 
}
#endif
#endif