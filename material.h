#ifndef MATERIAL_STRUCT
#define MATERIAL_STRUCT
#ifdef HAVE_GL

#include <glm/glm.hpp>

namespace camp {

inline bool operator < (const glm::vec4& m1, const glm::vec4& m2) {
  return m1[0] < m2[0] || 
                 (m1[0] == m2[0] &&
                  (m1[1] < m2[1] || 
                   (m1[1] == m2[1] &&
                    (m1[2] < m2[2] || 
                     (m1[2] == m2[2] &&
                      (m1[3] < m2[3]))))));
}

inline glm::vec4 GLparameters(GLfloat shininess, GLfloat metallic,
                             GLfloat fresnel0) {
  return glm::vec4(shininess,metallic,fresnel0,0.0);
}

struct Material {
public:
  glm::vec4 diffuse,emissive,specular;
  glm::vec4 parameters;
  Material() {}

  Material(const glm::vec4& diffuse, const glm::vec4& emissive,
           const glm::vec4& specular, double shininess, double metallic, double fresnel0) : 
    diffuse(diffuse), emissive(emissive), specular(specular),
    parameters(GLparameters(shininess,metallic,fresnel0)) {}

  Material(Material const& m):
    diffuse(m.diffuse), emissive(m.emissive),
    specular(m.specular), parameters(m.parameters) {}
  ~Material() {}

  Material& operator=(Material const& m)
  {
    diffuse=m.diffuse;
    emissive=m.emissive;
    specular=m.specular;
    parameters=m.parameters;
    return *this; 
  }
  friend bool operator < (const Material& m1, const Material& m2) {
    return m1.diffuse < m2.diffuse ||
                        (m1.diffuse == m2.diffuse && 
                         (m1.emissive < m2.emissive ||
                        (m1.emissive == m2.emissive && 
                         (m1.specular < m2.specular ||
                        (m1.specular == m2.specular && 
                         (m1.parameters < m2.parameters))))));
  }
}; 

extern size_t Nmaterials; // Number of materials compiled in shader
extern size_t nmaterials; // Current size of materials buffer
extern size_t Maxmaterials; // Maxinum size of materials buffer
void clearMaterialBuffer(bool draw=false);

}
#endif
#endif
