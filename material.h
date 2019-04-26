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

struct Material {
public:
  glm::vec4 diffuse, ambient, emissive, specular;
  GLfloat shininess; 
  GLfloat padding[3];

  Material() {}

  Material(const glm::vec4& diffuse, const glm::vec4& ambient,
           const glm::vec4& emissive, const glm::vec4& specular,
           double shininess) : 
    diffuse(diffuse), ambient(ambient), emissive(emissive), specular(specular),
    shininess(128*shininess) {}

  Material(Material const& m):
    diffuse(m.diffuse), ambient(m.ambient), emissive(m.emissive),
    specular(m.specular), shininess(m.shininess) {}
  ~Material() {}

  Material& operator=(Material const& m)
  {
    diffuse=m.diffuse;
    ambient=m.ambient;
    emissive=m.emissive;
    specular=m.specular;
    shininess=m.shininess;
    return *this; 
  }
      
  friend bool operator < (const Material& m1, const Material& m2) {
    return m1.diffuse < m2.diffuse ||
                        (m1.diffuse == m2.diffuse && 
                         (m1.ambient < m2.ambient ||
                        (m1.ambient == m2.ambient && 
                         (m1.emissive < m2.emissive ||
                        (m1.emissive == m2.emissive && 
                         (m1.specular < m2.specular ||
                        (m1.specular == m2.specular && 
                         (m1.shininess < m2.shininess))))))));
  }
      
}; 

extern size_t Nmaterials; // Number of materials compiled in shader
extern size_t nmaterials; // Current size of materials buffer
extern size_t Maxmaterials; // Maxinum size of materials buffer
void clearMaterialBuffer(bool draw=false);

}
#endif
#endif
