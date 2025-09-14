#pragma once

#include <chrono>
#include <cmath>
#include <utility>
#include <memory>
#include <set>
#include <sstream>
#include <stdlib.h>
#include <string>
#include <unordered_map>
#include <vector>
#include <array>

#include "glmCommon.h"
#include "material.h"
#include "triple.h"

namespace camp
{

static const double pixelResolution=1.0; // Adaptive rendering constant.
extern size_t materialIndex;

typedef std::map<const Material, size_t> MaterialMap;

template<class T>
inline void extendOffset(std::vector<T>& a, const std::vector<T>& b, T offset)
{
  size_t n = a.size();
  size_t m = b.size();
  a.resize(n + m);
  for (size_t i = 0; i < m; ++i)
    a[n + i] = b[i] + offset;
}

#define POSITION_LOCATION 0
#define NORMAL_LOCATION   1
#define MATERIAL_LOCATION 2
#define COLOR_LOCATION    3
#define WIDTH_LOCATION    4

struct MaterialVertex
{
  glm::vec3 position;
  glm::vec3 normal;
  glm::i32 material;

#ifdef HAVE_VULKAN
  static vk::VertexInputBindingDescription getBindingDescription()
  {
    return vk::VertexInputBindingDescription(0, sizeof(MaterialVertex), vk::VertexInputRate::eVertex);
  }

  static std::vector<vk::VertexInputAttributeDescription> getAttributeDescriptions(bool count = false)
  {
    std::vector<vk::VertexInputAttributeDescription> attributeDescriptions;
    attributeDescriptions.push_back(
            vk::VertexInputAttributeDescription(POSITION_LOCATION, 0, vk::Format::eR32G32B32Sfloat, offsetof(MaterialVertex, position)));

    if (!count) {
      attributeDescriptions.push_back(
            vk::VertexInputAttributeDescription(NORMAL_LOCATION, 0, vk::Format::eR32G32B32Sfloat, offsetof(MaterialVertex, normal)));
      attributeDescriptions.push_back(
            vk::VertexInputAttributeDescription(MATERIAL_LOCATION, 0, vk::Format::eR32Sint, offsetof(MaterialVertex, material)));
    }
    return attributeDescriptions;
  }
#endif
};

struct ColorVertex
{
  glm::vec3 position;
  glm::vec3 normal;
  glm::i32 material;
  glm::vec4 color;

#ifdef HAVE_VULKAN
  static vk::VertexInputBindingDescription getBindingDescription()
  {
    return vk::VertexInputBindingDescription(0, sizeof(ColorVertex), vk::VertexInputRate::eVertex);
  }

  static std::vector<vk::VertexInputAttributeDescription> getAttributeDescriptions(bool count = false)
  {
    std::vector<vk::VertexInputAttributeDescription> attributeDescriptions;
    attributeDescriptions.push_back(
            vk::VertexInputAttributeDescription(POSITION_LOCATION, 0, vk::Format::eR32G32B32Sfloat, offsetof(ColorVertex, position)));

    if (!count) {
      attributeDescriptions.push_back(
            vk::VertexInputAttributeDescription(NORMAL_LOCATION, 0, vk::Format::eR32G32B32Sfloat, offsetof(ColorVertex, normal)));
      attributeDescriptions.push_back(
            vk::VertexInputAttributeDescription(MATERIAL_LOCATION, 0, vk::Format::eR32Sint, offsetof(ColorVertex, material)));
      attributeDescriptions.push_back(
            vk::VertexInputAttributeDescription(COLOR_LOCATION, 0, vk::Format::eR32G32B32A32Sfloat, offsetof(ColorVertex, color)));
    }
    return attributeDescriptions;
  }
#endif
};

struct PointVertex
{
  glm::vec3 position;
  glm::f32 width;
  glm::i32 material;

#ifdef HAVE_VULKAN
  static vk::VertexInputBindingDescription getBindingDescription()
  {
    return vk::VertexInputBindingDescription(0, sizeof(PointVertex), vk::VertexInputRate::eVertex);
  }

  static std::vector<vk::VertexInputAttributeDescription> getAttributeDescriptions(bool count = false)
  {
    std::vector<vk::VertexInputAttributeDescription> attributeDescriptions;
    attributeDescriptions.push_back(
            vk::VertexInputAttributeDescription(POSITION_LOCATION, 0, vk::Format::eR32G32B32Sfloat, offsetof(PointVertex, position)));

    // Always include width for points
    attributeDescriptions.push_back(
            vk::VertexInputAttributeDescription(WIDTH_LOCATION, 0, vk::Format::eR32Sfloat, offsetof(PointVertex, width)));

    if (!count) {
      attributeDescriptions.push_back(
            vk::VertexInputAttributeDescription(MATERIAL_LOCATION, 0, vk::Format::eR32Sint, offsetof(PointVertex, material)));
    }
    return attributeDescriptions;
  }
#endif
};

struct VertexBuffer {
  std::vector<MaterialVertex> materialVertices;
  std::vector<ColorVertex> colorVertices;
  std::vector<PointVertex> pointVertices;
  std::vector<std::uint32_t> indices;

  int renderCount=0;  // Are all patches in this buffer fully rendered?
  bool copiedThisFrame=false;

  void clear()
  {
    materialVertices.clear();
    colorVertices.clear();
    pointVertices.clear();
    indices.clear();
  }

  std::uint32_t addVertex(const MaterialVertex& vertex)
  {
    std::uint32_t nvertices = materialVertices.size();
    materialVertices.push_back(vertex);
    return nvertices;
  }

  std::uint32_t addVertex(const ColorVertex& vertex)
  {
    std::uint32_t nvertices = colorVertices.size();
    colorVertices.push_back(vertex);
    return nvertices;
  }

  std::uint32_t addVertex(const PointVertex& vertex)
  {
    std::uint32_t nvertices = pointVertices.size();
    pointVertices.push_back(vertex);
    return nvertices;
  }

  void extendMaterial(const VertexBuffer& other)
  {
    extendOffset<std::uint32_t>(indices, other.indices, materialVertices.size());
    materialVertices.insert(materialVertices.end(), other.materialVertices.begin(), other.materialVertices.end());
  }

  void extendColor(const VertexBuffer& other)
  {
    extendOffset<std::uint32_t>(indices, other.indices, colorVertices.size());
    colorVertices.insert(colorVertices.end(), other.colorVertices.begin(), other.colorVertices.end());
  }

  void extendPoint(const VertexBuffer& other)
  {
    extendOffset<std::uint32_t>(indices, other.indices, pointVertices.size());
    pointVertices.insert(pointVertices.end(), other.pointVertices.begin(), other.pointVertices.end());
  }
};

extern VertexBuffer materialData;    // material Bezier patches & triangles
extern VertexBuffer colorData;       // colored Bezier patches & triangles
extern VertexBuffer triangleData;    // opaque indexed triangles
extern VertexBuffer transparentData; // transparent patches & triangles

extern VertexBuffer pointData;       // pixels
extern VertexBuffer lineData;        // material Bezier curves

extern glm::dmat4 projViewMat;
extern glm::dmat4 normMat;

inline triple billboardTransform(const triple& center, const triple& v)
{
  double cx = center.getx();
  double cy = center.gety();
  double cz = center.getz();

  double x = v.getx() - cx;
  double y = v.gety() - cy;
  double z = v.getz() - cz;

  const double* BBT = glm::value_ptr(normMat);

  return triple(x * BBT[0] + y * BBT[4] + z * BBT[8] + cx,
                x * BBT[1] + y * BBT[5] + z * BBT[9] + cy,
                x * BBT[2] + y * BBT[6] + z * BBT[10] + cz);
}

} // namespace camp
