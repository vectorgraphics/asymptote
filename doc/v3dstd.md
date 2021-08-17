# Draft V3D Standard
# Compress with gzip -9

unsigned int version
unsigned int type [material,transform,element,
                         vertex,
#MATERIAL ELEMENTS:
                         line(2),triangle(3),quad(4),
                         curve(4),bezierTriangle(10),bezierPatch(16),
                         triangles,
                         disk,cylinder,tube,sphere,halfSphere,

#UNFILL ELEMENTS:        triangle,quad,
                         curve,bezierTriangle,bezierPatch,

                         element2D,
                         animations,

#COLORED:
                         vertexColor,
                         lineColor,triangleColor,quadColor,curveColor,
                         bezierTriangleColor,bezierPatchColor]
MATERIAL:
unsigned int index
float[4] diffuse
float[4] emissive
float[4] specular
float shininess
float metallic
float fresnel0

TRANSFORM:
unsigned int index
double[16] 4x4 array

VERTEX:
index
double x,y,z

COLORED VERTEX:
index
double x,y,z
float r,g,b,a

MATERIAL ELEMENT:
double control point
...
unsigned int center index
unsigned int material index

ELEMENT
element index
material index
transform index
